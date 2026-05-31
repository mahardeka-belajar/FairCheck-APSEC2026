
import json
import os
import numpy as np
import pandas as pd
from z3 import Solver, Real, RealVal, If, Or, sat

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# =========================
# CONFIG
# =========================
TRAIN_PATH = 'datasets/adult/adult.data'
TEST_PATH = 'datasets/adult/adult.test'
LABEL = 'income'
SENSITIVE = 'sex'
RESULT_PATH = 'results/processed/smt_adult_tree.json'
TREE_MAX_DEPTH = 5
RANDOM_STATE = 42

os.makedirs('results/processed', exist_ok=True)

ADULT_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income'
]


def load_adult():
    train_df = pd.read_csv(
        TRAIN_PATH,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values='?'
    )
    test_df = pd.read_csv(
        TEST_PATH,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values='?',
        comment='|'
    )
    test_df['income'] = test_df['income'].astype(str).str.replace('.', '', regex=False)
    train_df = train_df.dropna(subset=['income'])
    test_df = test_df.dropna(subset=['income'])
    return train_df, test_df


def build_and_fit_pipeline(train_df: pd.DataFrame):
    X = train_df.drop(columns=[LABEL])
    y = (train_df[LABEL] == '>50K').astype(int)

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_cols),
        ]
    )

    clf = DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=RANDOM_STATE)
    model = Pipeline([
        ('prep', preprocessor),
        ('clf', clf)
    ])
    model.fit(X, y)
    return model, X, y, num_cols, cat_cols


def get_feature_schema(model, num_cols, cat_cols):
    prep: ColumnTransformer = model.named_steps['prep']
    ohe: OneHotEncoder = prep.named_transformers_['cat'].named_steps['onehot']
    scaler: StandardScaler = prep.named_transformers_['num'].named_steps['scaler']
    transformed_names = prep.get_feature_names_out().tolist()

    cat_group_ranges = {}
    offset = len(num_cols)
    for col, cats in zip(cat_cols, ohe.categories_):
        start = offset
        end = start + len(cats)
        cat_group_ranges[col] = {'start': start, 'end': end, 'categories': list(map(str, cats))}
        offset = end

    numeric_meta = {}
    for i, col in enumerate(num_cols):
        numeric_meta[col] = {'index': i, 'mean': float(scaler.mean_[i]), 'scale': float(scaler.scale_[i]) if float(scaler.scale_[i]) != 0 else 1.0}

    return transformed_names, numeric_meta, cat_group_ranges


def compute_transformed_bounds(model, X):
    Xt = model.named_steps['prep'].transform(X)
    if hasattr(Xt, 'toarray'):
        Xt = Xt.toarray()
    return Xt.min(axis=0), Xt.max(axis=0)


def exactly_one(vars_group):
    return sum(vars_group) == 1


def zero_one(v):
    return Or(v == 0, v == 1)


def tree_pred_expr(tree, zvars, node_id=0):
    left = tree.children_left[node_id]
    right = tree.children_right[node_id]
    if left == right:
        value = tree.value[node_id][0]
        pred_cls = int(np.argmax(value))
        return RealVal(pred_cls)
    feat = int(tree.feature[node_id])
    thresh = float(tree.threshold[node_id])
    return If(zvars[feat] <= RealVal(thresh), tree_pred_expr(tree, zvars, left), tree_pred_expr(tree, zvars, right))


def reverse_decode_solution(numeric_meta, cat_group_ranges, vars_x, solver_model):
    decoded = {}
    for col, meta in numeric_meta.items():
        z = solver_model.eval(vars_x[meta['index']], model_completion=True)
        try:
            zv = float(z.as_decimal(12).replace('?', ''))
        except Exception:
            zv = float(z.numerator_as_long()) / float(z.denominator_as_long())
        decoded[col] = zv * meta['scale'] + meta['mean']
    for col, meta in cat_group_ranges.items():
        active = None
        for idx, cat in enumerate(meta['categories']):
            z = solver_model.eval(vars_x[meta['start'] + idx], model_completion=True)
            if str(z) == '1':
                active = cat
                break
        decoded[col] = active
    return decoded


def main():
    train_df, test_df = load_adult()
    model, X_train, y_train, num_cols, cat_cols = build_and_fit_pipeline(train_df)
    transformed_names, numeric_meta, cat_group_ranges = get_feature_schema(model, num_cols, cat_cols)
    mins, maxs = compute_transformed_bounds(model, X_train)
    tree = model.named_steps['clf'].tree_

    n_features = len(transformed_names)
    x_vars = [Real(f'x_{i}') for i in range(n_features)]
    y_vars = [Real(f'y_{i}') for i in range(n_features)]
    s = Solver()

    for col, meta in numeric_meta.items():
        i = meta['index']
        s.add(x_vars[i] >= RealVal(float(mins[i])))
        s.add(x_vars[i] <= RealVal(float(maxs[i])))
        s.add(y_vars[i] >= RealVal(float(mins[i])))
        s.add(y_vars[i] <= RealVal(float(maxs[i])))
        s.add(x_vars[i] == y_vars[i])

    for col, meta in cat_group_ranges.items():
        xs = [x_vars[i] for i in range(meta['start'], meta['end'])]
        ys = [y_vars[i] for i in range(meta['start'], meta['end'])]
        for v in xs + ys:
            s.add(zero_one(v))
        s.add(exactly_one(xs))
        s.add(exactly_one(ys))
        if col != SENSITIVE:
            for xv, yv in zip(xs, ys):
                s.add(xv == yv)

    pred_x = tree_pred_expr(tree, x_vars, 0)
    pred_y = tree_pred_expr(tree, y_vars, 0)

    sensitive_meta = cat_group_ranges[SENSITIVE]
    cats = sensitive_meta['categories']
    pair_results = []
    found_violation = False
    sat_counterexample = None

    for src in cats:
        for dst in cats:
            if src == dst:
                continue
            local = Solver()
            local.add(s.assertions())
            for idx, cat in enumerate(cats):
                xv = x_vars[sensitive_meta['start'] + idx]
                yv = y_vars[sensitive_meta['start'] + idx]
                local.add(xv == (1 if cat == src else 0))
                local.add(yv == (1 if cat == dst else 0))
            local.add(pred_x != pred_y)
            status = local.check()
            pair_info = {'source_group': src, 'target_group': dst, 'status': str(status)}
            if status == sat and not found_violation:
                found_violation = True
                m = local.model()
                pair_info['prediction_x'] = str(m.eval(pred_x, model_completion=True))
                pair_info['prediction_y'] = str(m.eval(pred_y, model_completion=True))
                pair_info['example_x'] = reverse_decode_solution(numeric_meta, cat_group_ranges, x_vars, m)
                pair_info['example_y'] = reverse_decode_solution(numeric_meta, cat_group_ranges, y_vars, m)
                sat_counterexample = pair_info
            pair_results.append(pair_info)

    result = {
        'dataset': 'adult',
        'model': 'Decision Tree',
        'sensitive_attribute': SENSITIVE,
        'tree_max_depth': TREE_MAX_DEPTH,
        'encoding_scope': 'exact tree over transformed feature space (one-hot categoricals + standardized numerics)',
        'fairness_property': 'existence of two inputs equal on all non-sensitive transformed features, different on sex, with different model predictions',
        'overall_status': 'SAT' if found_violation else 'UNSAT',
        'pair_checks': pair_results,
        'first_counterexample': sat_counterexample,
        'n_train_rows': int(len(train_df)),
        'n_test_rows': int(len(test_df)),
    }
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'saved_to': RESULT_PATH, 'overall_status': result['overall_status'], 'checked_pairs': len(pair_results)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
