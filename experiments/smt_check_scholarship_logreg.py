
import time
import json
import os
import numpy as np
import pandas as pd
from z3 import Solver, Real, RealVal, Or, sat

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# =========================
# CONFIG
# =========================
CSV_PATH = 'datasets/scholarship/dataset_kelayakan_beasiswa.csv'
LABEL = 'Status_Kelayakan'
SENSITIVE = 'Asal_Daerah'   # multi-valued sensitive attribute: 3T / Pedesaan / Perkotaan
LEAKAGE_COL = 'Jumlah_Beasiswa_Per_Semester'
RESULT_PATH = 'results/processed/formal/smt_scholarship_logreg.json'

os.makedirs('results/processed', exist_ok=True)
os.makedirs('results/processed/formal', exist_ok=True)

RANDOM_STATE = 42
LOGREG_MAX_ITER = 500


def load_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def build_and_fit_pipeline(df: pd.DataFrame):
    X = df.drop(columns=[LABEL, LEAKAGE_COL])
    y = (df[LABEL] == 'Layak').astype(int)

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

    clf = LogisticRegression(max_iter=LOGREG_MAX_ITER)
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
        cat_group_ranges[col] = {
            'start': start,
            'end': end,
            'categories': list(map(str, cats))
        }
        offset = end

    numeric_meta = {}
    for i, col in enumerate(num_cols):
        numeric_meta[col] = {
            'index': i,
            'mean': float(scaler.mean_[i]),
            'scale': float(scaler.scale_[i]) if float(scaler.scale_[i]) != 0 else 1.0,
        }

    return transformed_names, numeric_meta, cat_group_ranges


def compute_transformed_bounds(model, X):
    Xt = model.named_steps['prep'].transform(X)
    if hasattr(Xt, 'toarray'):
        Xt = Xt.toarray()
    mins = Xt.min(axis=0)
    maxs = Xt.max(axis=0)
    return mins, maxs


def exactly_one(vars_group):
    return sum(vars_group) == 1


def zero_one(v):
    return Or(v == 0, v == 1)


def linear_score_expr(coef, intercept, zvars):
    expr = RealVal(float(intercept))
    for i, c in enumerate(coef):
        if float(c) != 0.0:
            expr = expr + RealVal(float(c)) * zvars[i]
    return expr


def reverse_decode_solution(numeric_meta, cat_group_ranges, vars_x, solver_model):
    decoded = {}
    for col, meta in numeric_meta.items():
        z = solver_model.eval(vars_x[meta['index']], model_completion=True)
        try:
            zv = float(z.as_decimal(12).replace('?', ''))
        except Exception:
            zv = float(z.numerator_as_long()) / float(z.denominator_as_long())
        original = zv * meta['scale'] + meta['mean']
        decoded[col] = original

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
    start_time = time.perf_counter()
    df = load_data()
    model, X, y, num_cols, cat_cols = build_and_fit_pipeline(df)

    transformed_names, numeric_meta, cat_group_ranges = get_feature_schema(model, num_cols, cat_cols)
    mins, maxs = compute_transformed_bounds(model, X)

    clf: LogisticRegression = model.named_steps['clf']
    coef = clf.coef_[0]
    intercept = clf.intercept_[0]

    n_features = len(transformed_names)
    x_vars = [Real(f'x_{i}') for i in range(n_features)]
    y_vars = [Real(f'y_{i}') for i in range(n_features)]

    s = Solver()

    # Numeric constraints + equality on non-sensitive numerics
    for col, meta in numeric_meta.items():
        i = meta['index']
        s.add(x_vars[i] >= RealVal(float(mins[i])))
        s.add(x_vars[i] <= RealVal(float(maxs[i])))
        s.add(y_vars[i] >= RealVal(float(mins[i])))
        s.add(y_vars[i] <= RealVal(float(maxs[i])))
        s.add(x_vars[i] == y_vars[i])

    # Categorical constraints
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

    # Exact linear-threshold encoding in transformed feature space
    score_x = linear_score_expr(coef, intercept, x_vars)
    score_y = linear_score_expr(coef, intercept, y_vars)

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

            # Pin sensitive groups
            for idx, cat in enumerate(cats):
                xv = x_vars[sensitive_meta['start'] + idx]
                yv = y_vars[sensitive_meta['start'] + idx]
                local.add(xv == (1 if cat == src else 0))
                local.add(yv == (1 if cat == dst else 0))

            # Binary logistic threshold: class 1 iff score > 0, else class 0
            local.add(Or(
                And(score_x > RealVal(0), score_y <= RealVal(0)),
                And(score_x <= RealVal(0), score_y > RealVal(0))
            ))

            status = local.check()
            pair_info = {
                'source_group': src,
                'target_group': dst,
                'status': str(status)
            }
            if status == sat and not found_violation:
                found_violation = True
                m = local.model()
                decoded_x = reverse_decode_solution(numeric_meta, cat_group_ranges, x_vars, m)
                decoded_y = reverse_decode_solution(numeric_meta, cat_group_ranges, y_vars, m)
                sx = m.eval(score_x, model_completion=True)
                sy = m.eval(score_y, model_completion=True)
                pair_info['score_x'] = str(sx)
                pair_info['score_y'] = str(sy)
                pair_info['prediction_x'] = 1 if 'True' not in str(m.eval(score_x <= RealVal(0), model_completion=True)) else 0
                pair_info['prediction_y'] = 1 if 'True' not in str(m.eval(score_y <= RealVal(0), model_completion=True)) else 0
                pair_info['example_x'] = decoded_x
                pair_info['example_y'] = decoded_y
                sat_counterexample = pair_info
            pair_results.append(pair_info)

    runtime_seconds = round(
        time.perf_counter() - start_time,
        4
    )

    ce_found = sum(
        1
        for p in pair_results
        if p["status"] == "sat"
    )

    ce_bound = 100

    timeout = False

    result = {
        'dataset': 'scholarship',
        'model': 'Logistic Regression',
        'sensitive_attribute': SENSITIVE,
        'encoding_scope': 'exact linear threshold over transformed feature space (one-hot categoricals + standardized numerics)',
        'fairness_property': 'existence of two inputs equal on all non-sensitive transformed features, different on Asal_Daerah, with different model predictions',
        'overall_status': 'SAT' if found_violation else 'UNSAT',
        'pair_checks': pair_results,
        'runtime_seconds': runtime_seconds,
        'timeout': timeout,
        'ce_bound': ce_bound,
        'ce_found': ce_found,
        'first_counterexample': sat_counterexample,
        'logreg_intercept': float(intercept),
    }

    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        'saved_to': RESULT_PATH,
        'overall_status': result['overall_status'],
        'checked_pairs': len(pair_results)
    }, ensure_ascii=False, indent=2))


# z3 And import placed at bottom to avoid linter issues in some envs
from z3 import And

if __name__ == '__main__':
    main()
