
import json
import os
import numpy as np
import pandas as pd
from z3 import Solver, Real, RealVal, If, Or, sat

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# =========================
# CONFIG
# =========================
CSV_PATH = 'datasets/scholarship/dataset_kelayakan_beasiswa.csv'
LABEL = 'Status_Kelayakan'
SENSITIVE = 'Asal_Daerah'
LEAKAGE_COL = 'Jumlah_Beasiswa_Per_Semester'
RESULT_PATH = 'results/processed/smt_scholarship_rulelist.json'

RANDOM_STATE = 42
RULELIST_MAX_DEPTH = 3

os.makedirs('results/processed', exist_ok=True)


class SimpleRuleListClassifier(BaseEstimator, ClassifierMixin):
    """
    Ordered rule-list sederhana yang diinduksi dari shallow decision tree.
    Aturan diekstrak sebagai conjunction root-to-leaf dan diprediksi secara berurutan.
    """
    def __init__(self, max_depth=3, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        self._tree_clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        self._tree_clf.fit(X_arr, y)
        self.default_class_ = int(np.bincount(np.asarray(y, dtype=int)).argmax())
        self.rules_ = []  # list of (conditions, pred_class, prob_pos)
        self._extract_rules(0, [])
        return self

    def _extract_rules(self, node_id, conditions):
        tree = self._tree_clf.tree_
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == right:
            value = tree.value[node_id][0]
            pred_class = int(np.argmax(value))
            prob_pos = float(value[1] / value.sum()) if value.sum() > 0 and len(value) > 1 else float(pred_class)
            self.rules_.append((list(conditions), pred_class, prob_pos))
            return
        feat = int(tree.feature[node_id])
        thresh = float(tree.threshold[node_id])
        self._extract_rules(left, conditions + [(feat, '<=', thresh)])
        self._extract_rules(right, conditions + [(feat, '>', thresh)])

    def _match_rule(self, row, conditions):
        for feat, op, thresh in conditions:
            val = row[feat]
            if op == '<=' and not (val <= thresh):
                return False
            if op == '>' and not (val > thresh):
                return False
        return True

    def predict(self, X):
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        preds = []
        for row in X_arr:
            pred = self.default_class_
            for conditions, pred_class, _ in self.rules_:
                if self._match_rule(row, conditions):
                    pred = pred_class
                    break
            preds.append(pred)
        return np.array(preds)

    def predict_proba(self, X):
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        probs = []
        for row in X_arr:
            prob_pos = 1.0 if self.default_class_ == 1 else 0.0
            for conditions, _, p in self.rules_:
                if self._match_rule(row, conditions):
                    prob_pos = p
                    break
            probs.append([1 - prob_pos, prob_pos])
        return np.array(probs)


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

    clf = SimpleRuleListClassifier(max_depth=RULELIST_MAX_DEPTH, random_state=RANDOM_STATE)
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


def rule_pred_expr(rules, default_class, zvars):
    """
    Ordered rule-list encoding: first matching rule decides prediction,
    otherwise use default class.
    Each rule is list of conditions (feat, op, thresh).
    """
    expr = RealVal(float(default_class))
    # reverse fold so the first rule has highest priority in final expression
    for conditions, pred_class, _ in reversed(rules):
        cond_expr = None
        for feat, op, thresh in conditions:
            atom = (zvars[int(feat)] <= RealVal(float(thresh))) if op == '<=' else (zvars[int(feat)] > RealVal(float(thresh)))
            cond_expr = atom if cond_expr is None else And(cond_expr, atom)
        expr = If(cond_expr, RealVal(float(pred_class)), expr)
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
    df = load_data()
    model, X, y, num_cols, cat_cols = build_and_fit_pipeline(df)

    transformed_names, numeric_meta, cat_group_ranges = get_feature_schema(model, num_cols, cat_cols)
    mins, maxs = compute_transformed_bounds(model, X)

    clf: SimpleRuleListClassifier = model.named_steps['clf']
    rules = clf.rules_
    default_class = clf.default_class_

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

    pred_x = rule_pred_expr(rules, default_class, x_vars)
    pred_y = rule_pred_expr(rules, default_class, y_vars)

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

            local.add(pred_x != pred_y)
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
                pair_info['prediction_x'] = str(m.eval(pred_x, model_completion=True))
                pair_info['prediction_y'] = str(m.eval(pred_y, model_completion=True))
                pair_info['example_x'] = decoded_x
                pair_info['example_y'] = decoded_y
                sat_counterexample = pair_info
            pair_results.append(pair_info)

    result = {
        'dataset': 'scholarship',
        'model': 'Rule List',
        'sensitive_attribute': SENSITIVE,
        'rulelist_max_depth': RULELIST_MAX_DEPTH,
        'encoding_scope': 'exact ordered rule-list over transformed feature space (one-hot categoricals + standardized numerics)',
        'fairness_property': 'existence of two inputs equal on all non-sensitive transformed features, different on Asal_Daerah, with different model predictions',
        'overall_status': 'SAT' if found_violation else 'UNSAT',
        'num_rules': len(rules),
        'pair_checks': pair_results,
        'first_counterexample': sat_counterexample,
    }

    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        'saved_to': RESULT_PATH,
        'overall_status': result['overall_status'],
        'checked_pairs': len(pair_results),
        'num_rules': len(rules)
    }, ensure_ascii=False, indent=2))


from z3 import And

if __name__ == '__main__':
    main()
