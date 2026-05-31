
import json
import os
import numpy as np
import pandas as pd
from z3 import Solver, Real, RealVal, If, Or, sat, And

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

DATA_PATH = 'datasets/german_credit/german.data'
LABEL = 'class'
SENSITIVE = 'sex'
RESULT_PATH = 'results/processed/smt_german_rulelist.json'
RULELIST_MAX_DEPTH = 3
RANDOM_STATE = 42
TEST_SIZE = 0.3
os.makedirs('results/processed', exist_ok=True)

GERMAN_COLUMNS = [
    'status_checking', 'duration_months', 'credit_history', 'purpose', 'credit_amount',
    'savings', 'employment_since', 'installment_rate', 'personal_status_sex', 'other_debtors',
    'present_residence_since', 'property', 'age_years', 'other_installment_plans', 'housing',
    'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'class'
]

class SimpleRuleListClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3, random_state=42):
        self.max_depth = max_depth; self.random_state = random_state
    def fit(self, X, y):
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        self._tree_clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        self._tree_clf.fit(X_arr, y)
        self.default_class_ = int(np.bincount(np.asarray(y, dtype=int)).argmax())
        self.rules_ = []; self._extract_rules(0, []); return self
    def _extract_rules(self, node_id, conditions):
        tree = self._tree_clf.tree_; left = tree.children_left[node_id]; right = tree.children_right[node_id]
        if left == right:
            value = tree.value[node_id][0]; pred_class = int(np.argmax(value)); prob_pos = float(value[1] / value.sum()) if value.sum() > 0 and len(value) > 1 else float(pred_class)
            self.rules_.append((list(conditions), pred_class, prob_pos)); return
        feat = int(tree.feature[node_id]); thresh = float(tree.threshold[node_id])
        self._extract_rules(left, conditions + [(feat, '<=', thresh)]); self._extract_rules(right, conditions + [(feat, '>', thresh)])


def load_german():
    df = pd.read_csv(DATA_PATH, sep='\\s+', header=None, names=GERMAN_COLUMNS)
    male_codes = {'A91', 'A93', 'A94'}; female_codes = {'A92', 'A95'}
    df['sex'] = df['personal_status_sex'].apply(lambda x: 'male' if x in male_codes else ('female' if x in female_codes else 'unknown'))
    df['class'] = (df['class'] == 1).astype(int)
    return df


def build_and_fit_pipeline(train_df):
    X = train_df.drop(columns=[LABEL]); y = train_df[LABEL].astype(int)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist(); num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
    ])
    clf = SimpleRuleListClassifier(max_depth=RULELIST_MAX_DEPTH, random_state=RANDOM_STATE)
    model = Pipeline([('prep', pre), ('clf', clf)])
    model.fit(X, y)
    return model, X, y, num_cols, cat_cols


def get_feature_schema(model, num_cols, cat_cols):
    prep = model.named_steps['prep']; ohe = prep.named_transformers_['cat'].named_steps['onehot']; scaler = prep.named_transformers_['num'].named_steps['scaler']
    transformed_names = prep.get_feature_names_out().tolist(); cat_group_ranges = {}; offset = len(num_cols)
    for col, cats in zip(cat_cols, ohe.categories_):
        cat_group_ranges[col] = {'start': offset, 'end': offset + len(cats), 'categories': list(map(str, cats))}; offset += len(cats)
    numeric_meta = {col: {'index': i, 'mean': float(scaler.mean_[i]), 'scale': float(scaler.scale_[i]) if float(scaler.scale_[i]) != 0 else 1.0} for i, col in enumerate(num_cols)}
    return transformed_names, numeric_meta, cat_group_ranges


def compute_transformed_bounds(model, X):
    Xt = model.named_steps['prep'].transform(X)
    if hasattr(Xt, 'toarray'): Xt = Xt.toarray()
    return Xt.min(axis=0), Xt.max(axis=0)

def exactly_one(vars_group): return sum(vars_group) == 1

def zero_one(v): return Or(v == 0, v == 1)

def rule_pred_expr(rules, default_class, zvars):
    expr = RealVal(float(default_class))
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
        try: zv = float(z.as_decimal(12).replace('?', ''))
        except Exception: zv = float(z.numerator_as_long()) / float(z.denominator_as_long())
        decoded[col] = zv * meta['scale'] + meta['mean']
    for col, meta in cat_group_ranges.items():
        active = None
        for idx, cat in enumerate(meta['categories']):
            z = solver_model.eval(vars_x[meta['start'] + idx], model_completion=True)
            if str(z) == '1': active = cat; break
        decoded[col] = active
    return decoded


def main():
    df = load_german(); X_all = df.drop(columns=[LABEL]); y_all = df[LABEL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all)
    X_train = X_train.reset_index(drop=True); y_train = y_train.reset_index(drop=True); X_test = X_test.reset_index(drop=True); y_test = y_test.reset_index(drop=True)
    train_df = X_train.copy(); train_df[LABEL] = y_train
    test_df = X_test.copy(); test_df[LABEL] = y_test
    model, X_train_only, y_train_only, num_cols, cat_cols = build_and_fit_pipeline(train_df)
    transformed_names, numeric_meta, cat_group_ranges = get_feature_schema(model, num_cols, cat_cols)
    mins, maxs = compute_transformed_bounds(model, X_train_only)
    clf = model.named_steps['clf']; rules = clf.rules_; default_class = clf.default_class_
    n_features = len(transformed_names); x_vars = [Real(f'x_{i}') for i in range(n_features)]; y_vars = [Real(f'y_{i}') for i in range(n_features)]
    s = Solver()
    for col, meta in numeric_meta.items():
        i = meta['index']; s.add(x_vars[i] >= RealVal(float(mins[i]))); s.add(x_vars[i] <= RealVal(float(maxs[i]))); s.add(y_vars[i] >= RealVal(float(mins[i]))); s.add(y_vars[i] <= RealVal(float(maxs[i]))); s.add(x_vars[i] == y_vars[i])
    for col, meta in cat_group_ranges.items():
        xs = [x_vars[i] for i in range(meta['start'], meta['end'])]; ys = [y_vars[i] for i in range(meta['start'], meta['end'])]
        for v in xs + ys: s.add(zero_one(v))
        s.add(exactly_one(xs)); s.add(exactly_one(ys))
        if col != SENSITIVE:
            for xv, yv in zip(xs, ys): s.add(xv == yv)
    pred_x = rule_pred_expr(rules, default_class, x_vars); pred_y = rule_pred_expr(rules, default_class, y_vars)
    sensitive_meta = cat_group_ranges[SENSITIVE]; cats = sensitive_meta['categories']
    pair_results = []; found_violation = False; sat_counterexample = None
    for src in cats:
        for dst in cats:
            if src == dst: continue
            local = Solver(); local.add(s.assertions())
            for idx, cat in enumerate(cats):
                xv = x_vars[sensitive_meta['start'] + idx]; yv = y_vars[sensitive_meta['start'] + idx]
                local.add(xv == (1 if cat == src else 0)); local.add(yv == (1 if cat == dst else 0))
            local.add(pred_x != pred_y)
            status = local.check(); pair_info = {'source_group': src, 'target_group': dst, 'status': str(status)}
            if status == sat and not found_violation:
                found_violation = True; m = local.model()
                pair_info['prediction_x'] = str(m.eval(pred_x, model_completion=True)); pair_info['prediction_y'] = str(m.eval(pred_y, model_completion=True))
                pair_info['example_x'] = reverse_decode_solution(numeric_meta, cat_group_ranges, x_vars, m); pair_info['example_y'] = reverse_decode_solution(numeric_meta, cat_group_ranges, y_vars, m)
                sat_counterexample = pair_info
            pair_results.append(pair_info)
    result = {
        'dataset': 'german_credit', 'model': 'Rule List', 'sensitive_attribute': SENSITIVE,
        'rulelist_max_depth': RULELIST_MAX_DEPTH,
        'encoding_scope': 'exact ordered rule-list over transformed feature space (one-hot categoricals + standardized numerics)',
        'fairness_property': 'existence of two inputs equal on all non-sensitive transformed features, different on sex, with different model predictions',
        'overall_status': 'SAT' if found_violation else 'UNSAT', 'num_rules': len(rules), 'pair_checks': pair_results,
        'first_counterexample': sat_counterexample, 'n_train_rows': int(len(train_df)), 'n_test_rows': int(len(test_df))
    }
    with open(RESULT_PATH, 'w', encoding='utf-8') as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'saved_to': RESULT_PATH, 'overall_status': result['overall_status'], 'checked_pairs': len(pair_results), 'num_rules': len(rules)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
