
import os
import time
import json
import random
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# =========================
# CONFIG UMUM GERMAN CREDIT
# =========================
DATA_PATH = 'datasets/german_credit/german.data'
LABEL = 'class'
RANDOM_STATE = 42
RAW_DIR = 'results/raw'
PROCESSED_DIR = 'results/processed'
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

GERMAN_COLUMNS = [
    'status_checking', 'duration_months', 'credit_history', 'purpose', 'credit_amount',
    'savings', 'employment_since', 'installment_rate', 'personal_status_sex', 'other_debtors',
    'present_residence_since', 'property', 'age_years', 'other_installment_plans', 'housing',
    'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'class'
]

class SimpleRuleListClassifier(BaseEstimator, ClassifierMixin):
    """
    Ordered rule-list sederhana yang diinduksi dari shallow decision tree.
    Dipakai sebagai implementasi ringan model Rule List untuk eksperimen awal.
    """
    def __init__(self, max_depth=3, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        self._tree_clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        self._tree_clf.fit(X_arr, y)
        self.default_class_ = int(np.bincount(np.asarray(y, dtype=int)).argmax())
        self.rules_ = []
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


def load_german():
    df = pd.read_csv(DATA_PATH, sep='\s+', header=None, names=GERMAN_COLUMNS)

    male_codes = {'A91', 'A93', 'A94'}
    female_codes = {'A92', 'A95'}
    df['sex'] = df['personal_status_sex'].apply(
        lambda x: 'male' if x in male_codes else ('female' if x in female_codes else 'unknown')
    )

    # 1 = good, 2 = bad -> good=1, bad=0
    df['class'] = (df['class'] == 1).astype(int)
    return df


def build_xy(df, sensitive_col, include_sensitive=True):
    X = df.drop(columns=[LABEL]).copy()
    if not include_sensitive and sensitive_col in X.columns:
        X = X.drop(columns=[sensitive_col])
    y = df[LABEL].astype(int)
    return X, y


def make_preprocessor(X):
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_cols)
        ]
    )


def make_model(model_type):
    if model_type == 'tree':
        return DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    elif model_type == 'logreg':
        return LogisticRegression(max_iter=500)
    elif model_type == 'rulelist':
        return SimpleRuleListClassifier(max_depth=3, random_state=RANDOM_STATE)
    else:
        raise ValueError("model_type harus 'tree', 'logreg', atau 'rulelist'")


def train_model(sensitive_col, model_type='logreg', include_sensitive=True):
    df = load_german()
    X, y = build_xy(df, sensitive_col=sensitive_col, include_sensitive=include_sensitive)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    preprocessor = make_preprocessor(X_train)
    model = Pipeline([
        ('prep', preprocessor),
        ('clf', make_model(model_type))
    ])

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'train_time_sec': float(train_time),
        'test_size': int(len(X_test))
    }
    return model, X_test.reset_index(drop=True), y_test.reset_index(drop=True), metrics


def alt_groups(current_value, all_groups):
    return [g for g in all_groups if g != current_value]


def flip_to_alt_group(row_df, sensitive_col, alt_value):
    x_cf = row_df.copy()
    x_cf[sensitive_col] = alt_value
    return x_cf


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8')

# =========================
# GERMAN BASELINE 3: BOUNDARY-FOCUSED SAMPLING (3 MODEL)
# =========================
SENSITIVE = 'sex'
MODELS = ['tree', 'logreg', 'rulelist']
INCLUDE_SENSITIVE_MODEL = True
TOP_K = 300


def boundary_distances(model, X):
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[:, 1]
        return np.abs(proba - 0.5)
    raise ValueError('Model harus mendukung predict_proba untuk baseline ini.')


def run_boundary_for_model(model_type):
    model, X_test, y_test, metrics = train_model(
        sensitive_col=SENSITIVE, model_type=model_type, include_sensitive=INCLUDE_SENSITIVE_MODEL
    )
    if SENSITIVE not in X_test.columns:
        raise ValueError('Sensitive attribute tidak ada di fitur model. Set INCLUDE_SENSITIVE_MODEL=True.')
    groups = sorted(X_test[SENSITIVE].dropna().unique().tolist())
    dist = boundary_distances(model, X_test)
    order = pd.Series(dist).sort_values().index.tolist()[:min(TOP_K, len(X_test))]
    details = []
    flip_count = 0
    total_checks = 0
    start = time.time()
    for idx in order:
        x = X_test.iloc[[idx]].copy()
        orig_group = x.iloc[0][SENSITIVE]
        pred_orig = int(model.predict(x)[0])
        p_orig = float(model.predict_proba(x)[:, 1][0])
        for alt in alt_groups(orig_group, groups):
            x_cf = flip_to_alt_group(x, SENSITIVE, alt)
            pred_cf = int(model.predict(x_cf)[0])
            p_cf = float(model.predict_proba(x_cf)[:, 1][0])
            flipped = int(pred_orig != pred_cf)
            total_checks += 1
            flip_count += flipped
            details.append({
                'model': model_type,
                'row_index': int(idx),
                'boundary_distance': float(abs(p_orig - 0.5)),
                'orig_group': orig_group,
                'alt_group': alt,
                'pred_orig': pred_orig,
                'pred_cf': pred_cf,
                'proba_orig': p_orig,
                'proba_cf': p_cf,
                'flipped': flipped
            })
    runtime = time.time() - start
    summary = {
        'baseline': 'boundary_focused_sampling',
        'dataset': 'german_credit',
        'model': model_type,
        'sensitive_attribute': SENSITIVE,
        'include_sensitive_model': INCLUDE_SENSITIVE_MODEL,
        'top_k': len(order),
        'total_checks': total_checks,
        'flip_count': flip_count,
        'flip_rate': (flip_count / total_checks) if total_checks else 0.0,
        'runtime_sec': runtime,
        **metrics,
    }
    return summary, details


if __name__ == '__main__':
    all_summaries = []
    all_details = []
    for model_type in MODELS:
        summary, details = run_boundary_for_model(model_type)
        all_summaries.append(summary)
        all_details.extend(details)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    save_json(f'{RAW_DIR}/german_boundary_focused_sampling_details_3models.json', all_details)
    save_csv(f'{PROCESSED_DIR}/german_boundary_focused_sampling_summary_3models.csv', all_summaries)
    print(f'[Saved] {PROCESSED_DIR}/german_boundary_focused_sampling_summary_3models.csv')
