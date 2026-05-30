
import os
import time
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
# CONFIG
# =========================
DATA_PATH = 'datasets/german_credit/german.data'
LABEL = 'class'
SENSITIVE = 'sex'   # derived from personal_status_sex
RESULTS_DIR = 'results/processed'
os.makedirs(RESULTS_DIR, exist_ok=True)

GERMAN_COLUMNS = [
    'status_checking', 'duration_months', 'credit_history', 'purpose', 'credit_amount',
    'savings', 'employment_since', 'installment_rate', 'personal_status_sex', 'other_debtors',
    'present_residence_since', 'property', 'age_years', 'other_installment_plans', 'housing',
    'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'class'
]

class SimpleRuleListClassifier(BaseEstimator, ClassifierMixin):
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


def build_xy(df, include_sensitive=True):
    X = df.drop(columns=[LABEL]).copy()
    if not include_sensitive and SENSITIVE in X.columns:
        X = X.drop(columns=[SENSITIVE])
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
        return DecisionTreeClassifier(max_depth=5, random_state=42)
    elif model_type == 'logreg':
        return LogisticRegression(max_iter=500)
    elif model_type == 'rulelist':
        return SimpleRuleListClassifier(max_depth=3, random_state=42)
    else:
        raise ValueError("model_type harus tree / logreg / rulelist")


def demographic_parity_gap(y_pred, sensitive_series):
    # Fix: align indexes before boolean masking
    y_pred_s = pd.Series(y_pred).reset_index(drop=True)
    sensitive_s = pd.Series(sensitive_series).reset_index(drop=True)
    groups = sorted(sensitive_s.dropna().unique().tolist())
    rates = {}
    for g in groups:
        mask = (sensitive_s == g)
        rates[g] = float(y_pred_s[mask].mean()) if int(mask.sum()) > 0 else 0.0
    if not rates:
        return {}, None
    gap = max(rates.values()) - min(rates.values())
    return rates, gap


def run_one(model_type, include_sensitive=True):
    df = load_german()
    X, y = build_xy(df, include_sensitive=include_sensitive)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # Extra safety against index misalignment
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

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if include_sensitive:
        rates, dp_gap = demographic_parity_gap(y_pred, X_test[SENSITIVE])
    else:
        rates, dp_gap = {}, None

    return {
        'dataset': 'german_credit',
        'model': model_type,
        'include_sensitive': include_sensitive,
        'sensitive_attribute': SENSITIVE,
        'accuracy': acc,
        'f1': f1,
        'dp_gap': dp_gap,
        'dp_rates': str(rates),
        'train_time_sec': train_time
    }


if __name__ == '__main__':
    rows = []
    for model_type in ['tree', 'logreg', 'rulelist']:
        rows.append(run_one(model_type, True))
        rows.append(run_one(model_type, False))
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, 'german_summary.csv')
    out_df.to_csv(out_path, index=False, encoding='utf-8')
    print(out_df)
    print(f'[Saved] {out_path}')
