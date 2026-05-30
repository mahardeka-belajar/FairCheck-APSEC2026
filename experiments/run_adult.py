import os
import pandas as pd
import numpy as np
import time

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# =========================
# CONFIG
# =========================
TRAIN_PATH = "datasets/adult/adult.data"
TEST_PATH = "datasets/adult/adult.test"

LABEL = "income"
SENSITIVE = "sex"   # nanti bisa diganti ke "race"

RESULTS_DIR = "results/processed"
os.makedirs(RESULTS_DIR, exist_ok=True)

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]


# =========================
# SIMPLE RULE LIST
# =========================
class SimpleRuleListClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        self.tree_ = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        self.tree_.fit(X_arr, y)
        self.default_class_ = int(np.bincount(np.asarray(y, dtype=int)).argmax())
        self.rules_ = []
        self._extract_rules(0, [])
        return self

    def _extract_rules(self, node_id, conditions):
        tree = self.tree_.tree_
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

        self._extract_rules(left, conditions + [(feat, "<=", thresh)])
        self._extract_rules(right, conditions + [(feat, ">", thresh)])

    def _match_rule(self, row, conditions):
        for feat, op, thresh in conditions:
            val = row[feat]
            if op == "<=" and not (val <= thresh):
                return False
            if op == ">" and not (val > thresh):
                return False
        return True

    def predict(self, X):
        X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
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
        X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        probs = []
        for row in X_arr:
            prob_pos = 1.0 if self.default_class_ == 1 else 0.0
            for conditions, _, p in self.rules_:
                if self._match_rule(row, conditions):
                    prob_pos = p
                    break
            probs.append([1 - prob_pos, prob_pos])
        return np.array(probs)


# =========================
# DATA LOADING
# =========================
def load_adult():
    train_df = pd.read_csv(
        TRAIN_PATH,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values="?"
    )

    test_df = pd.read_csv(
        TEST_PATH,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values="?",
        comment="|"
    )

    # adult.test biasanya punya titik di label test, misalnya '>50K.'
    test_df["income"] = test_df["income"].astype(str).str.replace(".", "", regex=False)

    # Buang baris kosong / rusak
    train_df = train_df.dropna(subset=["income"])
    test_df = test_df.dropna(subset=["income"])

    return train_df, test_df


# =========================
# PREP
# =========================
def build_xy(train_df, test_df, include_sensitive=True):
    drop_cols = [LABEL]

    X_train = train_df.drop(columns=drop_cols).copy()
    X_test = test_df.drop(columns=drop_cols).copy()

    if not include_sensitive and SENSITIVE in X_train.columns:
        X_train = X_train.drop(columns=[SENSITIVE])
        X_test = X_test.drop(columns=[SENSITIVE])

    y_train = (train_df[LABEL] == ">50K").astype(int)
    y_test = (test_df[LABEL] == ">50K").astype(int)

    return X_train, X_test, y_train, y_test


def make_preprocessor(X_train):
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )
    return preprocessor


def make_model(model_type):
    if model_type == "tree":
        return DecisionTreeClassifier(max_depth=5, random_state=42)
    elif model_type == "logreg":
        return LogisticRegression(max_iter=500)
    elif model_type == "rulelist":
        return SimpleRuleListClassifier(max_depth=3, random_state=42)
    else:
        raise ValueError("model_type harus tree / logreg / rulelist")


def demographic_parity_gap(y_pred, sensitive_series):
    groups = sorted(pd.Series(sensitive_series).dropna().unique().tolist())
    rates = {}
    for g in groups:
        idx = (sensitive_series == g)
        rates[g] = float(pd.Series(y_pred)[idx].mean()) if idx.sum() > 0 else 0.0

    if not rates:
        return {}, None

    gap = max(rates.values()) - min(rates.values())
    return rates, gap


def run_one(model_type, include_sensitive=True):
    train_df, test_df = load_adult()

    X_train, X_test, y_train, y_test = build_xy(
        train_df, test_df, include_sensitive=include_sensitive
    )

    preprocessor = make_preprocessor(X_train)

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", make_model(model_type))
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
        "dataset": "adult",
        "model": model_type,
        "include_sensitive": include_sensitive,
        "sensitive_attribute": SENSITIVE,
        "accuracy": acc,
        "f1": f1,
        "dp_gap": dp_gap,
        "dp_rates": str(rates),
        "train_time_sec": train_time
    }


if __name__ == "__main__":
    rows = []

    for model_type in ["tree", "logreg", "rulelist"]:
        rows.append(run_one(model_type, True))
        rows.append(run_one(model_type, False))

    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "adult_summary.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(df)
    print(f"[Saved] {out_path}")