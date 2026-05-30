import os
import time
from datetime import datetime

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


# =========================
# CONFIG
# =========================
CSV_PATH = 'datasets/scholarship/dataset_kelayakan_beasiswa.csv'
LABEL = 'Status_Kelayakan'
SENSITIVE = 'Asal_Daerah'   # bisa ganti ke 'Jenis_Kelamin' kalau mau bandingkan
LEAKAGE_COL = 'Jumlah_Beasiswa_Per_Semester'

RAW_RESULTS_DIR = 'results/raw'
PROCESSED_RESULTS_DIR = 'results/processed'

os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv(CSV_PATH)
    return df


# =========================
# BUILD MODEL PIPELINE
# =========================
def build_model(df, model_type="tree", include_sensitive=True):
    drop_cols = [LABEL]

    # buang kolom leakage kalau ada
    if LEAKAGE_COL in df.columns:
        drop_cols.append(LEAKAGE_COL)

    X = df.drop(columns=drop_cols)

    # buang sensitive kalau mode blind
    if not include_sensitive and SENSITIVE in X.columns:
        X = X.drop(columns=[SENSITIVE])

    # label biner: Layak = 1, Tidak Layak = 0
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
            ]), cat_cols)
        ]
    )

    if model_type == "tree":
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)

    elif model_type == "logreg":
        clf = LogisticRegression(max_iter=500)

    else:
        raise ValueError("model_type harus 'tree' atau 'logreg'")

    model = Pipeline([
        ('prep', preprocessor),
        ('clf', clf)
    ])

    return X, y, model


# =========================
# FAIRNESS METRICS
# =========================
def demographic_parity_gap(y_pred, sensitive_series):
    groups = sorted(sensitive_series.dropna().unique().tolist())

    rates = {}
    for g in groups:
        idx = (sensitive_series == g)
        rates[g] = float(y_pred[idx].mean()) if idx.sum() > 0 else 0.0

    if not rates:
        return {}, None

    gap = max(rates.values()) - min(rates.values())
    return rates, gap


def counterfactual_flip_rate_binary(model, X, sensitive_col=SENSITIVE):
    """
    Hanya jalan jika sensitive attribute biner.
    Contoh cocok untuk: Jenis_Kelamin (Laki-laki / Perempuan)
    Kalau sensitive attribute multi-kategori (misal Asal_Daerah), hasil = None.
    """
    if sensitive_col not in X.columns:
        return None

    X1 = X.copy()
    values = sorted(X1[sensitive_col].dropna().unique().tolist())

    if len(values) != 2:
        return None

    a, b = values[0], values[1]

    pred_original = model.predict(X1)

    X_cf = X1.copy()
    X_cf[sensitive_col] = X_cf[sensitive_col].map(
        lambda v: b if v == a else a if v == b else v
    )

    pred_cf = model.predict(X_cf)

    flips = (pred_original != pred_cf).sum()
    return flips / len(X1), int(flips)


# =========================
# RUN EXPERIMENT
# =========================
def run_experiment(model_type="tree", include_sensitive=True):
    df = load_data()

    X, y, model = build_model(df, model_type=model_type, include_sensitive=include_sensitive)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # train
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    # predict
    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred

    # metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if SENSITIVE in X_test.columns:
        dp_rates, dp_gap = demographic_parity_gap(
            pd.Series(y_pred, index=X_test.index),
            X_test[SENSITIVE]
        )
    else:
        dp_rates, dp_gap = {}, None

    cf = counterfactual_flip_rate_binary(model, X_test, sensitive_col=SENSITIVE) if include_sensitive else None

    results = {
        'model': model_type,
        'include_sensitive': include_sensitive,
        'sensitive_attribute': SENSITIVE,
        'accuracy': acc,
        'f1': f1,
        'train_time_sec': train_time,
        'prediction_time_sec': pred_time,
        'dp_gap': dp_gap,
        'cf_flip_rate': None if cf is None else cf[0],
        'cf_flips': None if cf is None else cf[1],
        'dp_rates': dp_rates
    }

    return results, y_test, y_pred


# =========================
# SAVE TXT RESULT
# =========================
def save_results_txt(name, results):
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(RAW_RESULTS_DIR, f"{name}_{now}.txt")

    with open(filepath, 'w', encoding='utf-8') as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    print(f"[Saved to {filepath}]")
    return filepath


# =========================
# SAVE CSV SUMMARY
# =========================
def save_results_csv(all_results):
    df_res = pd.DataFrame(all_results)

    # dp_rates dict dijadikan string agar aman disimpan
    if 'dp_rates' in df_res.columns:
        df_res['dp_rates'] = df_res['dp_rates'].astype(str)

    filepath = os.path.join(PROCESSED_RESULTS_DIR, 'scholarship_summary.csv')
    df_res.to_csv(filepath, index=False, encoding='utf-8')
    print(f"[CSV summary saved to {filepath}]")
    return filepath


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    all_results = []

    for model_type in ["tree", "logreg"]:

        print(f"\n=== MODEL: {model_type.upper()} | WITH SENSITIVE ===")
        res1, y_test1, y_pred1 = run_experiment(model_type=model_type, include_sensitive=True)
        print(res1)
        print(classification_report(y_test1, y_pred1))
        save_results_txt(f"scholarship_{model_type}_with_sensitive", res1)
        all_results.append(res1)

        print(f"\n=== MODEL: {model_type.upper()} | WITHOUT SENSITIVE ===")
        res2, y_test2, y_pred2 = run_experiment(model_type=model_type, include_sensitive=False)
        print(res2)
        print(classification_report(y_test2, y_pred2))
        save_results_txt(f"scholarship_{model_type}_without_sensitive", res2)
        all_results.append(res2)

    save_results_csv(all_results)
