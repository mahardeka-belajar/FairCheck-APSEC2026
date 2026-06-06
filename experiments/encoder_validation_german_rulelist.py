import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from run_german_fixed import (
    load_german,
    build_xy,
    make_preprocessor,
    make_model,
)

RESULT_DIR = "results/processed/formal"
os.makedirs(RESULT_DIR, exist_ok=True)


def rule_predict_from_encoding(rules, default_class, row):
    pred = default_class
    for conditions, pred_class, _ in rules:
        matched = True
        for feat, op, thresh in conditions:
            val = row[int(feat)]
            if op == "<=":
                if not (val <= thresh):
                    matched = False
                    break
            elif op == ">":
                if not (val > thresh):
                    matched = False
                    break
        if matched:
            pred = pred_class
            break
    return int(pred)


def validate_rulelist(n_samples=1000):
    df = load_german()
    X, y = build_xy(df, include_sensitive=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    model = Pipeline([
        ("prep", make_preprocessor(X_train)),
        ("clf", make_model("rulelist"))
    ])
    model.fit(X_train, y_train)
    sample_size = min(n_samples, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    Xt = prep.transform(X_sample)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    python_pred = clf.predict(Xt)
    rules = clf.rules_
    default_class = clf.default_class_
    encoded_pred = np.array([
        rule_predict_from_encoding(rules, default_class, row)
        for row in Xt
    ])
    mismatch_mask = python_pred != encoded_pred
    mismatch_count = int(np.sum(mismatch_mask))
    match_rate = round(((sample_size - mismatch_count) / sample_size) * 100.0, 2)
    return {
        "dataset": "german_credit",
        "model": "Rule List",
        "samples_checked": sample_size,
        "mismatch_count": mismatch_count,
        "match_rate": match_rate,
    }


def main():
    result = validate_rulelist()
    df = pd.DataFrame([result])
    out_path = os.path.join(RESULT_DIR, "encoder_validation_german_rulelist.csv")
    df.to_csv(out_path, index=False)
    print()
    print("=== Encoder Validation ===")
    print(df[["dataset", "model", "samples_checked", "mismatch_count", "match_rate"]])
    print()
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
