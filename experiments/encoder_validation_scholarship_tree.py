import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from run_scholarship import (
    load_scholarship,
    build_xy,
    make_preprocessor,
    make_model,
)

RESULT_DIR = "results/processed/formal"
os.makedirs(RESULT_DIR, exist_ok=True)


def tree_predict_from_encoding(tree, row):
    node = 0
    while True:
        left = tree.children_left[node]
        right = tree.children_right[node]
        if left == right:
            value = tree.value[node][0]
            return int(np.argmax(value))
        feat = int(tree.feature[node])
        thresh = float(tree.threshold[node])
        if row[feat] <= thresh:
            node = left
        else:
            node = right


def validate_tree(n_samples=1000):
    df = load_scholarship()
    X, y = build_xy(df, include_sensitive=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    model = Pipeline([
        ("prep", make_preprocessor(X_train)),
        ("clf", make_model("tree"))
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
    encoded_pred = np.array([
        tree_predict_from_encoding(clf.tree_, row)
        for row in Xt
    ])
    mismatch_mask = python_pred != encoded_pred
    mismatch_count = int(np.sum(mismatch_mask))
    match_rate = round(((sample_size - mismatch_count) / sample_size) * 100.0, 2)
    return {
        "dataset": "scholarship",
        "model": "Decision Tree",
        "samples_checked": sample_size,
        "mismatch_count": mismatch_count,
        "match_rate": match_rate,
    }


def main():
    result = validate_tree()
    df = pd.DataFrame([result])
    out_path = os.path.join(RESULT_DIR, "encoder_validation_scholarship_tree.csv")
    df.to_csv(out_path, index=False)
    print()
    print("=== Encoder Validation ===")
    print(df)
    print()
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
