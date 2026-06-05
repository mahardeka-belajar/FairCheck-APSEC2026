import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from run_adult import (
    load_adult,
    build_xy,
    make_preprocessor,
    make_model,
)

RESULT_DIR = "results/processed/formal"
os.makedirs(RESULT_DIR, exist_ok=True)


def logreg_predict_from_encoding(coef, intercept, row):
    """
    Re-implement SMT encoding semantics:

    score = w·x + b

    prediction =
        1 if score > 0
        0 otherwise
    """

    score = float(np.dot(coef, row) + intercept)

    return 1 if score > 0 else 0


def validate_logreg(n_samples=1000):

    train_df, test_df = load_adult()

    X_train, X_test, y_train, y_test = build_xy(
        train_df,
        test_df,
        include_sensitive=True
    )

    model = Pipeline([
        ("prep", make_preprocessor(X_train)),
        ("clf", make_model("logreg"))
    ])

    model.fit(X_train, y_train)

    sample_size = min(n_samples, len(X_test))

    X_sample = X_test.iloc[:sample_size]

    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]

    Xt = prep.transform(X_sample)

    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    # Python prediction
    python_pred = clf.predict(Xt)

    coef = clf.coef_[0]
    intercept = float(clf.intercept_[0])

    # Encoding prediction
    encoded_pred = np.array([
        logreg_predict_from_encoding(
            coef,
            intercept,
            row
        )
        for row in Xt
    ])

    mismatch_mask = python_pred != encoded_pred

    mismatch_count = int(
        np.sum(mismatch_mask)
    )

    match_rate = round(
        (
            (sample_size - mismatch_count)
            / sample_size
        ) * 100.0,
        2
    )

    return {
        "dataset": "adult",
        "model": "Logistic Regression",
        "samples_checked": sample_size,
        "mismatch_count": mismatch_count,
        "match_rate": match_rate,
    }


def main():

    result = validate_logreg()

    df = pd.DataFrame([result])

    out_path = os.path.join(
        RESULT_DIR,
        "encoder_validation_adult_logreg.csv"
    )

    df.to_csv(
        out_path,
        index=False
    )

    print()
    print("=== Encoder Validation ===")
    print(df)
    print()
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()