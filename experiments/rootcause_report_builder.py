import pandas as pd

INPUT_FILE = (
    "results/processed/formal/"
    "formal_counterexample_report.csv"
)

OUTPUT_FILE = (
    "results/processed/formal/"
    "rootcause_report.csv"
)


def determine_root_cause(row):

    changed = str(row["changed_features"])

    if "sex" in changed.lower():
        return (
            "Sensitive attribute change causes "
            "decision-boundary crossing"
        )

    if "asal_daerah" in changed.lower():
        return (
            "Sensitive region attribute change causes "
            "decision-boundary crossing"
        )

    return "Fairness violation detected"


def determine_action(row):

    changed = str(row["changed_features"])

    if "sex" in changed.lower():
        return (
            "Remove sensitive attribute from training "
            "or apply fairness constraints"
        )

    if "asal_daerah" in changed.lower():
        return (
            "Remove regional attribute from training "
            "or apply fairness-aware retraining"
        )

    return "Investigate feature influence"


def main():

    df = pd.read_csv(INPUT_FILE)

    df["boundary_distance"] = (
        df["score_y_float"] -
        df["score_x_float"]
    ).abs()

    df["root_cause"] = df.apply(
        determine_root_cause,
        axis=1
    )

    df["suggested_action"] = df.apply(
        determine_action,
        axis=1
    )

    cols = [
        "dataset",
        "model",
        "sensitive_attribute",
        "counterexample_pair",
        "prediction_x",
        "prediction_y",
        "score_x_float",
        "score_y_float",
        "boundary_distance",
        "root_cause",
        "suggested_action"
    ]

    out_df = df[cols]

    out_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(out_df)

    print()
    print(f"[Saved] {OUTPUT_FILE}")


if __name__ == "__main__":
    main()