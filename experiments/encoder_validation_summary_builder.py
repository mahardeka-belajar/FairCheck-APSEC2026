import pandas as pd
from pathlib import Path

INPUT_FILES = [
    "results/processed/formal/encoder_validation_adult_tree.csv",
    "results/processed/formal/encoder_validation_adult_logreg.csv",
    "results/processed/formal/encoder_validation_adult_rulelist.csv",
]

OUTPUT_CSV = (
    "results/processed/formal/"
    "encoder_validation.csv"
)

OUTPUT_MD = (
    "results/processed/formal/"
    "encoder_validation.md"
)


def main():

    frames = []

    for path in INPUT_FILES:
        frames.append(pd.read_csv(path))

    df = pd.concat(
        frames,
        ignore_index=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    lines = []

    lines.append(
        "| Dataset | Model | Samples Checked | Mismatch Count | Match Rate (%) |"
    )

    lines.append(
        "|---|---|---:|---:|---:|"
    )

    for _, r in df.iterrows():

        lines.append(
            f"| {r['dataset']} "
            f"| {r['model']} "
            f"| {r['samples_checked']} "
            f"| {r['mismatch_count']} "
            f"| {r['match_rate']} |"
        )

    Path(OUTPUT_MD).write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(df)

    print()
    print(f"[Saved] {OUTPUT_CSV}")
    print(f"[Saved] {OUTPUT_MD}")


if __name__ == "__main__":
    main()