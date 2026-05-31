
import pandas as pd
from pathlib import Path

MAIN_FILES = {
    'scholarship': 'results/processed/scholarship_summary.csv',
    'adult': 'results/processed/adult_summary.csv',
    'german_credit': 'results/processed/german_summary.csv',
}

PRETTY_MODEL = {
    'tree': 'Decision Tree',
    'logreg': 'Logistic Regression',
    'rulelist': 'Rule List',
}


def must_read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Missing required file: {path}')
    return pd.read_csv(p)


def main():
    rows = []
    for dataset, file_name in MAIN_FILES.items():
        df = must_read(file_name).copy()
        if 'dataset' not in df.columns:
            df['dataset'] = dataset
        df['model_pretty'] = df['model'].map(PRETTY_MODEL).fillna(df['model'])

        for model_name in df['model'].unique():
            sub = df[df['model'] == model_name].copy()
            row_with = sub[sub['include_sensitive'] == True].iloc[0]
            row_without = sub[sub['include_sensitive'] == False].iloc[0]
            rows.append({
                'dataset': dataset,
                'model': row_with['model_pretty'],
                'sensitive_attribute': row_with['sensitive_attribute'],
                'accuracy_with_sensitive': row_with['accuracy'],
                'accuracy_without_sensitive': row_without['accuracy'],
                'delta_accuracy': row_without['accuracy'] - row_with['accuracy'],
                'f1_with_sensitive': row_with['f1'],
                'f1_without_sensitive': row_without['f1'],
                'delta_f1': row_without['f1'] - row_with['f1'],
                'dp_gap_with_sensitive': row_with.get('dp_gap', None),
            })

    out_df = pd.DataFrame(rows)
    out_path = 'ablation_sensitive_removal.csv'
    out_df.to_csv(out_path, index=False, encoding='utf-8')
    print(f'[Saved] {out_path}')
    print(out_df)


if __name__ == '__main__':
    main()
