
import pandas as pd
from pathlib import Path

PRETTY_MODEL = {
    'tree': 'Decision Tree',
    'logreg': 'Logistic Regression',
    'rulelist': 'Rule List',
}

BASELINE_FILES = {
    'scholarship': {
        'random_pair_sampling': 'results/processed/random_pair_sampling_summary_3models.csv',
        'counterfactual_augmentation': 'results/processed/counterfactual_augmentation_summary_3models.csv',
        'boundary_focused_sampling': 'results/processed/boundary_focused_sampling_summary_3models.csv',
    },
    'adult': {
        'random_pair_sampling': 'results/processed/adult_random_pair_sampling_summary_3models.csv',
        'counterfactual_augmentation': 'results/processed/adult_counterfactual_augmentation_summary_3models.csv',
        'boundary_focused_sampling': 'results/processed/adult_boundary_focused_sampling_summary_3models.csv',
    },
    'german_credit': {
        'random_pair_sampling': 'results/processed/german_random_pair_sampling_summary_3models.csv',
        'counterfactual_augmentation': 'results/processed/german_counterfactual_augmentation_summary_3models.csv',
        'boundary_focused_sampling': 'results/processed/german_boundary_focused_sampling_summary_3models.csv',
    }
}


def must_read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Missing required file: {path}')
    return pd.read_csv(p)


def main():
    rows = []
    for dataset, mapping in BASELINE_FILES.items():
        loaded = {}
        for baseline_name, file_name in mapping.items():
            df = must_read(file_name).copy()
            loaded[baseline_name] = df

        models = sorted(set(loaded['random_pair_sampling']['model'].tolist()))
        for model_name in models:
            row_rand = loaded['random_pair_sampling'][loaded['random_pair_sampling']['model'] == model_name].iloc[0]
            row_cf = loaded['counterfactual_augmentation'][loaded['counterfactual_augmentation']['model'] == model_name].iloc[0]
            row_boundary = loaded['boundary_focused_sampling'][loaded['boundary_focused_sampling']['model'] == model_name].iloc[0]
            rates = {
                'random_pair_sampling': float(row_rand['flip_rate']),
                'counterfactual_augmentation': float(row_cf['flip_rate']),
                'boundary_focused_sampling': float(row_boundary['flip_rate']),
            }
            strongest = max(rates, key=rates.get)
            rows.append({
                'dataset': dataset,
                'model': PRETTY_MODEL.get(model_name, model_name),
                'sensitive_attribute': row_rand['sensitive_attribute'],
                'random_flip_rate': rates['random_pair_sampling'],
                'counterfactual_flip_rate': rates['counterfactual_augmentation'],
                'boundary_flip_rate': rates['boundary_focused_sampling'],
                'strongest_baseline': strongest,
            })

    out_df = pd.DataFrame(rows)
    out_path = 'ablation_baseline_effectiveness.csv'
    out_df.to_csv(out_path, index=False, encoding='utf-8')
    print(f'[Saved] {out_path}')
    print(out_df)


if __name__ == '__main__':
    main()
