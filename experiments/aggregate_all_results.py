import pandas as pd
from pathlib import Path

# Aggregate cross-dataset results into one final CSV.
# Expected files in working directory / repo root:
# - scholarship_summary.csv
# - adult_summary.csv
# - german_summary.csv
# - random_pair_sampling_summary_3models.csv
# - counterfactual_augmentation_summary_3models.csv
# - boundary_focused_sampling_summary_3models.csv
# - adult_random_pair_sampling_summary_3models.csv
# - adult_counterfactual_augmentation_summary_3models.csv
# - adult_boundary_focused_sampling_summary_3models.csv
# - german_random_pair_sampling_summary_3models.csv
# - german_counterfactual_augmentation_summary_3models.csv
# - german_boundary_focused_sampling_summary_3models.csv

pretty_model = {
    'tree': 'Decision Tree',
    'logreg': 'Logistic Regression',
    'rulelist': 'Rule List',
}

def must_read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Missing required file: {path}')
    return pd.read_csv(p)

# Main model summaries
sch = must_read('scholarship_summary.csv')
adult = must_read('adult_summary.csv')
german = must_read('german_summary.csv')

main_parts = []
for df, ds in [(sch, 'scholarship'), (adult, 'adult'), (german, 'german_credit')]:
    cur = df.copy()
    if 'dataset' not in cur.columns:
        cur['dataset'] = ds
    cur['model_pretty'] = cur['model'].map(pretty_model).fillna(cur['model'])
    main_parts.append(cur)
main_df = pd.concat(main_parts, ignore_index=True)

# Baseline summaries
baseline_files = {
    'scholarship': {
        'random_pair_sampling': 'random_pair_sampling_summary_3models.csv',
        'counterfactual_augmentation': 'counterfactual_augmentation_summary_3models.csv',
        'boundary_focused_sampling': 'boundary_focused_sampling_summary_3models.csv',
    },
    'adult': {
        'random_pair_sampling': 'adult_random_pair_sampling_summary_3models.csv',
        'counterfactual_augmentation': 'adult_counterfactual_augmentation_summary_3models.csv',
        'boundary_focused_sampling': 'adult_boundary_focused_sampling_summary_3models.csv',
    },
    'german_credit': {
        'random_pair_sampling': 'german_random_pair_sampling_summary_3models.csv',
        'counterfactual_augmentation': 'german_counterfactual_augmentation_summary_3models.csv',
        'boundary_focused_sampling': 'german_boundary_focused_sampling_summary_3models.csv',
    }
}

baseline_parts = []
for dataset_name, mapping in baseline_files.items():
    for baseline_name, file_name in mapping.items():
        df = must_read(file_name).copy()
        if 'dataset' not in df.columns:
            df['dataset'] = dataset_name
        df['baseline'] = baseline_name
        df['model_pretty'] = df['model'].map(pretty_model).fillna(df['model'])
        baseline_parts.append(df)

baseline_df = pd.concat(baseline_parts, ignore_index=True)

baseline_wide = baseline_df.pivot_table(
    index=['dataset', 'model', 'model_pretty', 'sensitive_attribute'],
    columns='baseline',
    values='flip_rate',
    aggfunc='first'
).reset_index()
baseline_wide.columns.name = None

main_with = main_df[main_df['include_sensitive'] == True].copy()
final_df = main_with.merge(
    baseline_wide,
    on=['dataset', 'model', 'model_pretty', 'sensitive_attribute'],
    how='left'
)

final_df = final_df[[
    'dataset', 'model_pretty', 'sensitive_attribute', 'accuracy', 'f1', 'dp_gap',
    'random_pair_sampling', 'counterfactual_augmentation', 'boundary_focused_sampling',
    'dp_rates', 'train_time_sec'
]].rename(columns={
    'model_pretty': 'model',
    'random_pair_sampling': 'baseline_random_flip_rate',
    'counterfactual_augmentation': 'baseline_counterfactual_flip_rate',
    'boundary_focused_sampling': 'baseline_boundary_flip_rate',
})

# Stable sort order
order_ds = {'scholarship': 0, 'adult': 1, 'german_credit': 2}
order_model = {'Decision Tree': 0, 'Logistic Regression': 1, 'Rule List': 2}
final_df['_ds_order'] = final_df['dataset'].map(order_ds)
final_df['_m_order'] = final_df['model'].map(order_model)
final_df = final_df.sort_values(['_ds_order', '_m_order']).drop(columns=['_ds_order', '_m_order'])

final_df.to_csv('final_3datasets_summary.csv', index=False, encoding='utf-8')
print('[Saved] final_3datasets_summary.csv')
print(final_df)
