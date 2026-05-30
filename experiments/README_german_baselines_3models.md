
# German Credit baseline suite (3 models)

Files:
- german_baseline_random_pair_sampling_3models.py
- german_baseline_counterfactual_augmentation_3models.py
- german_baseline_boundary_focused_sampling_3models.py

Default sensitive attribute:
- sex (derived from personal_status_sex)

Default models run together:
- tree
- logreg
- rulelist

## Cara run
Dari root project:

```bash
python experiments/german_baseline_random_pair_sampling_3models.py
python experiments/german_baseline_counterfactual_augmentation_3models.py
python experiments/german_baseline_boundary_focused_sampling_3models.py
```

## Output
- results/raw/german_*_details_3models.json
- results/processed/german_*_summary_3models.csv
