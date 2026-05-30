
# Adult baseline suite (3 models)

Files:
- adult_baseline_random_pair_sampling_3models.py
- adult_baseline_counterfactual_augmentation_3models.py
- adult_baseline_boundary_focused_sampling_3models.py

Default sensitive attribute:
- sex

Default models run together:
- tree
- logreg
- rulelist

## Cara run
Dari root project:

```bash
python experiments/adult_baseline_random_pair_sampling_3models.py
python experiments/adult_baseline_counterfactual_augmentation_3models.py
python experiments/adult_baseline_boundary_focused_sampling_3models.py
```

## Output
- results/raw/adult_*_details_3models.json
- results/processed/adult_*_summary_3models.csv
