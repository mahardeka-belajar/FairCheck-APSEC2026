
# Updated baseline scripts (support 3 model sekaligus)

Baseline minimum:
1. random_pair_sampling_3models.py
2. counterfactual_augmentation_3models.py
3. boundary_focused_sampling_3models.py

Ketiga script akan menjalankan baseline untuk 3 model sekaligus:
- tree
- logreg
- rulelist

## Cara run
```bash
python baseline_random_pair_sampling_3models.py
python baseline_counterfactual_augmentation_3models.py
python baseline_boundary_focused_sampling_3models.py
```

## Output
- detail raw di `results/raw/*_3models.json`
- ringkasan di `results/processed/*_3models.csv`
