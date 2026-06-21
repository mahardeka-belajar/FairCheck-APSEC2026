# FairCheck-APSEC2026

**FairCheck-CI: SMT-Based Fairness Regression Testing for AI Decision Pipelines**

This repository accompanies the APSEC 2026 SEIP submission on **FairCheck-CI**, an SMT-based fairness regression testing framework for tabular AI decision pipelines. The artifact provides the implementation, experimental scripts, processed results, and CI-style evidence used in the paper.

## 1. Artifact Overview

FairCheck-CI operationalizes fairness assessment as a regression-testing workflow. The repository includes:

* empirical fairness screening scripts,
* SMT-based formal verification scripts,
* encoder equivalence validation,
* counterexample extraction and root-cause diagnostics,
* a CI gate that maps solver outcomes to model-promotion decisions,
* processed results used in the manuscript.

The paper evaluates the framework on **three datasets** and **three model classes**, yielding **nine formal verification checks** in total.

## 2. Repository Structure

```text
FairCheck-APSEC2026/
├── .github/
│   └── workflows/
│       └── faircheck-ci.yml
├── ci/
│   ├── ci_fairness_demo.py
│   ├── ci_gate_runner.py
│   ├── ci_policy.md
│   └── README_ci.md
├── datasets/
│   ├── adult/
│   ├── german_credit/
│   └── scholarship/
├── experiments/
│   ├── run_scholarship.py
│   ├── run_adult.py
│   ├── run_german_fixed.py
│   ├── baseline_random_pair_sampling_3models.py
│   ├── baseline_counterfactual_augmentation_3models.py
│   ├── baseline_boundary_focused_sampling_3models.py
│   ├── encoder_validation_*.py
│   ├── formal_solver_summary_builder.py
│   ├── formal_counterexample_report_builder.py
│   ├── rootcause_report_builder.py
│   ├── smt_check_*.py
│   └── aggregate_all_results.py
├── notes/
│   └── experiment_log.md
├── results/
│   └── processed/
│       ├── ci/
│       ├── empirical/
│       └── formal/
├── README.md
└── requirements.txt
```

## 3. Datasets and Model Classes

The artifact evaluates FairCheck-CI on the following tabular decision-making tasks:

* **Scholarship**: scholarship eligibility prediction
* **Adult Income**: income classification
* **German Credit**: credit risk classification

The following model classes are used throughout the evaluation:

* **Decision Tree**
* **Rule List**
* **Logistic Regression**

The empirical layer additionally compares three screening baselines:

* **Random Pair Sampling**
* **Counterfactual Augmentation**
* **Boundary-Focused Sampling**

## 4. Main Claims Supported by the Artifact

The repository supports the core claims reported in the paper:

1. **Encoder equivalence**: the SMT encoding matches the Python model predictions across all evaluated dataset-model combinations.
2. **Formal verification results**: the nine SMT checks yield a consistent pattern of PASS/FAIL outcomes.
3. **CI gate behavior**: solver outcomes are translated into model-promotion decisions.
4. **Developer-facing diagnostics**: counterexamples are summarized in a root-cause report.

### Observed processed results

From the processed outputs included in this artifact:

* **Formal solver summary**: 9 checks in total, with **6 PASS**, **3 FAIL**, and **0 WARNING**.
* **CI gate status**: overall **FAIL** with exit code **1**, reflecting the presence of three failing Logistic Regression checks.
* **Encoder validation**: all evaluated encoder checks report **100% match** with **0 mismatches**.
* **Counterexample report**: all reported formal counterexamples occur in the Logistic Regression experiments.

These results are stored in the `results/processed/formal/` and `results/processed/ci/` directories.

## 5. File Locations for Key Evidence

### Formal verification evidence

* `results/processed/formal/formal_solver_summary.csv`
* `results/processed/formal/formal_solver_summary.json`
* `results/processed/formal/formal_solver_summary.md`
* `results/processed/formal/formal_counterexample_report.csv`
* `results/processed/formal/formal_counterexample_report.json`
* `results/processed/formal/formal_counterexample_report.md`
* `results/processed/formal/rootcause_report.csv`
* `results/processed/formal/encoder_validation.csv`
* `results/processed/formal/encoder_validation.md`

### CI evidence

* `results/processed/ci/ci_gate_status.json`
* `results/processed/ci/ci_gate_report.md`
* `results/processed/ci/ci_fairness_demo_summary.json`
* `results/processed/ci/ci_fairness_demo_report.md`

### Empirical screening evidence

* `results/processed/empirical/`

## 6. Reproducibility Instructions

### 6.1 Environment setup

Install the Python dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

If your environment requires a dedicated virtual environment, create one first and activate it before installing dependencies.

### 6.2 Run empirical experiments

The main experiment entry points are located in the `experiments/` directory.

```bash
python experiments/run_scholarship.py
python experiments/run_adult.py
python experiments/run_german_fixed.py
```

### 6.3 Run baseline screening experiments

The repository provides scripts for the three empirical baselines used in the paper.

```bash
python experiments/baseline_random_pair_sampling_3models.py
python experiments/baseline_counterfactual_augmentation_3models.py
python experiments/baseline_boundary_focused_sampling_3models.py
```

### 6.4 Run SMT verification

The formal verification layer is implemented as nine dataset-model checks.

```bash
python experiments/smt_check_scholarship_tree.py
python experiments/smt_check_scholarship_rulelist.py
python experiments/smt_check_scholarship_logreg.py
python experiments/smt_check_adult_tree.py
python experiments/smt_check_adult_rulelist.py
python experiments/smt_check_adult_logreg.py
python experiments/smt_check_german_tree.py
python experiments/smt_check_german_rulelist.py
python experiments/smt_check_german_logreg.py
```

### 6.5 Build result summaries

After running the experiments, aggregate the outputs into the summary files used by the manuscript.

```bash
python experiments/encoder_validation_summary_builder.py
python experiments/formal_solver_summary_builder.py
python experiments/formal_counterexample_report_builder.py
python experiments/rootcause_report_builder.py
python experiments/aggregate_all_results.py
```

### 6.6 Run the CI gate

The CI gate consumes the formal verification outputs and maps them into model-promotion decisions.

```bash
python ci/ci_gate_runner.py
```

The repository also includes a GitHub Actions workflow under `.github/workflows/faircheck-ci.yml`.

## 7. Interpretation of CI Outcomes

FairCheck-CI uses the following decision policy:

* **UNSAT → PASS**: no counterexample is found within the encoded model, transformed feature space, selected sensitive attribute, domain predicate, and fairness property.
* **SAT → FAIL**: at least one counterexample is found.
* **TIMEOUT / inconclusive → WARNING**: the fairness property cannot be confirmed within the solver budget.

This interpretation is intentionally scoped to the encoded verification setting and should not be read as a universal fairness guarantee.

## 8. Current Artifact Status

At the time of this repository snapshot, the processed results indicate the following:

* **Formal solver summary**: PASS = 6, FAIL = 3, WARNING = 0
* **CI gate status**: overall FAIL, exit code 1
* **Encoder validation**: 100% match, 0 mismatches
* **Counterexamples**: concentrated in the Logistic Regression models

These outcomes are consistent with the reported figures and tables in the manuscript.

## 9. Notes for Reviewers and Artifact Evaluators

This repository is organized as a research artifact for the APSEC 2026 SEIP submission. The intended use is to support:

* reproducibility of the reported experiments,
* inspection of the formal SMT verification pipeline,
* validation of the encoder equivalence checks,
* verification of CI gate behavior,
* examination of counterexample and root-cause diagnostics.

## 10. Citation

If you use this artifact in academic work, please cite the FairCheck-CI paper associated with APSEC 2026 SEIP.

## 11. License

Unless otherwise specified in a separate license file, the repository contents should be treated according to the policies of the hosting project and the authors' submission requirements.

## 12. Contact

For questions regarding the artifact, please contact the corresponding author or maintainers listed in the paper.
