# FairCheck-APSEC2026

**FairCheck-CI: SMT-Based Fairness Regression Testing for AI Decision Pipelines**

This repository accompanies the APSEC 2026 Software Engineering in Practice (SEIP) submission on **FairCheck-CI**, an SMT-based fairness regression testing framework for tabular AI decision pipelines. The artifact contains the implementation, datasets, experimental scripts, formal verification components, CI-style automation, and processed results used to support the claims reported in the manuscript.

---

# 1. Artifact Overview

FairCheck-CI operationalizes fairness assessment as a regression-testing workflow that combines empirical screening, formal verification, diagnostic reporting, and CI-style decision support.

The repository includes:

* empirical fairness screening scripts,
* SMT-based formal verification scripts,
* model-to-SMT encoder validation,
* counterexample extraction,
* root-cause diagnostics,
* CI gate automation,
* processed experimental results,
* reproducibility artifacts.

The evaluation reported in the paper covers:

* **3 datasets**
* **3 model classes**
* **3 empirical screening baselines**
* **9 SMT verification checks**

---

# 2. Artifact Snapshot

| Item                          | Value |
| ----------------------------- | ----- |
| Datasets                      | 3     |
| Model Classes                 | 3     |
| Empirical Baselines           | 3     |
| SMT Verification Checks       | 9     |
| PASS Results                  | 6     |
| FAIL Results                  | 3     |
| WARNING Results               | 0     |
| Encoder Validation Match Rate | 100%  |
| Counterexamples Detected      | 3     |

## Key Findings

The processed results included in this repository reveal a consistent verification pattern across all evaluated datasets.

| Model Class         | Verification Outcome |
| ------------------- | -------------------- |
| Decision Tree       | PASS                 |
| Rule List           | PASS                 |
| Logistic Regression | FAIL                 |

All reported formal counterexamples were associated with Logistic Regression models. Decision Tree and Rule List models satisfied the evaluated fairness property across all datasets included in the study.

---

# 3. Repository Structure

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
├── results/
│   └── processed/
│       ├── ci/
│       ├── empirical/
│       └── formal/
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 4. Datasets and Model Classes

The artifact evaluates FairCheck-CI on three tabular decision-making datasets.

## Datasets

### Scholarship

A scholarship eligibility prediction dataset used to evaluate fairness properties in educational decision-making.

### Adult Income

The Adult Income dataset used for income classification and fairness analysis.

### German Credit

The German Credit dataset used for credit-risk classification and fairness assessment.

## Model Classes

The following model classes are evaluated throughout the study:

* Decision Tree
* Rule List
* Logistic Regression

## Empirical Screening Baselines

The empirical layer compares three fairness-screening approaches:

* Random Pair Sampling
* Counterfactual Augmentation
* Boundary-Focused Sampling

---

# 5. Main Claims Supported by the Artifact

The repository supports the following claims reported in the manuscript.

## Claim 1: Encoder Equivalence

The SMT encoding faithfully reproduces the behavior of the trained Python models across all evaluated dataset-model combinations.

Evidence:

* `results/processed/formal/encoder_validation.csv`
* `results/processed/formal/encoder_validation.md`

## Claim 2: Formal Fairness Verification

The SMT verification layer successfully identifies fairness violations through counterexample generation.

Evidence:

* `results/processed/formal/formal_solver_summary.csv`
* `results/processed/formal/formal_counterexample_report.csv`

## Claim 3: CI Gate Integration

Formal verification outcomes can be translated into CI-style model-promotion decisions.

Evidence:

* `results/processed/ci/ci_gate_status.json`
* `results/processed/ci/ci_gate_report.md`

## Claim 4: Developer-Facing Diagnostics

Detected fairness violations can be summarized as actionable root-cause reports.

Evidence:

* `results/processed/formal/rootcause_report.csv`

---

# 6. Processed Results Summary

The processed outputs currently included in the repository show the following results.

## Formal Verification Summary

| Outcome | Count |
| ------- | ----- |
| PASS    | 6     |
| FAIL    | 3     |
| WARNING | 0     |
| TOTAL   | 9     |

## CI Gate Summary

| Metric         | Value |
| -------------- | ----- |
| Overall Status | FAIL  |
| Exit Code      | 1     |
| Passed Models  | 6     |
| Failed Models  | 3     |
| Warning Models | 0     |

## Encoder Validation Summary

| Metric                     | Value |
| -------------------------- | ----- |
| Dataset–Model Combinations | 9     |
| Match Rate                 | 100%  |
| Mismatches                 | 0     |

## Counterexample Summary

Three formal counterexamples were detected.

All reported counterexamples occurred in Logistic Regression models:

* Scholarship Dataset
* Adult Income Dataset
* German Credit Dataset

---

# 7. File Locations for Key Evidence

## Formal Verification Evidence

* `results/processed/formal/formal_solver_summary.csv`
* `results/processed/formal/formal_solver_summary.json`
* `results/processed/formal/formal_solver_summary.md`
* `results/processed/formal/formal_counterexample_report.csv`
* `results/processed/formal/formal_counterexample_report.json`
* `results/processed/formal/formal_counterexample_report.md`
* `results/processed/formal/rootcause_report.csv`
* `results/processed/formal/encoder_validation.csv`
* `results/processed/formal/encoder_validation.md`

## CI Evidence

* `results/processed/ci/ci_gate_status.json`
* `results/processed/ci/ci_gate_report.md`
* `results/processed/ci/ci_fairness_demo_summary.json`
* `results/processed/ci/ci_fairness_demo_report.md`

## Empirical Screening Evidence

* `results/processed/empirical/`

---

# 8. Reproducibility Instructions

## Environment Setup

Install the project dependencies.

```bash
pip install -r requirements.txt
```

If desired, create and activate a dedicated Python virtual environment before installing dependencies.

---

## Run Empirical Experiments

```bash
python experiments/run_scholarship.py
python experiments/run_adult.py
python experiments/run_german_fixed.py
```

---

## Run Baseline Screening Experiments

```bash
python experiments/baseline_random_pair_sampling_3models.py

python experiments/baseline_counterfactual_augmentation_3models.py

python experiments/baseline_boundary_focused_sampling_3models.py
```

---

## Run SMT Verification

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

---

## Build Result Summaries

```bash
python experiments/encoder_validation_summary_builder.py

python experiments/formal_solver_summary_builder.py

python experiments/formal_counterexample_report_builder.py

python experiments/rootcause_report_builder.py

python experiments/aggregate_all_results.py
```

---

## Run the CI Gate

```bash
python ci/ci_gate_runner.py
```

The repository also includes a GitHub Actions workflow located at:

```text
.github/workflows/faircheck-ci.yml
```

---

# 9. Interpretation of CI Outcomes

FairCheck-CI maps SMT solver outcomes into CI-style decisions.

| SMT Result             | CI Decision |
| ---------------------- | ----------- |
| UNSAT                  | PASS        |
| SAT                    | FAIL        |
| TIMEOUT / Inconclusive | WARNING     |

Interpretation:

* **PASS (UNSAT)** indicates that no counterexample was found under the encoded verification conditions.
* **FAIL (SAT)** indicates that at least one fairness counterexample was found.
* **WARNING** indicates that verification could not be completed within the configured resource limits.

This interpretation is intentionally scoped to the encoded verification setting and should not be interpreted as a universal guarantee of fairness.

---

# 10. Notes for Reviewers and Artifact Evaluators

This repository is organized as a research artifact accompanying the APSEC 2026 SEIP submission.

The artifact is intended to support:

* reproducibility of the reported experiments,
* inspection of the SMT verification workflow,
* validation of encoder equivalence,
* examination of generated counterexamples,
* inspection of root-cause diagnostics,
* evaluation of CI gate behavior.

All processed outputs included in the repository correspond to the evidence reported in the manuscript.

---


# 11. Contact

For questions regarding the artifact, please contact the corresponding author identified in the associated manuscript.
