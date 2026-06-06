# README for FairCheck-CI

This document explains how to run the **FairCheck-CI** gate, what files it expects, what outputs it produces, and how to interpret the final decision.

---

## 1. Purpose

FairCheck-CI is the repository-level fairness gate for the current project. It interprets solver-based fairness checking results and converts them into a CI-style decision:

- **PASS**: no solver-level counterexample was found for the active fairness property.
- **FAIL**: at least one solver-level counterexample was found.
- **WARNING / INCONCLUSIVE**: one or more checks are missing, inconclusive, or use a non-standard status.

At the current stage of the project, the final gate is driven by the formal solver artifacts produced for:

- Scholarship
- Adult
- German Credit

across three model classes:

- Decision Tree
- Rule List
- Logistic Regression

---

## 2. Main CI Entry Point

The main CI gate entry point is:

```bash
python ci/ci_gate_runner.py
```

If the script is still temporarily stored at repository root, it can also be run as:

```bash
python ci_gate_runner.py
```

---

## 3. Required Inputs

The final gate expects the following inputs to already exist:

### 3.1 Formal solver summary

```text
results/processed/formal/formal_solver_summary.json
```

This artifact is generated from all solver results and contains the cross-dataset, cross-model formal status summary.

### 3.2 Formal counterexample report

```text
results/processed/formal/formal_counterexample_report.json
```

This artifact contains all SAT / FAIL cases and their detailed counterexamples.

---

## 4. Expected Upstream Workflow

The intended workflow before running the CI gate is:

1. run the SMT solver checks for the active datasets/models,
2. build the consolidated formal summary,
3. build the formal counterexample report,
4. run the CI gate.

A typical sequence is:

```bash
python experiments/formal/smt_check_scholarship_tree.py
python experiments/formal/smt_check_scholarship_rulelist.py
python experiments/formal/smt_check_scholarship_logreg.py
python experiments/formal/smt_check_adult_tree.py
python experiments/formal/smt_check_adult_rulelist.py
python experiments/formal/smt_check_adult_logreg.py
python experiments/formal/smt_check_german_tree.py
python experiments/formal/smt_check_german_rulelist.py
python experiments/formal/smt_check_german_logreg.py

python experiments/utils/formal_solver_summary_builder.py
python experiments/utils/formal_counterexample_report_builder.py
python ci/ci_gate_runner.py
```

If your scripts are currently stored directly under `experiments/`, adapt the paths accordingly.

---

## 5. Outputs Produced by the CI Gate

Running the CI gate produces:

### 5.1 Machine-readable status

```text
results/processed/ci_gate_status.json
```

This file contains:

- overall gate status,
- exit code,
- counts of PASS / FAIL / WARNING,
- passed/failed model list,
- references to input artifacts.

### 5.2 Human-readable report

```text
results/processed/ci_gate_report.md
```

This file contains:

- final gate decision,
- status counts,
- per-model solver statuses,
- short rationale for the gate result,
- counterexample references.

---

## 6. Exit Codes

The gate uses the following exit codes:

- `0` -> **PASS**
- `1` -> **FAIL**
- `2` -> **WARNING / INCONCLUSIVE**

This makes the script suitable for use in a CI pipeline (e.g., GitHub Actions, local pre-merge checks, or nightly fairness regression runs).

---

## 7. Interpreting the Result

### PASS

A PASS result means all active solver checks returned **UNSAT**, so no fairness counterexample was found within the encoded scope.

### FAIL

A FAIL result means at least one solver check returned **SAT**, meaning a fairness counterexample exists. In that case, inspect:

- `results/processed/formal/formal_solver_summary.md`
- `results/processed/formal/formal_counterexample_report.md`
- `results/processed/ci_gate_report.md`

### WARNING / INCONCLUSIVE

A WARNING result means the gate could not safely conclude PASS/FAIL because one or more required solver artifacts were missing or an unexpected status was encountered.

---

## 8. Current Project Interpretation

At the current stage of the repository, the gate operates over the following fairness property:

> Existence of two inputs that are identical on all non-sensitive transformed features, differ only in the sensitive attribute, and produce different model predictions.

The transformed feature space uses:

- one-hot encoding for categorical variables,
- standardization for numerical variables,
- bounded constraints derived from observed transformed ranges.

This means the CI gate should be interpreted as a fairness regression check for the **encoded individual/counterfactual property**, not as a universal decision over all possible fairness definitions.

---

## 9. Recommended Repository Placement

A clean final repository structure for the CI logic is:

```text
ci/
├── ci_gate_runner.py
├── ci_fairness_demo.py
├── README_ci.md
└── ci_policy.md
```

---

## 10. Suggested Usage in Practice

### Quick local check

```bash
python ci/ci_gate_runner.py
```

### Inspect only the final status

```bash
cat results/processed/ci_gate_status.json
```

### Inspect the full gate report

```bash
cat results/processed/ci_gate_report.md
```

### Inspect detailed counterexamples

```bash
cat results/processed/formal_counterexample_report.md
```

---

## 11. Related Files

### Formal summary
- `results/processed/formal/formal_solver_summary.csv`
- `results/processed/formal/formal_solver_summary.json`
- `results/processed/formal/formal_solver_summary.md`

### Counterexample report
- `results/processed/formal/formal_counterexample_report.csv`
- `results/processed/formal/formal_counterexample_report.json`
- `results/processed/formal/formal_counterexample_report.md`

### Gate output
- `results/processed/ci_gate_status.json`
- `results/processed/ci_gate_report.md`

---

## 12. Notes

- The `ci_fairness_demo.py` script is a **mini demonstration** and is not the final repository-wide gate.
- The final repository-wide gate is `ci_gate_runner.py`.
- If the gate exits with code `1`, this is the expected behavior for a formal **FAIL** case, not a crash.
