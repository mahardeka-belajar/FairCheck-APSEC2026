# Formal Solver Summary

## Status Interpretation

- **UNSAT** = **PASS** (no counterexample found for the encoded fairness property)
- **SAT** = **FAIL** (counterexample found)
- **OTHER** = **WARNING / INCONCLUSIVE**

## Cross-Dataset Solver Results

| Dataset | Model | SMT Status | CI Status | CE Found | CE Bound | Runtime(s) | Timeout |
|---|---|---|---|---:|---:|---:|---|
| scholarship | Decision Tree | UNSAT | PASS | 0 | 100 | 0.3283 | False |
| scholarship | Rule List | UNSAT | PASS | 0 | 100 | 0.2489 | False |
| scholarship | Logistic Regression | SAT | FAIL | 6 | 100 | 0.3214 | False |
| adult | Decision Tree | UNSAT | PASS | 0 | 100 | 1.1775 | False |
| adult | Rule List | UNSAT | PASS | 0 | 100 | 0.874 | False |
| adult | Logistic Regression | SAT | FAIL | 2 | 100 | 1.1302 | False |
| german_credit | Decision Tree | UNSAT | PASS | 0 | 100 | 0.3058 | False |
| german_credit | Rule List | UNSAT | PASS | 0 | 100 | 0.2763 | False |
| german_credit | Logistic Regression | SAT | FAIL | 2 | 100 | 0.2988 | False |

## Aggregate Counts

- **PASS**: 6
- **FAIL**: 3
- **WARNING**: 0

## Quick Reading Guide

- Use this summary to compare model-class behavior under the same encoded fairness property across Scholarship, Adult, and German Credit.
- In the current artifact, **UNSAT** indicates no solver-level counterexample was found within the encoded scope, while **SAT** indicates a counterexample exists and can be inspected in the corresponding JSON file.