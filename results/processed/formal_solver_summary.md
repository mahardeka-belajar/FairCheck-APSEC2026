# Formal Solver Summary

## Status Interpretation

- **UNSAT** = **PASS** (no counterexample found for the encoded fairness property)
- **SAT** = **FAIL** (counterexample found)
- **OTHER** = **WARNING / INCONCLUSIVE**

## Cross-Dataset Solver Results

| Dataset | Model | SMT Status | CI Status | CE Found | CE Bound | Runtime(s) | Timeout |
|---|---|---|---|---:|---:|---:|---|
| scholarship | Decision Tree | UNSAT | PASS | None | None | None | None |
| scholarship | Rule List | UNSAT | PASS | None | None | None | None |
| scholarship | Logistic Regression | SAT | FAIL | None | None | None | None |
| adult | Decision Tree | UNSAT | PASS | None | None | None | None |
| adult | Rule List | UNSAT | PASS | None | None | None | None |
| adult | Logistic Regression | SAT | FAIL | None | None | None | None |
| german_credit | Decision Tree | UNSAT | PASS | None | None | None | None |
| german_credit | Rule List | UNSAT | PASS | None | None | None | None |
| german_credit | Logistic Regression | SAT | FAIL | None | None | None | None |

## Aggregate Counts

- **PASS**: 6
- **FAIL**: 3
- **WARNING**: 0

## Quick Reading Guide

- Use this summary to compare model-class behavior under the same encoded fairness property across Scholarship, Adult, and German Credit.
- In the current artifact, **UNSAT** indicates no solver-level counterexample was found within the encoded scope, while **SAT** indicates a counterexample exists and can be inspected in the corresponding JSON file.