# Formal Solver Summary

## Status Interpretation

- **UNSAT** = **PASS** (no counterexample found for the encoded fairness property)
- **SAT** = **FAIL** (counterexample found)
- **OTHER** = **WARNING / INCONCLUSIVE**

## Cross-Dataset Solver Results

| Dataset | Model | Sensitive Attribute | Solver Status | CI Status | Checked Pairs | Counterexample Pair |
|---|---|---|---|---|---:|---|
| scholarship | Decision Tree | Asal_Daerah | UNSAT | PASS | 6 | None |
| scholarship | Rule List | Asal_Daerah | UNSAT | PASS | 6 | None |
| scholarship | Logistic Regression | Asal_Daerah | SAT | FAIL | 6 | 3T -> Pedesaan |
| adult | Decision Tree | sex | UNSAT | PASS | 2 | None |
| adult | Rule List | sex | UNSAT | PASS | 2 | None |
| adult | Logistic Regression | sex | SAT | FAIL | 2 | Female -> Male |
| german_credit | Decision Tree | sex | UNSAT | PASS | 2 | None |
| german_credit | Rule List | sex | UNSAT | PASS | 2 | None |
| german_credit | Logistic Regression | sex | SAT | FAIL | 2 | female -> male |

## Aggregate Counts

- **PASS**: 6
- **FAIL**: 3
- **WARNING**: 0

## Quick Reading Guide

- Use this summary to compare model-class behavior under the same encoded fairness property across Scholarship, Adult, and German Credit.
- In the current artifact, **UNSAT** indicates no solver-level counterexample was found within the encoded scope, while **SAT** indicates a counterexample exists and can be inspected in the corresponding JSON file.