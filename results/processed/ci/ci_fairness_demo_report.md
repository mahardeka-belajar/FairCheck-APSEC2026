# FairCheck-CI Mini Demo (Scholarship)

## CI-style Status Mapping

- **UNSAT** = **PASS** (no counterexample found for the encoded fairness property)
- **SAT** = **FAIL** (counterexample found)
- **OTHER** = **WARNING / INCONCLUSIVE**

## Model Results

| Dataset | Model | Sensitive Attribute | Solver Status | CI Status | Checked Pairs | Counterexample |
|---|---|---|---|---|---:|---|
| scholarship | Decision Tree | Asal_Daerah | UNSAT | PASS | 6 | None |
| scholarship | Rule List | Asal_Daerah | UNSAT | PASS | 6 | None |
| scholarship | Logistic Regression | Asal_Daerah | SAT | FAIL | 6 | 3T -> Pedesaan |

## Summary

- **PASS models**: Decision Tree, Rule List
- **FAIL models**: Logistic Regression
- Decision Tree returns **UNSAT**, making it a natural **PASS** case in a FairCheck-CI demonstration.
- Rule List returns **UNSAT**, making it a natural **PASS** case in a FairCheck-CI demonstration.
- Logistic Regression produces a solver-level counterexample, making it the natural **FAIL** case in a FairCheck-CI demonstration.