# CI Gate Report

## Overall Status

- **Overall gate status**: **FAIL**
- **Exit code**: `1`
- **Generated at (UTC)**: `2026-06-06T04:07:04.310312Z`

## Policy

- **PASS**: no solver-level counterexample found in any active fairness check
- **FAIL**: at least one solver-level counterexample found (SAT case)
- **WARNING**: inconclusive or missing-status case detected

## Counts

- **PASS**: 6
- **FAIL**: 3
- **WARNING**: 0
- **TOTAL**: 9

## Per-Model Solver Gate Results

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

## Gate Decision Rationale

- The gate is marked **FAIL** because one or more solver checks returned **SAT**.
- **Failed models**: scholarship / Logistic Regression, adult / Logistic Regression, german_credit / Logistic Regression
- **Passed models**: scholarship / Decision Tree, scholarship / Rule List, adult / Decision Tree, adult / Rule List, german_credit / Decision Tree, german_credit / Rule List

## Counterexample Summary

- **scholarship / Logistic Regression** triggered **3T -> Pedesaan** with prediction change `0` -> `1`.
- **adult / Logistic Regression** triggered **Female -> Male** with prediction change `0` -> `1`.
- **german_credit / Logistic Regression** triggered **female -> male** with prediction change `0` -> `1`.

## Encoder Validation

- Total validation entries: 9
- Total samples checked: 5100
- Total mismatch count: 0
- Overall match rate: 100.0%
- Status: **PASS**

## Inputs

- Formal solver summary: `results/processed/formal/formal_solver_summary.json`
- Counterexample report: `results/processed/formal/formal_counterexample_report.json`
- Encoder validation: `results/processed/formal/encoder_validation.csv`