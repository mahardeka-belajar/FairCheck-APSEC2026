# CI Policy for FairCheck-CI

This document defines the decision policy used by the repository-level fairness gate.

---

## 1. Scope

The current CI policy applies to the solver-based fairness artifacts generated in this repository.

The policy is evaluated over formal solver results that encode the following property:

> There exist two inputs that are identical on all non-sensitive transformed features, differ only in the sensitive attribute, and produce different model predictions.

This is an **individual/counterfactual fairness check under the encoded transformed-space model**, not a universal fairness guarantee across all fairness definitions.

---

## 2. Status Mapping

The gate maps solver outcomes to CI outcomes as follows:

- **UNSAT** -> **PASS**
- **SAT** -> **FAIL**
- **OTHER / non-standard / missing / inconclusive** -> **WARNING / INCONCLUSIVE**

---

## 3. Decision Rules

### Rule 1 — FAIL dominates

If **any** active solver result is `SAT`, the overall CI gate status is:

```text
FAIL
```

This means at least one model/dataset combination admits a fairness counterexample under the encoded property.

### Rule 2 — WARNING if result set is incomplete or ambiguous

If there are **no FAIL cases**, but at least one result is missing, malformed, or uses a non-standard/inconclusive status, the overall CI gate status is:

```text
WARNING
```

### Rule 3 — PASS only if all active checks are UNSAT

The overall CI gate status is:

```text
PASS
```

only if **all** active solver checks return `UNSAT`.

---

## 4. Exit Codes

The gate uses these process exit codes:

- `0` -> PASS
- `1` -> FAIL
- `2` -> WARNING / INCONCLUSIVE

These are intended to support use in CI/CD workflows.

---

## 5. Input Artifacts

The CI gate expects the following consolidated inputs:

### 5.1 Formal solver summary

```text
results/processed/formal_solver_summary.json
```

### 5.2 Formal counterexample report

```text
results/processed/formal_counterexample_report.json
```

The final gate should not be run before these artifacts are available.

---

## 6. Interpretation Guidance

### PASS means

No solver-level counterexample was found **within the current encoding scope**.

A PASS result does **not** mean:

- the model is fair under all possible fairness definitions,
- the model is free of group-level disparity,
- the model is universally safe under all data distributions.

### FAIL means

At least one solver-level counterexample exists. For a FAIL case, the corresponding counterexample report should be inspected.

### WARNING means

The gate could not safely decide PASS/FAIL due to incomplete or inconclusive evidence.

---

## 7. Encoded Assumptions

The current solver encoding uses:

- one-hot encoding for categorical variables,
- standardization for numerical variables,
- bounded feature constraints derived from observed transformed feature ranges,
- equality on all non-sensitive transformed features,
- variation only on the selected sensitive attribute.

Therefore, all CI conclusions are **scoped to this encoding**.

---

## 8. Active Repository Policy

At the current stage of this repository, the policy is applied across:

- Scholarship
- Adult
- German Credit

and across:

- Decision Tree
- Rule List
- Logistic Regression

The final gate decision is obtained from the consolidated formal solver summary rather than from any single dataset-specific demo script.

---

## 9. Relationship to `ci_fairness_demo.py`

`ci_fairness_demo.py` is a **Scholarship-only illustration script** and should be treated as a local demo.

`ci_gate_runner.py` is the **project-level gate** and is the authoritative CI policy implementation.

---

## 10. Reviewer-Facing Summary

A concise reviewer-facing reading of the policy is:

> The FairCheck-CI gate fails the build if a solver-generated fairness counterexample is found for any active model/dataset combination under the encoded individual fairness property. It passes only when all active checks are UNSAT, and issues a warning when the evidence is incomplete or inconclusive.
