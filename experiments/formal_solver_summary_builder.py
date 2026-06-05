import json
from pathlib import Path
import pandas as pd

# Formal solver summary builder for FairCheck artifacts.
# Expected JSON inputs:
# - Scholarship: smt_scholarship_tree.json, smt_scholarship_rulelist.json, smt_scholarship_logreg.json
# - Adult: smt_adult_tree.json, smt_adult_rulelist.json, smt_adult_logreg.json
# - German: smt_german_tree.json, smt_german_rulelist.json, smt_german_logreg.json

INPUT_FILES = {
    'scholarship': {
        'Decision Tree': 'results/processed/formal/smt_scholarship_tree.json',
        'Rule List': 'results/processed/formal/smt_scholarship_rulelist.json',
        'Logistic Regression': 'results/processed/formal/smt_scholarship_logreg.json',
    },
    'adult': {
        'Decision Tree': 'results/processed/formal/smt_adult_tree.json',
        'Rule List': 'results/processed/formal/smt_adult_rulelist.json',
        'Logistic Regression': 'results/processed/formal/smt_adult_logreg.json',
    },
    'german_credit': {
        'Decision Tree': 'results/processed/formal/smt_german_tree.json',
        'Rule List': 'results/processed/formal/smt_german_rulelist.json',
        'Logistic Regression': 'results/processed/formal/smt_german_logreg.json',
    },
}

CSV_OUT = 'results/processed/formal_solver_summary.csv'
MD_OUT = 'results/processed/formal_solver_summary.md'
JSON_OUT = 'results/processed/formal_solver_summary.json'


def must_load_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Missing required file: {path}')
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def solver_to_ci(status: str) -> str:
    s = str(status).upper()
    if s == 'UNSAT':
        return 'PASS'
    if s == 'SAT':
        return 'FAIL'
    return 'WARNING'


def format_pair(first_ce):
    if not first_ce:
        return None
    src = first_ce.get('source_group')
    dst = first_ce.get('target_group')
    if src is None or dst is None:
        return None
    return f'{src} -> {dst}'


def main():
    rows = []
    for dataset, model_map in INPUT_FILES.items():
        for model_name, file_path in model_map.items():
            obj = must_load_json(file_path)
            solver_status = obj.get('overall_status', 'UNKNOWN')
            ci_status = solver_to_ci(solver_status)
            first_ce = obj.get('first_counterexample')
            pair_checks = obj.get('pair_checks', [])
            rows.append({
                'dataset': obj.get('dataset', dataset),
                'model': model_name,
                'sensitive_attribute': obj.get('sensitive_attribute'),

                'runtime_seconds': obj.get('runtime_seconds'),
                'timeout': obj.get('timeout'),
                'ce_bound': obj.get('ce_bound'),
                'ce_found': obj.get('ce_found'),

                'solver_status': solver_status,
                'ci_status': ci_status,

                'checked_pairs': len(pair_checks),
                'has_counterexample': first_ce is not None,
                'first_counterexample_pair': format_pair(first_ce),

                'encoding_scope': obj.get('encoding_scope'),
                'fairness_property': obj.get('fairness_property'),
            })

    out_df = pd.DataFrame(rows)

    ds_order = {'scholarship': 0, 'adult': 1, 'german_credit': 2}
    model_order = {'Decision Tree': 0, 'Rule List': 1, 'Logistic Regression': 2}
    out_df['_ds_order'] = out_df['dataset'].map(ds_order)
    out_df['_m_order'] = out_df['model'].map(model_order)
    out_df = out_df.sort_values(['_ds_order', '_m_order']).drop(columns=['_ds_order', '_m_order'])

    Path(CSV_OUT).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(CSV_OUT, index=False, encoding='utf-8')

    summary = {
        'interpretation': {
            'UNSAT': 'PASS (no counterexample found for the encoded fairness property)',
            'SAT': 'FAIL (counterexample found)',
            'OTHER': 'WARNING / INCONCLUSIVE',
        },
        'results': out_df.to_dict(orient='records'),
    }
    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append('# Formal Solver Summary\n')
    lines.append('## Status Interpretation\n')
    lines.append('- **UNSAT** = **PASS** (no counterexample found for the encoded fairness property)')
    lines.append('- **SAT** = **FAIL** (counterexample found)')
    lines.append('- **OTHER** = **WARNING / INCONCLUSIVE**\n')

    lines.append('## Cross-Dataset Solver Results\n')
    lines.append('| Dataset | Model | SMT Status | CI Status | CE Found | CE Bound | Runtime(s) | Timeout |')
    lines.append('|---|---|---|---|---:|---:|---:|---|')
    
    for _, r in out_df.iterrows():
        lines.append(
            f"| {r['dataset']} "
            f"| {r['model']} "
            f"| {r['solver_status']} "
            f"| {r['ci_status']} "
            f"| {r['ce_found']} "
            f"| {r['ce_bound']} "
            f"| {r['runtime_seconds']} "
            f"| {r['timeout']} |"
        )
    pass_count = int((out_df['ci_status'] == 'PASS').sum())
    fail_count = int((out_df['ci_status'] == 'FAIL').sum())
    warn_count = int((out_df['ci_status'] == 'WARNING').sum())
    lines.append('\n## Aggregate Counts\n')
    lines.append(f'- **PASS**: {pass_count}')
    lines.append(f'- **FAIL**: {fail_count}')
    lines.append(f'- **WARNING**: {warn_count}')

    # Highlight recurring model-class pattern if present
    lines.append('\n## Quick Reading Guide\n')
    lines.append('- Use this summary to compare model-class behavior under the same encoded fairness property across Scholarship, Adult, and German Credit.')
    lines.append('- In the current artifact, **UNSAT** indicates no solver-level counterexample was found within the encoded scope, while **SAT** indicates a counterexample exists and can be inspected in the corresponding JSON file.')

    with open(MD_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print({
        'saved_csv': CSV_OUT,
        'saved_md': MD_OUT,
        'saved_json': JSON_OUT,
        'rows': len(out_df),
        'pass_count': pass_count,
        'fail_count': fail_count,
        'warning_count': warn_count,
    })


if __name__ == '__main__':
    main()
