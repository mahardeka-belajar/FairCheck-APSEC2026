import json
from pathlib import Path

INPUT_FILES = {
    'Decision Tree': 'results/processed/smt_scholarship_tree.json',
    'Rule List': 'results/processed/smt_scholarship_rulelist.json',
    'Logistic Regression': 'results/processed/smt_scholarship_logreg.json',
}

JSON_OUT = 'results/processed/ci_fairness_demo_summary.json'
MD_OUT = 'results/processed/ci_fairness_demo_report.md'


def load_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Missing required file: {path}')
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def status_to_ci(overall_status: str) -> str:
    s = str(overall_status).upper()
    if s == 'UNSAT':
        return 'PASS'
    if s == 'SAT':
        return 'FAIL'
    return 'WARNING'


def main():
    rows = []
    for model_name, file_path in INPUT_FILES.items():
        obj = load_json(file_path)
        solver_status = obj.get('overall_status', 'UNKNOWN')
        ci_status = status_to_ci(solver_status)
        first_ce = obj.get('first_counterexample', None)
        rows.append({
            'dataset': obj.get('dataset', 'scholarship'),
            'model': model_name,
            'sensitive_attribute': obj.get('sensitive_attribute', 'Asal_Daerah'),
            'solver_status': solver_status,
            'ci_status': ci_status,
            'checked_pairs': len(obj.get('pair_checks', [])),
            'has_counterexample': first_ce is not None,
            'counterexample_pair': None if first_ce is None else f"{first_ce.get('source_group')} -> {first_ce.get('target_group')}",
        })

    final = {
        'dataset': 'scholarship',
        'interpretation': {
            'UNSAT': 'PASS (no counterexample found for the encoded fairness property)',
            'SAT': 'FAIL (counterexample found)',
            'OTHER': 'WARNING / INCONCLUSIVE'
        },
        'results': rows
    }

    Path(JSON_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append('# FairCheck-CI Mini Demo (Scholarship)\n')
    lines.append('## CI-style Status Mapping\n')
    lines.append('- **UNSAT** = **PASS** (no counterexample found for the encoded fairness property)')
    lines.append('- **SAT** = **FAIL** (counterexample found)')
    lines.append('- **OTHER** = **WARNING / INCONCLUSIVE**\n')

    lines.append('## Model Results\n')
    lines.append('| Dataset | Model | Sensitive Attribute | Solver Status | CI Status | Checked Pairs | Counterexample |')
    lines.append('|---|---|---|---|---|---:|---|')
    for r in rows:
        ce = 'None' if not r['has_counterexample'] else r['counterexample_pair']
        lines.append(f"| {r['dataset']} | {r['model']} | {r['sensitive_attribute']} | {r['solver_status']} | {r['ci_status']} | {r['checked_pairs']} | {ce} |")

    lines.append('\n## Summary\n')
    pass_models = [r['model'] for r in rows if r['ci_status'] == 'PASS']
    fail_models = [r['model'] for r in rows if r['ci_status'] == 'FAIL']
    warn_models = [r['model'] for r in rows if r['ci_status'] == 'WARNING']

    if pass_models:
        lines.append(f"- **PASS models**: {', '.join(pass_models)}")
    if fail_models:
        lines.append(f"- **FAIL models**: {', '.join(fail_models)}")
    if warn_models:
        lines.append(f"- **WARNING models**: {', '.join(warn_models)}")

    for r in rows:
        if r['model'] == 'Logistic Regression' and r['has_counterexample']:
            lines.append('- Logistic Regression produces a solver-level counterexample, making it the natural **FAIL** case in a FairCheck-CI demonstration.')
        if r['model'] in ('Decision Tree', 'Rule List') and r['ci_status'] == 'PASS':
            lines.append(f'- {r["model"]} returns **UNSAT**, making it a natural **PASS** case in a FairCheck-CI demonstration.')

    with open(MD_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print({'saved_json': JSON_OUT, 'saved_md': MD_OUT, 'rows': len(rows)})


if __name__ == '__main__':
    main()
