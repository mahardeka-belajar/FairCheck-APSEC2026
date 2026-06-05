import json
from pathlib import Path
import sys
from datetime import datetime

FORMAL_SUMMARY_JSON = (
    'results/processed/formal/'
    'formal_solver_summary.json'
)

COUNTEREXAMPLE_JSON = (
    'results/processed/formal/'
    'formal_counterexample_report.json'
)
STATUS_JSON = 'results/processed/ci_gate_status.json'
REPORT_MD = 'results/processed/ci_gate_report.md'


def must_load_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Missing required file: {path}')
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    formal = must_load_json(FORMAL_SUMMARY_JSON)
    counter = must_load_json(COUNTEREXAMPLE_JSON)

    results = formal.get('results', [])
    pass_rows = [r for r in results if str(r.get('ci_status', '')).upper() == 'PASS']
    fail_rows = [r for r in results if str(r.get('ci_status', '')).upper() == 'FAIL']
    warn_rows = [r for r in results if str(r.get('ci_status', '')).upper() == 'WARNING']

    if fail_rows:
        overall_status = 'FAIL'
        exit_code = 1
    elif warn_rows:
        overall_status = 'WARNING'
        exit_code = 2
    else:
        overall_status = 'PASS'
        exit_code = 0

    failed_models = [f"{r['dataset']} / {r['model']}" for r in fail_rows]
    warning_models = [f"{r['dataset']} / {r['model']}" for r in warn_rows]
    passed_models = [f"{r['dataset']} / {r['model']}" for r in pass_rows]

    payload = {
        'generated_at_utc': datetime.utcnow().isoformat() + 'Z',
        'overall_status': overall_status,
        'exit_code': exit_code,
        'counts': {
            'pass': len(pass_rows),
            'fail': len(fail_rows),
            'warning': len(warn_rows),
            'total': len(results),
        },
        'failed_models': failed_models,
        'warning_models': warning_models,
        'passed_models': passed_models,
        'inputs': {
            'formal_solver_summary_json': FORMAL_SUMMARY_JSON,
            'formal_counterexample_report_json': COUNTEREXAMPLE_JSON,
        }
    }

    Path(STATUS_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append('# CI Gate Report\n')
    lines.append('## Overall Status\n')
    lines.append(f"- **Overall gate status**: **{overall_status}**")
    lines.append(f"- **Exit code**: `{exit_code}`")
    lines.append(f"- **Generated at (UTC)**: `{payload['generated_at_utc']}`\n")

    lines.append('## Policy\n')
    lines.append('- **PASS**: no solver-level counterexample found in any active fairness check')
    lines.append('- **FAIL**: at least one solver-level counterexample found (SAT case)')
    lines.append('- **WARNING**: inconclusive or missing-status case detected\n')

    lines.append('## Counts\n')
    lines.append(f"- **PASS**: {len(pass_rows)}")
    lines.append(f"- **FAIL**: {len(fail_rows)}")
    lines.append(f"- **WARNING**: {len(warn_rows)}")
    lines.append(f"- **TOTAL**: {len(results)}\n")

    lines.append('## Per-Model Solver Gate Results\n')
    lines.append('| Dataset | Model | Sensitive Attribute | Solver Status | CI Status | Checked Pairs | Counterexample Pair |')
    lines.append('|---|---|---|---|---|---:|---|')
    for r in results:
        ce = r.get('first_counterexample_pair') or 'None'
        lines.append(f"| {r.get('dataset')} | {r.get('model')} | {r.get('sensitive_attribute')} | {r.get('solver_status')} | {r.get('ci_status')} | {r.get('checked_pairs')} | {ce} |")

    lines.append('\n## Gate Decision Rationale\n')
    if fail_rows:
        lines.append('- The gate is marked **FAIL** because one or more solver checks returned **SAT**, meaning a fairness counterexample was found.')
        if failed_models:
            lines.append(f"- **Failed models**: {', '.join(failed_models)}")
    elif warn_rows:
        lines.append('- The gate is marked **WARNING** because at least one result was inconclusive or had a non-standard status.')
        if warning_models:
            lines.append(f"- **Warning models**: {', '.join(warning_models)}")
    else:
        lines.append('- The gate is marked **PASS** because all solver checks returned **UNSAT**, so no counterexample was found in the active scope.')

    if passed_models:
        lines.append(f"- **Passed models**: {', '.join(passed_models)}")

    # Add counterexample references if available
    entries = counter.get('entries', []) if isinstance(counter, dict) else []
    if entries:
        lines.append('\n## Counterexample Summary\n')
        for entry in entries:
            lines.append(f"- **{entry.get('dataset')} / {entry.get('model')}** triggered **{entry.get('counterexample_pair')}** with prediction change `{entry.get('prediction_x')}` -> `{entry.get('prediction_y')}`.")

    lines.append('\n## Inputs\n')
    lines.append(f"- Formal solver summary: `{FORMAL_SUMMARY_JSON}`")
    lines.append(f"- Counterexample report: `{COUNTEREXAMPLE_JSON}`")

    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(json.dumps({
        'saved_status_json': STATUS_JSON,
        'saved_report_md': REPORT_MD,
        'overall_status': overall_status,
        'exit_code': exit_code,
        'pass_count': len(pass_rows),
        'fail_count': len(fail_rows),
        'warning_count': len(warn_rows),
    }, ensure_ascii=False, indent=2))

    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
