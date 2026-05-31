import json
from pathlib import Path
import pandas as pd

# Build a human-readable report for SAT counterexamples extracted from formal solver outputs.
# It reads the consolidated solver summary and the underlying per-model JSON artifacts.

SUMMARY_JSON = 'results/processed/formal_solver_summary.json'
OUT_MD = 'results/processed/formal_counterexample_report.md'
OUT_JSON = 'results/processed/formal_counterexample_report.json'
OUT_CSV = 'results/processed/formal_counterexample_report.csv'

# Mapping aligned with the solver artifact names already used in the project.
DETAIL_FILES = {
    ('scholarship', 'Decision Tree'): 'results/processed/smt_scholarship_tree.json',
    ('scholarship', 'Rule List'): 'results/processed/smt_scholarship_rulelist.json',
    ('scholarship', 'Logistic Regression'): 'results/processed/smt_scholarship_logreg.json',
    ('adult', 'Decision Tree'): 'results/processed/smt_adult_tree.json',
    ('adult', 'Rule List'): 'results/processed/smt_adult_rulelist.json',
    ('adult', 'Logistic Regression'): 'results/processed/smt_adult_logreg.json',
    ('german_credit', 'Decision Tree'): 'results/processed/smt_german_tree.json',
    ('german_credit', 'Rule List'): 'results/processed/smt_german_rulelist.json',
    ('german_credit', 'Logistic Regression'): 'results/processed/smt_german_logreg.json',
}


def must_load_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Missing required file: {path}')
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def to_float_if_possible(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if '/' in text:
        num, den = text.split('/', 1)
        try:
            return float(num) / float(den)
        except Exception:
            return None
    try:
        return float(text)
    except Exception:
        return None


def normalize_prediction(value):
    if value is None:
        return None
    text = str(value).strip()
    # Handles JSONs where predictions are stored as numbers or strings
    if text in {'0', '0.0'}:
        return 0
    if text in {'1', '1.0'}:
        return 1
    try:
        f = float(text)
        if f in (0.0, 1.0):
            return int(f)
    except Exception:
        pass
    return text


def build_example_diff(example_x, example_y):
    if not isinstance(example_x, dict) or not isinstance(example_y, dict):
        return []
    diffs = []
    keys = sorted(set(example_x.keys()) | set(example_y.keys()))
    for k in keys:
        vx = example_x.get(k)
        vy = example_y.get(k)
        if vx != vy:
            diffs.append({
                'feature': k,
                'value_x': vx,
                'value_y': vy,
            })
    return diffs


def main():
    summary_obj = must_load_json(SUMMARY_JSON)
    results = summary_obj.get('results', [])

    # Keep only FAIL/SAT cases
    sat_rows = [r for r in results if str(r.get('solver_status', '')).upper() == 'SAT']

    report_rows = []
    detailed_entries = []

    for row in sat_rows:
        dataset = row['dataset']
        model = row['model']
        sensitive_attribute = row['sensitive_attribute']
        detail_path = DETAIL_FILES.get((dataset, model))
        if not detail_path:
            raise KeyError(f'No detail file mapping for {(dataset, model)}')
        detail = must_load_json(detail_path)
        ce = detail.get('first_counterexample') or {}

        score_x_raw = ce.get('score_x')
        score_y_raw = ce.get('score_y')
        pred_x = normalize_prediction(ce.get('prediction_x'))
        pred_y = normalize_prediction(ce.get('prediction_y'))
        example_x = ce.get('example_x')
        example_y = ce.get('example_y')
        pair = row.get('first_counterexample_pair')
        src = ce.get('source_group')
        dst = ce.get('target_group')
        diffs = build_example_diff(example_x, example_y)

        report_rows.append({
            'dataset': dataset,
            'model': model,
            'sensitive_attribute': sensitive_attribute,
            'solver_status': row.get('solver_status'),
            'ci_status': row.get('ci_status'),
            'counterexample_pair': pair,
            'prediction_x': pred_x,
            'prediction_y': pred_y,
            'score_x_raw': score_x_raw,
            'score_y_raw': score_y_raw,
            'score_x_float': to_float_if_possible(score_x_raw),
            'score_y_float': to_float_if_possible(score_y_raw),
            'changed_features_count': len(diffs),
            'changed_features': '; '.join([f"{d['feature']}: {d['value_x']} -> {d['value_y']}" for d in diffs]),
        })

        detailed_entries.append({
            'dataset': dataset,
            'model': model,
            'sensitive_attribute': sensitive_attribute,
            'solver_status': row.get('solver_status'),
            'ci_status': row.get('ci_status'),
            'counterexample_pair': pair,
            'source_group': src,
            'target_group': dst,
            'prediction_x': pred_x,
            'prediction_y': pred_y,
            'score_x_raw': score_x_raw,
            'score_y_raw': score_y_raw,
            'score_x_float': to_float_if_possible(score_x_raw),
            'score_y_float': to_float_if_possible(score_y_raw),
            'example_x': example_x,
            'example_y': example_y,
            'changed_features': diffs,
            'encoding_scope': detail.get('encoding_scope'),
            'fairness_property': detail.get('fairness_property'),
        })

    out_df = pd.DataFrame(report_rows)
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False, encoding='utf-8')

    out_json = {
        'report_type': 'formal_counterexample_report',
        'num_counterexamples': len(detailed_entries),
        'entries': detailed_entries,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append('# Formal Counterexample Report\n')
    lines.append('This report summarizes all **SAT / FAIL** solver cases found in the formal fairness checks.\n')

    if not detailed_entries:
        lines.append('No counterexamples were found. All recorded cases are UNSAT / PASS.')
    else:
        lines.append('## Overview\n')
        lines.append('| Dataset | Model | Sensitive Attribute | Counterexample Pair | Prediction x | Prediction y | Score x | Score y | Changed Features |')
        lines.append('|---|---|---|---|---:|---:|---|---|---:|')
        for row in report_rows:
            sx = row['score_x_raw'] if row['score_x_raw'] is not None else 'None'
            sy = row['score_y_raw'] if row['score_y_raw'] is not None else 'None'
            lines.append(
                f"| {row['dataset']} | {row['model']} | {row['sensitive_attribute']} | {row['counterexample_pair']} | "
                f"{row['prediction_x']} | {row['prediction_y']} | {sx} | {sy} | {row['changed_features_count']} |"
            )

        lines.append('\n## Detailed Counterexamples\n')
        for entry in detailed_entries:
            lines.append(f"### {entry['dataset']} / {entry['model']}\n")
            lines.append(f"- **Sensitive attribute**: {entry['sensitive_attribute']}")
            lines.append(f"- **Counterexample pair**: {entry['counterexample_pair']}")
            lines.append(f"- **Prediction before**: {entry['prediction_x']}")
            lines.append(f"- **Prediction after**: {entry['prediction_y']}")
            lines.append(f"- **Score before**: {entry['score_x_raw']}")
            lines.append(f"- **Score after**: {entry['score_y_raw']}")
            lines.append(f"- **Encoding scope**: {entry['encoding_scope']}")
            lines.append(f"- **Fairness property**: {entry['fairness_property']}\n")

            lines.append('#### Changed Feature(s)\n')
            if not entry['changed_features']:
                lines.append('- No explicit per-feature difference could be reconstructed from the saved examples.\n')
            else:
                for diff in entry['changed_features']:
                    lines.append(f"- **{diff['feature']}**: `{diff['value_x']}` -> `{diff['value_y']}`")
                lines.append('')

            lines.append('#### Example x\n')
            lines.append('```json')
            lines.append(json.dumps(entry['example_x'], ensure_ascii=False, indent=2))
            lines.append('```\n')

            lines.append('#### Example y\n')
            lines.append('```json')
            lines.append(json.dumps(entry['example_y'], ensure_ascii=False, indent=2))
            lines.append('```\n')

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print({
        'saved_csv': OUT_CSV,
        'saved_json': OUT_JSON,
        'saved_md': OUT_MD,
        'num_counterexamples': len(detailed_entries),
    })


if __name__ == '__main__':
    main()
