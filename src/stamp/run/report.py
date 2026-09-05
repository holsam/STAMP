'''
STAMP: combined report for a full `stamp run`
'''

# Import external dependencies
import json
from pathlib import Path

# Import STAMP objects
from stamp.run.orchestrate import RunOutcome

# _decoy_banner: the PASS/FAIL/NOT RUN line the report leads with
def _decoy_banner(outcome: RunOutcome) -> str:
    if not outcome.decoy_enabled:
        return 'DECOY CONTROL: NOT RUN (--no-decoy)'
    if outcome.decoy_control is None:
        return 'DECOY CONTROL: NOT RUN (stopped before identify)'
    verdict = 'PASS' if outcome.decoy_control['passed'] else 'FAIL'
    return f'DECOY CONTROL: {verdict} — {outcome.decoy_control["reason"]}'

# write_report: write report.md and report.json
def write_report(outcome: RunOutcome, output_dir: Path) -> tuple[Path, Path]:
    payload = {
        'decoy_control': outcome.decoy_control,
        'decoy_enabled': outcome.decoy_enabled,
        'stop_after': outcome.stop_after,
        'identifications': outcome.identifications,
        'refine_results': outcome.refine_results,
        'provenance': outcome.sidecars,
    }
    json_path = output_dir / 'report.json'
    json_path.write_text(json.dumps(payload, indent=2))

    lines = [f'# STAMP run report', '', f'**{_decoy_banner(outcome)}**', '']
    failed = outcome.decoy_control is not None and not outcome.decoy_control['passed']
    if failed:
        lines += ['> The decoy control failed: a decoy class fits a predicted structure as',
                  '> well as the real data. Numbers below should not be treated as trustworthy.',
                  '']

    lines += ['## Identifications', '', 'class | candidate | fit score | gap to runner-up',
              '--|--|--|--']
    for row in outcome.identifications:
        lines.append(f'{row["cluster_id"]} | {row["candidate_protein"]} | {row["fit_score"]:.3f} | {row["score_gap_to_runner_up"]:.3f}')
    lines.append('')

    if outcome.refine_results:
        lines += ['## Resolution (FSC @ 0.143)', '', 'class | resolution (Å)',
                  '--|--']
        for row in outcome.refine_results:
            resolution = row['resolution_angstrom']
            lines.append(f'{row["class_id"]} | {resolution:.1f} |' if resolution else f'| {row["class_id"]} | n/a')
        lines.append('')

    lines += ['## Provenance', '']
    for name, sidecar in outcome.sidecars.items():
        lines.append(f'- `{name}` — commit `{sidecar.get("stamp_commit", "?")}`, '
                     f'{len(sidecar.get("input_checksums", {}))} input hashes')

    md_path = output_dir / 'report.md'
    md_path.write_text('\n'.join(lines) + '\n')
    return md_path, json_path
