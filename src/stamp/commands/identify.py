'''
STAMP: identification against predicted structures
'''

# Import external dependencies
import json, mrcfile, numpy as np, re
from collections import defaultdict
from pathlib import Path

# Import STAMP objects
from stamp.adapters.base import AdapterInputs
from stamp.adapters.mock import get_mock_adapter
from stamp.backends.base import ToolCommand
from stamp.backends.local import LocalRunner
from stamp.backends.mock import MockRunner
from stamp.identify.decoy_check import evaluate_decoy_control
from stamp.run.state import stage_dir
from stamp.identify.fit import fit_candidate, rank_candidates
from stamp.identify.panel import load_candidate_panel
from stamp.identify.simulate import simulate_density, to_comparable
from stamp.utils.io import write_sidecar

# _CLASS_ID: leading cNN token of a class-average filename
_CLASS_ID = re.compile(r'^(c\d+)')

# load_class_averages: {class_id: (mean_volume, voxel_size)} merged across half-set MRCs
def load_class_averages(directory: Path) -> dict[str, tuple[np.ndarray, float]]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    voxel_sizes: dict[str, float] = {}
    for path in sorted(directory.glob('*.mrc')):
        match = _CLASS_ID.match(path.stem)
        if not match:
            continue
        with mrcfile.open(str(path), permissive=True) as mrc:
            grouped[match.group(1)].append(np.transpose(np.asarray(mrc.data), (2, 1, 0)))
            voxel_sizes[match.group(1)] = float(mrc.voxel_size.x)
    if not grouped:
        raise ValueError(f'no cNN-named class averages in {directory}')
    return {cid: (np.mean(volumes, axis=0), voxel_sizes[cid]) for cid, volumes in grouped.items()}

# _score_panel: fit every candidate to every class average, returns {class_id: {candidate: score}}
def _score_panel(class_averages, panel, resolution, fitter, backend):
    all_scores: dict[str, dict[str, float]] = {}
    for class_id, (average, voxel_size) in class_averages.items():
        box_voxels = average.shape[-1]
        comparable_average = to_comparable(average, voxel_size, resolution)
        scores: dict[str, float] = {}
        for candidate in panel:
            simulated = simulate_density(candidate.structure_path, box_voxels, voxel_size, resolution)
            simulated = to_comparable(simulated, voxel_size, resolution, already_bandlimited=True)
            scores[candidate.name] = fit_candidate(comparable_average, simulated)
        all_scores[class_id] = scores
    return all_scores

# run_identify: fit the candidate panel to each class average and rank per class
def run_identify(
    classes: Path,
    candidates: Path,
    output_dir: Path,
    decoy_classes: Path | None,
    resolution: float | None,
    backend: str,
    fitter: str,
    fetch_missing: bool
) -> None:
    if resolution is None:
        raise ValueError('resolution is required: use --resolution or set [stage.identify].resolution')
    panel = load_candidate_panel(candidates, fetch_missing=fetch_missing)
    print(f'Loaded {len(panel)} candidates.')
    class_averages = load_class_averages(classes)
    print(f'Loaded {len(class_averages)} class averages.')

    real_scores = _score_panel(class_averages, panel, resolution, fitter, backend)
    results = [rank_candidates(class_id, scores, method=f'stamp-{fitter}') for class_id, scores in sorted(real_scores.items())]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'identification.json').write_text(json.dumps([r.model_dump() for r in results], indent=2))

    decoy_control = None
    if decoy_classes is not None:
        decoy_scores = _score_panel(load_class_averages(decoy_classes), panel, resolution, fitter, backend)
        real_best = [max(s.values()) for s in real_scores.values()]
        decoy_best = [max(s.values()) for s in decoy_scores.values()]
        decoy_control = evaluate_decoy_control(real_best, decoy_best)
        (output_dir / 'decoy_control.json').write_text(json.dumps(decoy_control.model_dump(), indent=2))

    _write_report(output_dir / 'identification_report.txt', results, real_scores, decoy_control, resolution)

    write_sidecar(
        output_dir,
        stage='identify',
        tool=f'stamp-{fitter}',
        tool_version=None,
        parameters={
            'fitter': fitter,
            'backend': backend,
            'resolution': resolution,
            'effective_resolution_angstrom': resolution,
            'symmetry': 'Cinf_z',
            'n_candidates': len(panel), 'n_classes': len(class_averages),
            'decoy_control': decoy_control.model_dump() if decoy_control else None,
        },
        inputs=[('class_averages', classes), ('candidates', candidates)]
        + ([('decoy_classes', decoy_classes)] if decoy_classes else []),
    )
    print(f'Wrote identification for {len(results)} classes to {output_dir}')

# build_identify_commands: create ToolCommand for `stamp identify`
def build_identify_commands(config, output_dir: Path) -> list[ToolCommand]:
    target = stage_dir(output_dir, 'real', 'identify')
    real_classes = stage_dir(output_dir, 'real', 'classify') / 'class_averages'
    decoy_classes = stage_dir(output_dir, 'decoy', 'classify') / 'class_averages'
    argv = [
        'stamp', 'identify',
        '--classes', str(real_classes),
        '--candidates', str(config.stage.identify.candidates),
        '--output-dir', str(target),
        '--fitter', config.stage.identify.fitter,
        '--backend', 'local',
    ]
    if config.decoy.enabled:
        argv += ['--decoy-classes', str(decoy_classes)]
    if config.stage.identify.resolution is not None:
        argv += ['--resolution', str(config.stage.identify.resolution)]
    if config.stage.identify.fetch_missing:
        argv.append('--fetch-missing')
    return [ToolCommand(tool='identify', argv=argv, working_directory=target, output_paths=[target / 'identification.json'])]

# _write_report: human-readable ranked table per class, decoy verdict first if present
def _write_report(path: Path, results, all_scores, decoy_control, resolution) -> None:
    lines: list[str] = [
        f'Comparison space: low-pass {resolution:.1f} Å, azimuthal average about the membrane normal (Cinf assumed; asymmetric features discarded, symmetry discrimination is done in classify).',
        '',
    ]
    if decoy_control is not None:
        banner = 'PASS' if decoy_control.passed else 'FAIL'
        lines += [f'DECOY CONTROL: {banner}', f'  {decoy_control.reason}', '']
    for result in results:
        gap = result.score_gap_to_runner_up
        gap_text = f'{gap:.3f}' if gap is not None else 'n/a (single candidate)'
        lines.append(f'{result.cluster_id}: {result.candidate_protein}  '
                     f'score={result.fit_score:.3f}  gap={gap_text}')
        for name, score in sorted(all_scores[result.cluster_id].items(), key=lambda kv: -kv[1]):
            lines.append(f'    {name:<24} {score:.3f}')
        lines.append('')
    path.write_text('\n'.join(lines))
