'''
STAMP: identification against predicted structures
'''

# Import external dependencies
import json, matplotlib.pyplot as plt, mrcfile, numpy as np, re, tomllib
from collections import defaultdict
from dataclasses import dataclass
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
from stamp.identify.panel import Candidate, load_candidate_panel
from stamp.identify.simulate import simulate_density, to_comparable
from stamp.utils.errors import StampPipelineError
from stamp.utils.io import write_sidecar
from stamp.utils.log import log
from stamp.utils.parallel import run_parallel
from stamp.utils.plotting.core import PlotFormat, finish, plot_path
from stamp.utils.plotting.identify import decoy_hist, score_heatmap

# _CLASS_ID: leading cNN token of a class-average filename
_CLASS_ID = re.compile(r'^(c\d+)')

# _classify_was_inplane_aligned: read the classify sidecar
def _classify_was_inplane_aligned(classes: Path) -> bool:
    sidecar = classes.parent / 'params.toml'
    if not sidecar.is_file():
        return False
    return bool(tomllib.loads(sidecar.read_text()).get('parameters', {}).get('inplane_alignment'))

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
        raise StampPipelineError(f'No cNN-named class averages in {directory}')
    return {cid: (np.mean(volumes, axis=0), voxel_sizes[cid]) for cid, volumes in grouped.items()}

# _FitJob: one (class, candidate) pair
@dataclass(frozen=True)
class _FitJob:
    class_id: str
    comparable_average: np.ndarray
    box_voxels: int
    voxel_size: float
    resolution: float
    candidate: Candidate

# _fit_one: multiprocessing worker entry point
def _fit_one(job: _FitJob) -> tuple[str, str, float]:
    simulated = simulate_density(job.candidate.structure_path, job.box_voxels, job.voxel_size, job.resolution)
    simulated = to_comparable(simulated, job.voxel_size, job.resolution, already_bandlimited=True)
    score = fit_candidate(job.comparable_average, simulated)
    return job.class_id, job.candidate.name, score

# _score_panel: fit every candidate to every class average, returns {class_id: {candidate: score}}
def _score_panel(class_averages, panel, resolution, fitter, backend, n_workers: int = 1):
    jobs = [
        _FitJob(class_id, to_comparable(average, voxel_size, resolution), average.shape[-1], voxel_size, resolution, candidate)
        for class_id, (average, voxel_size) in class_averages.items()
        for candidate in panel
    ]
    all_scores: dict[str, dict[str, float]] = {class_id: {} for class_id in class_averages}

    def _on_success(job: _FitJob, result: tuple[str, str, float]) -> None:
        class_id, candidate_name, score = result
        all_scores[class_id][candidate_name] = score
        log.debug(f'{class_id}/{candidate_name}: score={score:.3f}')

    run_parallel(jobs, _fit_one, max_workers=n_workers, label='identify-fit', on_success=_on_success)
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
    fetch_missing: bool,
    n_workers: int = 1,
    make_plots: bool = True,
    plot_format: str = 'tiff',
) -> None:
    log.progress('Identifying classes against candidate panel')
    if resolution is None:
        raise StampPipelineError('Resolution is required: use --resolution or set [stage.identify].resolution')
    panel = load_candidate_panel(candidates, fetch_missing=fetch_missing)
    log.info(f'Loaded {len(panel)} candidates')
    class_averages = load_class_averages(classes)
    log.info(f'Loaded {len(class_averages)} class averages')
    inplane_aligned = _classify_was_inplane_aligned(classes)

    log.progress(f'Fitting {len(panel)} candidates against {len(class_averages)} classes')
    real_scores = _score_panel(class_averages, panel, resolution, fitter, backend, n_workers)
    results = [rank_candidates(class_id, scores, method=f'stamp-{fitter}') for class_id, scores in sorted(real_scores.items())]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'identification.json').write_text(json.dumps([r.model_dump() for r in results], indent=2))

    decoy_control = None
    real_best, decoy_best = None, None
    if decoy_classes is not None:
        log.progress('Fitting candidates against decoy classes for control')
        decoy_scores = _score_panel(load_class_averages(decoy_classes), panel, resolution, fitter, backend, n_workers)
        real_best = [max(s.values()) for s in real_scores.values()]
        decoy_best = [max(s.values()) for s in decoy_scores.values()]
        decoy_control = evaluate_decoy_control(real_best, decoy_best)
        (output_dir / 'decoy_control.json').write_text(json.dumps(decoy_control.model_dump(), indent=2))
        if decoy_control.passed:
            log.info('Decoy control passed')
        else:
            log.warning(f'Decoy control failed: {decoy_control.reason}')

    _write_report(output_dir / 'identification_report.txt', results, real_scores, decoy_control, resolution, inplane_aligned)

    write_sidecar(
        output_dir,
        stage='identify',
        tool=f'stamp-{fitter}',
        tool_version=None,
        parameters={
            'fitter': fitter,
            'backend': backend,
            'resolution_angstrom': resolution,
            'inplane_aligned_class_averages': inplane_aligned,
            'symmetry': 'C1' if inplane_aligned else 'Cinf_z',
            'n_candidates': len(panel), 'n_classes': len(class_averages),
            'n_workers': n_workers,
            'decoy_control': decoy_control.model_dump() if decoy_control else None,
        },
        inputs=[('class_averages', classes), ('candidates', candidates)]
        + ([('decoy_classes', decoy_classes)] if decoy_classes else []),
    )
    log.info(f'Wrote identification for {len(results)} classes to {output_dir}')

    if make_plots:
        _plot_identify(real_scores, results, decoy_control, real_best, decoy_best, output_dir, plot_format)

# _plot_identify: CC heatmap (+ decoy histogram, if a decoy control ran)
def _plot_identify(real_scores, results, decoy_control, real_best, decoy_best, output_dir: Path, fmt: PlotFormat) -> None:
    cluster_ids = [r.cluster_id for r in results]
    names = sorted({name for scores in real_scores.values() for name in scores})
    row_labels = [f'{r.cluster_id} = {r.candidate_protein} (score={r.fit_score:.2f})' for r in results]
    fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(cluster_ids) + 2))
    score_heatmap(ax, cluster_ids, names, real_scores, row_labels=row_labels, title='identify: fit score per candidate')
    finish(fig, plot_path(output_dir, 'identify_scores', fmt))
    if decoy_control is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        decoy_hist(ax, real_best, decoy_best, decoy_control.passed)
        finish(fig, plot_path(output_dir, 'decoy_control', fmt))

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
        '-n', str(config.stage.identify.n_workers),
    ]
    if config.decoy.enabled:
        argv += ['--decoy-classes', str(decoy_classes)]
    if config.stage.identify.resolution is not None:
        argv += ['--resolution', str(config.stage.identify.resolution)]
    if config.stage.identify.fetch_missing:
        argv.append('--fetch-missing')
    argv.append('--plots' if config.plots.enabled else '--no-plots')
    argv += ['--plot-format', config.plots.format]
    return [ToolCommand(tool='identify', argv=argv, working_directory=target, output_paths=[target / 'identification.json'])]

# _write_report: human-readable ranked table per class, decoy verdict first if present
def _write_report(path: Path, results, all_scores, decoy_control, resolution, inplane_aligned) -> None:
    space = f'low-pass {resolution:.1f} Å, in-plane angles estimated in classify (C1)' if inplane_aligned else f'low-pass {resolution:.1f} Å, azimuthal average about the membrane normal (Cinf assumed; asymmetric features discarded)'
    lines: list[str] = [
        f'Comparison space: {space}.',
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
