'''
STAMP: identification against predicted structures
'''

# Import external dependencies
import json, shutil
from pathlib import Path

# Import STAMP objects
from stamp.adapters.base import AdapterInputs
from stamp.adapters.mock import get_mock_adapter
from stamp.backends.local import LocalRunner
from stamp.backends.mock import MockRunner
from stamp.classify.cluster import prior_of_class
from stamp.backends.base import ToolCommand
from stamp.identify.averages import load_class_averages
from stamp.identify.decoy_check import evaluate_decoy_control
from stamp.identify.scoring import score_panel
from stamp.run.state import stage_dir
from stamp.identify.fit import rank_candidates
from stamp.identify.panel import load_candidate_panel
from stamp.utils.checkpoint import directory_signatures, signatures, sync_checkpoint
from stamp.utils.errors import StampPipelineError
from stamp.utils.log import log
from stamp.utils.options.argv import plot_argv
from stamp.utils.plotting.identify import plot_identify
from stamp.utils.provenance import write_sidecar

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

    # everything that changes scores; n_workers, backend and plots are left out so a run can resume with different resources
    fingerprint = {
        'classes': directory_signatures(classes),
        'decoy_classes': directory_signatures(decoy_classes),
        'candidates': signatures({'panel': candidates, **{f'structure:{candidate.name}': candidate.structure_path for candidate in panel}}),
        'resolution': resolution,
        'fitter': fitter,
    }     # the prior is part of each class id, so the class signatures already cover it
    checkpoint_dir = output_dir / 'raw'
    if sync_checkpoint(checkpoint_dir, fingerprint):
        log.info('Found matching partial output, resuming')

    log.progress(f'Fitting {len(panel)} candidates against {len(class_averages)} classes')
    real_scores = score_panel(class_averages, panel, resolution, fitter, backend, n_workers, checkpoint_dir / 'real')
    results = [rank_candidates(class_id, scores, method=f'stamp-{fitter}') for class_id, scores in sorted(real_scores.items())]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'identification.json').write_text(json.dumps([r.model_dump() for r in results], indent=2))

    decoy_control = None
    real_best, decoy_best = None, None
    if decoy_classes is not None:
        log.progress('Fitting candidates against decoy classes for control')
        decoy_scores = score_panel(load_class_averages(decoy_classes, allow_empty=True), panel, resolution, fitter, backend, n_workers, checkpoint_dir / 'decoy')
        real_best = [max(s.values()) for s in real_scores.values()]
        decoy_best = [max(s.values()) for s in decoy_scores.values()]
        decoy_control = evaluate_decoy_control(real_best, decoy_best)
        (output_dir / 'decoy_control.json').write_text(json.dumps(decoy_control.model_dump(), indent=2))
        if decoy_control.passed:
            log.info('Decoy control passed')
        else:
            log.warning(f'Decoy control failed: {decoy_control.reason}')
    else:
        log.debug('No --decoy-classes given, skipping decoy control')

    log.progress('Writing identification report')
    _write_report(output_dir / 'identification_report.txt', results, real_scores, decoy_control, resolution)

    write_sidecar(
        output_dir,
        stage='identify',
        tool=f'stamp-{fitter}',
        tool_version=None,
        parameters={
            'fitter': fitter,
            'backend': backend,
            'resolution_angstrom': resolution,
            'symmetry': 'C1',
            'orientation_priors': sorted({prior_of_class(class_id).value for class_id in class_averages}),
            'n_candidates': len(panel), 'n_classes': len(class_averages),
            'n_workers': n_workers,
            'decoy_control': decoy_control.model_dump() if decoy_control else None,
        },
        inputs=[('class_averages', classes), ('candidates', candidates)]
        + ([('decoy_classes', decoy_classes)] if decoy_classes else []),
    )
    log.info(f'Wrote identification for {len(results)} classes to {output_dir}')
    shutil.rmtree(checkpoint_dir, ignore_errors=True)  # identify has no keep_raw, the checkpoint is only needed until it completes

    if make_plots:
        plot_identify(real_scores, results, decoy_control, real_best, decoy_best, output_dir, plot_format)

# build_identify_commands: create ToolCommand for `stamp identify`
def build_identify_commands(config, output_dir: Path) -> list[ToolCommand]:
    target = stage_dir(output_dir, 'real', 'identify')
    real_classes = stage_dir(output_dir, 'real', 'classify') / 'class_averages'
    decoy_classes = stage_dir(output_dir, 'decoy', 'classify') / 'class_averages'
    argv = [
        'stamp', 'identify',
        '--classes', str(real_classes),
        '--candidates', str(config.stage.identify.candidates),
        '--out-dir', str(target),
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
    argv += plot_argv(config.plots)
    return [ToolCommand(tool='identify', argv=argv, working_directory=target, output_paths=[target / 'identification.json'])]

# _write_report: human-readable ranked table per class, decoy verdict first if present
def _write_report(path: Path, results, all_scores, decoy_control, resolution) -> None:
    space = f'low-pass {resolution:.1f} Å, class averages aligned in 3D by classify (C1), candidates registered over each class\'s own orientation prior (see id prefix)'
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
