'''
STAMP: unsupervised classification logic
'''

# Import external dependencies
import json, os, numpy as np
from dataclasses import dataclass
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.cluster import ClusteringConfig, ClusteringResult
from stamp.utils.checkpoint import file_signature, signatures, sync_checkpoint
from stamp.classify.polish import PolishConfig
from stamp.classify.extract import extract_particle_set
from stamp.classify.prepare import working_box
from stamp.classify.shfeatures import build_sh_feature_matrix
from stamp.classify.features import build_feature_matrix
from stamp.backends.base import ToolCommand
from stamp.decoy.validate import is_decoy_particle_set
from stamp.picking.manifests import match_by_stem
from stamp.run.state import stage_dir
from stamp.schemas.particles import ClassAssignment, OrientationPrior, Particle, load_particle_set
from stamp.utils.errors import StampPipelineError
from stamp.utils.io import cleanup_raw
from stamp.utils.log import log, log_config
from stamp.utils.mrc import glob_mrc, paths_by_stem, require_voxel_size
from stamp.utils.options.argv import keep_raw_argv, plot_argv, seg_dir_argv
from stamp.utils.provenance import write_sidecar
from stamp.utils.plotting.classify import plot_classify
from stamp.classify.strategies import classify_combined, classify_strict

# run_classify: cluster picked particles by structural similarity per orientation prior
def run_classify(
    particles,
    raw_tomogram_dir,
    segmentation_dir,
    output_dir,
    voxel_size_angstrom,
    box_angstrom,
    n_radial_bins,
    method,
    min_cluster_size,
    n_clusters,
    n_components,
    strict_halfset_independence,
    random_state,
    inplane_angular_step_degrees = 10.0,
    azimuthal_modes = 4,
    min_radius_fraction = 0.25,
    n_azimuthal_samples = 64,
    feature_normalisation = 'both',
    n_workers: int = 1,
    make_plots: bool = True,
    plot_format: str = 'tiff',
    keep_raw: bool = False,
    tilt_max_degrees: float | None = None,
    tilt_axis_degrees: float = 90.0,
    bin_factor: int = 2,
    n_rotations: int = 3000,
    shift_range_voxels: int = 3,
    polish_rounds: int = 2,
    null_sigmas: float = 3.0,
    reject_quantile: float = 0.95,
    merge_fraction: float = 1.0,
) -> None:
    voxel_size_angstrom = require_voxel_size(voxel_size_angstrom, raw_tomogram_dir)
    log.progress('Classifying particle set')
    particle_set = load_particle_set(particles)
    is_decoy = is_decoy_particle_set(particle_set)
    groups = particle_set.by_prior()
    log.info(f'Loaded {len(particle_set.particles)} {"decoy" if is_decoy else "real"} particles ({", ".join(f"{len(group)} {prior.value}" for prior, group in groups.items())})')
    tomogram_ids = sorted({particle.tomogram_id for particle in particle_set.particles})
    matched, unmatched = match_by_stem([Path(f'{tomogram_id}.mrc') for tomogram_id in tomogram_ids], glob_mrc(raw_tomogram_dir))
    for stem_path in unmatched:
        log.warning(f'No raw tomogram matching {stem_path.stem}, its particles will be skipped')
    tomogram_paths = {stem_path.stem: raw_path for stem_path, raw_path in matched.items()}
    segmentation_paths = (
        {stem: str(path) for stem, path in paths_by_stem(segmentation_dir).items()} if segmentation_dir is not None else {}
    )
    box_voxels = int(round(box_angstrom / voxel_size_angstrom))
    if box_voxels % 2 == 0:
        box_voxels += 1  # odd box keeps the particle exactly centred
        log.debug(f'Using odd box voxel for centred particles: {box_voxels - 1} -> {box_voxels}')
    log.progress(f'Using box voxels: {box_voxels}')

    # everything that changes results; n_workers, plots and keep_raw are left out so a run can resume with different resources
    fingerprint = {
        'particles': file_signature(particles, content=True),
        'raw': signatures(tomogram_paths),
        'segmentation': signatures(segmentation_paths),
        'voxel_size_angstrom': voxel_size_angstrom,
        'box_voxels': box_voxels,
        'n_radial_bins': n_radial_bins,
        'azimuthal_modes': azimuthal_modes,
        'n_azimuthal_samples': n_azimuthal_samples,
        'min_radius_fraction': min_radius_fraction,
        'feature_normalisation': feature_normalisation,
        'method': method,
        'min_cluster_size': min_cluster_size,
        'n_clusters': n_clusters,
        'n_components': n_components,
        'strict_halfset_independence': strict_halfset_independence,
        'inplane_angular_step_degrees': inplane_angular_step_degrees,
        'random_state': random_state,
        'tilt_max_degrees': tilt_max_degrees,
        'tilt_axis_degrees': tilt_axis_degrees,
        'bin_factor': bin_factor,
        'n_rotations': n_rotations,
        'shift_range_voxels': shift_range_voxels,
        'polish_rounds': polish_rounds,
        'null_sigmas': null_sigmas,
        'reject_quantile': reject_quantile,
        'merge_fraction': merge_fraction,
    }    # the particle set signature already covers every prior
    if sync_checkpoint(output_dir / 'raw', fingerprint):
        log.info('Found matching partial output, resuming')

    config = ClusteringConfig(
        method=method,
        n_components=n_components,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    log_config('classify.clustering', config)

    # set environment variable NUMBA_NUM_THREADS so fast_hdbscan doesn't try to use all CPUs on compute clusters
    os.environ['NUMBA_NUM_THREADS'] = str(n_workers)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not strict_halfset_independence:
        log.warning('--no-strict-halfset-independence used, half-A and half-B particles are classified together; FSC built on these classes will be inflated')

    settings = _GroupSettings(
        box_voxels=box_voxels, voxel_size_angstrom=voxel_size_angstrom, n_radial_bins=n_radial_bins,
        azimuthal_modes=azimuthal_modes, n_azimuthal_samples=n_azimuthal_samples, min_radius_fraction=min_radius_fraction,
        feature_normalisation=feature_normalisation, clustering=config, strict_halfset_independence=strict_halfset_independence,
        inplane_angular_step_degrees=inplane_angular_step_degrees, n_rotations=n_rotations, shift_range_voxels=shift_range_voxels,
        polish_rounds=polish_rounds, min_cluster_size=min_cluster_size, merge_fraction=merge_fraction, null_sigmas=null_sigmas,
        reject_quantile=reject_quantile, random_state=random_state, bin_factor=bin_factor,
        tilt_max_degrees=tilt_max_degrees, tilt_axis_degrees=tilt_axis_degrees, n_workers=n_workers,
    )
    results: dict[OrientationPrior, _GroupResult] = {}
    failures: dict[OrientationPrior, str] = {}
    for prior, group in groups.items():
        log.progress(f'Classifying {len(group)} {prior.value}-prior particles')
        try:
            results[prior] = _classify_prior(prior, group, {k: str(v) for k, v in tomogram_paths.items()}, segmentation_paths, settings, output_dir)
        except StampPipelineError as exc:
            if len(groups) == 1:
                raise                       # a single prior fails
            log.warning(f'Skipping the {prior.value} prior: {exc}')
            failures[prior] = str(exc)
    if not results:
        raise StampPipelineError('No particles could be classified: ' + '; '.join(f'{prior.value}: {reason}' for prior, reason in failures.items()))
    assignments = [assignment for result in results.values() for assignment in result.assignments]

    assignments_path = output_dir / 'class_assignments.json'
    assignments_path.write_text(json.dumps([a.model_dump() for a in assignments], indent=2))

    cleanup_raw(output_dir, keep_raw=keep_raw, label='raw classify output')

    write_sidecar(
        output_dir,
        stage='classify',
        tool='stamp-polish-classifier',
        tool_version=None,
        parameters={
            'box_angstrom': box_angstrom,
            'box_voxels': box_voxels,
            'n_radial_bins': n_radial_bins,
            'method': method,
            'min_cluster_size': min_cluster_size,
            'n_clusters': n_clusters,
            'n_components': n_components,
            'strict_halfset_independence': strict_halfset_independence,
            'inplane_angular_step_degrees': inplane_angular_step_degrees,
            'random_state': random_state,
            'orientation_priors': {prior.value: len(group) for prior, group in groups.items()},
            'skipped_priors': {prior.value: reason for prior, reason in failures.items()},
            'tilt_max_degrees': tilt_max_degrees,
            'tilt_axis_degrees': tilt_axis_degrees,
            'bin_factor': bin_factor,
            'n_rotations': n_rotations,
            'shift_range_voxels': shift_range_voxels,
            'polish_rounds': polish_rounds,
            'null_sigmas': null_sigmas,
            'reject_quantile': reject_quantile,
            'merge_fraction': merge_fraction,
            'azimuthal_modes': azimuthal_modes,
            'n_azimuthal_samples': n_azimuthal_samples,
            'min_radius_fraction': min_radius_fraction,
            'feature_normalisation': feature_normalisation,
            'n_workers': n_workers,
            'is_decoy': is_decoy,
            'membrane_subtracted': bool(segmentation_paths),
            'n_extracted': sum(len(result.kept) for result in results.values()),
            'n_skipped': sum(len(result.skipped) for result in results.values()),
        },
        inputs=[('particle_set', particles)]
        + [(f'raw_tomogram:{stem}', path) for stem, path in sorted(tomogram_paths.items())],
    )
    log.info(f'Wrote {len(assignments)} class assignments to {assignments_path}')
    if make_plots:
        for prior, result in results.items():
            plot_classify(result.result_by_half, result.averages_by_half, output_dir, plot_format, prior.value)

# _GroupSettings: classify settings shared by every orientation prior group
@dataclass(frozen=True)
class _GroupSettings:
    box_voxels: int
    voxel_size_angstrom: float
    n_radial_bins: int
    azimuthal_modes: int
    n_azimuthal_samples: int
    min_radius_fraction: float
    feature_normalisation: str
    clustering: ClusteringConfig
    strict_halfset_independence: bool
    inplane_angular_step_degrees: float
    n_rotations: int
    shift_range_voxels: int
    polish_rounds: int
    min_cluster_size: int
    merge_fraction: float
    null_sigmas: float
    reject_quantile: float
    random_state: int
    bin_factor: int
    tilt_max_degrees: float | None
    tilt_axis_degrees: float
    n_workers: int

# _GroupResult: results of classifying one orientation prior
@dataclass
class _GroupResult:
    assignments: list[ClassAssignment]
    kept: list[Particle]
    skipped: list[Particle]
    result_by_half: dict[str, ClusteringResult]
    averages_by_half: dict[str, dict[str, tuple[np.ndarray, int]]]

# _classify_prior: extract, featurise, seed and polish one orientation prior group (cached to raw/<prior>)
def _classify_prior(
    prior: OrientationPrior,
    particles: list[Particle],
    tomogram_paths: dict[str, str],
    segmentation_paths: dict[str, str],
    settings: _GroupSettings,
    output_dir: Path,
) -> _GroupResult:
    work_dir = output_dir / 'raw' / prior.value
    log.progress(f'Starting {prior.value} subvolume extraction')
    subvols, kept, skipped = extract_particle_set(particles, tomogram_paths, settings.box_voxels, segmentation_paths=segmentation_paths, n_workers=settings.n_workers, cache_dir=work_dir / 'subvolumes', prior=prior)
    if skipped:
        log.warning(f'Skipped {len(skipped)} {prior.value} particles whose {settings.box_voxels}-voxel box fell outside the volume or had no matching tomogram or had no orientation')
    if not kept:
        raise StampPipelineError('No particles could be extracted. Check --raw-dir and --box-length-a')

    log.progress(f'Starting {prior.value} feature matrix construction')
    if prior is OrientationPrior.NONE:
        features = build_sh_feature_matrix(subvols, settings.tilt_max_degrees, settings.tilt_axis_degrees, n_workers=settings.n_workers)
    else:
        features = build_feature_matrix(
            subvols,
            n_radial_bins=settings.n_radial_bins,
            max_azimuthal_mode=settings.azimuthal_modes,
            n_azimuthal_samples=settings.n_azimuthal_samples,
            min_radius_fraction=settings.min_radius_fraction,
            n_workers=settings.n_workers,
            cache_dir=work_dir / 'features',
            normalisation=settings.feature_normalisation,
            axial_invariant=prior is OrientationPrior.AXIS,
        )
    log.progress(f'Feature matrix complete: {features.shape[0]} particles x {features.shape[1]} features ({prior.value} prior)')

    degenerate = ~features.any(axis=1)
    if degenerate.any():
        log.warning(f'Dropped {int(degenerate.sum())} {prior.value} particles with a constant/empty subvolume (edge fill)')
        keep_indices = [i for i, bad in enumerate(degenerate) if not bad]
        features = features[~degenerate]
        subvols = subvols.take(keep_indices)
        kept = [particle for particle, bad in zip(kept, degenerate) if not bad]
    if not kept:
        raise StampPipelineError('No particles left after dropping degenerate subvolumes')

    polish_config = PolishConfig(
        prior=prior, angular_step_degrees=settings.inplane_angular_step_degrees, n_rotations=settings.n_rotations, shift_range_voxels=settings.shift_range_voxels,
        n_polish_rounds=settings.polish_rounds, min_seed_size=settings.min_cluster_size, merge_fraction=settings.merge_fraction, null_sigmas=settings.null_sigmas,
        reject_quantile=settings.reject_quantile, seed=settings.random_state, threads=settings.n_workers,
    )
    classify = classify_strict if settings.strict_halfset_independence else classify_combined
    assignments, result_by_half, averages_by_half = classify(kept, subvols, features, settings.clustering, polish_config, working_box(settings.box_voxels, settings.bin_factor), (settings.tilt_max_degrees, settings.tilt_axis_degrees), output_dir, work_dir, settings.voxel_size_angstrom)
    return _GroupResult(assignments, kept, skipped, result_by_half, averages_by_half)

# build_classify_commands: create ToolCommand for `stamp classify`
def build_classify_commands(config, output_dir: Path, track: str = 'real') -> list[ToolCommand]:
    pick_name = 'particle_set.json' if track == 'real' else 'decoy_particle_set.json'
    particles = stage_dir(output_dir, track, 'pick') / pick_name
    target = stage_dir(output_dir, track, 'classify')
    settings = config.stage.classify
    argv = [
        'stamp', 'classify',
        '--particles', str(particles),
        '--raw-dir', str(config.run.raw_tomogram_dir),
        '--out-dir', str(target),
        '--voxel-size-a', str(config.run.voxel_size_angstrom),
        '--box-length-a', str(settings.box_angstrom),
        '--n-bins', str(settings.n_radial_bins),
        '--method', settings.method,
        '--min-cluster-size', str(settings.min_cluster_size),
        '--n-clusters', str(settings.n_clusters),
        '--n-components', str(settings.n_components),
        '--seed', str(settings.random_state),
        '--inplane-step-deg', str(settings.inplane_angular_step_degrees),
        '-n', str(settings.n_workers),
        '--tilt-axis', str(config.run.tilt_axis_degrees),
        '--bin-factor', str(settings.bin_factor),
        '--n-rots', str(settings.n_rotations),
        '--shift-range', str(settings.shift_range_voxels),
        '--polish-rounds', str(settings.polish_rounds),
        '--null-sigmas', str(settings.null_sigmas),
        '--reject-quantile', str(settings.reject_quantile),
        '--merge-fraction', str(settings.merge_fraction),
    ]
    if config.run.tilt_max_degrees is not None:
        argv += ['--tilt-max', str(config.run.tilt_max_degrees)]
    argv += seg_dir_argv(config.run.segmentation_dir)
    if not settings.strict_halfset_independence:
        argv.append('--no-strict-halfset-independence')
    argv += keep_raw_argv(settings.keep_raw)
    argv += plot_argv(config.plots)
    return [ToolCommand(tool='classify', argv=argv, working_directory=target, output_paths=[target / 'class_averages'])]

