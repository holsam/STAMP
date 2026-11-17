'''
STAMP: unsupervised classification logic
'''

# Import external dependencies
import json, os, numpy as np
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.cluster import ClusteringConfig, ClusteringResult
from stamp.utils.checkpoint import file_signature, signatures, sync_checkpoint
from stamp.classify.polish import PolishConfig
from stamp.classify.extract import extract_particle_set
from stamp.classify.shfeatures import build_sh_feature_matrix
from stamp.classify.features import build_feature_matrix
from stamp.decoy.validate import is_decoy_particle_set
from stamp.picking.manifests import match_by_stem
from stamp.schemas.particles import ClassAssignment, OrientationPrior, Particle, ParticleSet, load_particle_set
from stamp.utils.errors import StampPipelineError
from stamp.utils.io import cleanup_raw
from stamp.utils.log import log, log_config
from stamp.utils.mrc import glob_mrc, paths_by_stem, require_voxel_size
from stamp.utils.provenance import write_sidecar
from stamp.utils.plotting.classify import plot_classify
from stamp.classify.settings import GroupSettings
from stamp.classify.strategies import classify_combined, classify_strict

# build_group_settings: settings from classify options
def build_group_settings(
    box_voxels: int,
    voxel_size_angstrom: float,
    *,
    n_radial_bins: int,
    method: str,
    min_cluster_size: int,
    n_clusters: int,
    n_components: int,
    strict_halfset_independence: bool,
    random_state: int,
    inplane_angular_step_degrees: float,
    azimuthal_modes: int,
    min_radius_fraction: float,
    n_azimuthal_samples: int,
    feature_normalisation: str,
    tilt_max_degrees: float | None,
    tilt_axis_degrees: float,
    bin_factor: int,
    n_rotations: int,
    shift_range_voxels: int,
    polish_rounds: int,
    null_sigmas: float,
    reject_quantile: float,
    merge_fraction: float,
    n_workers: int,
) -> GroupSettings:
    config = ClusteringConfig(
        method=method,
        n_components=n_components,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    log_config('classify.clustering', config)
    return GroupSettings(
        box_voxels=box_voxels, voxel_size_angstrom=voxel_size_angstrom, n_radial_bins=n_radial_bins,
        azimuthal_modes=azimuthal_modes, n_azimuthal_samples=n_azimuthal_samples, min_radius_fraction=min_radius_fraction,
        feature_normalisation=feature_normalisation, clustering=config, strict_halfset_independence=strict_halfset_independence,
        inplane_angular_step_degrees=inplane_angular_step_degrees, n_rotations=n_rotations, shift_range_voxels=shift_range_voxels,
        polish_rounds=polish_rounds, min_cluster_size=min_cluster_size, merge_fraction=merge_fraction, null_sigmas=null_sigmas,
        reject_quantile=reject_quantile, random_state=random_state, bin_factor=bin_factor,
        tilt_max_degrees=tilt_max_degrees, tilt_axis_degrees=tilt_axis_degrees, n_workers=n_workers,
    )

# classify_particle_set: classify every prior group of a particle set and write assignments, sidecar and plots
def classify_particle_set(
    particle_set: ParticleSet,
    settings: GroupSettings,
    output_dir: Path,
    *,
    tomogram_paths: dict[str, str],
    segmentation_paths: dict[str, str],
    fingerprint: dict,
    sidecar_inputs: list[tuple[str, Path | str]],
    extra_parameters: dict,
    keep_raw: bool,
    make_plots: bool,
    plot_format: str,
    cache_only: bool = False,   # stream: subvolumes come from the append-time cache, no tomogram is read
    subvolume_cache_dir: Callable[[OrientationPrior], Path] | None = None,   # stream: cache location per prior
    membrane_subtracted: bool | None = None,   # stream: recorded at append time, not from segmentation_paths
) -> None:
    # set environment variable NUMBA_NUM_THREADS so fast_hdbscan doesn't try to use all CPUs on compute clusters
    os.environ['NUMBA_NUM_THREADS'] = str(settings.n_workers)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not settings.strict_halfset_independence:
        log.warning('--no-strict-halfset-independence used, half-A and half-B particles are classified together; FSC built on these classes will be inflated')
    is_decoy = is_decoy_particle_set(particle_set)
    groups = particle_set.by_prior()

    if sync_checkpoint(output_dir / 'raw', fingerprint):
        log.info('Found matching partial output, resuming')
    results: dict[OrientationPrior, _GroupResult] = {}
    failures: dict[OrientationPrior, str] = {}
    for prior, group in groups.items():
        log.progress(f'Classifying {len(group)} {prior.value}-prior particles')
        try:
            results[prior] = _classify_prior(prior, group, tomogram_paths, segmentation_paths, settings, output_dir, cache_only=cache_only, subvolume_cache_dir=subvolume_cache_dir)
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
            **extra_parameters,
            **asdict(settings),
            'orientation_priors': {prior.value: len(group) for prior, group in groups.items()},
            'skipped_priors': {prior.value: reason for prior, reason in failures.items()},
            'is_decoy': is_decoy,
            'membrane_subtracted': bool(segmentation_paths) if membrane_subtracted is None else membrane_subtracted,
            'n_extracted': sum(len(result.kept) for result in results.values()),
            'n_skipped': sum(len(result.skipped) for result in results.values()),
        },
        inputs=sidecar_inputs,
    )
    log.info(f'Wrote {len(assignments)} class assignments to {assignments_path}')
    if make_plots:
        for prior, result in results.items():
            plot_classify(result.result_by_half, result.averages_by_half, output_dir, plot_format, prior.value)

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

    settings = build_group_settings(
        box_voxels, voxel_size_angstrom,
        n_radial_bins=n_radial_bins, method=method, min_cluster_size=min_cluster_size, n_clusters=n_clusters,
        n_components=n_components, strict_halfset_independence=strict_halfset_independence, random_state=random_state,
        inplane_angular_step_degrees=inplane_angular_step_degrees, azimuthal_modes=azimuthal_modes,
        min_radius_fraction=min_radius_fraction, n_azimuthal_samples=n_azimuthal_samples,
        feature_normalisation=feature_normalisation, tilt_max_degrees=tilt_max_degrees, tilt_axis_degrees=tilt_axis_degrees,
        bin_factor=bin_factor, n_rotations=n_rotations, shift_range_voxels=shift_range_voxels, polish_rounds=polish_rounds,
        null_sigmas=null_sigmas, reject_quantile=reject_quantile, merge_fraction=merge_fraction, n_workers=n_workers,
    )
    # everything that changes results; n_workers is left out so a run can resume with different resources
    fingerprint = {
        'particles': file_signature(particles, content=True),
        'raw': signatures(tomogram_paths),
        'segmentation': signatures(segmentation_paths),
        **{key: value for key, value in asdict(settings).items() if key != 'n_workers'},
    }    # the particle set signature already covers every prior

    classify_particle_set(
        particle_set, settings, output_dir,
        tomogram_paths={k: str(v) for k, v in tomogram_paths.items()},
        segmentation_paths=segmentation_paths,
        fingerprint=fingerprint,
        sidecar_inputs=[('particle_set', particles)] + [(f'raw_tomogram:{stem}', path) for stem, path in sorted(tomogram_paths.items())],
        extra_parameters={'box_angstrom': box_angstrom},
        keep_raw=keep_raw,
        make_plots=make_plots,
        plot_format=plot_format,
    )

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
    settings: GroupSettings,
    output_dir: Path,
    *,
    cache_only: bool = False,
    subvolume_cache_dir: Callable[[OrientationPrior], Path] | None = None,
) -> _GroupResult:
    work_dir = output_dir / 'raw' / prior.value
    log.progress(f'Starting {prior.value} subvolume extraction')
    cache_dir = subvolume_cache_dir(prior) if subvolume_cache_dir is not None else work_dir / 'subvolumes'
    subvols, kept, skipped = extract_particle_set(particles, tomogram_paths, settings.box_voxels, segmentation_paths=segmentation_paths, n_workers=settings.n_workers, cache_dir=cache_dir, prior=prior, cache_only=cache_only)
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
    assignments, result_by_half, averages_by_half = classify(kept, subvols, features, settings, polish_config, output_dir, work_dir)
    return _GroupResult(assignments, kept, skipped, result_by_half, averages_by_half)
