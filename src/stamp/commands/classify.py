'''
STAMP: unsupervised classification logic
'''

# Import external dependencies
import json, os, matplotlib.pyplot as plt, numpy as np
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.average import write_class_averages
from stamp.classify.cluster import ClusteringConfig, ClusteringResult, label_to_cluster_id, reduce_and_cluster
from stamp.utils.checkpoint import file_signature, signatures, sync_checkpoint
from stamp.classify.polish import PolishConfig, posed_class_averages, run_polish
from stamp.classify.extract import extract_particle_set
from stamp.classify.halves import express_in_frame_of_a, run_polish_by_half
from stamp.classify.frames import azimuth_degrees, rotation_to_quaternion
from stamp.classify.prepare import working_box
from stamp.classify.shfeatures import build_sh_feature_matrix
from stamp.classify.store import ParticleStore, build_store
from stamp.classify.features import build_feature_matrix
from stamp.backends.base import ToolCommand
from stamp.decoy.validate import is_decoy_particle_set
from stamp.run.state import stage_dir
from stamp.utils.halfset import split_by_half_set
from stamp.schemas.particles import ClassAssignment, HalfSet, OrientationPrior, ParticleSet
from stamp.schemas.subvolumes import Subvolumes
from stamp.utils.errors import StampPipelineError
from stamp.utils.io import archive_and_remove_directory, glob_mrc, match_by_stem, resolve_directory_voxel_size_angstrom, write_sidecar
from stamp.utils.log import log, log_config
from stamp.utils.plotting.core import PlotFormat, central_slice, finish, plot_path
from stamp.utils.plotting.classify import class_average_grid, scatter_labels

# run_classify: cluster picked particles by structural similarity
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
    if voxel_size_angstrom is None:
        voxel_size_angstrom = resolve_directory_voxel_size_angstrom(glob_mrc(raw_tomogram_dir))
        if voxel_size_angstrom is None:
            raise StampPipelineError('Voxel size not given and not found in any MRC header in --raw-dir.')
        log.debug(f'Resolved voxel size from MRC headers: {voxel_size_angstrom} Å')
    log.progress('Classifying particle set')
    particle_set = ParticleSet.model_validate(json.loads(particles.read_text()))
    is_decoy = is_decoy_particle_set(particle_set)
    prior = particle_set.orientation_prior
    log.info(f'Loaded {len(particle_set.particles)} {"decoy" if is_decoy else "real"} particles')
    tomogram_ids = sorted({particle.tomogram_id for particle in particle_set.particles})
    matched, unmatched = match_by_stem([Path(f'{tomogram_id}.mrc') for tomogram_id in tomogram_ids], glob_mrc(raw_tomogram_dir))
    for stem_path in unmatched:
        log.warning(f'No raw tomogram matching {stem_path.stem}, its particles will be skipped')
    tomogram_paths = {stem_path.stem: raw_path for stem_path, raw_path in matched.items()}
    segmentation_paths = (
        {path.stem: str(path) for path in glob_mrc(segmentation_dir)} if segmentation_dir is not None else {}
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
        'orientation_prior': prior.value,
        'tilt_max_degrees': tilt_max_degrees,
        'tilt_axis_degrees': tilt_axis_degrees,
        'bin_factor': bin_factor,
        'n_rotations': n_rotations,
        'shift_range_voxels': shift_range_voxels,
        'polish_rounds': polish_rounds,
        'null_sigmas': null_sigmas,
        'reject_quantile': reject_quantile,
        'merge_fraction': merge_fraction,
    }
    if sync_checkpoint(output_dir / 'raw', fingerprint):
        log.info('Found matching partial output, resuming')

    log.progress('Starting subvolume extraction')
    subvols, kept, skipped = extract_particle_set(
        particle_set.particles, {k: str(v) for k, v in tomogram_paths.items()}, box_voxels,
        segmentation_paths=segmentation_paths, n_workers=n_workers, cache_dir=output_dir / 'raw' / 'subvolumes', prior=prior,
    )
    if skipped:
        log.warning(f'Skipped {len(skipped)} particles whose {box_voxels}-voxel box fell outside the volume or had no matching tomogram or had no orientation')
    if not kept:
        raise StampPipelineError('No particles could be extracted. Check --raw-dir and --box-length-a')

    log.progress('Starting feature matrix construction')
    if prior is OrientationPrior.NONE:
        features = build_sh_feature_matrix(subvols, tilt_max_degrees, tilt_axis_degrees, n_workers=n_workers)
    else:
        features = build_feature_matrix(
            subvols,
            n_radial_bins=n_radial_bins,
            max_azimuthal_mode=azimuthal_modes,
            n_azimuthal_samples=n_azimuthal_samples,
            min_radius_fraction=min_radius_fraction,
            n_workers=n_workers,
            cache_dir=output_dir / 'raw' / 'features',
            normalisation=feature_normalisation,
            axial_invariant=prior is OrientationPrior.AXIS,
        )
    log.progress(f'Feature matrix complete: {features.shape[0]} particles x {features.shape[1]} features ({prior.value} prior)')

    degenerate = ~features.any(axis=1)
    if degenerate.any():
        log.warning(f'Dropped {int(degenerate.sum())} particles with a constant/empty subvolume (edge fill)')
        keep_indices = [i for i, bad in enumerate(degenerate) if not bad]
        features = features[~degenerate]
        subvols = subvols.take(keep_indices)
        kept = [particle for particle, bad in zip(kept, degenerate) if not bad]
    if not kept:
        raise StampPipelineError('No particles left after dropping degenerate subvolumes')

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

    polish_config = PolishConfig(
        prior=prior, angular_step_degrees=inplane_angular_step_degrees, n_rotations=n_rotations, shift_range_voxels=shift_range_voxels,
        n_polish_rounds=polish_rounds, min_seed_size=min_cluster_size, merge_fraction=merge_fraction, null_sigmas=null_sigmas,
        reject_quantile=reject_quantile, seed=random_state, threads=n_workers,
    )
    classify = _classify_strict if strict_halfset_independence else _classify_combined
    if not strict_halfset_independence:
        log.warning('--no-strict-halfset-independence used, half-A and half-B particles are classified together; FSC built on these classes will be inflated')
    assignments, result_by_half, averages_by_half = classify(kept, subvols, features, config, polish_config, working_box(box_voxels, bin_factor), (tilt_max_degrees, tilt_axis_degrees), output_dir, voxel_size_angstrom)

    assignments_path = output_dir / 'class_assignments.json'
    assignments_path.write_text(json.dumps([a.model_dump() for a in assignments], indent=2))

    if not keep_raw:
        raw_dir = output_dir / 'raw'
        if raw_dir.is_dir():
            archive_path = archive_and_remove_directory(raw_dir)
            log.info(f'Archived raw classify output to {archive_path}')

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
            'orientation_prior': prior.value,
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
            'n_extracted': len(kept),
            'n_skipped': len(skipped),
        },
        inputs=[('particle_set', particles)]
        + [(f'raw_tomogram:{stem}', path) for stem, path in sorted(tomogram_paths.items())],
    )
    log.info(f'Wrote {len(assignments)} class assignments to {assignments_path}')
    if make_plots:
        _plot_classify(result_by_half, averages_by_half, output_dir, plot_format)

# _plot_classify: create plots (PCA embedding scatter + class-average thumbnail grid)
def _plot_classify(result_by_half: dict[str, ClusteringResult], averages_by_half: dict[str, dict[str, tuple[np.ndarray, int]]], output_dir: Path, fmt: PlotFormat) -> None:
    halves = sorted(result_by_half)
    fig, axes = plt.subplots(1, len(halves), figsize=(5.5 * len(halves), 4), squeeze=False)
    for ax, half in zip(axes[0], halves):
        result = result_by_half[half]
        scatter_labels(ax, result.embedding[:, :2], result.labels, f'half {half}: PCA -> {result.labels.max() + 1} cluster(s)')
    finish(fig, plot_path(output_dir, 'embedding', fmt))

    for half, averages in averages_by_half.items():
        ids = sorted(averages)
        if not ids:
            continue
        fig, axes = plt.subplots(1, max(2, len(ids)), figsize=(2.6 * max(2, len(ids)), 3), squeeze=False)
        class_average_grid(axes[0], ids, {cid: central_slice(volume) for cid, (volume, _count) in averages.items()}, lambda c: c)
        finish(fig, plot_path(output_dir, f'class_averages_half{half}', fmt))

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
    if config.run.segmentation_dir is not None:
        argv += ['--seg-dir', str(config.run.segmentation_dir)]
    if not settings.strict_halfset_independence:
        argv.append('--no-strict-halfset-independence')
    argv.append('--keep-raw' if settings.keep_raw else '--no-keep-raw')
    argv.append('--plots' if config.plots.enabled else '--no-plots')
    argv += ['--plot-format', config.plots.format]
    return [ToolCommand(tool='classify', argv=argv, working_directory=target, output_paths=[target / 'class_averages'])]

# _pose_assignment: a class assignment carrying polish's pose, the in-plane angle only where it means something (membrane normal prior)
def _pose_assignment(particle, cluster_id: str, classifier: str, prior: OrientationPrior, rotation: np.ndarray, shift_voxels: np.ndarray, cc: float, margin: float, polarity: int) -> ClassAssignment:
    if cluster_id == 'noise':
        return ClassAssignment(particle_id=particle.particle_id, cluster_id=cluster_id, classifier=classifier, vesicle_id=particle.vesicle_id, cc=cc, margin=margin)
    return ClassAssignment(
        particle_id=particle.particle_id,
        cluster_id=cluster_id,
        classifier=classifier,
        inplane_angle_degrees=azimuth_degrees(rotation.T) if prior is OrientationPrior.NORMAL else None,
        vesicle_id=particle.vesicle_id,
        rotation=rotation_to_quaternion(rotation),
        shift_voxels=tuple(float(v) for v in shift_voxels),
        cc=cc,
        margin=margin,
        polarity=polarity if prior is OrientationPrior.AXIS else None,
    )

# _classify: seed clusters from the features, run polish over all particles together, then average each half at polish's poses
def _classify_combined(
    particles,
    subvols: Subvolumes,
    features,
    config,
    polish_config: PolishConfig,
    working_voxels: int,
    tilt: tuple[float | None, float],
    output_dir: Path,
    voxel_size_angstrom: float,
) -> tuple[list[ClassAssignment], dict[str, ClusteringResult], dict[str, dict[str, tuple[np.ndarray, int]]]]:
    seeds = reduce_and_cluster(features, config)
    (output_dir / 'class_averages').mkdir(parents=True, exist_ok=True)     # present even when polish rejects everything, so a decoy that matched nothing is still compared
    orientations = [particle.orientation for particle in particles]
    store = build_store(subvols, orientations, working_voxels, *tilt, output_dir / 'raw' / 'working')
    outcome = run_polish(store, seeds.embedding, seeds.labels, polish_config, checkpoint_dir=output_dir / 'raw' / 'polish')
    cluster_ids = [label_to_cluster_id(int(label)) for label in outcome.labels]
    _report_clusters(cluster_ids, seeds.explained_variance_ratio)
    full = ParticleStore(subvols, orientations, *tilt)
    scale = subvols.box_voxels / working_voxels     # polish shifts are in working voxels
    averages_by_half: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
    for half_set in (HalfSet.A, HalfSet.B):
        half_indices = [i for i, particle in enumerate(particles) if particle.half_set == half_set]
        if not half_indices:
            continue
        averages = posed_class_averages(full.take(half_indices), outcome.labels[half_indices], outcome.rotations[half_indices], outcome.shifts[half_indices] * scale)
        write_class_averages(averages, output_dir / 'class_averages', voxel_size_angstrom, half_set.value)
        averages_by_half[half_set.value] = averages
    assignments = [
        _pose_assignment(particle, cluster_id, 'stamp-polish-classifier', polish_config.prior, outcome.rotations[index], outcome.shifts[index] * scale, float(outcome.cc[index]), float(outcome.margin[index]), int(outcome.polarity[index]))
        for index, (particle, cluster_id) in enumerate(zip(particles, cluster_ids))
    ]
    return assignments, {'combined': seeds}, averages_by_half

# _classify_strict: seed and polish each half on its own, match half-B clusters to half A by reference similarity and express half-B poses in the half-A frame so the two half averages can be compared
def _classify_strict(
    particles,
    subvols: Subvolumes,
    features,
    config,
    polish_config: PolishConfig,
    working_voxels: int,
    tilt: tuple[float | None, float],
    output_dir: Path,
    voxel_size_angstrom: float,
) -> tuple[list[ClassAssignment], dict[str, ClusteringResult], dict[str, dict[str, tuple[np.ndarray, int]]]]:
    half_a, half_b = split_by_half_set(list(particles))
    index_of = {particle.particle_id: i for i, particle in enumerate(particles)}
    members = {'A': [index_of[p.particle_id] for p in half_a], 'B': [index_of[p.particle_id] for p in half_b]}
    if not members['A'] or not members['B']:
        raise StampPipelineError('Strict mode needs particles in both half-sets')
    seeds = {half: reduce_and_cluster(features[indices], config) for half, indices in members.items()}   # one PCA per half, so nothing in half A depends on half B
    (output_dir / 'class_averages').mkdir(parents=True, exist_ok=True)     # present even when polish rejects everything, so a decoy that matched nothing is still compared
    orientations = [particle.orientation for particle in particles]
    store = build_store(subvols, orientations, working_voxels, *tilt, output_dir / 'raw' / 'working')
    results, matches = run_polish_by_half(store, members, {h: seeds[h].embedding for h in members}, {h: seeds[h].labels for h in members}, polish_config, checkpoint_dir=output_dir / 'raw' / 'polish')
    for label_b, match in sorted(matches.items()):
        log.debug(f'Cross-half cluster matching (B -> A, by reference): c{label_b:02d} -> c{match.label_a:02d}  similarity={match.similarity:.3f}')
    n_a = len(results['A'].references)
    renumber = {label_b: match.label_a for label_b, match in matches.items()}
    unmatched = [label_b for label_b in range(len(results['B'].references)) if label_b not in matches]
    renumber.update({label_b: n_a + position for position, label_b in enumerate(unmatched)})   # unmatched half-B clusters get ids that cannot collide with half A
    rotations_b, shifts_b = express_in_frame_of_a(results['B'], matches)
    poses = {'A': (results['A'].rotations, results['A'].shifts), 'B': (rotations_b, shifts_b)}
    labels = {'A': results['A'].labels, 'B': np.array([renumber[int(l)] if l >= 0 else -1 for l in results['B'].labels])}
    full = ParticleStore(subvols, orientations, *tilt)
    scale = subvols.box_voxels / working_voxels
    assignments: list[ClassAssignment] = []
    averages_by_half: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
    for half, indices in members.items():
        averages = posed_class_averages(full.take(indices), labels[half], poses[half][0], poses[half][1] * scale)
        write_class_averages(averages, output_dir / 'class_averages', voxel_size_angstrom, half)
        averages_by_half[half] = averages
        result = results[half]
        for local, index in enumerate(indices):
            cluster_id = label_to_cluster_id(int(labels[half][local]))
            assignments.append(_pose_assignment(particles[index], cluster_id, 'stamp-polish-classifier-strict', polish_config.prior, poses[half][0][local], poses[half][1][local] * scale, float(result.cc[local]), float(result.margin[local]), int(result.polarity[local])))
    return assignments, {half: seeds[half] for half in seeds}, averages_by_half

# _report_clusters: echo PCA variance and per-cluster sizes
def _report_clusters(cluster_ids: list[str], explained_variance: np.ndarray) -> None:
    counts: dict[str, int] = {}
    for cluster_id in cluster_ids:
        counts[cluster_id] = counts.get(cluster_id, 0) + 1
    log.debug(f'PCA: top 5 components explain {explained_variance[:5].sum():.1%} of variance')
    for cluster_id, count in sorted(counts.items()):
        log.debug(f'cluster {cluster_id}: {count}')
