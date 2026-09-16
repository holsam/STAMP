'''
STAMP: unsupervised classification logic
'''

# Import external dependencies
import json, matplotlib.pyplot as plt, numpy as np
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.align import align_inplane, roll_about_normal
from stamp.classify.average import compute_class_averages, write_class_averages
from stamp.classify.cluster import (
    ClusteringConfig,
    ClusteringResult,
    label_to_cluster_id,
    match_clusters_across_halves,
    reduce_and_cluster,
    reduce_and_cluster_shared,
)
from stamp.classify.extract import extract_particle_set
from stamp.classify.features import build_feature_matrix
from stamp.backends.base import ToolCommand
from stamp.decoy.validate import is_decoy_particle_set
from stamp.run.state import stage_dir
from stamp.utils.halfset import split_by_half_set
from stamp.schemas.particles import ClassAssignment, HalfSet, ParticleSet
from stamp.utils.io import write_sidecar
from stamp.utils.log import log
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
    inplane_alignment = True,
    inplane_angular_step_degrees = 10.0,
    inplane_iterations = 3,
    azimuthal_modes = 4,
    min_radius_fraction = 0.25,
    n_azimuthal_samples = 64,
    make_plots: bool = True,
    plot_format: str = 'tiff',
) -> None:
    log.progress('Classifying particle set')
    particle_set = ParticleSet.model_validate(json.loads(particles.read_text()))
    is_decoy = is_decoy_particle_set(particle_set)
    log.info(f'Loaded {len(particle_set.particles)} {"decoy" if is_decoy else "real"} particles')
    tomogram_paths = {
        path.stem: path for path in sorted(raw_tomogram_dir.glob('*.mrc'))
    }
    segmentation_paths = (
        {path.stem: str(path) for path in sorted(segmentation_dir.glob('*.mrc'))} if segmentation_dir is not None else {}
    )
    box_voxels = int(round(box_angstrom / voxel_size_angstrom))
    if box_voxels % 2 == 0:
        box_voxels += 1  # odd box keeps the particle exactly centred

    subvolumes, kept, skipped = extract_particle_set(
        particle_set.particles, {k: str(v) for k, v in tomogram_paths.items()}, box_voxels, segmentation_paths=segmentation_paths,
    )
    if skipped:
        log.warning(f'Skipped {len(skipped)} particles whose {box_voxels}-voxel box fell outside the volume or had no matching tomogram or had no orientation')
    if not kept:
        log.error('No particles could be extracted. Check --raw-dir and --box-length-a')
        raise SystemExit(1)
    log.info(f'Extracted {len(kept)} subvolumes at box {box_voxels}{" (membrane subtracted)" if segmentation_paths else ""}')

    features = build_feature_matrix(
        subvolumes,
        n_radial_bins=n_radial_bins,
        max_azimuthal_mode=azimuthal_modes,
        n_azimuthal_samples=n_azimuthal_samples,
        min_radius_fraction=min_radius_fraction,
    )
    log.debug(f'Feature vector: {features.shape[1]} dimensions (modes 0-{azimuthal_modes})')

    degenerate = ~features.any(axis=1)
    if degenerate.any():
        log.warning(f'Dropped {int(degenerate.sum())} particles with a constant/empty subvolume (edge fill)')
        features = features[~degenerate]
        subvolumes = subvolumes[~degenerate]
        kept = [particle for particle, bad in zip(kept, degenerate) if not bad]
    if not kept:
        log.error('No particles left after dropping degenerate subvolumes')
        raise SystemExit(1)

    config = ClusteringConfig(
        method=method,
        n_components=n_components,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    align_settings = dict(
        enabled=inplane_alignment,
        angular_step_degrees=inplane_angular_step_degrees,
        iterations=inplane_iterations,
    )
    if not inplane_alignment:
        log.warning('--no-inplane-alignment: class averages are a rotational average about the membrane normal (Cinf assumed). Downstream fit scores and half-map FSC will reflect the shared radial profile, not a 3D structure.')

    if not strict_halfset_independence:
        log.warning('--no-strict-halfset-independence used, half-A and half-B particles are clustered together; FSC built on these classes will be inflated')
    if strict_halfset_independence:
        assignments, result_by_half, averages_by_half = _classify_strict(kept, subvolumes, features, config, output_dir, voxel_size_angstrom, align_settings)
    else:
        assignments, result_by_half, averages_by_half = _classify_combined(kept, subvolumes, features, config, output_dir, voxel_size_angstrom, align_settings)

    assignments_path = output_dir / 'class_assignments.json'
    assignments_path.write_text(json.dumps([a.model_dump() for a in assignments], indent=2))

    write_sidecar(
        output_dir,
        stage='classify',
        tool='stamp-native-classifier',
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
            'inplane_alignment': inplane_alignment,
            'inplane_angular_step_degrees': inplane_angular_step_degrees,
            'inplane_iterations': inplane_iterations,
            'random_state': random_state,
            'azimuthal_modes': azimuthal_modes,
            'n_azimuthal_samples': n_azimuthal_samples,
            'min_radius_fraction': min_radius_fraction,
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
        '--seg-dir', str(config.run.segmentation_dir),
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
        '--inplane-iterations', str(settings.inplane_iterations),
    ]
    if not settings.strict_halfset_independence:
        argv.append('--no-strict-halfset-independence')
    if not settings.inplane_alignment:
        argv.append('--no-inplane-alignment')
    argv.append('--plots' if config.plots.enabled else '--no-plots')
    argv += ['--plot-format', config.plots.format]
    return [ToolCommand(tool='classify', argv=argv, working_directory=target, output_paths=[target / 'class_averages'])]

# _resolve_and_apply_inplane: estimate per-row azimuth, return rolled subvolumes and {row: angle}
def _resolve_and_apply_inplane(
    subvolumes: np.ndarray, cluster_ids: list[str], align_settings: dict
) -> tuple[np.ndarray, dict[int, float]]:
    if not align_settings['enabled']:
        return subvolumes, {}
    angles = align_inplane(
        subvolumes,
        cluster_ids,
        angular_step_degrees=align_settings['angular_step_degrees'],
        iterations=align_settings['iterations'],
    )
    rolled = np.stack([roll_about_normal(volume, angles.get(index, 0.0)) for index, volume in enumerate(subvolumes)])
    return rolled, angles

# _classify_combined: cluster all particles together, then average each half separately
def _classify_combined(
    particles,
    subvolumes,
    features,
    config,
    output_dir: Path,
    voxel_size_angstrom: float,
    align_settings: dict,
) -> tuple[list[ClassAssignment], dict[str, ClusteringResult], dict[str, dict[str, tuple[np.ndarray, int]]]]:
    result = reduce_and_cluster(features, config)
    cluster_ids = [label_to_cluster_id(int(label)) for label in result.labels]

    _report_clusters(cluster_ids, result.explained_variance_ratio)

    aligned, angles = _resolve_and_apply_inplane(subvolumes, cluster_ids, align_settings)

    averages_by_half: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
    for half_set in (HalfSet.A, HalfSet.B):
        mask = np.array([p.half_set == half_set for p in particles])
        if not mask.any():
            continue
        averages = compute_class_averages(aligned[mask], [cid for cid, keep in zip(cluster_ids, mask) if keep])
        write_class_averages(averages, output_dir / 'class_averages', voxel_size_angstrom, half_set.value)
        averages_by_half[half_set.value] = averages

    assignments = [
        ClassAssignment(
            particle_id=particle.particle_id,
            cluster_id=cluster_id,
            classifier='stamp-native-classifier',
            inplane_angle_degrees=angles.get(index),
            vesicle_id=particle.vesicle_id,
        )
        for index, (particle, cluster_id) in enumerate(zip(particles, cluster_ids))
    ]
    return assignments, {'combined': result}, averages_by_half

# _classify_strict: cluster each half independently, then match clusters by centroid
def _classify_strict(
    particles,
    subvolumes,
    features,
    config,
    output_dir: Path,
    voxel_size_angstrom: float,
    align_settings: dict,
) -> tuple[list[ClassAssignment], dict[str, ClusteringResult], dict[str, dict[str, tuple[np.ndarray, int]]]]:
    half_a, half_b = split_by_half_set(list(particles))
    index_of = {particle.particle_id: i for i, particle in enumerate(particles)}
    indices_a = np.array([index_of[p.particle_id] for p in half_a])
    indices_b = np.array([index_of[p.particle_id] for p in half_b])
    if indices_a.size == 0 or indices_b.size == 0:
        log.error('Strict mode needs particles in both half-sets')
        raise SystemExit(1)

    results = reduce_and_cluster_shared(features, {'A': indices_a, 'B': indices_b}, config)
    result_a, result_b = results['A'], results['B']
    matches = match_clusters_across_halves(result_a.centroids, result_b.centroids)
    log.debug('Cross-half cluster matching (B -> A, shared PCA space):')
    for label_b, (label_a, distance) in sorted(matches.items()):
        log.debug(f'  c{label_b:02d} -> c{label_a:02d}  d={distance:.3f}')

    assignments: list[ClassAssignment] = []
    result_by_half: dict[str, ClusteringResult] = {}
    averages_by_half: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
    for half_label, half_particles, indices, result, remap in (
        ('A', half_a, indices_a, result_a, None),
        ('B', half_b, indices_b, result_b, matches),
    ):
        cluster_ids = [label_to_cluster_id(int(label)) for label in result.labels]
        aligned, local_angles = _resolve_and_apply_inplane(subvolumes[indices], cluster_ids, align_settings)
        averages = compute_class_averages(aligned, cluster_ids)
        write_class_averages(averages, output_dir / 'class_averages', voxel_size_angstrom, half_label)
        result_by_half[half_label] = result
        averages_by_half[half_label] = averages
        for local_row, (particle, label) in enumerate(zip(half_particles, result.labels)):
            cluster_id = label_to_cluster_id(int(label))
            if remap is not None and int(label) in remap:
                cluster_id = label_to_cluster_id(remap[int(label)][0])
            assignments.append(
                ClassAssignment(
                    particle_id=particle.particle_id,
                    cluster_id=cluster_id,
                    classifier='stamp-native-classifier-strict',
                    inplane_angle_degrees=local_angles.get(local_row),
                    vesicle_id=particle.vesicle_id,
                )
            )
    return assignments, result_by_half, averages_by_half

# _report_clusters: echo PCA variance and per-cluster sizes
def _report_clusters(cluster_ids: list[str], explained_variance: np.ndarray) -> None:
    counts: dict[str, int] = {}
    for cluster_id in cluster_ids:
        counts[cluster_id] = counts.get(cluster_id, 0) + 1
    log.debug(f'PCA: top 5 components explain {explained_variance[:5].sum():.1%} of variance')
    for cluster_id, count in sorted(counts.items()):
        log.debug(f'cluster {cluster_id}: {count}')
