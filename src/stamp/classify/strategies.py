'''
STAMP: classify strategies, combined or strict half-set independence
'''

# Import external dependencies
import numpy as np
from pathlib import Path

# Import STAMP objects
from stamp.classify.average import write_class_averages
from stamp.classify.cluster import ClusteringResult, label_to_cluster_id, reduce_and_cluster
from stamp.classify.frames import azimuth_degrees, rotation_to_quaternion
from stamp.classify.halves import express_in_frame_of_a, run_polish_by_half
from stamp.classify.polish import PolishConfig, posed_class_averages, run_polish
from stamp.classify.prepare import working_box
from stamp.classify.settings import GroupSettings
from stamp.classify.store import ParticleStore, build_store
from stamp.schemas.particles import ClassAssignment, OrientationPrior
from stamp.classify.subvolumes import Subvolumes
from stamp.utils.errors import StampPipelineError
from stamp.utils.halfset import half_set_indices
from stamp.utils.log import log

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

# classify_combined: seed clusters from the features, run polish over all particles together, then average each half at polish's poses
def classify_combined(
    particles,
    subvols: Subvolumes,
    features,
    settings: GroupSettings,
    polish_config: PolishConfig,
    output_dir: Path,
    work_dir: Path,
) -> tuple[list[ClassAssignment], dict[str, ClusteringResult], dict[str, dict[str, tuple[np.ndarray, int]]]]:
    working_voxels = working_box(settings.box_voxels, settings.bin_factor)
    tilt = (settings.tilt_max_degrees, settings.tilt_axis_degrees)
    voxel_size_angstrom = settings.voxel_size_angstrom
    seeds = reduce_and_cluster(features, settings.clustering)
    (output_dir / 'class_averages').mkdir(parents=True, exist_ok=True)     # present even when polish rejects everything, so a decoy that matched nothing is still compared
    orientations = [particle.orientation for particle in particles]
    store = build_store(subvols, orientations, working_voxels, *tilt, work_dir / 'working')
    outcome = run_polish(store, seeds.embedding, seeds.labels, polish_config, checkpoint_dir=work_dir / 'polish')
    cluster_ids = [label_to_cluster_id(int(label), polish_config.prior) for label in outcome.labels]
    _report_clusters(cluster_ids, seeds.explained_variance_ratio)
    full = ParticleStore(subvols, orientations, *tilt)
    scale = subvols.box_voxels / working_voxels     # polish shifts are in working voxels
    averages_by_half: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
    for half_set, half_indices in half_set_indices(particles).items():
        if not half_indices:
            continue
        averages = posed_class_averages(full.take(half_indices), outcome.labels[half_indices], outcome.rotations[half_indices], outcome.shifts[half_indices] * scale, polish_config.prior)
        write_class_averages(averages, output_dir / 'class_averages', voxel_size_angstrom, half_set.value)
        averages_by_half[half_set.value] = averages
    assignments = [
        _pose_assignment(particle, cluster_id, 'stamp-polish-classifier', polish_config.prior, outcome.rotations[index], outcome.shifts[index] * scale, float(outcome.cc[index]), float(outcome.margin[index]), int(outcome.polarity[index]))
        for index, (particle, cluster_id) in enumerate(zip(particles, cluster_ids))
    ]
    return assignments, {'combined': seeds}, averages_by_half

# classify_strict: seed and polish each half on its own, match half-B clusters to half A by reference similarity and express half-B poses in the half-A frame so the two half averages can be compared
def classify_strict(
    particles,
    subvols: Subvolumes,
    features,
    settings: GroupSettings,
    polish_config: PolishConfig,
    output_dir: Path,
    work_dir: Path,
) -> tuple[list[ClassAssignment], dict[str, ClusteringResult], dict[str, dict[str, tuple[np.ndarray, int]]]]:
    working_voxels = working_box(settings.box_voxels, settings.bin_factor)
    tilt = (settings.tilt_max_degrees, settings.tilt_axis_degrees)
    voxel_size_angstrom = settings.voxel_size_angstrom
    members = {half.value: indices for half, indices in half_set_indices(list(particles)).items()}
    if not members['A'] or not members['B']:
        raise StampPipelineError('Strict mode needs particles in both half-sets')
    seeds = {half: reduce_and_cluster(features[indices], settings.clustering) for half, indices in members.items()}   # one PCA per half, so nothing in half A depends on half B
    (output_dir / 'class_averages').mkdir(parents=True, exist_ok=True)     # present even when polish rejects everything, so a decoy that matched nothing is still compared
    orientations = [particle.orientation for particle in particles]
    store = build_store(subvols, orientations, working_voxels, *tilt, work_dir / 'working')
    results, matches = run_polish_by_half(store, members, {h: seeds[h].embedding for h in members}, {h: seeds[h].labels for h in members}, polish_config, checkpoint_dir=work_dir / 'polish')
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
        averages = posed_class_averages(full.take(indices), labels[half], poses[half][0], poses[half][1] * scale, polish_config.prior)
        write_class_averages(averages, output_dir / 'class_averages', voxel_size_angstrom, half)
        averages_by_half[half] = averages
        result = results[half]
        for local, index in enumerate(indices):
            cluster_id = label_to_cluster_id(int(labels[half][local]), polish_config.prior)
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
