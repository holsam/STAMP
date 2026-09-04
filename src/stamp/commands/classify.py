'''
STAMP: unsupervised classification logic
'''

# Import external dependencies
import json, numpy as np
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.average import compute_class_averages, write_class_averages
from stamp.classify.cluster import (
    ClusteringConfig,
    label_to_cluster_id,
    match_clusters_across_halves,
    reduce_and_cluster,
)
from stamp.classify.extract import extract_particle_set
from stamp.classify.features import build_feature_matrix
from stamp.decoy.validate import is_decoy_particle_set
from stamp.utils.halfset import split_by_half_set
from stamp.schemas.particles import ClassAssignment, HalfSet, ParticleSet
from stamp.utils.io import write_sidecar

# run_classify: cluster picked particles by structural similarity
def run_classify(
    particles,
    raw_tomogram_dir,
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
) -> None:
    particle_set = ParticleSet.model_validate(json.loads(particles.read_text()))
    is_decoy = is_decoy_particle_set(particle_set)
    print(f'Loaded {len(particle_set.particles)} {"decoy" if is_decoy else "real"} particles.')
    tomogram_paths = {
        path.stem: path for path in sorted(raw_tomogram_dir.glob('*.mrc'))
    }
    box_voxels = int(round(box_angstrom / voxel_size_angstrom))
    if box_voxels % 2 == 0:
        box_voxels += 1  # odd box keeps the particle exactly centred

    subvolumes, kept, skipped = extract_particle_set(
        particle_set.particles, {k: str(v) for k, v in tomogram_paths.items()}, box_voxels
    )
    if skipped:
        print(f'Skipped {len(skipped)} particles whose {box_voxels}-voxel box fell outside the volume or had no matching tomogram')
    if not kept:
        print('No particles could be extracted. Check --raw-tomogram-dir and --box-angstrom')
        raise SystemExit(1)
    print(f'Extracted {len(kept)} subvolumes at box {box_voxels}')

    features = build_feature_matrix(subvolumes, n_radial_bins=n_radial_bins)
    config = ClusteringConfig(
        method=method,
        n_components=n_components,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if strict_halfset_independence:
        assignments = _classify_strict(kept, subvolumes, features, config, output_dir, voxel_size_angstrom)
    else:
        assignments = _classify_combined(kept, subvolumes, features, config, output_dir, voxel_size_angstrom)

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
            'random_state': random_state,
            'is_decoy': is_decoy,
            'n_extracted': len(kept),
            'n_skipped': len(skipped),
        },
        inputs=[('particle_set', particles)]
        + [(f'raw_tomogram:{stem}', path) for stem, path in sorted(tomogram_paths.items())],
    )
    print(f'Wrote {len(assignments)} class assignments to {assignments_path}')

# _classify_combined: cluster all particles together, then average each half separately
def _classify_combined(
    particles, subvolumes, features, config, output_dir: Path, voxel_size_angstrom: float
) -> list[ClassAssignment]:
    result = reduce_and_cluster(features, config)
    cluster_ids = [label_to_cluster_id(int(label)) for label in result.labels]

    _report_clusters(cluster_ids, result.explained_variance_ratio)

    for half_set in (HalfSet.A, HalfSet.B):
        mask = np.array([p.half_set == half_set for p in particles])
        if not mask.any():
            continue
        averages = compute_class_averages(subvolumes[mask], [cid for cid, keep in zip(cluster_ids, mask) if keep])
        write_class_averages(averages, output_dir / 'class_averages', voxel_size_angstrom, half_set.value)

    return [
        ClassAssignment(
            particle_id=particle.particle_id,
            cluster_id=cluster_id,
            classifier='stamp-native-classifier',
        )
        for particle, cluster_id in zip(particles, cluster_ids)
    ]

# _classify_strict: cluster each half independently, then match clusters by centroid
def _classify_strict(
    particles,
    subvolumes,
    features,
    config,
    output_dir: Path,
    voxel_size_angstrom: float
) -> list[ClassAssignment]:
    half_a, half_b = split_by_half_set(list(particles))
    index_of = {particle.particle_id: i for i, particle in enumerate(particles)}
    indices_a = np.array([index_of[p.particle_id] for p in half_a])
    indices_b = np.array([index_of[p.particle_id] for p in half_b])
    if indices_a.size == 0 or indices_b.size == 0:
        print('Strict mode needs particles in both half-sets')
        raise SystemExit(1)

    result_a = reduce_and_cluster(features[indices_a], config)
    result_b = reduce_and_cluster(features[indices_b], config)
    matches = match_clusters_across_halves(result_a.centroids, result_b.centroids)
    print('Cross-half cluster matching (B -> A, centroid distance):')
    for label_b, (label_a, distance) in sorted(matches.items()):
        print(f'  c{label_b:02d} -> c{label_a:02d}  d={distance:.3f}')

    assignments: list[ClassAssignment] = []
    for particle, label in zip(half_a, result_a.labels):
        assignments.append(
            ClassAssignment(
                particle_id=particle.particle_id,
                cluster_id=label_to_cluster_id(int(label)),
                classifier='stamp-native-classifier-strict',
            )
        )
    for particle, label in zip(half_b, result_b.labels):
        matched = matches.get(int(label))
        cluster_id = (
            label_to_cluster_id(matched[0]) if matched else label_to_cluster_id(int(label))
        )
        assignments.append(
            ClassAssignment(
                particle_id=particle.particle_id,
                cluster_id=cluster_id,
                classifier='stamp-native-classifier-strict',
            )
        )
    for half_label, indices, result in (
        ('A', indices_a, result_a), ('B', indices_b, result_b)
    ):
        cluster_ids = [label_to_cluster_id(int(label)) for label in result.labels]
        averages = compute_class_averages(subvolumes[indices], cluster_ids)
        write_class_averages(averages, output_dir / 'class_averages', voxel_size_angstrom, half_label)
    return assignments

# _report_clusters: echo PCA variance and per-cluster sizes
def _report_clusters(cluster_ids: list[str], explained_variance: np.ndarray) -> None:
    counts: dict[str, int] = {}
    for cluster_id in cluster_ids:
        counts[cluster_id] = counts.get(cluster_id, 0) + 1
    print(f'PCA: top 5 components explain {explained_variance[:5].sum():.1%} of variance.')
    print('Cluster sizes:')
    for cluster_id, count in sorted(counts.items()):
        print(f'  {cluster_id}: {count}')
