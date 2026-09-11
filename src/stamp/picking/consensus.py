'''
STAMP: combines raw picks from multiple pickers into one half-set-tagged ParticleSet
'''

# Import external dependencies
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import Literal

# Import internal STAMP objects
from stamp.utils.halfset import assign_half_sets
from stamp.schemas.particles import Particle, ParticleSet
from stamp.schemas.picks import RawPick

# _components: complete-linkage groups of pick indices, each with diameter <= distance_threshold
def _components(positions: np.ndarray, distance_threshold: float) -> dict[int, list[int]]:
    if len(positions) == 1:
        return {0: [0]}
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='complete',
        metric='euclidean',
    ).fit_predict(positions)
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return groups

# reconcile_picks: reconcile raw picks from multiple pickers for a single tomogram
def reconcile_picks(
    picks_by_picker: dict[str, list[RawPick]],
    consensus_rule: Literal['intersection', 'union'],
    distance_threshold: float,
    tomogram_id: str,
) -> list[RawPick]:
    all_picks: list[RawPick] = []
    picker_of: list[str] = []
    for picker_name, picks in picks_by_picker.items():
        for pick in picks:
            if pick.tomogram_id != tomogram_id:
                continue
            all_picks.append(pick)
            picker_of.append(picker_name)
    if not all_picks:
        return []
    min_agreement = len(picks_by_picker) if consensus_rule == 'intersection' else 1
    positions = np.array([pick.position for pick in all_picks])
    reconciled: list[RawPick] = []
    for member_indices in _components(positions, distance_threshold).values():
        contributing_pickers = {picker_of[index] for index in member_indices}
        if len(contributing_pickers) < min_agreement:
            continue
        # one representative pick per picker (the member nearest that picker's own mean)
        picker_mean: dict[str, np.ndarray] = {}
        for picker in contributing_pickers:
            own_indices = [index for index in member_indices if picker_of[index] == picker]
            picker_mean[picker] = positions[own_indices].mean(axis=0)
        representative: dict[str, int] = {}
        for index in member_indices:
            picker = picker_of[index]
            distance = float(np.linalg.norm(positions[index] - picker_mean[picker]))
            if picker not in representative or distance < representative[picker][1]:
                representative[picker] = (index, distance)
        rep_indices = [index for index, _ in representative.values()]
        centroid = positions[rep_indices].mean(axis=0)
        confidences = [all_picks[i].confidence for i in rep_indices if all_picks[i].confidence is not None]
        # orientation from the representative closest to the centroid overall
        nearest = min(rep_indices, key=lambda i: np.linalg.norm(positions[i] - centroid))
        reconciled.append(
            RawPick(
                tomogram_id=tomogram_id,
                position=tuple(float(coord) for coord in centroid),
                orientation=all_picks[nearest].orientation,
                confidence=(sum(confidences) / len(confidences) if confidences else None),
                source_picker='+'.join(sorted(contributing_pickers)),
            )
        )
    return reconciled


# build_particle_set: turn reconciled picks into a half-set-tagged ParticleSet
def build_particle_set(
    reconciled_picks: list[RawPick],
    consensus_rule: Literal['intersection', 'union'],
    contributing_pickers: list[str],
    half_set_seed: int,
) -> ParticleSet:
    particle_ids = [f'p{index:06d}' for index in range(len(reconciled_picks))]
    group_of = {particle_id: pick.tomogram_id for particle_id, pick in zip(particle_ids, reconciled_picks)}
    if len(set(group_of.values())) < 2:
        print('WARNING: one tomogram only; half-sets are not independent (all particles in half A).')
    half_set_by_id = assign_half_sets(particle_ids, seed=half_set_seed, group_of=group_of)
    particles = [
        Particle(
            particle_id=particle_id,
            tomogram_id=pick.tomogram_id,
            position=pick.position,
            orientation=pick.orientation,
            source_picker=pick.source_picker,
            confidence=pick.confidence,
            half_set=half_set_by_id[particle_id],
        ) for particle_id, pick in zip(particle_ids, reconciled_picks)
    ]
    return ParticleSet(
        particles=particles,
        consensus_rule=consensus_rule,
        contributing_pickers=contributing_pickers,
    )
