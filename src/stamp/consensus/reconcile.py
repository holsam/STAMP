'''
STAMP: combines raw picks from multiple pickers into one half-set-tagged ParticleSet
'''

# Import external dependencies
import numpy as np
from scipy.spatial import cKDTree
from typing import Literal

# Import internal STAMP objects
from stamp.halfset.split import assign_half_sets
from stamp.schemas.particles import Particle, ParticleSet
from stamp.schemas.picks import RawPick

# _UnionFind: minimal disjoint-set structure for connected-components grouping
class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, index: int) -> int:
        while self._parent[index] != index:
            self._parent[index] = self._parent[self._parent[index]]
            index = self._parent[index]
        return index

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_a] = root_b

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
    tree = cKDTree(positions)
    union_find = _UnionFind(len(all_picks))
    for i, j in tree.query_pairs(r=distance_threshold):
        union_find.union(i, j)
    components: dict[int, list[int]] = {}
    for index in range(len(all_picks)):
        components.setdefault(union_find.find(index), []).append(index)
    reconciled: list[RawPick] = []
    for member_indices in components.values():
        contributing_pickers = {picker_of[i] for i in member_indices}
        if len(contributing_pickers) < min_agreement:
            continue
        centroid = positions[member_indices].mean(axis=0)
        confidences = [
            all_picks[i].confidence
            for i in member_indices
            if all_picks[i].confidence is not None
        ]
        orientation = next(
            (all_picks[i].orientation for i in member_indices if all_picks[i].orientation is not None),
            None,
        )
        reconciled.append(
            RawPick(
                tomogram_id=tomogram_id,
                position=tuple(float(coord) for coord in centroid),
                orientation=orientation,
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
    half_set_by_id = assign_half_sets(particle_ids, seed=half_set_seed)
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
