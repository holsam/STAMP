'''
STAMP: structural enforcement of half-set independence in refinement
'''

# Import external dependencies
from pathlib import Path

# Import STAMP schema
from stamp.schemas.particles import ClassAssignment, HalfSet, Particle

# split_class_by_half: particles behind one class, split A/B, both halves required
def split_class_by_half(
    class_id: str,
    assignments: list[ClassAssignment],
    particles: list[Particle]
) -> tuple[list[Particle], list[Particle]]:
    member_ids = {a.particle_id for a in assignments if a.cluster_id == class_id}
    if not member_ids:
        raise ValueError(f'no particles assigned to class {class_id}')
    members = [p for p in particles if p.particle_id in member_ids]
    half_a = [p for p in members if p.half_set == HalfSet.A]
    half_b = [p for p in members if p.half_set == HalfSet.B]
    if not half_a or not half_b:
        raise ValueError(f'class {class_id} has {len(half_a)} half-A and {len(half_b)} half-B particles; a class must be independently refinable on both halves')
    return half_a, half_b

# refine_output_tree: separate per-half trees so a cross-half leak cannot be expressed
def refine_output_tree(output_dir: Path, class_id: str) -> dict[str, Path]:
    tree = {
        'A': output_dir / class_id / 'halfA',
        'B': output_dir / class_id / 'halfB',
        'combined': output_dir / class_id,
    }
    for path in tree.values():
        path.mkdir(parents=True, exist_ok=True)
    return tree

# assert_distinct_references: refuse the same seed reference for both halves
def assert_distinct_references(ref_a: Path, ref_b: Path) -> None:
    if ref_a.resolve() == ref_b.resolve():
        raise ValueError(f'both halves would be seeded from {ref_a}; each half must start from its own class average')
