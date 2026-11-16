'''
STAMP: structural enforcement of half-set independence in refinement
'''

# Import external dependencies
import numpy as np
from pathlib import Path

# Import STAMP objects
from stamp.adapters.base import AdapterInputs
from stamp.picking.geometry import compose_roll_about_normal
from stamp.schemas.particles import ClassAssignment, HalfSet, Particle
from stamp.utils.errors import StampPipelineError, StampValidationError
from stamp.utils.log import log
from stamp.utils.rotation import quaternion_to_matrix

# split_class_by_half: particles behind one class, split A/B, both halves required
def split_class_by_half(
    class_id: str,
    assignments: list[ClassAssignment],
    particles: list[Particle]
) -> tuple[list[Particle], list[Particle]]:
    member_ids = {a.particle_id for a in assignments if a.cluster_id == class_id}
    if not member_ids:
        raise StampValidationError(f'no particles assigned to class {class_id}')
    members = [p for p in particles if p.particle_id in member_ids]
    half_a = [p for p in members if p.half_set == HalfSet.A]
    half_b = [p for p in members if p.half_set == HalfSet.B]
    if not half_a or not half_b:
        raise StampValidationError(f'class {class_id} has {len(half_a)} half-A and {len(half_b)} half-B particles; a class must be independently refinable on both halves')
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
        raise StampValidationError(f'both halves would be seeded from {ref_a}; each half must start from its own class average')

# seed_reference: this class + half's class average, from the class_averages dir
def seed_reference(class_averages_dir: Path, class_id: str, half: str) -> Path:
    matches = sorted(class_averages_dir.glob(f'{class_id}_half{half}_*.mrc'))
    if not matches:
        raise StampPipelineError(f'No class average for {class_id} half {half} in {class_averages_dir}')
    return matches[0]

# half_map: a half-map finished by an earlier run, still in its half directory or already moved to the class directory, else None
def half_map(tree: dict[str, Path], half: str) -> Path | None:
    for directory in (tree[half], tree['combined']):
        candidate = directory / f'final_{half}.mrc'
        if candidate.is_file():
            return candidate
    return None

# with_inplane: return a copy of the particle with the estimated in-plane roll folded into its orientation
def with_inplane(particle: Particle, angle_by_particle: dict[str, float]) -> Particle:
    angle = angle_by_particle.get(particle.particle_id)
    if angle is None or particle.orientation is None:
        return particle
    return particle.model_copy(update={'orientation': compose_roll_about_normal(particle.orientation, angle)})

# full_poses: {particle_id: (rotation matrix, shift)} for assignments the polish step posed, which replace the in-plane angle
def full_poses(assignments: list[ClassAssignment]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {row.particle_id: (quaternion_to_matrix(row.rotation), np.array(row.shift_voxels if row.shift_voxels is not None else (0.0, 0.0, 0.0))) for row in assignments if row.rotation is not None}

# run_half: one independent half-set refinement, returns the final map path
def run_half(adapter, runner, particles: list[Particle], reference: Path, workdir: Path, parameters: dict) -> Path:
    inputs = AdapterInputs(
        input_paths=[reference],
        output_directory=workdir,
        parameters=parameters,
    )
    adapter.write_particle_star(particles, workdir / 'particles.star', {}, poses=parameters.get('poses'), voxel_size_angstrom=parameters['voxel_size_angstrom'])
    command = adapter.build_command(inputs)
    log.debug(f'{workdir.name}: dispatching {adapter.name} refinement')
    result = runner.run(command)
    if not result.succeeded:
        raise StampPipelineError(f'Refine failed for {workdir.name}: {result.stderr}')
    final_map = workdir / f'final_{workdir.name[-1]}.mrc'
    adapter.collect_final_map(result, reference, final_map)
    return final_map
