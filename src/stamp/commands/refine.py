'''
STAMP: refinement with enforced half-set independence
'''

# Import external dependencies
import json, mrcfile, numpy as np
from pathlib import Path

# Import STAMP objects
from stamp.adapters.base import AdapterInputs
from stamp.adapters.m_refine import MRefineAdapter
from stamp.adapters.mock import get_mock_adapter
from stamp.adapters.relion import RelionRefineAdapter
from stamp.backends.local import LocalRunner
from stamp.backends.mock import MockRunner
from stamp.refine.fsc import compute_fsc, soft_sphere_mask, write_fsc_files
from stamp.refine.halfset_guard import (
    assert_distinct_references, refine_output_tree, split_class_by_half,
)
from stamp.schemas.particles import ClassAssignment, ParticleSet
from stamp.utils.io import write_sidecar

# _ADAPTERS: real refine adapters by --tool name
_ADAPTERS = {'relion': RelionRefineAdapter, 'm': MRefineAdapter}

# _seed_reference: this class + half's class average, from the class_averages dir
def _seed_reference(class_averages_dir: Path, class_id: str, half: str) -> Path:
    matches = sorted(class_averages_dir.glob(f'{class_id}_half{half}_*.mrc'))
    if not matches:
        raise ValueError(f'no class average for {class_id} half {half} in {class_averages_dir}')
    return matches[0]

# _run_half: one independent half-set refinement, returns the final map path
def _run_half(adapter, runner, particles, reference, workdir, parameters, backend) -> Path:
    inputs = AdapterInputs(
        input_paths=[reference],
        output_directory=workdir,
        parameters=parameters,
    )
    if backend != 'mock':
        adapter.write_particle_star(particles, workdir / 'particles.star', {})
    command = adapter.build_command(inputs)
    result = runner.run(command)
    if not result.succeeded:
        raise RuntimeError(f'refine failed for {workdir.name}: {result.stderr}')
    final_map = workdir / f'final_{workdir.name[-1]}.mrc'
    if backend == 'mock':
        # mock backend runs nothing; synthesise a deterministic map from the seed
        with mrcfile.open(str(reference), permissive=True) as mrc:
            seed = np.asarray(mrc.data, dtype=np.float32)
        with mrcfile.new(final_map, overwrite=True) as mrc:
            mrc.set_data(seed)
    else:
        produced = adapter.parse_output(result).parsed['final_map']
        Path(produced).replace(final_map)
    return final_map

# run_refine: refine one class (or all) with structurally independent half-sets
def run_refine(
    class_id: str,
    identification: Path,
    particles: Path,
    class_assignments: Path,
    raw_tomogram_dir: Path,
    output_dir: Path,
    tool: str,
    mask: Path | None,
    iterations: int,
    backend: str,
    voxel_size_angstrom: float,
    combined_halfset: bool
) -> None:
    particle_set = ParticleSet.model_validate(json.loads(particles.read_text()))
    assignments = [ClassAssignment.model_validate(row) for row in json.loads(class_assignments.read_text())]
    class_averages_dir = class_assignments.parent / 'class_averages'
    identified = {row['cluster_id'] for row in json.loads(identification.read_text())}

    targets = sorted(identified) if class_id == 'all' else [class_id]
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = get_mock_adapter('relion' if tool == 'relion' else 'm-refine') if backend == 'mock' else _ADAPTERS[tool]()
    runner = MockRunner() if backend == 'mock' else LocalRunner()
    parameters = {'voxel_size_angstrom': voxel_size_angstrom, 'iterations': iterations}

    for target in targets:
        if combined_halfset:
            raise NotImplementedError('--combined-halfset escape hatch is A2 future work')
        half_a, half_b = split_class_by_half(target, assignments, particle_set.particles)
        tree = refine_output_tree(output_dir, target)
        reference_a = _seed_reference(class_averages_dir, target, 'A')
        reference_b = _seed_reference(class_averages_dir, target, 'B')
        assert_distinct_references(reference_a, reference_b)

        final_a = _run_half(adapter, runner, half_a, reference_a, tree['A'], parameters, backend)
        final_b = _run_half(adapter, runner, half_b, reference_b, tree['B'], parameters, backend)

        with mrcfile.open(str(final_a), permissive=True) as mrc:
            map_a = np.asarray(mrc.data, dtype=np.float32)
        with mrcfile.open(str(final_b), permissive=True) as mrc:
            map_b = np.asarray(mrc.data, dtype=np.float32)
        user_mask = None
        if mask is not None:
            with mrcfile.open(str(mask), permissive=True) as mrc:
                user_mask = np.asarray(mrc.data, dtype=np.float32)
        fsc = compute_fsc(map_a, map_b, voxel_size_angstrom, mask=user_mask if user_mask is not None else soft_sphere_mask(map_a.shape))

        final_a.replace(tree['combined'] / 'final_A.mrc')
        final_b.replace(tree['combined'] / 'final_B.mrc')
        write_fsc_files(fsc, tree['combined'])
        print(f'{target}: resolution {fsc.resolution_angstrom:.1f} A @ FSC=0.143')

        write_sidecar(
            tree['combined'],
            stage='refine',
            tool=f'stamp-{tool}',
            tool_version=None,
            parameters={
                'class_id': target,
                'tool': tool,
                'iterations': iterations,
                'backend': backend,
                'resolution_angstrom': fsc.resolution_angstrom,
                'n_half_a': len(half_a),
                'n_half_b': len(half_b),
                'mask': str(mask) if mask else None,
            },
            inputs=[
                ('particle_set', particles),
                ('class_assignments', class_assignments),
                ('reference:A', reference_a),
                ('reference:B', reference_b),
            ] + ([('mask', mask)] if mask else []),
        )
