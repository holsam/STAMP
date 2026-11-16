'''
STAMP: refinement with enforced half-set independence
'''

# Import external dependencies
import json, numpy as np
from pathlib import Path

# Import STAMP objects
from stamp.adapters.m_refine import MRefineAdapter
from stamp.adapters.mock import get_mock_adapter
from stamp.adapters.relion import RelionRefineAdapter
from stamp.backends.factory import check_backend_supports, select_runner
from stamp.refine.fsc import compute_fsc, soft_sphere_mask, write_fsc_files
from stamp.refine.halfset_guard import (
    assert_distinct_references, full_poses, half_map, refine_output_tree, run_half, seed_reference, split_class_by_half, with_inplane,
)
from stamp.schemas.particles import load_class_assignments, load_particle_set
from stamp.utils.checkpoint import file_signature, signatures, sync_checkpoint
from stamp.utils.errors import StampAdapterError, StampPipelineError, StampValidationError
from stamp.utils.log import log
from stamp.utils.mrc import read_mrc, require_voxel_size
from stamp.utils.plotting.refine import plot_fsc
from stamp.utils.provenance import write_sidecar

# _ADAPTERS: real refine adapters by --tool name
_ADAPTERS = {'relion': RelionRefineAdapter, 'm': MRefineAdapter}

# run_refine: refine one class (or all) with structurally independent half-sets
def run_refine(
    class_id: str,
    identification: Path,
    particles: Path,
    class_assignments: Path,
    raw_tomogram_dir: Path,
    output_dir: Path,
    threshold: float,
    tool: str,
    mask: Path | None,
    iterations: int,
    backend: str,
    voxel_size_angstrom: float | None,
    combined_halfset: bool,
    make_plots: bool = True,
    plot_format: str = 'tiff',
) -> None:
    voxel_size_angstrom = require_voxel_size(voxel_size_angstrom, raw_tomogram_dir)
    particle_set = load_particle_set(particles)
    assignments = load_class_assignments(class_assignments)
    poses = full_poses(assignments)
    angle_by_particle = {row.particle_id: row.inplane_angle_degrees for row in assignments if row.inplane_angle_degrees is not None and row.particle_id not in poses}    # a full pose already contains the in-plane angle
    class_averages_dir = class_assignments.parent / 'class_averages'
    identified = {row['cluster_id'] for row in json.loads(identification.read_text())}

    targets = sorted(identified) if class_id == 'all' else [class_id]
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = get_mock_adapter('relion' if tool == 'relion' else 'm-refine') if backend == 'mock' else _ADAPTERS[tool]()
    check_backend_supports(adapter, backend)
    runner = select_runner(backend, requires_gpu=getattr(adapter, 'requires_gpu', False))
    parameters = {'voxel_size_angstrom': voxel_size_angstrom, 'iterations': iterations, 'poses': poses}     # poses ride along to the adapter that writes the particle file

    failed_targets: list[str] = []
    for target in targets:
        log.progress(f'Refining class {target}')
        if combined_halfset:
            raise StampPipelineError('--combined-halfset escape hatch is A2 future work')
        try:
            half_a, half_b = split_class_by_half(target, assignments, particle_set.particles)
            half_a = [with_inplane(p, angle_by_particle) for p in half_a]
            half_b = [with_inplane(p, angle_by_particle) for p in half_b]
            reference_a = seed_reference(class_averages_dir, target, 'A')
            reference_b = seed_reference(class_averages_dir, target, 'B')
            assert_distinct_references(reference_a, reference_b)

            # everything that changes a half-map; mask, threshold and plots only affect the FSC that always reruns, so they can change between runs
            fingerprint = {
                'particles': file_signature(particles, content=True),
                'class_assignments': file_signature(class_assignments, content=True),
                'references': signatures({'A': reference_a, 'B': reference_b}),
                'tool': tool,
                'iterations': iterations,
                'voxel_size_angstrom': voxel_size_angstrom,
                'backend': backend,
            }
            if sync_checkpoint(output_dir / target, fingerprint):  # the class directory is both output and work dir
                log.info(f'{target}: found matching partial output, resuming')
            tree = refine_output_tree(output_dir, target)

            final_a = half_map(tree, 'A')
            if final_a is None:
                final_a = run_half(adapter, runner, half_a, reference_a, tree['A'], parameters)
            else:
                log.info(f'{target}: half A loaded from checkpoint')
            final_b = half_map(tree, 'B')
            if final_b is None:
                final_b = run_half(adapter, runner, half_b, reference_b, tree['B'], parameters)
            else:
                log.info(f'{target}: half B loaded from checkpoint')

            map_a = read_mrc(final_a, dtype=np.float32)
            map_b = read_mrc(final_b, dtype=np.float32)
            user_mask = None
            if mask is not None:
                user_mask = read_mrc(mask, dtype=np.float32)
            log.progress(f'{target}: computing FSC')
            fsc = compute_fsc(map_a, map_b, voxel_size_angstrom, mask=user_mask if user_mask is not None else soft_sphere_mask(map_a.shape), threshold=threshold)
        except (StampValidationError, StampAdapterError) as exc:
            log.error(f'{target}: {exc}')
            failed_targets.append(target)
            continue

        final_a.replace(tree['combined'] / 'final_A.mrc')
        final_b.replace(tree['combined'] / 'final_B.mrc')
        write_fsc_files(fsc, tree['combined'], threshold=threshold)
        log.info(f'{target}: resolution {fsc.resolution_angstrom:.1f} A @ FSC=0.143')

        if make_plots:
            plot_fsc(fsc, tree['combined'], target, plot_format)

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
    if failed_targets:
        raise StampPipelineError(f'{len(failed_targets)} of {len(targets)} class(es) failed: {", ".join(failed_targets)}')
    log.info(f'Refinement complete for {len(targets)} class(es)')
