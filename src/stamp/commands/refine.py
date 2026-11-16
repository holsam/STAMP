'''
STAMP: refinement preparation and assessment with enforced half-set independence
'''

# Import external dependencies
import json, shutil, numpy as np
from pathlib import Path

# Import STAMP objects
from stamp.refine.fsc import compute_fsc, soft_sphere_mask, write_fsc_files
from stamp.refine.halfset_guard import (
    assert_distinct_references, assert_independent_outputs, full_poses, refine_output_tree, seed_reference, split_class_by_half, with_inplane,
)
from stamp.refine.relion_export import write_half_readme, write_particle_star
from stamp.schemas.particles import load_class_assignments, load_particle_set
from stamp.utils.errors import StampPipelineError, StampValidationError
from stamp.utils.log import log
from stamp.utils.mrc import read_mrc, read_voxel_size_angstrom, require_voxel_size
from stamp.utils.plotting.refine import plot_fsc
from stamp.utils.provenance import write_sidecar

# run_refine_prepare: write independent half-set inputs for an external RELION refinement
def run_refine_prepare(
    class_id: str,
    identification: Path,
    particles: Path,
    class_assignments: Path,
    raw_tomogram_dir: Path,
    output_dir: Path,
    voxel_size_angstrom: float | None,
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

    failed_targets: list[str] = []
    for target in targets:
        log.progress(f'Preparing class {target}')
        try:
            half_a, half_b = split_class_by_half(target, assignments, particle_set.particles)
            half_a = [with_inplane(p, angle_by_particle) for p in half_a]
            half_b = [with_inplane(p, angle_by_particle) for p in half_b]
            reference_a = seed_reference(class_averages_dir, target, 'A')
            reference_b = seed_reference(class_averages_dir, target, 'B')
            assert_distinct_references(reference_a, reference_b)
        except (StampValidationError, StampPipelineError) as exc:
            log.error(f'{target}: {exc}')
            failed_targets.append(target)
            continue
        tree = refine_output_tree(output_dir, target)
        for half, members, reference in (('A', half_a, reference_a), ('B', half_b, reference_b)):
            write_particle_star(members, tree[half] / 'particles.star', poses=poses, voxel_size_angstrom=voxel_size_angstrom)
            shutil.copyfile(reference, tree[half] / 'reference.mrc')
            write_half_readme(tree[half] / 'README.md', half, len(members), voxel_size_angstrom)
        (tree['combined'] / 'prepare.json').write_text(json.dumps({'class_id': target, 'voxel_size_angstrom': voxel_size_angstrom, 'n_half_a': len(half_a), 'n_half_b': len(half_b)}, indent=2))
        write_sidecar(
            tree['combined'],
            stage='refine-prepare',
            tool='stamp',
            tool_version=None,
            parameters={'class_id': target, 'voxel_size_angstrom': voxel_size_angstrom, 'n_half_a': len(half_a), 'n_half_b': len(half_b)},
            inputs=[('particle_set', particles), ('class_assignments', class_assignments), ('reference:A', reference_a), ('reference:B', reference_b)],
        )
        log.info(f'{target}: wrote half-set inputs to {tree["combined"]}')
    if failed_targets:
        raise StampPipelineError(f'{len(failed_targets)} of {len(targets)} class(es) failed: {", ".join(failed_targets)}')

# run_refine_assess: FSC of two externally refined half-maps, after independence checks
def run_refine_assess(
    class_id: str,
    prepared_dir: Path,
    half_a: Path,
    half_b: Path,
    threshold: float,
    mask: Path | None,
    make_plots: bool = True,
    plot_format: str = 'tiff',
) -> None:
    manifest = json.loads((prepared_dir / 'prepare.json').read_text())    # missing file raises, caught by the CLI error handler as a normal error
    voxel_size_angstrom = manifest['voxel_size_angstrom']
    map_a, map_b = read_mrc(half_a, dtype=np.float32), read_mrc(half_b, dtype=np.float32)
    ref_a = read_mrc(prepared_dir / 'halfA' / 'reference.mrc', dtype=np.float32)
    ref_b = read_mrc(prepared_dir / 'halfB' / 'reference.mrc', dtype=np.float32)
    assert_independent_outputs(half_a, half_b, map_a, map_b, ref_a, ref_b)
    for path in (half_a, half_b):
        measured = read_voxel_size_angstrom(path)    # None when the header has no voxel size, then nothing to compare
        if measured is not None and abs(measured - voxel_size_angstrom) > 1e-3 * voxel_size_angstrom:
            raise StampValidationError(f'{path.name} voxel size {measured} A does not match {voxel_size_angstrom} A from prepare')
    user_mask = read_mrc(mask, dtype=np.float32) if mask is not None else None
    log.progress(f'{class_id}: computing FSC')
    fsc = compute_fsc(map_a, map_b, voxel_size_angstrom, mask=user_mask if user_mask is not None else soft_sphere_mask(map_a.shape), threshold=threshold)
    write_fsc_files(fsc, prepared_dir, threshold=threshold)
    log.info(f'{class_id}: resolution {fsc.resolution_angstrom:.1f} A @ FSC={threshold}')
    if make_plots:
        plot_fsc(fsc, prepared_dir, class_id, plot_format)
    write_sidecar(
        prepared_dir,
        stage='refine-assess',
        tool='stamp',
        tool_version=None,
        parameters={'class_id': class_id, 'threshold': threshold, 'resolution_angstrom': fsc.resolution_angstrom, 'mask': str(mask) if mask else None},
        inputs=[('half_a', half_a), ('half_b', half_b)] + ([('mask', mask)] if mask else []),
    )
