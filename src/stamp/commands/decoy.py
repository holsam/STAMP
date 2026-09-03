'''
STAMP: decoy dataset generation logic
'''

# Import external dependencies
import json
from pathlib import Path

# Import internal STAMP objects
from stamp.decoy.generate import (
    METHOD_REJECTED_SURFACE,
    METHOD_SHIFTED,
    METHOD_SYNTHETIC_NOISE,
    generate_rejected_surface_decoys,
    generate_shifted_decoys,
    generate_synthetic_noise_decoys,
)
from stamp.picking.native import NativePickerConfig
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.particles import ParticleSet
from stamp.utils.io import write_sidecar

# METHODS: decoy generation methods
METHODS = (METHOD_REJECTED_SURFACE, METHOD_SHIFTED, METHOD_SYNTHETIC_NOISE)

# run_decoy: generate a decoy dataset to run through STAMP alongside real data
def run_decoy(
    output_dir,
    method,
    parameters,
    real_particle_set,
    segmentation_dir,
    raw_tomogram_dir,
    voxel_size_angstrom,
    n_decoys_per_tomogram,
    min_distance_from_real_angstrom,
    min_shift_angstrom,
    max_shift_angstrom,
    n_synthetic_tomograms,
    synthetic_shape,
    seed,
) -> None:
    '''Generate a decoy dataset to run through STAMP alongside real data'''
    config_fields = {
        key: value
        for key, value in parameters.items()
        if key in NativePickerConfig.__dataclass_fields__
    }
    config = NativePickerConfig(voxel_size_angstrom=voxel_size_angstrom, **config_fields)

    output_dir.mkdir(parents=True, exist_ok=True)

    if method == METHOD_SYNTHETIC_NOISE:
        shape = tuple(int(value) for value in synthetic_shape.split(','))
        if len(shape) != 3:
            print('--synthetic-shape must be three integers')
            raise SystemExit(code=1)
        decoy_set, decoy_manifests = generate_synthetic_noise_decoys(
            tomogram_shape=shape,
            n_tomograms=n_synthetic_tomograms,
            n_decoys_per_tomogram=n_decoys_per_tomogram,
            output_dir=output_dir,
            config=config,
            seed=seed,
        )
        (output_dir / 'decoy_manifests.json').write_text(
            json.dumps([m.model_dump(mode='json') for m in decoy_manifests], indent=2)
        )
    else:
        if not (real_particle_set and segmentation_dir and raw_tomogram_dir):
            print(f'--method {method} requires --real-particle-set, --segmentation-dir and --raw-tomogram-dir.')
            raise SystemExit(code=1)
        real_set = ParticleSet.model_validate(json.loads(real_particle_set.read_text()))
        manifests = _load_manifests(segmentation_dir, raw_tomogram_dir, voxel_size_angstrom)
        if not manifests:
            print('No matched segmentation/tomogram pairs found')
            raise SystemExit(code=1)

        if method == METHOD_REJECTED_SURFACE:
            decoy_set = generate_rejected_surface_decoys(
                real_particle_set=real_set,
                manifests=manifests,
                config=config,
                n_decoys_per_tomogram=n_decoys_per_tomogram,
                min_distance_from_real_angstrom=min_distance_from_real_angstrom,
                seed=seed,
            )
        else:
            decoy_set = generate_shifted_decoys(
                real_particle_set=real_set,
                manifests=manifests,
                config=config,
                min_shift_angstrom=min_shift_angstrom,
                max_shift_angstrom=max_shift_angstrom,
                min_distance_from_surface_angstrom=min_distance_from_real_angstrom,
                seed=seed,
            )

    if decoy_set is None or not decoy_set.particles:
        print('No decoy positions generated. For rejected-surface, check that --min-distance-from-real-angstrom isn\'t excluding the whole surface; for shifted, that the shift range fits inside the volume.')
        raise SystemExit(code=1)

    decoy_path = output_dir / 'decoy_particle_set.json'
    decoy_path.write_text(decoy_set.model_dump_json(indent=2))

    write_sidecar(
        output_dir,
        stage='decoy',
        tool=f'stamp-{method}',
        tool_version=None,
        parameters={
            'method': method,
            'voxel_size_angstrom': voxel_size_angstrom,
            'n_decoys_per_tomogram': n_decoys_per_tomogram,
            'min_distance_from_real_angstrom': min_distance_from_real_angstrom,
            'min_shift_angstrom': min_shift_angstrom,
            'max_shift_angstrom': max_shift_angstrom,
            'n_synthetic_tomograms': n_synthetic_tomograms,
            'synthetic_shape': synthetic_shape,
            'seed': seed,
            'picker_params': parameters,
        },
        inputs=(
            [('particle_set', real_particle_set)] if real_particle_set else []
        ) + (
            [('segmentation', segmentation_dir)] if segmentation_dir else []
        ),
    )
    print(f'Wrote {len(decoy_set.particles)} decoy particles to {decoy_path}')


# _load_manifests: match segmentations to raw tomograms by filename stem
def _load_manifests(
    segmentation_dir: Path, raw_tomogram_dir: Path, voxel_size_angstrom: float
) -> list[TomogramManifest]:
    raw_by_stem = {path.stem: path for path in sorted(raw_tomogram_dir.glob('*.mrc'))}
    return [
        TomogramManifest(
            tomogram_id=segmentation_path.stem,
            segmentation_path=segmentation_path,
            raw_tomogram_path=raw_by_stem[segmentation_path.stem],
            voxel_size_angstrom=voxel_size_angstrom,
        )
        for segmentation_path in sorted(segmentation_dir.glob('*.mrc'))
        if segmentation_path.stem in raw_by_stem
    ]
