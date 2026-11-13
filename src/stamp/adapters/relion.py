'''
STAMP: subprocess adapter for a single half-set RELION-5 tomography refinement
'''

# Import external dependencies
import numpy as np, starfile
from pathlib import Path
from scipy.spatial.transform import Rotation

# Import STAMP objects
from stamp.adapters.base import AdapterInputs, AdapterOutput
from stamp.backends.base import RunResult, ToolCommand
from stamp.classify.extract import quaternion_to_matrix
from stamp.schemas.particles import Particle
from stamp.utils.errors import StampAdapterError
from stamp.utils.log import log

# _REFERENCE_AXIS: the model axis STAMP's picker orientation quaternion maps onto the membrane normal
_REFERENCE_AXIS = np.array([0.0, 0.0, 1.0])

# normal_to_tilt_psi: RELION tilt/psi priors (degrees) from a particle's orientation quaternion
def normal_to_tilt_psi(orientation: tuple[float, float, float, float] | None) -> tuple[float, float]:
    if orientation is None:
        return 0.0, 0.0
    normal = quaternion_to_matrix(orientation) @ _REFERENCE_AXIS
    normal = normal / np.linalg.norm(normal)
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    psi = float(np.degrees(np.arctan2(normal[1], normal[0])))
    return tilt, psi

# ParticlePose: a particle's class pose as classify wrote it, class average moved by rotation then shift (box x, y, z voxels) gives the subvolume
ParticlePose = tuple[np.ndarray, np.ndarray]

# relion_euler_angles: (rot, tilt, psi) in degrees of rotation taking reference into tomogram frame, RELION's A = Rz(psi) Ry(tilt) Rz(rot) acting on column vectors, so scipy's extrinsic zyz
def relion_euler_angles(rotation: np.ndarray) -> tuple[float, float, float]:
    rot, tilt, psi = Rotation.from_matrix(rotation).as_euler('zyz', degrees=True)
    return float(rot), float(tilt), float(psi)

# pose_to_relion: Euler angles and origin (Angstrom) of a particle from its box orientation and class pose (reference-to-tomogram rotation is orientation @ rotation, and origin is minus of shift taken to tomogram frame because RELION's origin is shift that centres particle)
def pose_to_relion(orientation: tuple[float, float, float, float] | None, pose: ParticlePose, voxel_size_angstrom: float) -> dict[str, float]:
    box_to_tomogram = np.eye(3) if orientation is None else quaternion_to_matrix(orientation)
    rot, tilt, psi = relion_euler_angles(box_to_tomogram @ pose[0])
    origin = -(box_to_tomogram @ pose[1]) * voxel_size_angstrom
    return {'rlnAngleRot': rot, 'rlnAngleTilt': tilt, 'rlnAnglePsi': psi, 'rlnOriginXAngst': float(origin[0]), 'rlnOriginYAngst': float(origin[1]), 'rlnOriginZAngst': float(origin[2])}

# RelionRefineAdapter: one half-set RELION-5 Refine3D job
class RelionRefineAdapter:
    name = 'relion'
    stage = 'refine'
    mac_compatible = False
    requires_gpu = True
    automatable = True
    batches_natively = True
    runs_in_process = False

    # write_particle_star: RELION-5 particles.star, with the full class pose as angles and origins where classify estimated one, else normal-derived angle priors
    def write_particle_star(
        self,
        particles: list[Particle],
        path: Path,
        raw_tomogram_paths: dict[str, Path],
        poses: dict[str, ParticlePose] | None = None,
        voxel_size_angstrom: float | None = None,
    ) -> None:
        rows = []
        for particle in particles:
            tilt_prior, psi_prior = normal_to_tilt_psi(particle.orientation)
            pose = (poses or {}).get(particle.particle_id)
            if pose is not None:
                if voxel_size_angstrom is None:
                    raise StampAdapterError('voxel_size_angstrom is required to write particle poses')
                rows.append({
                    'rlnTomoName': particle.tomogram_id,
                    'rlnCoordinateX': particle.position[0],
                    'rlnCoordinateY': particle.position[1],
                    'rlnCoordinateZ': particle.position[2],
                    **pose_to_relion(particle.orientation, pose, voxel_size_angstrom),
                    'rlnTomoParticleName': particle.particle_id,
                })
                continue
            rows.append({
                'rlnTomoName': particle.tomogram_id,
                'rlnCoordinateX': particle.position[0],
                'rlnCoordinateY': particle.position[1],
                'rlnCoordinateZ': particle.position[2],
                'rlnAngleTiltPrior': tilt_prior,
                'rlnAnglePsiPrior': psi_prior,
                'rlnAngleRot': 0.0,
                'rlnTomoParticleName': particle.particle_id,
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        log.debug(f'Writing {len(rows)} particles to {path}')
        import pandas as pd
        starfile.write({'particles': pd.DataFrame(rows)}, path, overwrite=True)

    # build_command: one half-set Refine3D job seeded from this half's Stage D class average (A2)
    def build_command(self, inputs: AdapterInputs) -> ToolCommand:
        if len(inputs.input_paths) != 1:
            raise StampAdapterError('relion refine takes exactly one seed reference (this half\'s class average)')
        reference = inputs.input_paths[0]
        particle_star = inputs.output_directory / 'particles.star'
        output_prefix = inputs.output_directory / 'run'
        parameters = inputs.parameters
        argv = [
            'relion_refine',
            '--i', str(particle_star),
            '--ref', str(reference),
            '--o', str(output_prefix),
            '--angpix', str(parameters['voxel_size_angstrom']),
            '--particle_diameter', str(parameters.get('particle_diameter_angstrom', 300.0)),
            '--iter', str(parameters.get('iterations', 5)),
            '--healpix_order', '2',
            '--offset_range', '5', '--offset_step', '2',
            '--sym', parameters.get('symmetry', 'C1'),
            '--pad', '2', '--flatten_solvent', '--zero_mask',
            '--gpu', '',
        ]
        log.debug(f'{self.name}: {" ".join(argv)}')
        return ToolCommand(
            tool=self.name,
            argv=argv,
            working_directory=inputs.output_directory,
            output_paths=[
                inputs.output_directory / 'run_class001.mrc',
                inputs.output_directory / 'run_model.star',
            ],
        )

    # parse_output: final map path and current resolution from run_model.star
    def parse_output(self, result: RunResult) -> AdapterOutput:
        model_star = next((p for p in result.output_paths if p.name == 'run_model.star'), None)
        final_map = next((p for p in result.output_paths if p.name.endswith('class001.mrc')), None)
        resolution = None
        if model_star is not None and Path(model_star).is_file():
            model = starfile.read(model_star)
            table = model.get('model_general', model) if isinstance(model, dict) else model
            resolution = float(np.asarray(table['rlnCurrentResolution']).ravel()[0])
        else:
            log.warning('run_model.star not found in RELION output; resolution unknown')
        return AdapterOutput(
            output_paths=[p for p in (final_map,) if p is not None],
            parsed={'final_map': str(final_map) if final_map else None, 'resolution_angstrom': resolution},
        )
