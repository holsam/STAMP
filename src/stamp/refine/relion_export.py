'''
STAMP: RELION-5 particle export and suggested-settings README for prepared half-sets
'''

# Import external dependencies
import numpy as np, pandas as pd, starfile
from pathlib import Path
from typing import Literal
from scipy.spatial.transform import Rotation

# Import STAMP objects
from stamp.schemas.particles import Particle
from stamp.utils.errors import StampValidationError
from stamp.utils.log import log
from stamp.utils.rotation import quaternion_to_matrix

# _REFERENCE_AXIS: the model axis STAMP's picker orientation quaternion maps onto the membrane normal
_REFERENCE_AXIS = np.array([0.0, 0.0, 1.0])

# ParticlePose: a particle's class pose as classify wrote it, class average moved by rotation then shift (box x, y, z voxels) gives the subvolume
ParticlePose = tuple[np.ndarray, np.ndarray]

# normal_to_tilt_psi: RELION tilt/psi priors (degrees) from a particle's orientation quaternion
def normal_to_tilt_psi(orientation: tuple[float, float, float, float] | None) -> tuple[float, float]:
    if orientation is None:
        return 0.0, 0.0
    normal = quaternion_to_matrix(orientation) @ _REFERENCE_AXIS
    normal = normal / np.linalg.norm(normal)
    # with phi = 0, RELION's A(0, tilt, psi) maps the particle z axis to (-cos psi sin tilt, sin psi sin tilt, cos tilt), so psi takes -n_x (not n_x)
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    psi = float(np.degrees(np.arctan2(normal[1], -normal[0])))
    return tilt, psi

# relion_euler_angles: (rot, tilt, psi) in degrees such that RELION's matrix A(rot, tilt, psi) equals the reference-to-tomogram rotation; Euler::anglesToMatrix3 in RELION 5 gives A = (Rz(rot) Ry(tilt) Rz(psi)).T with active rotations, so decode R.T as scipy's extrinsic zyz and read the angles in reverse
def relion_euler_angles(rotation: np.ndarray) -> tuple[float, float, float]:
    psi, tilt, rot = Rotation.from_matrix(np.asarray(rotation).T).as_euler('zyz', degrees=True)
    return float(rot), float(tilt), float(psi)

# pose_to_relion: Euler angles and origin (Angstrom) of a particle from its box orientation and class pose (reference-to-tomogram rotation is orientation @ rotation, and origin is minus of shift taken to tomogram frame because RELION's origin is shift that centres particle)
def pose_to_relion(orientation: tuple[float, float, float, float] | None, pose: ParticlePose, voxel_size_angstrom: float) -> dict[str, float]:
    box_to_tomogram = np.eye(3) if orientation is None else quaternion_to_matrix(orientation)
    rot, tilt, psi = relion_euler_angles(box_to_tomogram @ pose[0])
    origin = -(box_to_tomogram @ pose[1]) * voxel_size_angstrom
    return {'rlnAngleRot': rot, 'rlnAngleTilt': tilt, 'rlnAnglePsi': psi, 'rlnOriginXAngst': float(origin[0]), 'rlnOriginYAngst': float(origin[1]), 'rlnOriginZAngst': float(origin[2])}

# centred_coordinates_angstrom: (x, y, z) voxel position to Angstrom from the tomogram centre, as RELION-5 rlnCenteredCoordinate{X,Y,Z}Angst
def centred_coordinates_angstrom(
    position: tuple[float, float, float],
    shape_xyz: tuple[int, int, int],
    voxel_size_angstrom: float,
) -> tuple[float, float, float]:
    # RELION's tomogram centre is integer division (getMatrix4x4 uses ((int)w)/2), so an odd size centres half a voxel lower than size/2
    return tuple((p - size // 2) * voxel_size_angstrom for p, size in zip(position, shape_xyz))

# _position_columns: RELION position columns for one particle, centred Angstrom from the tomogram centre (default) or the STAMP voxel position as rlnCoordinate* (pixel)
def _position_columns(
    particle: Particle,
    coordinate_system: Literal['centred', 'pixel'],
    shape_by_tomogram: dict[str, tuple[int, int, int]] | None,
    voxel_size_angstrom: float,
) -> dict[str, float]:
    if coordinate_system == 'pixel':
        x, y, z = particle.position
        return {'rlnCoordinateX': x, 'rlnCoordinateY': y, 'rlnCoordinateZ': z}
    if shape_by_tomogram is None or particle.tomogram_id not in shape_by_tomogram:
        raise StampValidationError(f'no raw tomogram size for {particle.tomogram_id!r}, cannot write its centred coordinates (pass -r/--raw-dir with that tomogram, or use --coordinate-system pixel)')
    x, y, z = centred_coordinates_angstrom(particle.position, shape_by_tomogram[particle.tomogram_id], voxel_size_angstrom)
    return {'rlnCenteredCoordinateXAngst': x, 'rlnCenteredCoordinateYAngst': y, 'rlnCenteredCoordinateZAngst': z}

# write_particle_star: RELION-5 particles.star with centred Angstrom coordinates (default) or pixel rlnCoordinate* columns, and the full class pose as angles and origins where classify estimated one, else normal-derived angle priors
def write_particle_star(
    particles: list[Particle],
    path: Path,
    *,
    voxel_size_angstrom: float,
    shape_by_tomogram: dict[str, tuple[int, int, int]] | None = None,   # required for centred coordinates only
    coordinate_system: Literal['centred', 'pixel'] = 'centred',
    poses: dict[str, ParticlePose] | None = None,
) -> None:
    rows = []
    for particle in particles:
        coordinates = {'rlnTomoName': particle.tomogram_id, **_position_columns(particle, coordinate_system, shape_by_tomogram, voxel_size_angstrom)}
        pose = (poses or {}).get(particle.particle_id)
        if pose is not None:
            rows.append({
                **coordinates,
                **pose_to_relion(particle.orientation, pose, voxel_size_angstrom),
                'rlnTomoParticleName': particle.particle_id,
            })
            continue
        tilt_prior, psi_prior = normal_to_tilt_psi(particle.orientation)
        rows.append({
            **coordinates,
            'rlnAngleTiltPrior': tilt_prior,
            'rlnAnglePsiPrior': psi_prior,
            'rlnAngleRot': 0.0,
            'rlnTomoParticleName': particle.particle_id,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f'Writing {len(rows)} particles to {path}')
    starfile.write({'particles': pd.DataFrame(rows)}, path, overwrite=True)

# write_half_readme: suggested RELION-5 settings for one prepared half, nothing here is executed by STAMP
def write_half_readme(path: Path, half: str, n_particles: int, voxel_size_angstrom: float) -> None:
    path.write_text('\n'.join([
        f'# STAMP half {half}: {n_particles} particles',
        '',
        'Run an independent Refine3D for this half. Do not share references, masks derived from the other half, or output directories.',
        '',
        '- input particles: particles.star',
        '- initial reference: reference.mrc',
        f'- pixel size: {voxel_size_angstrom} A',
        '- suggested: --healpix_order 2 --offset_range 5 --offset_step 2 --pad 2 --flatten_solvent --zero_mask --sym C1',
        '- STAMP angle priors (rlnAngleTiltPrior/PsiPrior) or full poses are in the star file; enable priors in the RELION job if only priors are present',
        '',
        'Then run: stamp refine assess --class-id <class> --prepared-dir <this class directory> --half-a <final map A> --half-b <final map B>',
        '',
    ]))
