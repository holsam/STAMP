'''
STAMP: subvolume extraction
'''

# Import external dependencies
import mrcfile, numpy as np
from scipy.ndimage import map_coordinates

# Import STAMP schema
from stamp.schemas.particles import Particle

# quaternion_to_matrix: return a rotation matrix from a unit quaternion (w, x, y, z)
def quaternion_to_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )

# canonical_grid: sampling grid for a cubic box centred on the origin
def canonical_grid(box_voxels: int) -> np.ndarray:
    if box_voxels < 2:
        raise ValueError('box_voxels must be at least 2')
    half = (box_voxels - 1) / 2.0
    axis = np.arange(box_voxels) - half
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    return np.stack([x.ravel(), y.ravel(), z.ravel()])

# extract_subvolume: sample one subvolume in the particle's canonical frame
def extract_subvolume(
    tomogram: np.ndarray,
    position_xyz: tuple[float, float, float],
    orientation: tuple[float, float, float, float] | None,
    box_voxels: int,
) -> np.ndarray:
    grid_xyz = canonical_grid(box_voxels)
    if orientation is not None:
        grid_xyz = quaternion_to_matrix(orientation) @ grid_xyz

    sample_xyz = grid_xyz + np.array(position_xyz)[:, None]
    sample_zyx = sample_xyz[::-1]  # single xyz -> zyx conversion

    values = map_coordinates(
        tomogram.astype(np.float32), sample_zyx, order=1, mode='constant', cval=0.0
    )
    return values.reshape(box_voxels, box_voxels, box_voxels)

# box_fits_inside: whether a rotated box at this position stays inside the volume
def box_fits_inside(
    position_xyz: tuple[float, float, float], shape_zyx: tuple[int, ...], box_voxels: int
) -> bool:
    radius = (box_voxels - 1) / 2.0 * np.sqrt(3.0)
    extent_xyz = np.array(shape_zyx)[::-1] - 1
    position = np.array(position_xyz)
    return bool(np.all(position >= radius) and np.all(position <= extent_xyz - radius))

# extract_particle_set: extract subvolumes for a list of particles
def extract_particle_set(
    particles: list[Particle],
    tomogram_paths: dict[str, str],
    box_voxels: int,
    segmentation_paths: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[Particle], list[Particle]]:
    by_tomogram: dict[str, list[Particle]] = {}
    for particle in particles:
        by_tomogram.setdefault(particle.tomogram_id, []).append(particle)

    segmentation_paths = segmentation_paths or {}
    subvolumes: list[np.ndarray] = []
    kept: list[Particle] = []
    skipped: list[Particle] = []

    for tomogram_id, group in by_tomogram.items():
        path = tomogram_paths.get(tomogram_id)
        if path is None:
            skipped.extend(group)
            continue

        with mrcfile.open(str(path), permissive=True) as mrc:
            tomogram = np.asarray(mrc.data).astype(np.float32)

        seg_path = segmentation_paths.get(tomogram_id)
        if seg_path is not None:
            with mrcfile.open(str(seg_path), permissive=True) as mrc:
                segmentation = np.asarray(mrc.data)
            if segmentation.shape != tomogram.shape:
                raise ValueError(f'segmentation shape {segmentation.shape} does not match tomogram shape {tomogram.shape} for {tomogram_id!r}; they must be the same volume at the same binning')
            # replace membrane voxels with the background median so the class average is not dominated by the membrane slab
            tomogram[segmentation > 0] = np.median(tomogram[segmentation <= 0])

        for particle in group:
            if not box_fits_inside(particle.position, tomogram.shape, box_voxels):
                skipped.append(particle)
                continue
            subvolumes.append(
                extract_subvolume(tomogram, particle.position, particle.orientation, box_voxels)
            )
            kept.append(particle)

    if not subvolumes:
        return np.empty((0, box_voxels, box_voxels, box_voxels)), kept, skipped
    return np.stack(subvolumes), kept, skipped
