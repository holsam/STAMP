'''
STAMP: class averages from subvolumes
'''

# Import external dependencies
import numpy as np
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.frames import rotate_density
from stamp.utils.io import new_mrc
from stamp.utils.log import log

# write_class_averages: write each class average as an MRC with its voxel size set
def write_class_averages(
    averages: dict[str, tuple[np.ndarray, int]],
    output_dir: Path,
    voxel_size_angstrom: float,
    half_set_label: str | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for cluster_id, (average, count) in averages.items():
        suffix = f'_half{half_set_label}' if half_set_label else ''
        path = output_dir / f'{cluster_id}{suffix}_n{count}.mrc'
        log.debug(f'Writing class average {cluster_id} (half-set: {half_set_label})')
        new_mrc(path, np.transpose(average, (2, 1, 0)).astype(np.float32), voxel_size=voxel_size_angstrom)
        written.append(path)
    return written

# WeightedAverage: running Fourier-space sum of aligned particles and of their rotated wedge coverage, so averages can be built from batches
class WeightedAverage:
    def __init__(self, box_voxels: int) -> None:
        self.total = np.zeros((box_voxels,) * 3, dtype=np.complex128)
        self.coverage = np.zeros((box_voxels,) * 3)
        self.count = 0

    # add: undo each particle's shift and rotation, then accumulate its spectrum and rotated wedge
    def add(self, particles: np.ndarray, wedges: np.ndarray, rotations: np.ndarray, shifts: np.ndarray) -> None:
        for particle, wedge, rotation, shift in zip(particles, wedges, rotations, shifts):
            aligned = rotate_density(np.roll(particle, -np.round(shift).astype(int), axis=(0, 1, 2)), rotation.T)
            self.total += np.fft.fftshift(np.fft.fftn(aligned))
            self.coverage += rotate_density(np.fft.fftshift(wedge), rotation.T)   # fftshift puts the Fourier origin on the voxel the rotation turns about
        self.count += len(particles)

    # result: coverage-corrected average, unobserved directions are filled by particles
    def result(self, *, eps: float = 0.05) -> np.ndarray:
        average = np.fft.ifftn(np.fft.ifftshift(self.total / (self.coverage + eps * max(self.count, 1)))).real
        return average.astype(np.float32)

# weighted_average: coverage-corrected average of aligned particles in one call
def weighted_average(particles: np.ndarray, wedges: np.ndarray, rotations: np.ndarray, shifts: np.ndarray, *, eps: float = 0.05) -> np.ndarray:
    accumulator = WeightedAverage(particles.shape[1])
    accumulator.add(particles, wedges, rotations, shifts)
    return accumulator.result(eps=eps)
