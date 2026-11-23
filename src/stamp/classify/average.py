'''
STAMP: class averages from subvolumes
'''

# Import external dependencies
import numpy as np
import scipy.fft as sfft
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.frames import rotate_density
from stamp.utils.mrc import write_class_average_mrc
from stamp.utils.log import log
from stamp.utils.parallel import thread_map

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
        write_class_average_mrc(path, average, voxel_size=voxel_size_angstrom)
        written.append(path)
    return written

# WeightedAverage: running Fourier-space sum of aligned particles and of their rotated wedge coverage, so averages can be built from batches
class WeightedAverage:
    def __init__(self, box_voxels: int) -> None:
        self.total = np.zeros((box_voxels,) * 3, dtype=np.complex128)
        self.coverage = np.zeros((box_voxels,) * 3)
        self.count = 0

    # add: undo each particle's shift and rotation, then accumulate its spectrum and rotated wedge, with threads each summing a slice of the batch before the partial sums are combined
    def add(self, particles: np.ndarray, wedges: np.ndarray, rotations: np.ndarray, shifts: np.ndarray, *, threads: int = 1) -> None:
        # _slice: partial sums over some of the batch, so no two threads touch the same array
        def _slice(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            total = np.zeros_like(self.total)
            coverage = np.zeros_like(self.coverage)
            for i in indices:
                aligned = rotate_density(np.roll(particles[i], -np.round(shifts[i]).astype(int), axis=(0, 1, 2)), rotations[i].T)
                total += np.fft.fftshift(sfft.fftn(aligned))
                coverage += rotate_density(np.fft.fftshift(wedges[i]), rotations[i].T)   # fftshift puts the Fourier origin on the voxel the rotation turns about
            return total, coverage
        for total, coverage in thread_map(_slice, np.array_split(np.arange(len(particles)), max(threads, 1)), threads):
            self.total += total
            self.coverage += coverage
        self.count += len(particles)

    # result: coverage-corrected average, unobserved directions are filled by particles
    def result(self, *, eps: float = 0.05) -> np.ndarray:
        average = np.fft.ifftn(np.fft.ifftshift(self.total / (self.coverage + eps * max(self.count, 1)))).real
        return average.astype(np.float32)

# weighted_average: coverage-corrected average of aligned particles in one call
def weighted_average(particles: np.ndarray, wedges: np.ndarray, rotations: np.ndarray, shifts: np.ndarray, *, eps: float = 0.05, threads: int = 1) -> np.ndarray:
    accumulator = WeightedAverage(particles.shape[1])
    accumulator.add(particles, wedges, rotations, shifts, threads=threads)
    return accumulator.result(eps=eps)
