'''
STAMP: 2D mocked registration of a simulated map to a class average
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import rotate

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior
from stamp.tools.diagram.utils.align import align_to_references
from stamp.tools.diagram.utils.search import SearchSpace

# Registration: best pose of a simulated map against a class average, with the coarse and refinement curves the diagram draws
@dataclass
class Registration:
    score: float
    angle: float
    shift: tuple[int, int]
    sign: float                 # contrast of the simulated map that fitted, the density sign is unknown
    fitted: np.ndarray          # the simulated map at the best pose
    coarse_angles: np.ndarray
    coarse_cc: np.ndarray       # best correlation over shifts at each coarse angle, for the winning contrast
    refined: list[tuple[float, float]]  # (angle, cc) after each local refinement round

# _cc_over_angles: best correlation over shifts at each of the given angles
def _cc_over_angles(average: np.ndarray, simulated: np.ndarray, angles: np.ndarray, prior: OrientationPrior, max_shift: int) -> np.ndarray:
    ones = np.ones((1, *average.shape), dtype=np.float32)
    return np.array([align_to_references(average[None], ones, simulated[None], SearchSpace(prior, np.array([angle]), max_shift)).cc[0, 0] for angle in angles])

# register_score: best normalised correlation of the simulated map with the class average over every rotation: a coarse angular scan, then two rounds of local refinement, for both contrasts
def register_score(average: np.ndarray, simulated: np.ndarray, prior: OrientationPrior, *, max_shift: int = 3, step: float | None = None) -> Registration:
    step = step or (15.0 if prior is OrientationPrior.NORMAL else 10.0)   # the membrane candidate is slab-oriented, so only the azimuth is scanned
    coarse = np.arange(0.0, 360.0, step)
    best: Registration | None = None
    for sign in (1.0, -1.0):
        reference = sign * simulated
        curve = _cc_over_angles(average, reference, coarse, prior, max_shift)
        angle, score, spread, refined = float(coarse[curve.argmax()]), float(curve.max()), step / 2, []
        for _ in range(2):
            local = angle + np.linspace(-spread, spread, 5)
            local_curve = _cc_over_angles(average, reference, local, prior, max_shift)
            if local_curve.max() > score:
                angle, score = float(local[local_curve.argmax()]), float(local_curve.max())
            refined.append((angle, score))
            spread /= 2
        if best is None or score > best.score:
            size = average.shape[0]
            pose = align_to_references(average[None], np.ones((1, size, size), dtype=np.float32), reference[None], SearchSpace(prior, np.array([angle]), max_shift))
            shift = (int(pose.shift[0, 0, 0]), int(pose.shift[0, 0, 1]))
            fitted = np.roll(rotate(reference, angle, reshape=False, order=1, mode='constant'), shift, axis=(0, 1))
            best = Registration(score, angle, shift, sign, fitted, coarse, curve, refined)
    return best
