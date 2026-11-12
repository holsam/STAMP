'''
STAMP: azimuthal roll of subvolumes about membrane normal
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import rotate

# _ROLL_AXES: box axes spanning the plane perpendicular to the membrane normal
_ROLL_AXES = (0, 1)

# roll_about_normal: rotate a subvolume about its normal by angle_degrees
def roll_about_normal(subvolume: np.ndarray, angle_degrees: float) -> np.ndarray:
    if abs(angle_degrees) < 1e-6:
        return subvolume
    return rotate(subvolume, angle_degrees, axes=_ROLL_AXES, reshape=False, order=1, mode='constant')
