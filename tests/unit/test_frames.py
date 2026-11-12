'''
STAMP: unit tests for frame and rotation conventions
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import gaussian_filter

# Import internal STAMP objects
from stamp.classify.align import roll_about_normal
from stamp.classify.extract import extract_subvolume, quaternion_to_matrix
from stamp.classify.frames import azimuth_degrees, azimuth_matrix, flip_matrix, rotate_density, rotation_to_quaternion
from stamp.picking.geometry import quaternion_from_reference_to

# _blob: volume with one gaussian blob at the given integer index
def _blob(shape: tuple[int, int, int], index: tuple[int, int, int]) -> np.ndarray:
    volume = np.zeros(shape, dtype=np.float32)
    volume[index] = 1.0
    return gaussian_filter(volume, 1.0)

# _peak: index of the maximum of a volume
def _peak(volume: np.ndarray) -> tuple[int, ...]:
    return tuple(int(i) for i in np.unravel_index(volume.argmax(), volume.shape))

# TestFrames: test for src/stamp/classify/frames.py
class TestFrames:
    # test_rotate_density_moves_features_by_the_matrix: a feature at +x moves to +y under a +90 degree azimuth and to -z under the flip
    def test_rotate_density_moves_features_by_the_matrix(self) -> None:
        volume = _blob((15, 15, 15), (10, 7, 9))    # offset (+3, 0, +2) from the centre
        assert _peak(rotate_density(volume, azimuth_matrix(90.0))) == (7, 10, 9)
        assert _peak(rotate_density(volume, flip_matrix())) == (10, 7, 5)

    # test_roll_about_normal_is_azimuth_matrix: the existing scipy roll equals rotate_density with azimuth_matrix for random angles
    def test_roll_about_normal_is_azimuth_matrix(self) -> None:
        rng = np.random.default_rng(0)
        volume = gaussian_filter(rng.normal(size=(17, 17, 17)).astype(np.float32), 2.0)
        for angle in rng.uniform(-180, 180, 5):
            rolled = roll_about_normal(volume, float(angle))
            moved = rotate_density(volume, azimuth_matrix(float(angle)))
            inside = np.s_[4:-4, 4:-4, 4:-4]
            assert np.allclose(rolled[inside], moved[inside], atol=1e-3)

    # test_azimuth_degrees_inverts_azimuth_matrix: angle round trip across the full range
    def test_azimuth_degrees_inverts_azimuth_matrix(self) -> None:
        for angle in (-170.0, -45.0, 0.0, 30.0, 179.0):
            assert np.isclose(azimuth_degrees(azimuth_matrix(angle)), angle)

    # test_rotation_to_quaternion_round_trip: quaternion_to_matrix recovers the matrix, w is non-negative
    def test_rotation_to_quaternion_round_trip(self) -> None:
        rotation = azimuth_matrix(70.0) @ flip_matrix()
        quaternion = rotation_to_quaternion(rotation)
        assert quaternion[0] >= 0 and np.allclose(quaternion_to_matrix(quaternion), rotation)

    # test_off_axis_density_lands_on_the_normal: a density planted 8 voxels along a tilted normal sits on +z of the extracted box
    def test_off_axis_density_lands_on_the_normal(self) -> None:
        normal = np.array([0.6, 0.0, 0.8])
        position = np.array([30.0, 30.0, 30.0])
        tomogram = np.zeros((61, 61, 61), dtype=np.float32)    # zyx array
        target_xyz = position + 8.0 * normal
        tomogram[int(round(target_xyz[2])), int(round(target_xyz[1])), int(round(target_xyz[0]))] = 1.0
        tomogram = gaussian_filter(tomogram, 1.0)
        quaternion = quaternion_from_reference_to(normal)
        box = extract_subvolume(tomogram, tuple(position), quaternion, 21)
        assert _peak(box) == (10, 10, 18)   # centre 10, plus 8 along +z
