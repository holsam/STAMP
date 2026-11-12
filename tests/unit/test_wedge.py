'''
STAMP: unit tests for per-particle missing wedge
'''

# Import external dependencies
import numpy as np

# Import internal STAMP objects
from stamp.classify.extract import quaternion_to_matrix
from stamp.classify.wedge import apply_wedge, box_wedge, observed_directions
from stamp.picking.geometry import quaternion_from_reference_to

# _brute_wedge: explicit triple loop, tomogram-frame vectors from the rotation matrix, tilt about the axis at tilt_axis_degrees
def _brute_wedge(box: int, orientation, tilt_max: float, tilt_axis: float) -> np.ndarray:
    rotation = np.eye(3) if orientation is None else quaternion_to_matrix(orientation)
    axis = np.radians(tilt_axis)
    direction = np.array([np.cos(axis), np.sin(axis), 0.0])     # tilt axis itself
    freq = np.fft.fftfreq(box)
    out = np.zeros((box,) * 3, dtype=np.float32)
    for i in range(box):
        for j in range(box):
            for k in range(box):
                tomo = rotation @ np.array([freq[i], freq[j], freq[k]])
                across = np.sqrt(max(tomo[0] ** 2 + tomo[1] ** 2 - (tomo @ direction) ** 2, 0.0))   # in-plane component perpendicular to the tilt axis
                out[i, j, k] = abs(tomo[2]) <= across * np.tan(np.radians(tilt_max)) + 1e-12
    return out

# TestWedge: unit tests for classify/wedge.py
class TestWedge:
    # test_identity_orientation_is_the_global_mask: tilt about y with identity orientation keeps |kz| <= |kx| tan(tilt)
    def test_identity_orientation_is_the_global_mask(self) -> None:
        box = 15
        freq = np.fft.fftfreq(box)
        kx, _ky, kz = np.meshgrid(freq, freq, freq, indexing='ij')
        expected = (np.abs(kz) <= np.abs(kx) * np.tan(np.radians(60.0))).astype(np.float32)
        assert np.array_equal(box_wedge(box, None, 60.0, 90.0), expected)

    # test_rotated_wedge_matches_brute_force: random orientations and tilt axes agree with the explicit construction
    def test_rotated_wedge_matches_brute_force(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(3):
            normal = rng.normal(size=3)
            orientation = quaternion_from_reference_to(normal)
            tilt_axis = float(rng.uniform(0, 180))
            fast = box_wedge(9, orientation, 55.0, tilt_axis)
            slow = _brute_wedge(9, orientation, 55.0, tilt_axis)
            assert (fast != slow).mean() < 0.01     # only voxels exactly on the edge may differ by rounding

    # test_wedge_is_centrosymmetric: the mask is symmetric under k -> -k so filtered volumes stay real
    def test_wedge_is_centrosymmetric(self) -> None:
        orientation = quaternion_from_reference_to(np.array([0.3, -0.5, 0.8]))
        wedge = box_wedge(11, orientation, 50.0, 30.0)
        assert np.array_equal(wedge, np.roll(wedge[::-1, ::-1, ::-1], 1, axis=(0, 1, 2)))

    # test_no_tilt_means_no_wedge: unset tilt keeps everything
    def test_no_tilt_means_no_wedge(self) -> None:
        assert box_wedge(7, None, None, 90.0).all()
        assert observed_directions(np.eye(3), None, None, 90.0).all()

    # test_apply_wedge_is_idempotent: filtering twice changes nothing
    def test_apply_wedge_is_idempotent(self) -> None:
        rng = np.random.default_rng(2)
        wedge = box_wedge(11, None, 60.0, 90.0)
        once = apply_wedge(rng.normal(size=(11, 11, 11)), wedge)
        assert np.allclose(apply_wedge(once, wedge), once, atol=1e-5)
