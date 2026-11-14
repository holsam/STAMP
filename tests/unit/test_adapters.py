'''
STAMP: unit tests for adapters
'''

# Import external dependencies
import pytest
from pathlib import Path

# Import internal STAMP objects
from stamp.adapters.base import AdapterInputs
from stamp.adapters.mock import MOCK_ADAPTERS, get_mock_adapter
from stamp.backends.mock import MockRunner

# TestAdapters: class containing adapter unit tests
class TestAdapters:
    @pytest.mark.parametrize('name', sorted(MOCK_ADAPTERS))
    def test_mock_adapter_round_trips(self, tmp_path: Path, name: str) -> None:
        adapter = get_mock_adapter(name)
        inputs = AdapterInputs(input_paths=[], output_directory=tmp_path)

        command = adapter.build_command(inputs)
        result = MockRunner().run(command)
        output = adapter.parse_output(result)

        assert command.tool == name
        assert result.succeeded
        assert output.output_paths == []

    def test_unknown_adapter_name_raises_clear_error(self) -> None:
        with pytest.raises(KeyError, match='not-a-real-tool'):
            get_mock_adapter('not-a-real-tool')

    def test_capability_metadata(self) -> None:
        # Spot-check capability metadata
        assert MOCK_ADAPTERS['membrain-pick'].mac_compatible is True
        assert MOCK_ADAPTERS['tomotwin'].mac_compatible is True
        assert MOCK_ADAPTERS['domainfit'].mac_compatible is True
        assert MOCK_ADAPTERS['relion'].mac_compatible is False
        assert MOCK_ADAPTERS['relion'].requires_gpu is True

    def test_all_nine_tools_present(self) -> None:
        expected = {
            'membrain-pick',
            'pytom-match-pick',
            'tomotwin',
            'disca',
            'relion',
            'm-refine',
            'domainfit',
        }
        assert set(MOCK_ADAPTERS) == expected

# _relion_matrix: RELION 5 Euler::anglesToMatrix3 (jaz/math/Euler_angles_relion.h), the matrix mapping particle space to tomogram space
def _relion_matrix(rot: float, tilt: float, psi: float):
    import numpy as np
    p, t, c = np.radians([rot, tilt, psi])
    sp, cp, st, ct, sc, cc = np.sin(p), np.cos(p), np.sin(t), np.cos(t), np.sin(c), np.cos(c)
    return np.array([[cc * ct * cp - sc * sp, cc * ct * sp + sc * cp, -cc * st], [-sc * ct * cp - cc * sp, -sc * ct * sp + cc * cp, sc * st], [st * cp, st * sp, ct]])

# TestRelionPoses: class containing tests for writing class poses into RELION particle files
class TestRelionPoses:
    def test_euler_angles_round_trip(self) -> None:
        import numpy as np
        from scipy.spatial.transform import Rotation
        from stamp.adapters.relion import relion_euler_angles
        for seed in range(5):
            rotation = Rotation.random(random_state=seed).as_matrix()
            rot, tilt, psi = relion_euler_angles(rotation)
            assert np.allclose(_relion_matrix(rot, tilt, psi), rotation)
            assert 0.0 <= tilt <= 180.0

    def test_pose_is_composed_with_the_box_orientation_and_origin_is_in_angstrom(self) -> None:
        import numpy as np
        from scipy.spatial.transform import Rotation
        from stamp.adapters.relion import pose_to_relion
        from stamp.classify.frames import azimuth_matrix, rotation_to_quaternion
        orientation = rotation_to_quaternion(Rotation.from_euler('y', 90, degrees=True).as_matrix())
        row = pose_to_relion(orientation, (azimuth_matrix(30.0), np.array([2.0, 0.0, 0.0])), 5.0)
        composed = Rotation.from_euler('y', 90, degrees=True).as_matrix() @ azimuth_matrix(30.0)
        assert np.allclose(_relion_matrix(row['rlnAngleRot'], row['rlnAngleTilt'], row['rlnAnglePsi']), composed)
        # a 2 voxel box x shift is -z in the tomogram under this orientation, the origin is the negative of it, 5 A per voxel
        assert np.allclose([row['rlnOriginXAngst'], row['rlnOriginYAngst'], row['rlnOriginZAngst']], [0.0, 0.0, 10.0], atol=1e-6)

    def test_star_carries_poses_when_given_and_priors_otherwise(self, tmp_path: Path) -> None:
        import numpy as np, starfile
        from stamp.adapters.relion import RelionRefineAdapter
        from stamp.schemas.particles import HalfSet, Particle
        make = lambda i: Particle(particle_id=f'p{i}', tomogram_id='t', position=(1.0, 2.0, 3.0), orientation=(1.0, 0.0, 0.0, 0.0), source_picker='x', half_set=HalfSet.A)
        RelionRefineAdapter().write_particle_star([make(0), make(1)], tmp_path / 'p.star', {}, poses={'p0': (np.eye(3), np.zeros(3))}, voxel_size_angstrom=4.0)
        table = starfile.read(tmp_path / 'p.star').set_index('rlnTomoParticleName')
        assert 'rlnOriginXAngst' in table.columns and table.loc['p0', 'rlnOriginXAngst'] == 0.0
        assert table.loc['p1', 'rlnAngleTiltPrior'] == 0.0     # no pose for p1, so the old normal-derived prior path
