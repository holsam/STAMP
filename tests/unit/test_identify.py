'''
STAMP: unit tests for protein parsing, simulation, fitting and decoy checking
'''

# Import external dependencies
import numpy as np, pytest
from scipy.ndimage import shift as shift_volume
from scipy.spatial.transform import Rotation

# Import internal STAMP objects
from stamp.classify.frames import azimuth_matrix, rotate_density
from stamp.identify.decoy_check import evaluate_decoy_control
from stamp.identify.fit import rank_candidates
from stamp.identify.panel import load_candidate_panel
from stamp.identify.register import register_score
from stamp.identify.simulate import simulate_density
from stamp.schemas.particles import OrientationPrior
from synthetic import asymmetric_object

# _REGISTER_BOX: box edge of the registration tests
_REGISTER_BOX = 25

# _write_panel: helper to drop a candidates.yaml plus a stub PDB next to it
def _write_panel(tmp_path, body):
    (tmp_path / 'model.pdb').write_text('ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n')
    path = tmp_path / 'candidates.yaml'
    path.write_text(body)
    return path

# _write_cloud: a PDB of eight Gaussian-scattered atom clusters, different for each seed, so rotated copies are distinguishable from other clouds
def _write_cloud(path, seed):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=22.0, size=(8, 3))
    atoms = np.vstack([c + rng.normal(scale=4.0, size=(6, 3)) for c in centres])
    path.write_text(''.join(f'ATOM  {i + 1:>5}  CA  ALA A{i + 1:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n' for i, (x, y, z) in enumerate(atoms)))

# TestAlignedIdentify: class containing tests for scoring candidates against 3D-aligned class averages
class TestAlignedIdentify:
    # test_aligned_average_picks_its_own_candidate_in_any_orientation: a randomly rotated average of candidate a scores highest against a
    def test_aligned_average_picks_its_own_candidate_in_any_orientation(self, tmp_path):
        from stamp.commands.identify import _score_panel
        for name in ('a', 'b'):
            _write_cloud(tmp_path / f'{name}.pdb', {'a': 1, 'b': 2}[name])
        (tmp_path / 'candidates.yaml').write_text('candidates:\n  - name: a\n    structure_path: a.pdb\n  - name: b\n    structure_path: b.pdb\n')
        panel = load_candidate_panel(tmp_path / 'candidates.yaml')
        average = rotate_density(simulate_density(tmp_path / 'a.pdb', 25, 4.0, 16.0), Rotation.random(random_state=3).as_matrix())
        scores = _score_panel({'c00': (average, 4.0)}, panel, 16.0, 'native', 'local', aligned_prior=OrientationPrior.NONE)['c00']
        assert scores['a'] > 0.9 and scores['a'] > scores['b'] + 0.08

# TestPanel: class containing unit tests for src/stamp/identify/panel.py
class TestPanel:
    def test_loads_valid_panel(self, tmp_path):
        path = _write_panel(tmp_path, 'candidates:\n  - name: a\n    structure_path: model.pdb\n    expected_mass_kda: 30.4\n')
        panel = load_candidate_panel(path)
        assert panel[0].name == 'a' and panel[0].structure_path.is_file()
        assert panel[0].expected_mass_kda == pytest.approx(30.4)

    def test_rejects_missing_structure(self, tmp_path):
        path = _write_panel(tmp_path, 'candidates:\n  - name: a\n    structure_path: absent.pdb\n')
        with pytest.raises(ValueError, match='does not exist'):
            load_candidate_panel(path)

    def test_rejects_no_structure_and_no_fetch(self, tmp_path):
        path = _write_panel(tmp_path, 'candidates:\n  - name: a\n    uniprot: Q08733\n')
        with pytest.raises(ValueError, match='fetch_missing is False'):
            load_candidate_panel(path)

    def test_rejects_empty_list(self, tmp_path):
        path = _write_panel(tmp_path, 'candidates: []\n')
        with pytest.raises(ValueError, match='candidates'):
            load_candidate_panel(path)

# TestRegister: class containing unit tests for src/stamp/identify/register.py
class TestRegister:
    # test_registration_finds_the_matching_candidate_after_any_rotation: for the none prior a randomly rotated and shifted copy of candidate A scores far above candidate B
    def test_registration_finds_the_matching_candidate_after_any_rotation(self) -> None:
        candidate_a, candidate_b = asymmetric_object(_REGISTER_BOX, 1), asymmetric_object(_REGISTER_BOX, 2)
        average = shift_volume(rotate_density(candidate_a, Rotation.random(random_state=5).as_matrix()), (1, -1, 0), order=1)
        score_a = register_score(average, candidate_a, OrientationPrior.NONE, n_rotations=800)
        score_b = register_score(average, candidate_b, OrientationPrior.NONE, n_rotations=800)
        assert score_a > 0.85 and score_a > score_b + 0.2

    # test_registration_matches_through_inverted_contrast: a class average with opposite contrast still matches when match_contrast is on
    def test_registration_matches_through_inverted_contrast(self) -> None:
        candidate = asymmetric_object(_REGISTER_BOX, 3)
        average = -rotate_density(candidate, azimuth_matrix(40.0))
        assert register_score(average, candidate, OrientationPrior.NORMAL) > 0.85
        assert register_score(average, candidate, OrientationPrior.NORMAL, match_contrast=False) < 0.5

    # test_normal_prior_searches_only_azimuth_and_flip: an off-axis tilt of 60 degrees is not recovered by the membrane search, unlike the none prior
    def test_normal_prior_searches_only_azimuth_and_flip(self) -> None:
        candidate = asymmetric_object(_REGISTER_BOX, 4)
        tilted = rotate_density(candidate, Rotation.from_euler('x', 70, degrees=True).as_matrix())
        assert register_score(tilted, candidate, OrientationPrior.NORMAL, n_local=1) < 0.7
        assert register_score(tilted, candidate, OrientationPrior.NONE, n_rotations=800) > 0.85

# TestSimulate: class containing unit tests for src/stamp/identify/simulate.py
class TestSimulate:
    def test_simulated_map_is_centred(self, tmp_path):
        pdb = tmp_path / 'point.pdb'
        pdb.write_text('ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n')
        volume = simulate_density(pdb, box_voxels=21, voxel_size_angstrom=3.0, resolution_angstrom=12.0)
        centre = np.array(np.unravel_index(np.argmax(volume), volume.shape))
        assert np.allclose(centre, 10, atol=1)

    def test_orient_puts_slab_normal_on_z(self):
        from stamp.identify.simulate import orient_to_membrane_slab
        rng = np.random.default_rng(0)
        # a pancake in x/y, thin in z, then rotated into a random frame
        slab = rng.normal(scale=[40.0, 40.0, 6.0], size=(2000, 3))
        random_rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        oriented = orient_to_membrane_slab(slab @ random_rotation.T, np.ones(len(slab)))
        spread = oriented.std(axis=0)
        assert spread[2] == pytest.approx(min(spread), rel=0.05)

    def test_blank_element_column_is_hydrogen(self, tmp_path):
        from stamp.identify.simulate import read_pdb_atoms
        pdb = tmp_path / 'blank.pdb'
        pdb.write_text('ATOM      1  H   ALA A   1       0.000   0.000   0.000  1.00  0.00\n')
        _coords, weights, elements = read_pdb_atoms(pdb)
        assert weights[0] == 1.0 and elements[0] == 'H'

# TestRank: class containing unit tests for src/stamp/identify/fit.py
class TestRank:
    def test_gap_widens_without_distractor(self):
        with_distractor = rank_candidates('c00', {'real': 0.9, 'near': 0.85, 'far': 0.2}, 'm')
        without = rank_candidates('c00', {'real': 0.9, 'far': 0.2}, 'm')
        assert without.score_gap_to_runner_up > with_distractor.score_gap_to_runner_up
        assert with_distractor.candidate_protein == 'real'

    def test_single_candidate_has_no_gap(self):
        result = rank_candidates('c00', {'only': 0.7}, 'm')
        assert result.score_gap_to_runner_up is None

# TestDecoyCheck: class containing unit tests for src/stamp/identify/decoy_check.py
class TestDecoyControl:
    def test_clear_separation_passes(self):
        result = evaluate_decoy_control([0.80, 0.78, 0.76], [0.30, 0.28, 0.31, 0.29])
        assert result.passed and result.separation_sigma > 2.0

    def test_overlap_fails_on_sigma(self):
        result = evaluate_decoy_control([0.42], [0.40, 0.39, 0.41, 0.38])
        assert not result.passed and 'sigma' in result.reason

    def test_fires_on_overlapping_distributions(self):
        result = evaluate_decoy_control([0.6, 0.55, 0.58], [0.59, 0.57, 0.56])
        assert not result.passed

    def test_passes_on_separated_distributions(self):
        result = evaluate_decoy_control([0.9, 0.88, 0.91, 0.87], [0.3, 0.25, 0.31, 0.28])
        assert result.passed

    def test_small_class_counts_pass_on_margin_alone(self):
        result = evaluate_decoy_control([0.80, 0.03], [0.13, 0.02])
        assert result.passed
        assert result.mann_whitney_p != result.mann_whitney_p or result.mann_whitney_p > 0.05

    def test_rank_test_still_vetoes_when_powered(self):
        result = evaluate_decoy_control([0.95, 0.50, 0.49, 0.10], [0.60, 0.55, 0.45, 0.05])
        assert not result.passed
        assert 'Mann-Whitney' in result.reason

    # test_decoy_control_passes_when_the_decoy_matched_nothing: no decoy classes is the best outcome, not an error
    def test_decoy_control_passes_when_the_decoy_matched_nothing(self) -> None:
        result = evaluate_decoy_control([0.8, 0.7], [])
        assert result.passed and 'no classes' in result.reason
