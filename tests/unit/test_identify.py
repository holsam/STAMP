'''
STAMP: unit tests for protein parsing, simulation, fitting and decoy checking
'''

# Import external dependencies
import numpy as np, pytest
from scipy.ndimage import gaussian_filter

# Import internal STAMP objects
from stamp.identify.decoy_check import evaluate_decoy_control
from stamp.identify.fit import fit_candidate, rank_candidates
from stamp.identify.panel import load_candidate_panel
from stamp.identify.simulate import azimuthal_smear, simulate_density

# _write_panel: helper to drop a candidates.yaml plus a stub PDB next to it
def _write_panel(tmp_path, body):
    (tmp_path / 'model.pdb').write_text('ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n')
    path = tmp_path / 'candidates.yaml'
    path.write_text(body)
    return path

def _blob(box, offset=0):
    volume = np.zeros((box, box, box))
    volume[box // 2 + offset, box // 2, box // 2] = 1.0
    return gaussian_filter(volume, 2.0)

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

# TestSimulate: class containing unit tests for src/stamp/identify/simulate.py
class TestSimulate:
    def test_simulated_map_is_centred(self, tmp_path):
        pdb = tmp_path / 'point.pdb'
        pdb.write_text('ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n')
        volume = simulate_density(pdb, box_voxels=21, voxel_size_angstrom=3.0, resolution_angstrom=12.0)
        centre = np.array(np.unravel_index(np.argmax(volume), volume.shape))
        assert np.allclose(centre, 10, atol=1)

    def test_smear_is_rotation_invariant(self):
        rng = np.random.default_rng(0)
        volume = rng.random((16, 16, 16))
        smeared = azimuthal_smear(volume, n_angles=36)
        rotated = azimuthal_smear(np.rot90(volume, k=1, axes=(0, 1)), n_angles=36)
        assert np.corrcoef(smeared.ravel(), np.rot90(rotated, k=-1, axes=(0, 1)).ravel())[0, 1] > 0.95

# TestFit: class containing unit tests for src/stamp/identify/fit.py
class TestFit:
    def test_fit_recovers_planted_match(self):
        target = _blob(21)
        match_score = fit_candidate(target, target.copy())
        mismatch_score = fit_candidate(target, np.random.default_rng(1).random((21, 21, 21)))
        assert match_score > mismatch_score
        assert match_score > 0.8

    def test_gap_widens_without_distractor(self):
        with_distractor = rank_candidates('c00', {'real': 0.9, 'near': 0.85, 'far': 0.2}, 'm')
        without = rank_candidates('c00', {'real': 0.9, 'far': 0.2}, 'm')
        assert without.score_gap_to_runner_up > with_distractor.score_gap_to_runner_up
        assert with_distractor.candidate_protein == 'real'

    def test_species_discrimination(self):
        box, voxel, resolution = 32, 4.0, 20.0

        def cylinder(height_voxels):
            centre = box // 2
            yy, xx = np.ogrid[-centre:box - centre, -centre:box - centre]
            disk = (xx * xx + yy * yy) <= 16
            mask = np.zeros((box, box, box), dtype=bool)
            z0, z1 = centre - height_voxels // 2, centre + height_voxels // 2
            mask[..., z0:z1] = disk[..., None]
            volume = np.zeros((box, box, box))
            volume[mask] = 1.0
            return volume

        tall_template = to_comparable(cylinder(20), voxel, resolution)
        short_template = to_comparable(cylinder(6), voxel, resolution)

        rng = np.random.default_rng(0)
        average = -1.0 * cylinder(20) + rng.normal(0.0, 0.2, (box, box, box))
        comparable_average = to_comparable(average, voxel, resolution)

        tall_score = fit_candidate(comparable_average, tall_template)
        short_score = fit_candidate(comparable_average, short_template)
        assert tall_score > short_score
# TestDecoyCheck: class containing unit tests for src/stamp/identify/decoy_check.py
class TestDecoyCheck:
    def test_fires_on_overlapping_distributions(self):
        result = evaluate_decoy_control([0.6, 0.55, 0.58], [0.59, 0.57, 0.56])
        assert not result.passed

    def test_passes_on_separated_distributions(self):
        result = evaluate_decoy_control([0.9, 0.88, 0.91, 0.87], [0.3, 0.25, 0.31, 0.28])
        assert result.passed
