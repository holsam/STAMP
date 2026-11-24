'''
STAMP: unit tests for diagram's mocked pipeline
'''

# Import external dependencies
import numpy as np, pytest
from dataclasses import replace
from pathlib import Path
from scipy.ndimage import gaussian_filter, rotate
from types import SimpleNamespace

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior
from stamp.tools.diagram import classify, identify, overview, pick, stream
from stamp.tools.diagram.utils.align import align_stack, align_to_references, bin_stack
from stamp.tools.diagram.utils.classify import class_label
from stamp.tools.diagram.utils.carriers import make_carriers, nearest_surface, place_on
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.decoy import random_volume_decoys, shifted_decoys
from stamp.tools.diagram.utils.mixed import MIXED_CLASS, build_shared_scene
from stamp.tools.diagram.utils.pick import infer_orientations, reconcile
from stamp.tools.diagram.utils.polish import PolishConfig, Stacks, run_polish
from stamp.tools.diagram.utils.register import register_score
from stamp.tools.diagram.utils.search import search_space, wedge_keep

# _write_pdb: a small asymmetric cloud of atoms, enough for the diagram to project
def _write_pdb(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    atoms = np.vstack([rng.normal([0, 0, 0], [18, 8, 8], (300, 3)), rng.normal([25, 20, 0], [7, 16, 8], (200, 3))])
    path.write_text(''.join(f'ATOM  {i:5d}  CA  ALA A{i % 999 + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n' for i, (x, y, z) in enumerate(atoms, 1)))
    return path

# _blob: an asymmetric 2D test object, a bar with a knob, so every rotation is distinguishable
def _blob(size: int, knob: tuple[int, int]) -> np.ndarray:
    image = np.zeros((size, size))
    image[size // 2 - 2:size // 2 + 2, size // 4:3 * size // 4] = 1.0
    image[knob] = 2.0
    return gaussian_filter(image, 1.2)

# _ring: a second, clearly different object, a ring with a short arm
def _ring(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.hypot(yy - size / 2, xx - size / 2)
    image = np.exp(-((radius - size * 0.25) ** 2) / 4.0)
    image[size // 2 - 1:size // 2 + 1, size // 2 + 8:size - 4] = 1.0
    return gaussian_filter(image, 1.0)

# _config: a small diagram config that renders in a few seconds
def _config(tmp_path: Path, prior: OrientationPrior) -> DiagramConfig:
    return DiagramConfig(target=_write_pdb(tmp_path / 'a.pdb', 1), distractor=_write_pdb(tmp_path / 'b.pdb', 2), outdir=tmp_path / 'out', orientation_prior=prior, box_size=32, field_px=500, n_vesicles=8, n_filaments=12,
                         n_target=150, n_distractor=120, n_extra=60, n_synthetic_unknowns=1, min_cluster_size=8)

# TestCarriers: class containing tests for utils/carriers.py
class TestCarriers:
    # check each prior gets its own carrier kind and none gets no carrier
    def test_carrier_kind_per_prior(self) -> None:
        rng = np.random.default_rng(0)
        kinds = {prior: {c['kind'] for c in make_carriers(prior, 600, 6, 6, rng)} for prior in OrientationPrior}
        assert kinds == {OrientationPrior.NORMAL: {'circle'}, OrientationPrior.AXIS: {'line'}, OrientationPrior.NONE: set()}

    # check a particle placed on a vesicle sits near its surface and the nearest-surface angle is the outward normal
    def test_place_and_normal_angle(self) -> None:
        vesicle = {'kind': 'circle', 'cx': 200.0, 'cy': 200.0, 'r': 80.0}
        x, y, angle = place_on(vesicle, np.random.default_rng(1))
        distance, normal, owner = nearest_surface([vesicle], np.array([[x, y]]))
        assert abs(np.hypot(x - 200, y - 200) - 80) < 8 and distance[0] < 8 and owner[0] == 0
        assert np.cos(normal[0] - angle) > 0.99

# TestPick: class containing tests for utils/pick.py and utils/decoy.py
class TestPick:
    # check normal-prior picks beyond the surface distance keep no orientation, the other priors keep every pick
    def test_far_picks_are_unoriented_under_normal(self) -> None:
        vesicle = [{'kind': 'circle', 'cx': 200.0, 'cy': 200.0, 'r': 80.0}]
        points = np.array([[280.0, 200.0], [200.0, 400.0]])
        _angle, oriented = infer_orientations(OrientationPrior.NORMAL, vesicle, points, np.random.default_rng(0), 24.0)
        assert oriented.tolist() == [True, False]
        assert infer_orientations(OrientationPrior.NONE, [], points, np.random.default_rng(0), 24.0)[1].all()

    # check only groups both files contribute to are kept, and the members come back
    def test_reconcile_is_an_intersection(self) -> None:
        a = np.array([[0.0, 0.0], [100.0, 100.0]])
        b = np.array([[2.0, 1.0], [300.0, 300.0]])
        kept, rejected, members = reconcile(a, b, distance_px=10.0)
        assert len(kept) == 1 and len(rejected) == 2 and sorted(members[0].tolist()) == [0, 2]

    # check decoys stay clear of every real pick and inside the border
    def test_decoys_are_clear_of_picks(self) -> None:
        rng = np.random.default_rng(0)
        picks = rng.uniform(100, 400, size=(20, 2))
        for decoys in (shifted_decoys(picks, [], picks, 500, 32, rng), random_volume_decoys(20, picks, 500, 32, rng)):
            assert len(decoys) > 0
            assert np.linalg.norm(decoys[:, None] - picks[None], axis=2).min() > 0.8 * 32
            assert decoys.min() > 32 and decoys.max() < 500 - 32

    # check the synthetic-noise decoy panel exists for the normal prior only, as decoy/generate.SUPPORTED_PRIORS allows
    def test_noise_decoy_panel_is_normal_only(self) -> None:
        names = {prior: [name for name, _fn in pick._panels(SimpleNamespace(w=SimpleNamespace(prior=prior)))] for prior in OrientationPrior}
        assert 'noise_decoys' in names[OrientationPrior.NORMAL]
        assert all('noise_decoys' not in names[prior] for prior in (OrientationPrior.AXIS, OrientationPrior.NONE))

# TestClassify: class containing tests for utils/classify.py
class TestClassify:
    # check class labels carry the prior prefix, as classify writes them
    def test_class_label_is_prior_prefixed(self) -> None:
        assert class_label(OrientationPrior.AXIS, 3) == 'axis-c03'

# TestSearch: class containing tests for utils/search.py
class TestSearch:
    # check the search space opens what each prior leaves unknown
    def test_space_per_prior(self) -> None:
        normal, axis, none = (search_space(prior) for prior in (OrientationPrior.NORMAL, OrientationPrior.AXIS, OrientationPrior.NONE))
        assert len(normal.angles) == 36 and not normal.axial_only
        assert axis.angles.tolist() == [0.0, 180.0] and axis.axial_only
        assert len(none.angles) == 72

    # check the wedge keeps the mean and removes a fixed fraction of directions that rotates with the frame
    def test_wedge(self) -> None:
        keep = wedge_keep(32, 60.0)
        assert keep[0, 0] == 1.0 and 0.5 < keep.mean() < 0.9
        assert not np.array_equal(keep, wedge_keep(32, 60.0, 45.0))

# TestAlign: class containing tests for utils/align.py
class TestAlign:
    # check alignment recovers a known rotation and shift and the aligned particles match the reference
    def test_recovers_pose(self) -> None:
        reference = _blob(32, (9, 20))
        truth = [(30.0, (2, -3)), (120.0, (-1, 1)), (250.0, (0, 2))]
        particles = np.array([np.roll(rotate(reference, angle, reshape=False, order=1), shift, (0, 1)) for angle, shift in truth])
        masks = np.ones_like(particles, dtype=np.float32)
        alignment = align_to_references(particles, masks, reference[None], search_space(OrientationPrior.NORMAL))
        assert alignment.angle[:, 0].tolist() == [30.0, 120.0, 250.0] and alignment.shift[:, 0].tolist() == [[2, -3], [-1, 1], [0, 2]]
        aligned, _ = align_stack(particles, masks, alignment.angle[:, 0], alignment.shift[:, 0])
        assert all(np.corrcoef(image.ravel(), reference.ravel())[0, 1] > 0.99 for image in aligned)

# TestPolish: class containing tests for utils/polish.py
class TestPolish:
    # check polish separates two species from seeds that mix them and rejects pure noise
    def test_separates_species_and_rejects_noise(self) -> None:
        rng = np.random.default_rng(0)
        size = 32
        species = [_blob(size, (9, 20)), _ring(size)]
        images, truth = [], []
        for _ in range(30):
            for label, image in enumerate(species):
                images.append(rotate(image, rng.uniform(0, 360), reshape=False, order=1) + rng.normal(0, 0.25, image.shape))
                truth.append(label)
        for _ in range(10):
            images.append(rng.normal(0, 0.25, (size, size)))
            truth.append(-1)
        particles, truth = np.array(images), np.array(truth)
        stacks = Stacks(particles, np.ones_like(particles, dtype=np.float32), bin_stack(particles, 2), np.ones((len(particles), 16, 16), dtype=np.float32), 2)
        seeds = np.where(truth == -1, 0, truth)    # noise is seeded into class 0, polish has to push it out
        seeds[::3] = 1 - seeds[::3]                # and a third of the seeds are wrong
        result = run_polish(stacks, rng.normal(size=(len(particles), 2)), seeds, search_space(OrientationPrior.NORMAL), PolishConfig(min_seed_size=10))
        assert len(result.references) == 2
        assert (result.labels[truth == -1] == -1).mean() > 0.8
        for label in (0, 1):
            found = result.labels[truth == label]
            assert max(np.bincount(found[found >= 0])) / len(found) > 0.9

# TestRegister: class containing tests for utils/register.py
class TestRegister:
    # check the right candidate scores higher at any rotation, in either contrast
    def test_right_candidate_wins(self) -> None:
        target, other = _blob(32, (9, 20)), _ring(32)
        for angle, sign in ((70.0, 1.0), (200.0, -1.0)):
            average = sign * rotate(target, angle, reshape=False, order=1)
            assert register_score(average, target, OrientationPrior.NORMAL).score > register_score(average, other, OrientationPrior.NORMAL).score + 0.1

# TestMixed: class containing tests for utils/mixed.py
class TestMixed:
    # check the shared scene has carriers and particles for each prior, one species per prior
    def test_shared_scene_holds_every_prior(self, tmp_path: Path) -> None:
        scene = build_shared_scene(replace(_config(tmp_path, OrientationPrior.NORMAL), field_px=700))
        assert scene.carriers[OrientationPrior.NORMAL] and scene.carriers[OrientationPrior.AXIS] and scene.carriers[OrientationPrior.NONE] == []
        assert {p['class'] for p in scene.particles} == set(MIXED_CLASS.values())

    # check a mixed render writes one directory per prior group and the summary figure
    def test_mixed_pick_render(self, tmp_path: Path) -> None:
        outdir = pick.render_mixed_priors(_config(tmp_path, OrientationPrior.NORMAL))
        assert (outdir / '00_pick_mixed.png').stat().st_size > 0
        assert all((outdir / prior.value / '00_pick_storyboard.png').exists() for prior in OrientationPrior)

# TestStream: class containing tests for stream.py
class TestStream:
    # check the stream diagram appends each tomogram and gives whole tomograms to one half
    def test_stream_render(self, tmp_path: Path) -> None:
        run = stream.StreamRun(replace(_config(tmp_path, OrientationPrior.NORMAL), n_tomograms=3))
        assert len(run.tomograms) == 3 and len(run.real_half) == 3
        outdir = stream.render(replace(_config(tmp_path, OrientationPrior.NORMAL), n_tomograms=3))
        assert (outdir / '00_stream_storyboard.png').stat().st_size > 0

# TestDiagram: class containing smoke tests for the diagram renderers
class TestDiagram:
    # check each renderer writes its storyboard for every prior
    @pytest.mark.parametrize('prior', list(OrientationPrior))
    @pytest.mark.parametrize('module, storyboard', [(pick, '00_pick_storyboard.png'), (classify, '00_classify_storyboard.png'), (identify, '00_identify_storyboard.png'), (overview, '00_storyboard.png')])
    def test_renders_storyboard(self, tmp_path: Path, module, storyboard: str, prior: OrientationPrior) -> None:
        outdir = module.render(_config(tmp_path, prior))
        assert (outdir / storyboard).stat().st_size > 0

    # check --all-priors renders one subdirectory per prior and the comparison figure
    def test_all_priors_comparison(self, tmp_path: Path) -> None:
        outdir = pick.render_all(_config(tmp_path, OrientationPrior.NORMAL))
        assert (outdir / '00_pick_priors.png').exists()
        assert all((outdir / prior.value / '00_pick_storyboard.png').exists() for prior in OrientationPrior)

    # check a particle count too small for any seed cluster fails with a message that names the fix
    def test_too_few_particles_explains_itself(self, tmp_path: Path) -> None:
        config = replace(_config(tmp_path, OrientationPrior.NORMAL), n_target=6, n_distractor=6, n_extra=4)
        with pytest.raises(ValueError, match='raise|few|min'):
            classify.render(config)
