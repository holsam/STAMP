'''
STAMP: test suite shared fixtures
'''

# Import external dependencies
import json, mrcfile, numpy as np, pytest
from pathlib import Path
from typer.testing import CliRunner
from stamp.cli.cli import stamp

# Initialise CLI runner
_runner = CliRunner()

# _make_dataset: 3 segmentation/tomogram pairs with a membrane slab + planted blobs, plus candidates.yaml
def _make_dataset(root: Path) -> Path:
    seg, tomo = root / 'seg', root / 'tomo'
    seg.mkdir(parents=True)
    tomo.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(12):
        shape = (60, 60, 60)
        segmentation = np.zeros(shape, dtype=np.float32)
        segmentation[:, :, 30] = 1.0  # flat membrane slab
        volume = rng.normal(0, 1, shape).astype(np.float32)
        # several large and small blobs, spread out, so each picker/cluster has enough
        # particles to survive the half-A/half-B split during refine
        # keep blobs clear of the membrane plane itself (index 30), within the picker's
        # 20-80 Angstrom (~1.5-6 voxel) offset window on one side of the surface
        # picker default is density_sign=-1 (dark-protein contrast), so blobs are dips
        for x, y in ((13, 13), (13, 35), (35, 13), (35, 35)):
            volume[x:x + 5, y:y + 5, 33:38] -= 6.0   # large blob
        for x, y in ((18, 28), (28, 18), (28, 38), (38, 28)):
            volume[x:x + 3, y:y + 3, 33:35] -= 4.0   # small blob
        for directory, data in ((seg, segmentation), (tomo, volume)):
            with mrcfile.new(directory / f't{index:02d}.mrc', overwrite=True) as mrc:
                mrc.set_data(data)
                mrc.voxel_size = 13.48
    (root / 'small.pdb').write_text('ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n')
    (root / 'large.pdb').write_text(
        '\n'.join(
            f'ATOM  {i:>5}  CA  ALA A{i:>4}    {i * 2.0:8.3f}   0.000   0.000  1.00  0.00           C'
            for i in range(1, 40)
        ) + '\n'
    )
    (root / 'candidates.yaml').write_text(
        'candidates:\n'
        '  - name: small\n    structure_path: small.pdb\n'
        '  - name: large\n    structure_path: large.pdb\n'
    )
    return root

# _write_config: an in-tmp stamp_run.toml pointing at the generated data
def _write_config(root: Path, **overrides) -> Path:
    body = f'''
[run]
segmentation_dir = "seg"
raw_tomogram_dir = "tomo"
output_dir = "out"
voxel_size_angstrom = 13.48
backend = "mock"
{overrides.get('run_extra', '')}

[decoy]
enabled = {str(overrides.get('decoy_enabled', True)).lower()}
method = "rejected-surface"

[stage.pick]
backend = "local"

[stage.classify]
method = "kmeans"
n_clusters = 2
n_components = 3
box_angstrom = 300

[stage.identify]
candidates = "candidates.yaml"

[stage.refine]
tool = "relion"
class_id = "all"
iterations = 2
'''
    path = root / 'stamp_run.toml'
    path.write_text(body)
    return path

# _run_mock_pipeline: generate data, write config, invoke `stamp run`, return the output dir
def _run_mock_pipeline(root: Path, args: list[str] | None = None) -> Path:
    _make_dataset(root)
    config = _write_config(root)
    result = _runner.invoke(stamp, ['run', '--config', str(config), *(args or [])])
    assert result.exit_code == 0, result.output
    return root / 'out'

# make_dataset: fixture to call _make_dataset
@pytest.fixture
def make_dataset():
    return _make_dataset

# write_config: fixture to call _write_config
@pytest.fixture
def write_config():
    return _write_config

# run_mock_pipeline: fixture to call _run_mock_pipeline
@pytest.fixture
def run_mock_pipeline():
    return _run_mock_pipeline
