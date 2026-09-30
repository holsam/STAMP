'''
STAMP: integration tests for `stamp identify`
'''

# Import external dependencies
import json, mrcfile, numpy as np
from scipy.ndimage import gaussian_filter
from typer.testing import CliRunner
from stamp.cli.cli import stamp_app

# Initialise runner
runner = CliRunner()

# _write_average: a smeared blob class average with its voxel size set
def _write_average(path, box, spread, voxel_size=3.0):
    volume = np.zeros((box, box, box))
    volume[box // 2, box // 2, box // 2] = 1.0
    volume = gaussian_filter(volume, spread)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.transpose(volume, (2, 1, 0)).astype(np.float32))
        mrc.voxel_size = voxel_size

# _write_pdb: a small cluster of CA atoms, `n` of them along x
def _write_pdb(path, n):
    rows = [
        f'ATOM  {i:>5}  CA  ALA A{i:>4}    {i * 2.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C'
        for i in range(1, n + 1)
    ]
    path.write_text('\n'.join(rows) + '\n')

# TestIdentifyCommand: class containing tests for test_decoy_command.py
class TestIdentifyCommand:
    def test_identify_end_to_end(self, tmp_path):
        classes = tmp_path / 'classes'
        decoys = tmp_path / 'decoys'
        classes.mkdir()
        decoys.mkdir()
        _write_average(classes / 'c00_halfA_n40.mrc', 21, 2.0)
        _write_average(classes / 'c01_halfA_n30.mrc', 21, 5.0)
        _write_average(decoys / 'c00_halfA_n40.mrc', 21, 9.0)

        _write_pdb(tmp_path / 'small.pdb', 6)
        _write_pdb(tmp_path / 'large.pdb', 60)
        (tmp_path / 'candidates.yaml').write_text(
            'candidates:\n'
            '  - name: small\n    structure_path: small.pdb\n'
            '  - name: large\n    structure_path: large.pdb\n'
        )

        result = runner.invoke(stamp_app, [
            '-d', str(tmp_path),
            'identify',
            '--classes', str(classes),
            '--candidates', str(tmp_path / 'candidates.yaml'),
            '--decoy-classes', str(decoys),
            '--output-dir', tmp_path,
            '--resolution', '30.0',
        ])
        assert result.exit_code == 0, result.output

        out = tmp_path / 'stamp' / 'identify'
        identifications = json.loads((out / 'identification.json').read_text())
        assert {row['cluster_id'] for row in identifications} == {'c00', 'c01'}
        assert all(row['score_gap_to_runner_up'] >= 0.0 for row in identifications)

        report = (out / 'identification_report.txt').read_text()
        assert any(line.startswith('DECOY CONTROL:') for line in report.splitlines())
        assert (out / 'decoy_control.json').is_file()
        assert (out / 'params.toml').is_file()

    def test_identify_resumes_after_interruption(self, tmp_path, monkeypatch):
        from stamp.commands import identify
        classes = tmp_path / 'classes'
        classes.mkdir()
        _write_average(classes / 'c00_halfA_n40.mrc', 21, 2.0)
        _write_average(classes / 'c01_halfA_n30.mrc', 21, 5.0)
        _write_pdb(tmp_path / 'small.pdb', 6)
        _write_pdb(tmp_path / 'large.pdb', 60)
        (tmp_path / 'candidates.yaml').write_text(
            'candidates:\n'
            '  - name: small\n    structure_path: small.pdb\n'
            '  - name: large\n    structure_path: large.pdb\n'
        )
        real = identify._fit_one
        fitted: list[tuple[str, str]] = []
        state = {'fail': True}

        # flaky: fail on the third fit until told otherwise, else fit normally
        def flaky(job):
            if state['fail'] and len(fitted) == 2:
                raise RuntimeError('boom')
            fitted.append((job.class_id, job.candidate.name))
            return real(job)

        monkeypatch.setattr(identify, '_fit_one', flaky)
        args = [
            '-d', str(tmp_path),
            'identify',
            '--classes', str(classes),
            '--candidates', str(tmp_path / 'candidates.yaml'),
            '--output-dir', str(tmp_path),
            '--resolution', '30.0',
            '-n', '1',
        ]
        assert runner.invoke(stamp_app, args).exit_code != 0
        finished = list(fitted)
        assert len(finished) == 2

        state['fail'] = False
        fitted.clear()
        result = runner.invoke(stamp_app, args)
        assert result.exit_code == 0, result.output
        assert len(fitted) == 2 and not set(fitted) & set(finished)
        out = tmp_path / 'stamp' / 'identify'
        assert {row['cluster_id'] for row in json.loads((out / 'identification.json').read_text())} == {'c00', 'c01'}
        assert not (out / 'raw').exists()
