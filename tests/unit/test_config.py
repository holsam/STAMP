'''
STAMP: unit tests for config management functions
'''

# Import external dependencies
import pytest
from pathlib import Path

# Import internal STAMP objects
from stamp.config.config import _resolve_config_path, edit_config, init_config, show_config
from stamp.config.template import TEMPLATE
from stamp.utils.errors import StampError

# TestResolveConfigPath: tests for _resolve_config_path
class TestConfig:
    # check a directory named `stamp` resolves to `<dir>/stamp_run.toml`
    def test_resolves_stamp_directory(self, tmp_path: Path) -> None:
        stamp_dir = tmp_path / 'stamp'
        stamp_dir.mkdir()
        path, exists = _resolve_config_path(stamp_dir)
        assert path == stamp_dir / 'stamp_run.toml'
        assert exists is False
    # check an arbitrary directory resolves to `<dir>/stamp/stamp_run.toml`
    def test_resolves_arbitrary_directory(self, tmp_path: Path) -> None:
        path, exists = _resolve_config_path(tmp_path)
        assert path == tmp_path / 'stamp' / 'stamp_run.toml'
        assert exists is False
    # check a direct stamp_run.toml path resolves to itself
    def test_resolves_direct_file_path(self, tmp_path: Path) -> None:
        target = tmp_path / 'stamp_run.toml'
        target.write_text(TEMPLATE)
        path, exists = _resolve_config_path(target)
        assert path == target
        assert exists is True

    # check init_config writes the template to a fresh path
    def test_writes_template(self, tmp_path: Path) -> None:
        init_config(tmp_path)
        assert (tmp_path / 'stamp' / 'stamp_run.toml').read_text() == TEMPLATE
    # check init_config refuses to overwrite without --force
    def test_refuses_overwrite_without_force(self, tmp_path: Path) -> None:
        init_config(tmp_path)
        with pytest.raises(StampError):
            init_config(tmp_path)
    # check init_config overwrites with force=True
    def test_overwrites_with_force(self, tmp_path: Path) -> None:
        init_config(tmp_path)
        config_path = tmp_path / 'stamp' / 'stamp_run.toml'
        config_path.write_text('mutated')
        init_config(tmp_path, force=True)
        assert config_path.read_text() == TEMPLATE

    # check show_config raises when no config file exists
    def test_raises_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(StampError):
            show_config(tmp_path)
    # check show_config succeeds when a config file exists
    def test_succeeds_when_present(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        init_config(tmp_path)
        show_config(tmp_path)
        assert capsys.readouterr().out != ''

    # check edit_config creates a config file first if none exists, then invokes the editor
    def test_creates_then_edits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr('stamp.config.config.subprocess.call', lambda args: calls.append(args))
        edit_config(tmp_path, editor='true')
        assert (tmp_path / 'stamp' / 'stamp_run.toml').exists()
        assert calls and calls[0][0].endswith('true')
