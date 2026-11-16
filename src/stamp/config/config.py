'''
STAMP: manage a stamp_run.toml file
'''

# -- Import external dependencies
import os, subprocess
from pathlib import Path

# Import internal stamp objects
from stamp.config.template import TEMPLATE
from stamp.utils.errors import StampError
from stamp.utils.log import log
from stamp.utils.toml_view import print_toml

# _resolve_config_path: resolve a given path to a stamp_run.toml and whether it exists
def _resolve_config_path(path: Path) -> tuple[Path, bool]:
    path = path.resolve()
    if path.match('stamp_run.toml'):
        return (path, path.exists())
    elif path.match('stamp'):
        path = path / 'stamp_run.toml'
    else:
        path = path / 'stamp' / 'stamp_run.toml'   
    return (path, path.exists())

# init_config: write a blank annotated stamp_run.toml to the target directory
def init_config(path: Path, force: bool = False):
    path, path_exists = _resolve_config_path(path)
    if path_exists and not force:
        raise StampError(f'{path} already exists, use --force to overwrite')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE)
    log.info(f'Wrote stamp_run.toml to {path}')

# edit_config: edit a stamp_run.toml file
def edit_config(path: Path, editor: str | None):
    path, path_exists = _resolve_config_path(path)
    if not path_exists:
        init_config(path=path)
    if editor is not None:
        from shutil import which
        editor_path = which(editor)
        if editor_path:
            try:
                subprocess.call([editor_path, str(path)])
                return
            except Exception as e:
                raise StampError(f'Could not open {path} in editor {editor}: {e}')
        else:
            raise StampError(f'editor {editor!r} not found on PATH')
    else:
        editor = os.environ.get('EDITOR', 'vi')
        subprocess.call([editor, str(path)])

# show_config: print a stamp_run.toml file
def show_config(path: Path):
    path, path_exists = _resolve_config_path(path)
    if not path_exists:
        raise StampError(f'No config file found at/under {path}, run stamp config init first.')
    print_toml(path)
