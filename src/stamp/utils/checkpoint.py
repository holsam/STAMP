'''
STAMP: fingerprint-gated working directories for resumable commands
'''

# Import external dependencies
import hashlib, json, numpy as np, os, shutil
from pathlib import Path

# Import internal STAMP objects
from stamp.utils.log import log

# _CHECKPOINT_NAME: fingerprint file stored inside the working directory
_CHECKPOINT_NAME = 'checkpoint.json'

# file_signature: identity of an input file, content hash when content is True else size and mtime
def file_signature(path: Path, *, content: bool = False) -> str:
    if content:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return f'{stat.st_size}:{stat.st_mtime_ns}'

# atomic_save_npz: write arrays via a temporary file so a partial file never looks complete
def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'wb') as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)

# atomic_write_json: write a json file via a temporary file so a partial file never looks complete
def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload))
    os.replace(temporary, path)

# sync_checkpoint: keep work_dir when its stored fingerprint matches exactly, else wipe it; returns True when resuming
def sync_checkpoint(work_dir: Path, fingerprint: dict) -> bool:
    path = work_dir / _CHECKPOINT_NAME
    try:
        stored = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        stored = None
    if stored == fingerprint:
        return True
    if work_dir.exists() and any(work_dir.iterdir()):
        log.warning(f'{work_dir} was produced by a different command or inputs, starting fresh')
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))
    return False
