'''
STAMP: input/output utilities
'''

# Import external dependencies
import shutil, tarfile
from pathlib import Path

# Import STAMP objects
from stamp.utils.log import log

# cleanup_raw: unless keep_raw, tar+gzip '<output_dir>/raw' to 'raw.tar.gz' next to it, then delete it and log under label
def cleanup_raw(output_dir: Path, *, keep_raw: bool, label: str) -> Path | None:
    if keep_raw:
        return None
    directory = output_dir / 'raw'
    if not directory.is_dir():
        return None
    archive_path = directory.with_suffix(directory.suffix + '.tar.gz')
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(directory, arcname=directory.name)
    shutil.rmtree(directory)
    log.info(f'Archived {label} to {archive_path}')
    return archive_path

# resolve_output_dir: append stamp/<command> to supplied output dir
def resolve_output_dir(output_dir: Path, command: str, track: str | None = None) -> Path:
    base = output_dir / 'stamp' / command
    return Path(f'{base}_{track}') if track else base
