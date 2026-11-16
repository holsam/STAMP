'''
STAMP: MRC volume reading, writing and voxel size resolution
'''

# Import external dependencies
import mrcfile, numpy as np
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from os import access, R_OK, X_OK
from pathlib import Path
from typing import Any

# Import STAMP objects
from stamp.utils.errors import StampConfigError, StampPipelineError, StampValidationError
from stamp.utils.log import log

# _mrc_error: translate an OSError/ValueError from mrcfile into a StampValidationError naming the path
def _mrc_error(action: str, path: Path, exc: Exception) -> Exception:
    if isinstance(exc, PermissionError):
        return StampValidationError(f'No permission to {action} MRC file {str(path)!r}')
    if isinstance(exc, FileNotFoundError):
        return StampValidationError(f'MRC file not found: {str(path)!r}')
    return StampValidationError(f'Could not {action} MRC file {str(path)!r}: {exc}')

# open_mrc: context manager opening an MRC with IO failures raised as StampValidationError
@contextmanager
def open_mrc(path: Path | str, *, mmap: bool = False, header_only: bool = False) -> Iterator[Any]:
    path = Path(path)
    # only wrap open so any errors raised in caller's with body aren't shown as IO errors instead
    try:
        if mmap:
            mrc = mrcfile.mmap(str(path), mode='r', permissive=True)
        else:
            mrc = mrcfile.open(str(path), mode='r', permissive=True, header_only=header_only)
    except (OSError, ValueError) as exc:
        raise _mrc_error('read', path, exc) from exc
    with mrc:
        yield mrc

# read_mrc: read an MRC file volume
def read_mrc(path: Path | str, *, dtype: type | None = None) -> np.ndarray:
    with open_mrc(path) as mrc:
        data = np.asarray(mrc.data)
    return data if dtype is None else data.astype(dtype, copy=False)

# new_mrc: write an array (and optional voxel size) to a new MRC, raising IO failures as StampValidationError
def new_mrc(path: Path | str, data: np.ndarray, *, voxel_size: Any = None) -> None:
    path = Path(path)
    try:
        with mrcfile.new(str(path), overwrite=True) as mrc:
            mrc.set_data(data)
            if voxel_size is not None:
                mrc.voxel_size = voxel_size
    except (OSError, ValueError) as exc:
        raise _mrc_error('write', path, exc) from exc

# write_class_average_mrc: write an (x, y, z) class average to MRC with axes reversed
def write_class_average_mrc(path: Path | str, average: np.ndarray, *, voxel_size: float) -> None:
    new_mrc(path, np.transpose(average, (2, 1, 0)).astype(np.float32), voxel_size=voxel_size)

# read_class_average_mrc: read a class average MRC back to (x, y, z) order
def read_class_average_mrc(path: Path | str) -> tuple[np.ndarray, float]:
    with open_mrc(path) as mrc:
        return np.transpose(np.asarray(mrc.data), (2, 1, 0)), float(mrc.voxel_size.x)

# paths_by_stem: {file stem: path} for MRC files in a directory
def paths_by_stem(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in glob_mrc(directory)}

# glob_mrc: sorted *.mrc paths in a directory, raising StampConfigError if it is missing or unreadable
def glob_mrc(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise StampConfigError(f'Not a directory: {str(directory)!r}')
    # X_OK is needed to list and open files inside a directory
    if not access(directory, R_OK | X_OK):
        raise StampConfigError(f'No permission to read directory {str(directory)!r}')
    return sorted(directory.glob('*.mrc'))

# read_voxel_size_angstrom: voxel size from an MRC header, Angstrom, or None if absent/zero/non-cubic
def read_voxel_size_angstrom(path: Path) -> float | None:
    with open_mrc(path, header_only=True) as mrc:
        voxel_size = mrc.voxel_size
    x, y, z = float(voxel_size.x), float(voxel_size.y), float(voxel_size.z)
    if x <= 0 or y <= 0 or z <= 0:
        log.debug(f'{path.name}: voxel size absent or zero in header')
        return None
    if not (abs(x - y) < 1e-3 and abs(y - z) < 1e-3):
        log.warning(f'{path.name}: non-cubic voxel size in header ({x}, {y}, {z}), using x={x}')
    return x

# resolve_directory_voxel_size_angstrom: most common voxel size across a directory's MRC headers, warning if they disagree
def resolve_directory_voxel_size_angstrom(paths: list[Path]) -> float | None:
    sizes = [size for size in (read_voxel_size_angstrom(path) for path in paths) if size is not None]
    if not sizes:
        return None
    # round to 1e-3 A so floating-point header noise doesn't split one true value into several counts
    rounded = [round(size, 3) for size in sizes]
    counts = Counter(rounded)
    most_common_value, _count = counts.most_common(1)[0]
    if len(counts) > 1:
        breakdown = ', '.join(f'{value}Å x{count}' for value, count in counts.most_common())
        log.warning(f'Multiple voxel sizes were found in MRC headers ({breakdown}) - using the most common {most_common_value}')
    return most_common_value

# require_voxel_size: if no voxel size given, resolve it from MRC headers or raise StampPipelineError if still unknown
def require_voxel_size(voxel_size_angstrom: float | None, directory_or_paths: Path | list[Path] | None) -> float:
    if voxel_size_angstrom is not None:
        return voxel_size_angstrom
    if directory_or_paths is None:
        raise StampPipelineError('No voxel size given and no tomogram directory to read it from; pass --voxel-size-a')
    if isinstance(directory_or_paths, Path):
        paths, where = glob_mrc(directory_or_paths), str(directory_or_paths)
    else:
        paths, where = directory_or_paths, f'{len(directory_or_paths)} tomogram(s)'
    resolved = resolve_directory_voxel_size_angstrom(paths)
    if resolved is None:
        raise StampPipelineError(f'No voxel size in any MRC header in {where} and --voxel-size-a not given')
    log.debug(f'Resolved voxel size from MRC headers: {resolved} Å')
    return resolved
