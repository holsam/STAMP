'''
STAMP: input/output utilities
'''

# Import external dependencies
import hashlib, json, mrcfile, shutil, subprocess, tarfile, tomli_w, tomllib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from rich.console import Console
from rich.text import Text
from rich.tree import Tree
from typing import Any

# Import STAMP objects
from stamp.schemas.provenance import ProvenanceSidecar

# _CACHE_NAME: per-directory checksum cache keyed on (path, size, mtime)
_CACHE_NAME = '.stamp_checksums.json'
_CHUNK = 1024 * 1024

# -- _resolve_abspath: returns a Path for the absolute path for a given path
def _resolve_abspath(path: Path):
    return path.expanduser().resolve()

# -- _is_writable: returns bool indicating if supplied directory is writable
def _is_writable(directory: Path):
    from os import access, W_OK
    directory = _resolve_abspath(directory)
    return access(directory, W_OK)

# toml_none_to_empty: map any None instances to an empty string for TOML serialisation
def toml_none_to_empty(obj):
    if isinstance(obj, dict):
        return {k: toml_none_to_empty(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [toml_none_to_empty(v) for v in obj]
    return '' if obj is None else obj

# resolve_stamp_commit: git HEAD (+ '-dirty'), else installed package version, else 'unknown'
def resolve_stamp_commit() -> str:
    source_tree = Path(__file__).resolve().parent.parent.parent.parent
    try:
        head = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=source_tree,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=source_tree,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f'{head}-dirty' if dirty else head
    except (subprocess.CalledProcessError, FileNotFoundError):
        from stamp.utils.log import log
        log.debug('git rev-parse failed, falling back to installed package version')
    try:
        return f'stamp-{version("stamp")}'
    except PackageNotFoundError:
        return 'unknown'

# _load_cache / _save_cache: the optional per-directory checksum cache
def _load_cache(cache_dir: Path | None) -> dict:
    if cache_dir is None:
        return {}
    cache_path = cache_dir / _CACHE_NAME
    if cache_path.is_file():
        try:
            return json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            from stamp.utils.log import log
            log.debug(f'{cache_path} is not valid JSON, treating checksum cache as empty')
            return {}
    return {}

def _save_cache(cache_dir: Path | None, cache: dict) -> None:
    if cache_dir is not None:
        (cache_dir / _CACHE_NAME).write_text(json.dumps(cache, indent=2))

# checksum_file: SHA-256 of contents, cached on (path, size, mtime) when cache_dir is given
def checksum_file(path: Path, cache_dir: Path | None = None, *, _cache: dict | None = None) -> str:
    from stamp.utils.log import log
    owns_cache = _cache is None
    cache = _load_cache(cache_dir) if owns_cache else _cache
    stat = path.stat()
    key = str(path.resolve())
    entry = cache.get(key)
    if entry and entry['size'] == stat.st_size and entry['mtime'] == stat.st_mtime:
        log.debug(f'{path.name}: checksum cache hit')
        return entry['sha256']
    log.debug(f'{path.name}: checksum cache miss, hashing')

    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b''):
            digest.update(chunk)
    hexdigest = digest.hexdigest()

    cache[key] = {'size': stat.st_size, 'mtime': stat.st_mtime, 'sha256': hexdigest}
    if owns_cache:
        _save_cache(cache_dir, cache)
    return hexdigest

# _digest_directory: digest of a directory's sorted (relative path, sha256) pairs, recursively
def _digest_directory(directory: Path, cache_dir: Path | None, cache: dict) -> str:
    parts = [
        f'{child.relative_to(directory).as_posix()}:{checksum_file(child, cache_dir, _cache=cache)}'
        for child in sorted(directory.rglob('*'))
        if child.is_file() and child.name != _CACHE_NAME
    ]
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()

# checksum_inputs: role -> hash; a directory role gets the recursive digest; a missing input is recorded as 'absent'
def checksum_inputs(inputs: list[tuple[str, Path]], cache_dir: Path | None = None) -> dict[str, str]:
    cache = _load_cache(cache_dir)
    checksums: dict[str, str] = {}
    for role, path in inputs:
        path = Path(path)
        if not path.exists():
            checksums[role] = 'absent'
            continue
        checksums[role] = (_digest_directory(path, cache_dir, cache) if path.is_dir() else checksum_file(path, cache_dir, _cache=cache))
    _save_cache(cache_dir, cache)
    return checksums

# write_sidecar: build a ProvenanceSidecar, resolve the commit, hash inputs, write params.toml
def write_sidecar(
    output_dir: Path,
    stage: str,
    tool: str,
    tool_version: str | None,
    parameters: dict,
    inputs: list[tuple[str, Path]],
    *,
    cache_inputs: bool = True
) -> Path:
    from stamp.utils.log import log
    log.debug(f'Writing sidecar for stage={stage!r} tool={tool!r} to {output_dir / "params.toml"}')
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir if cache_inputs else None
    sidecar = ProvenanceSidecar(
        stage=stage,
        tool=tool,
        tool_version=tool_version,
        parameters=parameters,
        stamp_commit=resolve_stamp_commit(),
        timestamp=datetime.now(timezone.utc),
        input_checksums=checksum_inputs(inputs, cache_dir),
    )
    path = output_dir / 'params.toml'
    path.write_text(tomli_w.dumps(toml_none_to_empty(sidecar.model_dump(mode='json'))))
    return path

# match_by_stem: pair segmentation files to raw tomogram files by longest common stem prefix
def match_by_stem(
    segmentation_paths: list[Path],
    raw_paths: list[Path]
) -> tuple[dict[Path, Path], list[Path]]:
    raw_by_stem = {path.stem: path for path in raw_paths}
    matched: dict[Path, Path] = {}
    unmatched: list[Path] = []
    for segmentation_path in segmentation_paths:
        stem = segmentation_path.stem
        raw_path = raw_by_stem.get(stem)
        if raw_path is None:
            candidates = [raw_stem for raw_stem in raw_by_stem if stem.startswith(raw_stem)]
            if candidates:
                raw_path = raw_by_stem[max(candidates, key=len)]
        if raw_path is None:
            unmatched.append(segmentation_path)
        else:
            matched[segmentation_path] = raw_path
    return matched, unmatched

# read_voxel_size_angstrom: voxel size from an MRC header, Angstrom, or None if absent/zero/non-cubic
def read_voxel_size_angstrom(path: Path) -> float | None:
    from stamp.utils.log import log
    with mrcfile.open(str(path), header_only=True, permissive=True) as mrc:
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
    from collections import Counter
    from stamp.utils.log import log
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

# archive_and_remove_directory: tar+gzip a directory to '<directory>.tar.gz' next to it, then delete the directory
def archive_and_remove_directory(directory: Path) -> Path:
    archive_path = directory.with_suffix(directory.suffix + '.tar.gz')
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(directory, arcname=directory.name)
    shutil.rmtree(directory)
    return archive_path

# resolve_output_dir: append stamp/<command> to supplied output dir
def resolve_output_dir(output_dir: Path, command: str, track: str | None = None) -> Path:
    base = output_dir / 'stamp' / command
    return Path(f'{base}_{track}') if track else base

# -- build_toml_tree: returns a Tree instance from a parsed TOML file 
def build_toml_tree(data: dict[str, Any], filename: str) -> Tree:
    tree = Tree(Text(filename, style='bold'))

    # -- format_value: returns Text instance containing stylised value
    def format_value(value: Any) -> Text:
        text = Text()
        if value in [None, '', list]:
            text.append("(empty)", style="dim italic")
        else:
            text.append(repr(value), style="cyan")
        return text

    # -- add_line: returns None, but adds a given line to the Tree
    def add_line(parent: Tree, key: str, value: Any) -> None:
        line = Text()
        line.append(key, style='yellow')
        line.append(': ')
        line.append_text(format_value(value))
        parent.add(line)

    # -- add_value: returns None, but adds a value to the Tree
    def add_value(parent: Tree, key: str, value: Any) -> None:
        # Empty nested table
        if isinstance(value, dict):
            if not value:
                add_line(parent, key, None)
                return
            branch = parent.add(Text(key, style='yellow'))
            for child_key, child_value in value.items():
                add_value(branch, child_key, child_value)
            return
        # List / array
        if isinstance(value, list):
            add_list(parent, key, value)
            return
        # Scalar value
        add_line(parent, key, value)

    # -- add_list: returns None, but renders a TOML array/array of tables to Tree
    def add_list(parent: Tree, key: str, values: list[Any]) -> None:
        # Array of tables ([[table]])
        is_array_of_tables = bool(values) and all(isinstance(item, dict) for item in values)
        if is_array_of_tables:
            branch = parent.add(Text(key, style='yellow'))
            for item in values:
                item_branch = branch.add(Text("[item]", style='dim'))
                if not item:
                    item_branch.add(Text('(empty)', style='dim italic'))
                    continue
                for child_key, child_value in item.items():
                    add_value(item_branch, child_key, child_value)
            return
        # Ordinary TOML array
        line = Text()
        line.append(key, style='yellow')
        line.append(": ", style='white')
        if values:
            line.append(repr(values), style='cyan')
        else:
            line.append('(empty)', style='dim italic')
        parent.add(line)

    for key, value in data.items():
        add_value(tree, key, value)
    return tree

# -- print_toml: print parsed TOML data as a tree
def print_toml(path: Path) -> None:
    with path.open('rb') as f:
        data = tomllib.load(f)
    console = Console()
    console.print()
    console.print(build_toml_tree(data, path.name))
    console.print()
