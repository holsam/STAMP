'''
STAMP: input/output utilities
'''

# Import external dependencies
import hashlib, json, subprocess, tomli_w
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Import STAMP objects
from stamp.schemas.provenance import ProvenanceSidecar

# _CACHE_NAME: per-directory checksum cache keyed on (path, size, mtime)
_CACHE_NAME = '.stamp_checksums.json'
_CHUNK = 1024 * 1024


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
        pass
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
            return {}
    return {}


def _save_cache(cache_dir: Path | None, cache: dict) -> None:
    if cache_dir is not None:
        (cache_dir / _CACHE_NAME).write_text(json.dumps(cache, indent=2))


# checksum_file: SHA-256 of contents, cached on (path, size, mtime) when cache_dir is given
def checksum_file(path: Path, cache_dir: Path | None = None) -> str:
    cache = _load_cache(cache_dir)
    stat = path.stat()
    key = str(path.resolve())
    entry = cache.get(key)
    if entry and entry['size'] == stat.st_size and entry['mtime'] == stat.st_mtime:
        return entry['sha256']

    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b''):
            digest.update(chunk)
    hexdigest = digest.hexdigest()

    cache[key] = {'size': stat.st_size, 'mtime': stat.st_mtime, 'sha256': hexdigest}
    _save_cache(cache_dir, cache)
    return hexdigest


# _digest_directory: digest of a directory's sorted (name, sha256) pairs
def _digest_directory(directory: Path, cache_dir: Path | None) -> str:
    parts = [
        f'{child.name}:{checksum_file(child, cache_dir)}'
        for child in sorted(directory.iterdir())
        if child.is_file()
    ]
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()


# checksum_inputs: role -> hash; a directory role gets the directory digest
def checksum_inputs(inputs: list[tuple[str, Path]], cache_dir: Path | None = None) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for role, path in inputs:
        path = Path(path)
        if not path.exists():
            continue
        checksums[role] = (_digest_directory(path, cache_dir) if path.is_dir() else checksum_file(path, cache_dir))
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
