'''
STAMP: tomogram manifests and per-tomogram pick caches
'''

# Import external dependencies
import json
from pathlib import Path
from typing import Literal

# Import STAMP objects
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.picks import RawPick
from stamp.utils.checkpoint import atomic_write_json
from stamp.utils.log import log
from stamp.utils.mrc import glob_mrc, require_voxel_size

# manifest_paths: {tomogram_id: path} for manifests that have given path set
def manifest_paths(manifests: list[TomogramManifest], attr: Literal['segmentation_path', 'raw_tomogram_path']) -> dict[str, Path]:
    return {m.tomogram_id: getattr(m, attr) for m in manifests if getattr(m, attr) is not None}

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

# load_tomogram_manifests: match segmentations to raw tomograms by filename stem, unmatched segmentations skipped with a warning; without a segmentation directory every raw tomogram gets a manifest with no segmentation
def load_tomogram_manifests(
    segmentation_dir: Path | None,
    raw_tomogram_dir: Path,
    voxel_size_angstrom: float | None,
) -> list[TomogramManifest]:
    raw_paths = glob_mrc(raw_tomogram_dir)
    if segmentation_dir is None:
        voxel_size_angstrom = require_voxel_size(voxel_size_angstrom, raw_paths)
        return [TomogramManifest(tomogram_id=path.stem, raw_tomogram_path=path, voxel_size_angstrom=voxel_size_angstrom) for path in raw_paths]
    segmentation_paths = glob_mrc(segmentation_dir)
    matched, unmatched = match_by_stem(segmentation_paths, raw_paths)
    for segmentation_path in unmatched:
        log.warning(f'No raw tomogram matching {segmentation_path.stem}, skipping')
    voxel_size_angstrom = require_voxel_size(voxel_size_angstrom, list(matched.values()))
    return [
        TomogramManifest(
            tomogram_id=segmentation_path.stem,
            segmentation_path=segmentation_path,
            raw_tomogram_path=raw_path,
            voxel_size_angstrom=voxel_size_angstrom,
        )
        for segmentation_path, raw_path in matched.items()
    ]

# cache_tomogram_picks: atomically write one tomogram's picks to <cache_dir>/<tomogram_id>.json
def cache_tomogram_picks(cache_dir: Path, tomogram_id: str, picks: list[RawPick]) -> None:
    atomic_write_json(cache_dir / f'{tomogram_id}.json', [pick.model_dump() for pick in picks])

# load_cached_picks: read back per-tomogram JSON pick caches written by cache_tomogram_picks, skipping unreadable ones
def load_cached_picks(cache_dir: Path, tomogram_ids: list[str]) -> list[RawPick]:
    picks: list[RawPick] = []
    for tomogram_id in tomogram_ids:
        cache_path = cache_dir / f'{tomogram_id}.json'
        if not cache_path.exists():
            continue
        try:
            raw_picks = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            log.warning(f'{cache_path} is not valid JSON, ignoring it')
            continue
        picks.extend(RawPick(**raw) for raw in raw_picks)
    return picks
