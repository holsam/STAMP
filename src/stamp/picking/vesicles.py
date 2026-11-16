'''
STAMP: import per-vesicle labels from an EValuator label MRC and attribute surface points to them
'''

# Import external dependencies
import json, numpy as np
from dataclasses import dataclass
from pathlib import Path

# Import internal STAMP objects
from stamp.picking.geometry import mesh_surface
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.picks import RawPick
from stamp.utils.errors import StampValidationError
from stamp.utils.log import log
from stamp.utils.mrc import paths_by_stem, read_mrc
from stamp.utils.parallel import run_parallel

# VESICLE_ID_TEMPLATE: '{tomogram_id}:v{label:04d}', built from EValuator's own integer label
VESICLE_ID_TEMPLATE = '{tomogram_id}:v{label:04d}'

# load_vesicle_labels: read an EValuator labelled MRC and check it matches the segmentation shape
def load_vesicle_labels(labels_mrc_path: Path, segmentation_shape: tuple[int, ...]) -> np.ndarray:
    labels = read_mrc(labels_mrc_path, dtype=np.int64)
    if labels.shape != segmentation_shape:
        raise StampValidationError(
            f'Vesicle labels MRC {str(labels_mrc_path)!r} has shape {labels.shape}, '
            f'expected {segmentation_shape} to match the segmentation'
        )
    log.debug(f'Loaded vesicle labels from {labels_mrc_path.name}, {int(labels.max())} label(s)')
    return labels

# vesicle_ids_at: vesicle_id string for each surface point (nearest-voxel lookup into the label volume), '' if the point lands on background
def vesicle_ids_at(label_volume: np.ndarray, points_zyx: np.ndarray, tomogram_id: str) -> list[str]:
    indices = np.round(points_zyx).astype(np.int64)
    indices = np.clip(indices, 0, np.array(label_volume.shape) - 1)
    voxel_labels = label_volume[indices[:, 0], indices[:, 1], indices[:, 2]]
    return [
        VESICLE_ID_TEMPLATE.format(tomogram_id=tomogram_id, label=int(label)) if label > 0 else ''
        for label in voxel_labels
    ]

# vesicle_surface_area_angstrom2: approximate area per labelled vesicle, from marching-cubes mesh face areas assigned to their nearest labelled voxel
def vesicle_surface_area_angstrom2(
    vertices_zyx: np.ndarray,
    faces: np.ndarray,
    label_volume: np.ndarray,
    voxel_size_angstrom: float,
    tomogram_id: str,
) -> dict[str, float]:
    triangle_vertices = vertices_zyx[faces]  # (n_faces, 3, 3)
    edge_a = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge_b = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    areas_voxel2 = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    centroids = triangle_vertices.mean(axis=1)
    face_ids = vesicle_ids_at(label_volume, centroids, tomogram_id)

    totals: dict[str, float] = {}
    for vesicle_id, area_voxel2 in zip(face_ids, areas_voxel2):
        if not vesicle_id:
            continue
        totals[vesicle_id] = totals.get(vesicle_id, 0.0) + float(area_voxel2) * voxel_size_angstrom ** 2
    return totals

# VesicleSummary: per-vesicle QC row for one tomogram
@dataclass(frozen=True)
class VesicleSummary:
    vesicle_id: str
    tomogram_id: str
    n_picks: int
    surface_area_angstrom2: float
    picks_per_1000_angstrom2: float

# summarise_vesicles: build a per-vesicle QC table from picks and their surface areas
def summarise_vesicles(
    picks: list, areas_by_vesicle: dict[str, float], tomogram_id: str,
) -> list[VesicleSummary]:
    counts: dict[str, int] = {}
    for pick in picks:
        if pick.vesicle_id:
            counts[pick.vesicle_id] = counts.get(pick.vesicle_id, 0) + 1
    all_ids = set(counts) | set(areas_by_vesicle)
    summaries = []
    for vesicle_id in sorted(all_ids):
        area = areas_by_vesicle.get(vesicle_id, 0.0)
        n_picks = counts.get(vesicle_id, 0)
        density = (n_picks / area * 1000.0) if area > 0 else 0.0
        summaries.append(VesicleSummary(vesicle_id, tomogram_id, n_picks, area, density))
    return summaries

# resolve_vesicle_labels: match --vesicle-labels-mrc (file or directory) to manifests by tomogram_id, warning per unmatched tomogram
def resolve_vesicle_labels(vesicle_labels_mrc: Path | None, manifests: list[TomogramManifest]) -> dict[str, Path]:
    if vesicle_labels_mrc is None:
        return {}
    if vesicle_labels_mrc.is_file():
        if len(manifests) > 1:
            log.warning('--vesicle-labels-mrc is a single file but multiple tomograms are being picked; ignoring --vesicle-labels-mrc')
        return {}

    by_stem = paths_by_stem(vesicle_labels_mrc)
    resolved: dict[str, Path] = {}
    for manifest in manifests:
        match = by_stem.get(f'{manifest.tomogram_id}_labelled') or next(
            (path for stem, path in by_stem.items() if stem.startswith(manifest.tomogram_id)), None
        )
        if match is None:
            log.warning(f'No vesicle labels MRC matching {manifest.tomogram_id} in {vesicle_labels_mrc}, picking without vesicle labelling for it')
            continue
        resolved[manifest.tomogram_id] = match
    return resolved

# _VesicleSummaryJob: one tomogram's summary inputs
@dataclass(frozen=True)
class _VesicleSummaryJob:
    tomogram_id: str
    segmentation_path: Path
    labels_mrc: Path
    voxel_size_angstrom: float
    picks: list[RawPick]

# _summarise_one_tomogram: multiprocessing worker entry point
def _summarise_one_tomogram(job: _VesicleSummaryJob) -> list[dict]:
    segmentation = read_mrc(job.segmentation_path)
    labels = load_vesicle_labels(job.labels_mrc, segmentation.shape)
    vertices, faces, _normals = mesh_surface(segmentation)     # one meshing pass, shared with nothing else on this volume
    areas = vesicle_surface_area_angstrom2(vertices, faces, labels, job.voxel_size_angstrom, job.tomogram_id)
    return [summary.__dict__ for summary in summarise_vesicles(job.picks, areas, job.tomogram_id)]

# write_vesicle_summary: per-vesicle pick count, area and density across all tomograms, written as vesicle_summary.json
def write_vesicle_summary(
    all_reconciled: list[RawPick],
    manifests: list[TomogramManifest],
    vesicle_labels_mrc_by_tomogram: dict[str, Path],
    output_dir: Path,
    n_workers: int = 1,
) -> None:
    picks_by_tomogram: dict[str, list[RawPick]] = {}
    for pick in all_reconciled:
        picks_by_tomogram.setdefault(pick.tomogram_id, []).append(pick)

    jobs: list[_VesicleSummaryJob] = []
    for manifest in manifests:
        labels_mrc = vesicle_labels_mrc_by_tomogram.get(manifest.tomogram_id)
        picks = picks_by_tomogram.get(manifest.tomogram_id, [])
        if manifest.segmentation_path is None or labels_mrc is None or not any(p.vesicle_id for p in picks):
            continue
        jobs.append(_VesicleSummaryJob(manifest.tomogram_id, manifest.segmentation_path, labels_mrc, manifest.voxel_size_angstrom, picks))

    rows_by_tomogram: dict[str, list[dict]] = {}

    def _on_success(job: _VesicleSummaryJob, result: list[dict]) -> None:
        rows_by_tomogram[job.tomogram_id] = result

    def _on_error(job: _VesicleSummaryJob, exc: Exception) -> None:
        if isinstance(exc, StampValidationError):
            log.error(f'{job.tomogram_id}: vesicle summary failed ({exc})')
        else:
            raise exc

    run_parallel(jobs, _summarise_one_tomogram, max_workers=n_workers, label='vesicle-summary', on_success=_on_success, on_error=_on_error)

    rows = [row for job in jobs for row in rows_by_tomogram.get(job.tomogram_id, [])]
    if not rows:
        return
    summary_path = output_dir / 'vesicle_summary.json'
    summary_path.write_text(json.dumps(rows, indent=2))
    n_vesicles = len(rows)
    total_picks = sum(r['n_picks'] for r in rows)
    log.info(f'{n_vesicles} vesicle(s) across {len(manifests)} tomogram(s), {total_picks} picks ({total_picks / n_vesicles:.1f} picks/vesicle on average)')
