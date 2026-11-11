'''
STAMP: import externally generated picks (CSV, RELION STAR, JSON) and infer missing orientations
'''

# Import external dependencies
import csv, json, numpy as np, starfile
from pathlib import Path
from scipy.spatial import cKDTree

# Import internal STAMP objects
from stamp.picking.geometry import beam_angle_deviation_degrees, extract_surface, quaternion_from_reference_to
from stamp.picking.vesicles import load_vesicle_labels, vesicle_ids_at
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.picks import RawPick
from stamp.utils.errors import StampError, StampValidationError
from stamp.utils.io import read_mrc
from stamp.utils.log import log

# CSV_REQUIRED: columns every CSV picks file must carry
CSV_REQUIRED = ('tomogram_id', 'x', 'y', 'z')

# CSV_QUATERNION: optional orientation columns, all or none
CSV_QUATERNION = ('qw', 'qx', 'qy', 'qz')

# STAR_COLUMNS: RELION columns read from a STAR file, positions in pixels
STAR_COLUMNS = ('rlnTomoName', 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ')

# STAR_CONFIDENCE: optional RELION column mapped onto RawPick.confidence
STAR_CONFIDENCE = 'rlnAutopickFigureOfMerit'

# TOMOGRAM_ID_SUFFIX: segmentation filename suffix ignored when matching pick ids to manifests
TOMOGRAM_ID_SUFFIX = '_segmented'

# read_picks: RawPicks from a CSV, STAR, JSON or JSONL picks file, tagged with source_picker
def read_picks(path: Path, *, source_picker: str) -> list[RawPick]:
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise StampValidationError(f'unsupported picks file {str(path)!r}, expected one of {sorted(_READERS)}')
    picks = reader(path, source_picker)
    if not picks:
        raise StampValidationError(f'{str(path)!r} contains no picks')
    return picks

# map_tomogram_ids: rewrite pick ids onto manifest ids (ignoring trailings suffix on either side), raising on ids with no manifest
def map_tomogram_ids(picks: list[RawPick], manifests: list[TomogramManifest]) -> list[RawPick]:
    by_key = {_id_key(manifest.tomogram_id): manifest.tomogram_id for manifest in manifests}
    unknown = sorted({pick.tomogram_id for pick in picks if _id_key(pick.tomogram_id) not in by_key})
    if unknown:
        raise StampValidationError(f'picks reference tomogram id(s) with no matching segmentation/tomogram pair: {unknown}; known ids: {sorted(by_key.values())}')
    return [pick.model_copy(update={'tomogram_id': by_key[_id_key(pick.tomogram_id)]}) for pick in picks]

# infer_orientations: fill missing orientation, beam angle and vesicle id from the nearest segmentation surface
def infer_orientations(
    picks: list[RawPick],
    manifests: list[TomogramManifest],
    *,
    vesicle_labels_by_tomogram: dict[str, Path] | None = None,
    max_surface_distance_angstrom: float = 300.0,
) -> list[RawPick]:
    labels_by_tomogram = vesicle_labels_by_tomogram or {}
    manifest_by_id = {manifest.tomogram_id: manifest for manifest in manifests}
    indices_by_tomogram: dict[str, list[int]] = {}
    for index, pick in enumerate(picks):
        indices_by_tomogram.setdefault(pick.tomogram_id, []).append(index)
    result = list(picks)
    # TODO: parallelise?
    for tomogram_id, indices in indices_by_tomogram.items():
        manifest = manifest_by_id[tomogram_id]
        labels_path = labels_by_tomogram.get(tomogram_id)
        todo = [index for index in indices if _needs_inference(picks[index], labels_path is not None)]
        if not todo:
            continue
        segmentation = read_mrc(manifest.segmentation_path)
        try:
            vertices_zyx, normals_zyx = extract_surface(segmentation)
        except StampValidationError as exc:
            log.warning(f'{tomogram_id}: cannot infer orientations ({exc})')
            continue
        positions_xyz = np.array([picks[index].position for index in todo])
        distances, nearest = cKDTree(vertices_zyx).query(positions_xyz[:, ::-1])  # surface is (z, y, x), picks are (x, y, z)
        within = distances * manifest.voxel_size_angstrom <= max_surface_distance_angstrom
        nearest_xyz = vertices_zyx[nearest][:, ::-1]
        normals_xyz = normals_zyx[nearest][:, ::-1]
        # the particle sits on the side it was picked from, so point the normal at the pick
        flip = np.einsum('ij,ij->i', normals_xyz, positions_xyz - nearest_xyz) < 0
        normals_xyz = np.where(flip[:, None], -normals_xyz, normals_xyz)
        vesicle_ids = vesicle_ids_at(load_vesicle_labels(labels_path, segmentation.shape), vertices_zyx[nearest], tomogram_id) if labels_path is not None else None
        for position, index in enumerate(todo):
            pick = picks[index]
            updates: dict = {}
            normal = None if pick.orientation is None else _normal_from_quaternion(pick.orientation)  # a supplied quaternion always wins
            if pick.orientation is None and within[position]:
                normal = normals_xyz[position]
                updates['orientation'] = quaternion_from_reference_to(normal)
            if pick.beam_angle_deviation_degrees is None and normal is not None:
                updates['beam_angle_deviation_degrees'] = beam_angle_deviation_degrees(normal)
            if vesicle_ids is not None and pick.vesicle_id is None and within[position] and vesicle_ids[position]:
                updates['vesicle_id'] = vesicle_ids[position]
            result[index] = pick.model_copy(update=updates)
    n_missing = sum(pick.orientation is None for pick in result)
    if n_missing:
        log.warning(f'{n_missing} of {len(result)} picks have no orientation (further than {max_surface_distance_angstrom} Å from any surface, or no usable segmentation); classify will skip them')
    return result

# _needs_inference: True when a pick lacks something infer_orientations can fill
def _needs_inference(pick: RawPick, want_vesicle_id: bool) -> bool:
    return pick.orientation is None or pick.beam_angle_deviation_degrees is None or (want_vesicle_id and pick.vesicle_id is None)

# _normal_from_quaternion: the image of +z under a (w, x, y, z) unit quaternion
def _normal_from_quaternion(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

# _id_key: comparison key for a tomogram id
def _id_key(tomogram_id: str) -> str:
    return tomogram_id.removesuffix(TOMOGRAM_ID_SUFFIX)

# _cell: stripped string value of a CSV cell, '' when absent
def _cell(row: dict, key: str) -> str:
    return (row.get(key) or '').strip()

# _read_csv: picks from a CSV with tomogram_id,x,y,z[,confidence][,qw,qx,qy,qz]
def _read_csv(path: Path, source_picker: str) -> list[RawPick]:
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = list(reader)
    missing = [column for column in CSV_REQUIRED if column not in header]
    if missing:
        raise StampValidationError(f'{str(path)!r} is missing required column(s) {missing}')
    has_quaternion = [column for column in CSV_QUATERNION if column in header]
    if has_quaternion and len(has_quaternion) != len(CSV_QUATERNION):
        raise StampValidationError(f'{str(path)!r} must carry all of {list(CSV_QUATERNION)} or none of them')

    picks: list[RawPick] = []
    for line, row in enumerate(rows, start=2):  # line 1 is the header
        quaternion = [_cell(row, column) for column in CSV_QUATERNION] if has_quaternion else []
        try:
            picks.append(RawPick(
                tomogram_id=_cell(row, 'tomogram_id'),
                position=(float(row['x']), float(row['y']), float(row['z'])),
                orientation=tuple(float(value) for value in quaternion) if quaternion and all(quaternion) else None,
                confidence=float(_cell(row, 'confidence')) if _cell(row, 'confidence') else None,
                source_picker=source_picker,
            ))
        except (ValueError, TypeError, StampError) as exc:  # pydantic's ValidationError is a ValueError; the quaternion validator raises StampPipelineError
            raise StampValidationError(f'{str(path)!r} line {line}: {exc}') from exc
    return picks

# _read_star: picks from a RELION STAR file (pixel coordinates not RELION 5 centred-Angstrom columns)
def _read_star(path: Path, source_picker: str) -> list[RawPick]:
    data = starfile.read(path)
    blocks = data if isinstance(data, dict) else {'': data}
    table = next((block for block in blocks.values() if hasattr(block, 'columns') and 'rlnCoordinateX' in block.columns), None)
    if table is None:
        raise StampValidationError(f'{str(path)!r} has no STAR block with rlnCoordinateX (RELION 5 centred-Angstrom columns are not yet supported)')
    missing = [column for column in STAR_COLUMNS if column not in table.columns]
    if missing:
        raise StampValidationError(f'{str(path)!r} is missing required column(s) {missing}')

    confidence = table[STAR_CONFIDENCE].tolist() if STAR_CONFIDENCE in table.columns else [None] * len(table)
    try:
        return [
            RawPick(
                tomogram_id=str(name),
                position=(float(x), float(y), float(z)),
                confidence=None if score is None else float(score),
                source_picker=source_picker,
            )
            for name, x, y, z, score in zip(table['rlnTomoName'], table['rlnCoordinateX'], table['rlnCoordinateY'], table['rlnCoordinateZ'], confidence)
        ]
    except (ValueError, TypeError, StampError) as exc:
        raise StampValidationError(f'{str(path)!r}: {exc}') from exc

# _read_json: picks from a JSON list or JSONL file of RawPick records; source_picker is overwritten
def _read_json(path: Path, source_picker: str) -> list[RawPick]:
    text = path.read_text()
    try:
        records = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix.lower() == '.jsonl' else json.loads(text)
        return [RawPick.model_validate({**record, 'source_picker': source_picker}) for record in records]
    except (ValueError, TypeError, StampError) as exc:
        raise StampValidationError(f'{str(path)!r}: {exc}') from exc

# _READERS: picks file reader per lowercase suffix
_READERS = {
    '.csv': _read_csv,
    '.star': _read_star,
    '.json': _read_json,
    '.jsonl': _read_json,
}
