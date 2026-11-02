'''
STAMP: import externally generated picks (CSV, RELION STAR, JSON) and infer missing orientations
'''

# Import external dependencies
import csv, json, starfile
from pathlib import Path

# Import internal STAMP objects
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.picks import RawPick
from stamp.utils.errors import StampError, StampValidationError

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
