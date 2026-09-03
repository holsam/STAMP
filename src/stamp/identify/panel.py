'''
STAMP: mass-spec candidate panel loader
'''

# Import external dependencies
import urllib.request, yaml
from dataclasses import dataclass
from pathlib import Path

# AFDB_URL: AlphaFold-DB predicted-model URL, keyed by UniProt accession
AFDB_URL = 'https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.pdb'

# Candidate: one protein candidate, pointing at a predicted structure
@dataclass(frozen=True)
class Candidate:
    name: str
    uniprot: str | None
    structure_path: Path
    expected_mass_kda: float | None
    oligomer: int | None

# load_candidate_panel: parse MS-derived candidates into Candidate records
def load_candidate_panel(
    path: Path,
    fetch_missing: bool = False,
    cache_dir: Path | None = None,
) -> list[Candidate]:
    document = yaml.safe_load(path.read_text())
    entries = document.get('candidates') if isinstance(document, dict) else None
    if not entries:
        raise ValueError(f'{path} has no non-empty "candidates:" list')

    cache_dir = cache_dir or path.parent
    panel: list[Candidate] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or 'name' not in entry:
            raise ValueError(f'candidate {index} has no "name"')
        name = str(entry['name'])
        if name in seen:
            raise ValueError(f'duplicate candidate name {name!r}')
        seen.add(name)

        uniprot = entry.get('uniprot')
        raw_path = entry.get('structure_path')
        if raw_path:
            structure_path = (path.parent / raw_path).resolve()
        elif uniprot and fetch_missing:
            structure_path = _fetch_afdb_model(str(uniprot), cache_dir)
        else:
            missing = 'no uniprot to fetch from' if not uniprot else 'fetch_missing is False'
            raise ValueError(f'candidate {name!r} has no structure_path and {missing}')
        if not structure_path.is_file():
            raise ValueError(f'candidate {name!r}: structure {structure_path} does not exist')

        oligomer = entry.get('oligomer')
        mass = entry.get('expected_mass_kda')
        panel.append(
            Candidate(
                name=name,
                uniprot=str(uniprot) if uniprot else None,
                structure_path=structure_path,
                expected_mass_kda=float(mass) if mass is not None else None,
                oligomer=int(oligomer) if oligomer is not None else None,
            )
        )
    return panel

# _fetch_afdb_model: download one AlphaFold-DB PDB, cached on disk
def _fetch_afdb_model(uniprot: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f'AF-{uniprot}-F1-model_v4.pdb'
    if destination.is_file():
        return destination
    url = AFDB_URL.format(uniprot=uniprot)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            destination.write_bytes(response.read())
    except Exception as error:
        raise ValueError(f'could not fetch AlphaFold model for {uniprot} from {url}: {error}')
    return destination
