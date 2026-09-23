'''
STAMP: filter-picks session state — load a particle set + manifests, track accept/reject
'''

# Import external dependencies
import json
from dataclasses import dataclass, field
from pathlib import Path

# Import internal STAMP objects
from stamp.schemas.particles import Particle, ParticleSet
from stamp.utils.errors import StampPipelineError

# FilterState: a filtering session's data and accept/reject decisions
@dataclass
class FilterState:
    particle_set: ParticleSet
    segmentation_path_by_tomogram: dict[str, Path]
    raw_tomogram_path_by_tomogram: dict[str, Path]
    rejected: set[str] = field(default_factory=set)  # particle_ids

    # tomogram_ids: sorted tomogram IDs present in the particle set
    @property
    def tomogram_ids(self) -> list[str]:
        return sorted({p.tomogram_id for p in self.particle_set.particles})

    # particles_for: this tomogram's particles
    def particles_for(self, tomogram_id: str) -> list[Particle]:
        return [p for p in self.particle_set.particles if p.tomogram_id == tomogram_id]

    # toggle: flip a particle's accept/reject state
    def toggle(self, particle_id: str) -> None:
        if particle_id in self.rejected:
            self.rejected.discard(particle_id)
        else:
            self.rejected.add(particle_id)

    # reject_all: mark every particle in a tomogram rejected
    def reject_all(self, tomogram_id: str) -> None:
        self.rejected.update(p.particle_id for p in self.particles_for(tomogram_id))

    # reset_tomogram: clear rejections for one tomogram only
    def reset_tomogram(self, tomogram_id: str) -> None:
        ids = {p.particle_id for p in self.particles_for(tomogram_id)}
        self.rejected -= ids

    # save: write particle_set.filtered.json (accepted only) and rejected_picks.json
    def save(self, output_dir: Path) -> tuple[Path, Path]:
        accepted = [p for p in self.particle_set.particles if p.particle_id not in self.rejected]
        if not accepted:
            raise StampPipelineError('Filtering rejected every particle, nothing to save')
        filtered_set = self.particle_set.model_copy(update={'particles': accepted})
        filtered_path = output_dir / 'particle_set.filtered.json'
        filtered_path.write_text(filtered_set.model_dump_json(indent=2))
        rejected_path = output_dir / 'rejected_picks.json'
        rejected_rows = [{'particle_id': p.particle_id, 'tomogram_id': p.tomogram_id} for p in self.particle_set.particles if p.particle_id in self.rejected]
        rejected_path.write_text(json.dumps(rejected_rows, indent=2))
        return filtered_path, rejected_path

# load_filter_state: read particle_set.json, match segmentation/raw dirs by stem, optionally resume prior rejections
def load_filter_state(
    particle_set_path: Path,
    segmentation_dir: Path | None,
    raw_tomogram_dir: Path | None,
    *,
    output_dir: Path | None = None,
) -> FilterState:
    particle_set = ParticleSet.model_validate_json(particle_set_path.read_text())
    tomogram_ids = {p.tomogram_id for p in particle_set.particles}

    def _match(directory: Path | None) -> dict[str, Path]:
        if directory is None:
            return {}
        return {path.stem: path for path in directory.glob('*.mrc') if path.stem in tomogram_ids}

    state = FilterState(particle_set, _match(segmentation_dir), _match(raw_tomogram_dir))

    # resume: seed rejected from a prior session's rejected_picks.json, if present
    if output_dir is not None:
        rejected_path = output_dir / 'rejected_picks.json'
        if rejected_path.exists():
            rows = json.loads(rejected_path.read_text())
            state.rejected = {row['particle_id'] for row in rows}

    return state
