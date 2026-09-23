'''
STAMP: standalone plot generation logic
'''

# Import external dependencies
import json, random
from pathlib import Path

# Import internal STAMP objects
from stamp.schemas.particles import Particle, ParticleSet
from stamp.utils.errors import StampPipelineError
from stamp.utils.log import log
from stamp.utils.plotting.picks import plot_positions

# _select_tomogram_ids: resolve which tomogram IDs to plot, by explicit list, random sample, or all
def _select_tomogram_ids(
    all_tomogram_ids: list[str],
    tomogram_ids: list[str] | None,
    n_tomograms: int | None,
    seed: int,
) -> set[str]:
    if tomogram_ids:
        missing = set(tomogram_ids) - set(all_tomogram_ids)
        if missing:
            raise StampPipelineError(f'Tomogram ID(s) not in particle set: {", ".join(sorted(missing))}')
        return set(tomogram_ids)
    if n_tomograms is not None:
        if n_tomograms >= len(all_tomogram_ids):
            return set(all_tomogram_ids)
        return set(random.Random(seed).sample(all_tomogram_ids, n_tomograms))
    return set(all_tomogram_ids)

# run_plots_pick: load a pick/decoy particle set and (re)render its position plots
def run_plots_pick(
    particle_set_path: Path,
    output_dir: Path,
    segmentation_dir: Path | None,
    raw_tomogram_dir: Path | None,
    pick_plot_style: str,
    plot_format: str,
    pick_zstack_movie: bool,
    pick_plot_3d: bool,
    tomogram_ids: list[str] | None,
    n_tomograms: int | None,
    seed: int,
    n_workers: int,
) -> None:
    particle_set = ParticleSet.model_validate(json.loads(particle_set_path.read_text()))
    all_tomogram_ids = sorted({p.tomogram_id for p in particle_set.particles})
    selected = _select_tomogram_ids(all_tomogram_ids, tomogram_ids, n_tomograms, seed)
    particles: list[Particle] = [p for p in particle_set.particles if p.tomogram_id in selected]
    log.info(f'Plotting {len(selected)} of {len(all_tomogram_ids)} tomogram(s)')

    segmentation_paths = {path.stem: path for path in sorted(segmentation_dir.glob('*.mrc'))} if segmentation_dir else {}
    raw_tomogram_paths = {path.stem: path for path in sorted(raw_tomogram_dir.glob('*.mrc'))} if raw_tomogram_dir else {}

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_positions(particles, output_dir, pick_plot_style, plot_format, segmentation_paths, raw_tomogram_paths, zstack_movie=pick_zstack_movie, plot_3d_view=pick_plot_3d, max_workers=n_workers)
