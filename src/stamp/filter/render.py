'''
STAMP: filter-picks slice render with per-pick accept/reject (or confidence) colouring
'''

# Import external dependencies
import mrcfile, numpy as np
from pathlib import Path

# Import internal STAMP objects
from stamp.schemas.particles import Particle
from stamp.utils.plotting.core import GREEN, RED, central_slice, show_segmentation

# load_central_slice: downsampled central-Z slice of an MRC volume
def load_central_slice(path: Path, *, max_side_px: int = 1024) -> tuple[np.ndarray, int]:
    with mrcfile.open(str(path), permissive=True) as mrc:
        volume = np.asarray(mrc.data)
    plane = central_slice(volume)
    step = max(1, max(plane.shape) // max_side_px)
    return plane[::step, ::step], step

# slice_at: a single Z plane of an MRC volume, downsampled to max_side_px
def slice_at(path: Path, z: int, *, max_side_px: int = 1024) -> tuple[np.ndarray, int]:
    with mrcfile.open(str(path), permissive=True) as mrc:
        volume = np.asarray(mrc.data)
    plane = volume[z]
    step = max(1, max(plane.shape) // max_side_px)
    return plane[::step, ::step], step

# volume_depth: Z extent of an MRC volume, for slider bounds
def volume_depth(path: Path) -> int:
    with mrcfile.open(str(path), permissive=True) as mrc:
        return mrc.data.shape[0]

# draw_picks: scatter picks on an existing axis (rejected: red; accepted: green; confidence-scaled when confidence_mode is on)
def draw_picks(
    ax,
    particles: list[Particle],
    rejected: set[str],
    step: int,
    *,
    confidence_mode: bool = False
) -> None:
    if not particles:
        return
    xy = np.array([[p.position[0], p.position[1]] for p in particles]) / step
    accepted_idx = [i for i, p in enumerate(particles) if p.particle_id not in rejected]
    rejected_idx = [i for i, p in enumerate(particles) if p.particle_id in rejected]
    if rejected_idx:
        ax.scatter(xy[rejected_idx, 0], xy[rejected_idx, 1], s=10, c=RED, alpha=0.9, picker=5)
    if accepted_idx:
        if confidence_mode:
            confidences = [particles[i].confidence if particles[i].confidence is not None else 0.0 for i in accepted_idx]
            ax.scatter(xy[accepted_idx, 0], xy[accepted_idx, 1], s=10, c=confidences, cmap='viridis', vmin=0, vmax=1, alpha=0.9, picker=5)
        else:
            ax.scatter(xy[accepted_idx, 0], xy[accepted_idx, 1], s=10, c=GREEN, alpha=0.9, picker=5)
