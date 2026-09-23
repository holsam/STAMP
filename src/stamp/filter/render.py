'''
STAMP: filter-picks slice render with per-pick accept/reject (or confidence) colouring
'''

# Import external dependencies
import mrcfile, numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Import internal STAMP objects
from stamp.schemas.particles import Particle
from stamp.utils.plotting.core import PROTEIN_CMAP, bare, central_slice

# load_central_slice: memory-mapped and downsampled central Z slice of an MRC volume
def load_central_slice(path: Path, *, max_side_px: int = 1024) -> tuple[np.ndarray, int]:
    with mrcfile.mmap(str(path), mode='r', permissive=True) as mrc:
        plane = np.asarray(central_slice(mrc.data))
    step = max(1, max(plane.shape) // max_side_px)
    return plane[::step, ::step], step

# slice_at: memory-mapped and downsampled slice of an MRC volume at a given Z
def slice_at(path: Path, z: int, *, max_side_px: int = 1024) -> tuple[np.ndarray, int]:
    with mrcfile.mmap(str(path), mode='r', permissive=True) as mrc:
        plane = np.asarray(mrc.data[z])
    step = max(1, max(plane.shape) // max_side_px)
    return plane[::step, ::step], step

# volume_depth: Z extent of an MRC volume, for slider bounds
def volume_depth(path: Path) -> int:
    with mrcfile.mmap(str(path), mode='r', permissive=True) as mrc:
        return mrc.data.shape[0]

# draw_segmentation: dark-theme segmentation render 
def draw_segmentation(ax, plane: np.ndarray) -> None:
    ax.imshow(plane == 1, cmap='Greys_r', alpha=0.9, vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(plane != 2, plane), cmap=PROTEIN_CMAP, alpha=0.95)
    height, width = plane.shape[-2], plane.shape[-1]
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    bare(ax)

# draw_picks: scatter picks on an existing axis (rejected: rejected_colour; accepted: accepted_colour; confidence-scaled between the two colours when confidence_mode is on)
def draw_picks(
    ax,
    particles: list[Particle],
    rejected: set[str],
    step: int,
    *,
    confidence_mode: bool = False,
    pick_size: float = 10,
    accepted_colour: str = '#5dff8f',
    rejected_colour: str = '#ff5d5d',
):
    if not particles:
        return None
    xy = np.array([[p.position[0], p.position[1]] for p in particles]) / step
    accepted_idx = [i for i, p in enumerate(particles) if p.particle_id not in rejected]
    rejected_idx = [i for i, p in enumerate(particles) if p.particle_id in rejected]
    mappable = None
    if rejected_idx:
        artist = ax.scatter(xy[rejected_idx, 0], xy[rejected_idx, 1], s=pick_size, c=rejected_colour, alpha=0.9, picker=5)
        artist.particle_ids = [particles[i].particle_id for i in rejected_idx]
    if accepted_idx:
        if confidence_mode:
            cmap = LinearSegmentedColormap.from_list('confidence', [rejected_colour, accepted_colour])
            confidences = [particles[i].confidence if particles[i].confidence is not None else 0.0 for i in accepted_idx]
            artist = ax.scatter(xy[accepted_idx, 0], xy[accepted_idx, 1], s=pick_size, c=confidences, cmap=cmap, vmin=0, vmax=1, alpha=0.9, picker=5)
            mappable = artist
        else:
            artist = ax.scatter(xy[accepted_idx, 0], xy[accepted_idx, 1], s=pick_size, c=accepted_colour, alpha=0.9, picker=5)
        artist.particle_ids = [particles[i].particle_id for i in accepted_idx]
    return mappable
