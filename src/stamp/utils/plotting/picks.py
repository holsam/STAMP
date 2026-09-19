'''
STAMP: pick/decoy position plots — xy scatter, segmentation overlay, Z-stack QC movie
'''

# Import external dependencies
import matplotlib.pyplot as plt, mrcfile, numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from pathlib import Path

# Import internal STAMP objects
from stamp.utils.plotting.core import GREEN, PlotFormat, central_slice, finish, plot_path, scatter_groups, show_segmentation
from stamp.utils.log import log

_HALF_COLOUR = {'A': '#4da6ff', 'B': '#ff9d4d'}

# _group_by_tomogram: {tomogram_id: [particle, ...]}
def _group_by_tomogram(particles) -> dict[str, list]:
    by_tomogram: dict[str, list] = {}
    for particle in particles:
        by_tomogram.setdefault(particle.tomogram_id, []).append(particle)
    return by_tomogram

# _scatter_panel: xy scatter of one tomogram's picks, coloured by half-set
def _scatter_panel(ax, particles, title: str) -> None:
    xy = np.array([[p.position[0], p.position[1]] for p in particles])
    half = np.array([p.half_set.value for p in particles])
    groups = [(half == letter, colour, f'half {letter} (n={int((half == letter).sum())})') for letter, colour in _HALF_COLOUR.items()]
    scatter_groups(ax, xy, groups, title)
    ax.invert_yaxis()
    ax.set_aspect('equal')

# plot_scatter: xy-scatter panel per tomogram
def plot_scatter(particles, output_dir: Path, fmt: PlotFormat, stem: str = 'consensus_picks') -> None:
    by_tomogram = _group_by_tomogram(particles)
    fig, axes = plt.subplots(1, len(by_tomogram), figsize=(4.2 * len(by_tomogram), 4), squeeze=False)
    for ax, (tomogram_id, members) in zip(axes[0], sorted(by_tomogram.items())):
        _scatter_panel(ax, members, tomogram_id)
    finish(fig, plot_path(output_dir, stem, fmt))

# plot_segmented: segmentation-slice-plus-picks panel per tomogram
def plot_segmented(particles, segmentation_path_by_tomogram: dict[str, Path], output_dir: Path, fmt: PlotFormat, stem: str = 'consensus_picks_segmented') -> None:
    by_tomogram = _group_by_tomogram(particles)
    usable = {tid: members for tid, members in by_tomogram.items() if tid in segmentation_path_by_tomogram}
    missing = set(by_tomogram) - set(usable)
    if missing:
        log.warning(f'No segmentation.mrc for {len(missing)} tomogram(s), skipping from segmented plot: {sorted(missing)}')
    if not usable:
        log.warning('No tomogram has a matching segmentation path, skipping segmented plot')
        return
    fig, axes = plt.subplots(1, len(usable), figsize=(4.2 * len(usable), 4), squeeze=False)
    for ax, (tomogram_id, members) in zip(axes[0], sorted(usable.items())):
        with mrcfile.open(str(segmentation_path_by_tomogram[tomogram_id]), permissive=True) as mrc:
            segmentation = central_slice(np.asarray(mrc.data))
        picks = np.array([[p.position[0], p.position[1]] for p in members])
        show_segmentation(ax, segmentation, field_px=segmentation.shape[-1], picks=picks)
        ax.set_title(tomogram_id, fontsize=9)
    finish(fig, plot_path(output_dir, stem, fmt))

# plot_zstack_movie: one GIF per tomogram, stepping through Z with picks near the current plane highlighted
def plot_zstack_movie(
    particles,
    segmentation_path_by_tomogram: dict[str, Path],
    output_dir: Path,
    *,
    slab_voxels: float = 3.0,
    fps: int = 8,
) -> None:
    have_ffmpeg = FFMpegWriter.isAvailable()
    if not have_ffmpeg:
        log.warning('ffmpeg not found on PATH, falling back to .gif for --pick-zstack-movie')
    by_tomogram = _group_by_tomogram(particles)
    for tomogram_id, members in sorted(by_tomogram.items()):
        segmentation_path = segmentation_path_by_tomogram.get(tomogram_id)
        if segmentation_path is None:
            log.warning(f'No segmentation.mrc for {tomogram_id}, skipping Z-stack movie')
            continue
        with mrcfile.open(str(segmentation_path), permissive=True) as mrc:
            volume = np.asarray(mrc.data)
        positions = np.array([p.position for p in members])  # (n, 3): x, y, z

        fig, ax = plt.subplots(figsize=(5, 5))
        image = ax.imshow(volume[0] == 1, cmap='Greys', alpha=0.55, vmin=0, vmax=1)
        protein = ax.imshow(np.ma.masked_where(volume[0] != 2, volume[0]), cmap='Oranges', alpha=0.85)
        scatter = ax.scatter([], [], s=18, c=GREEN)
        ax.set_xlim(0, volume.shape[-1])
        ax.set_ylim(volume.shape[-2], 0)
        ax.set_title(f'{tomogram_id}: z=0/{volume.shape[0] - 1}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        # _update: redraw the current Z plane and the picks within slab_voxels of it
        def _update(z: int):
            image.set_data(volume[z] == 1)
            protein.set_data(np.ma.masked_where(volume[z] != 2, volume[z]))
            near = np.abs(positions[:, 2] - z) <= slab_voxels if len(positions) else np.array([], dtype=bool)
            scatter.set_offsets(positions[near][:, :2] if near.any() else np.empty((0, 2)))
            ax.set_title(f'{tomogram_id}: z={z}/{volume.shape[0] - 1} (picks within ±{slab_voxels:g} vx)', fontsize=9)
            return image, protein, scatter

        animation = FuncAnimation(fig, _update, frames=volume.shape[0], blit=False)
        path = output_dir / f'{tomogram_id}_zstack.mp4'
        if have_ffmpeg:
            try:
                animation.save(path, writer=FFMpegWriter(fps=fps))
            except Exception as exc:
                log.warning(f'ffmpeg failed writing {path.name} ({exc}), falling back to .gif')
                have_ffmpeg = False
        if not have_ffmpeg:
            path = path.with_suffix('.gif')
            animation.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        log.info(f'Wrote Z-stack movie: {path}')

# plot_positions: segmented/movie degrades to scatter-only when no segmentation paths are given
def plot_positions(
    particles,
    output_dir: Path,
    style: str,  # 'scatter' | 'segmented' | 'both'
    fmt: PlotFormat,
    segmentation_path_by_tomogram: dict[str, Path] | None = None,
    *,
    zstack_movie: bool = False,
) -> None:
    if style in ('scatter', 'both'):
        plot_scatter(particles, output_dir, fmt)
    if style in ('segmented', 'both'):
        if not segmentation_path_by_tomogram:
            log.warning(f'--pick-plot-style={style} but no segmentation found, skipping the segmented panel')
        else:
            plot_segmented(particles, segmentation_path_by_tomogram, output_dir, fmt)
    if zstack_movie:
        if not segmentation_path_by_tomogram:
            log.warning('--pick-zstack-movie but no segmentation found, skipping')
        else:
            plot_zstack_movie(particles, segmentation_path_by_tomogram, output_dir)
