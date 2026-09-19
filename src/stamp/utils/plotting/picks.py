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
def plot_segmented(
    particles,
    segmentation_path_by_tomogram: dict[str, Path],
    output_dir: Path,
    fmt: PlotFormat,
    stem: str = 'consensus_picks_segmented'
) -> None:
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

# plot_raw: raw-tomogram-slice-plus-picks panel per tomogram
def plot_raw(
    particles,
    raw_tomogram_path_by_tomogram: dict[str, Path],
    output_dir: Path,
    fmt: PlotFormat,
    stem: str = 'consensus_picks_raw'
) -> None:
    by_tomogram = _group_by_tomogram(particles)
    usable = {tid: members for tid, members in by_tomogram.items() if tid in raw_tomogram_path_by_tomogram}
    missing = set(by_tomogram) - set(usable)
    if missing:
        log.warning(f'No raw tomogram for {len(missing)} tomogram(s), skipping from raw plot: {sorted(missing)}')
    if not usable:
        log.warning('No tomogram has a matching raw tomogram path, skipping raw plot')
        return
    fig, axes = plt.subplots(1, len(usable), figsize=(4.2 * len(usable), 4), squeeze=False)
    for ax, (tomogram_id, members) in zip(axes[0], sorted(usable.items())):
        with mrcfile.open(str(raw_tomogram_path_by_tomogram[tomogram_id]), permissive=True) as mrc:
            raw_slice = central_slice(np.asarray(mrc.data))
        picks = np.array([[p.position[0], p.position[1]] for p in members])
        ax.imshow(raw_slice, cmap='Greys_r')
        if len(picks):
            ax.scatter(picks[:, 0], picks[:, 1], s=18, c=GREEN)
        ax.set_xlim(0, raw_slice.shape[-1])
        ax.set_ylim(raw_slice.shape[-2], 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(tomogram_id, fontsize=9)
    finish(fig, plot_path(output_dir, stem, fmt))

# _zstack_movie_for_background: one movie, given a Z-stack volume and how to render each plane
def _zstack_movie_for_background(
    tomogram_id: str,
    positions: np.ndarray,
    volume: np.ndarray,
    output_dir: Path,
    stem_suffix: str,
    render_plane,
    fps: int,
    slab_voxels: float,
) -> None:
    have_ffmpeg = FFMpegWriter.isAvailable()
    fig, ax = plt.subplots(figsize=(5, 5))
    images = render_plane(ax, volume, 0)
    scatter = ax.scatter([], [], s=18, c=GREEN)
    ax.set_xlim(0, volume.shape[-1])
    ax.set_ylim(volume.shape[-2], 0)
    ax.set_title(f'{tomogram_id}: z=0/{volume.shape[0] - 1}', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    def _update(z: int):
        render_plane(ax, volume, z, images=images)
        near = np.abs(positions[:, 2] - z) <= slab_voxels if len(positions) else np.array([], dtype=bool)
        scatter.set_offsets(positions[near][:, :2] if near.any() else np.empty((0, 2)))
        ax.set_title(f'{tomogram_id}: z={z}/{volume.shape[0] - 1} (picks within ±{slab_voxels:g} vx)', fontsize=9)
        return (*images, scatter)

    animation = FuncAnimation(fig, _update, frames=volume.shape[0], blit=False)
    path = output_dir / f'{tomogram_id}_zstack{stem_suffix}.mp4'
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

# _render_segmentation_plane: segmentation background renderer for _zstack_movie_for_background
def _render_segmentation_plane(
    ax,
    volume,
    z,
    images=None
):
    if images is None:
        image = ax.imshow(volume[z] == 1, cmap='Greys', alpha=0.55, vmin=0, vmax=1)
        protein = ax.imshow(np.ma.masked_where(volume[z] != 2, volume[z]), cmap='Oranges', alpha=0.85)
        return image, protein
    image, protein = images
    image.set_data(volume[z] == 1)
    protein.set_data(np.ma.masked_where(volume[z] != 2, volume[z]))
    return images

# _render_raw_plane: raw tomogram background renderer for _zstack_movie_for_background
def _render_raw_plane(
    ax,
    volume,
    z,
    images=None
):
    if images is None:
        return (ax.imshow(volume[z], cmap='Greys_r'),)
    images[0].set_data(volume[z])
    return images

# plot_zstack_movie: one movie per tomogram (segmentation background, plus raw background if given)
def plot_zstack_movie(
    particles,
    segmentation_path_by_tomogram: dict[str, Path],
    output_dir: Path,
    raw_tomogram_path_by_tomogram: dict[str, Path] | None = None,
    *,
    slab_voxels: float = 3.0,
    fps: int = 8,
) -> None:
    by_tomogram = _group_by_tomogram(particles)
    for tomogram_id, members in sorted(by_tomogram.items()):
        positions = np.array([p.position for p in members])  # (n, 3): x, y, z
        segmentation_path = segmentation_path_by_tomogram.get(tomogram_id)
        if segmentation_path is None:
            log.warning(f'No segmentation.mrc for {tomogram_id}, skipping Z-stack movie')
        else:
            with mrcfile.open(str(segmentation_path), permissive=True) as mrc:
                volume = np.asarray(mrc.data)
            _zstack_movie_for_background(tomogram_id, positions, volume, output_dir, '_segmentation', _render_segmentation_plane, fps, slab_voxels)

        raw_path = (raw_tomogram_path_by_tomogram or {}).get(tomogram_id)
        if raw_path is not None:
            with mrcfile.open(str(raw_path), permissive=True) as mrc:
                raw_volume = np.asarray(mrc.data)
            _zstack_movie_for_background(tomogram_id, positions, raw_volume, output_dir, '_raw', _render_raw_plane, fps, slab_voxels)

# plot_positions: segmented/movie degrades to scatter-only when no segmentation paths are given
def plot_positions(
    particles,
    output_dir: Path,
    style: str,  # 'scatter' | 'segmented' | 'both'
    fmt: PlotFormat,
    segmentation_path_by_tomogram: dict[str, Path] | None = None,
    raw_tomogram_path_by_tomogram: dict[str, Path] | None = None,
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
    if raw_tomogram_path_by_tomogram:
        plot_raw(particles, raw_tomogram_path_by_tomogram, output_dir, fmt)
    if zstack_movie:
        if not segmentation_path_by_tomogram and not raw_tomogram_path_by_tomogram:
            log.warning('--pick-zstack-movie but no segmentation or raw tomogram found, skipping')
        else:
            plot_zstack_movie(particles, segmentation_path_by_tomogram or {}, output_dir, raw_tomogram_path_by_tomogram)