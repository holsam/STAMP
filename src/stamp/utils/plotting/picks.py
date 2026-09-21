'''
STAMP: pick/decoy position plots — segmentation/raw overlay, Z-stack QC movie, 3D view
'''

# Import external dependencies
import matplotlib.pyplot as plt, mrcfile, numpy as np
from dataclasses import dataclass
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from skimage import measure
from typing import Literal

# Import internal STAMP objects
from stamp.utils.plotting.core import GREEN, GREY, PlotFormat, central_slice, finish, plot_path, show_segmentation
from stamp.utils.log import log
from stamp.utils.parallel import run_parallel

# _group_by_tomogram: {tomogram_id: [particle, ...]}
def _group_by_tomogram(particles) -> dict[str, list]:
    by_tomogram: dict[str, list] = {}
    for particle in particles:
        by_tomogram.setdefault(particle.tomogram_id, []).append(particle)
    return by_tomogram

# _ensure_agg_backend: subprocess-safe headless backend, called at the top of each multiprocessing worker
def _ensure_agg_backend() -> None:
    import matplotlib
    matplotlib.use('Agg', force=True)

# _render_movie_job: multiprocessing worker entry point
def _render_movie_job(job: _ZstackJob) -> None:
    _ensure_agg_backend()
    with mrcfile.open(str(job.volume_path), permissive=True) as mrc:
        volume = np.asarray(mrc.data)
    _zstack_movie_for_background(
        job.tomogram_id, job.positions, volume, job.output_dir,
        job.stem_suffix, _RENDER_PLANE_BY_KIND[job.render_kind], job.fps, job.slab_voxels,
    )

# _SegmentedJob: one tomogram's segmentation-slice-plus-picks plot inputs
@dataclass(frozen=True)
class _SegmentedJob:
    tomogram_id: str
    picks: np.ndarray
    segmentation_path: Path
    output_path: Path

# _render_segmented_job: multiprocessing worker entry point for plot_segmented
def _render_segmented_job(job: _SegmentedJob) -> None:
    _ensure_agg_backend()
    with mrcfile.open(str(job.segmentation_path), permissive=True) as mrc:
        segmentation = central_slice(np.asarray(mrc.data))
    fig, ax = plt.subplots(figsize=(4.2, 4))
    show_segmentation(ax, segmentation, field_px=segmentation.shape[-1], picks=job.picks)
    ax.set_title(job.tomogram_id, fontsize=9)
    finish(fig, job.output_path)

# plot_segmented: segmentation-slice-plus-picks plot per tomogram
def plot_segmented(
    particles,
    segmentation_path_by_tomogram: dict[str, Path],
    output_dir: Path,
    fmt: PlotFormat,
    stem: str = 'consensus_picks_segmented',
    *,
    max_workers: int = 1,
) -> None:
    by_tomogram = _group_by_tomogram(particles)
    usable = {tid: members for tid, members in by_tomogram.items() if tid in segmentation_path_by_tomogram}
    missing = set(by_tomogram) - set(usable)
    if missing:
        log.warning(f'No segmentation.mrc for {len(missing)} tomogram(s), skipping from segmented plot: {sorted(missing)}')
    if not usable:
        log.warning('No tomogram has a matching segmentation path, skipping segmented plot')
        return
    jobs = [
        _SegmentedJob(
            tomogram_id,
            np.array([[p.position[0], p.position[1]] for p in members]),
            segmentation_path_by_tomogram[tomogram_id],
            plot_path(output_dir, f'{stem}_{tomogram_id}', fmt),
        )
        for tomogram_id, members in sorted(usable.items())
    ]
    run_parallel(jobs, _render_segmented_job, max_workers=max_workers, label='segmented plots')

# _RawJob: one tomogram's raw-tomogram-slice-plus-picks plot inputs
@dataclass(frozen=True)
class _RawJob:
    tomogram_id: str
    picks: np.ndarray
    raw_path: Path
    output_path: Path

# _render_raw_job: multiprocessing worker entry point for plot_raw
def _render_raw_job(job: _RawJob) -> None:
    _ensure_agg_backend()
    with mrcfile.open(str(job.raw_path), permissive=True) as mrc:
        raw_slice = central_slice(np.asarray(mrc.data))
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(raw_slice, cmap='Greys_r')
    if len(job.picks):
        ax.scatter(job.picks[:, 0], job.picks[:, 1], s=6, c=GREEN)
    ax.set_xlim(0, raw_slice.shape[-1])
    ax.set_ylim(raw_slice.shape[-2], 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(job.tomogram_id, fontsize=9)
    finish(fig, job.output_path)

# plot_raw: raw-tomogram-slice-plus-picks plot per tomogram
def plot_raw(
    particles,
    raw_tomogram_path_by_tomogram: dict[str, Path],
    output_dir: Path,
    fmt: PlotFormat,
    stem: str = 'consensus_picks_raw',
    *,
    max_workers: int = 1,
) -> None:
    by_tomogram = _group_by_tomogram(particles)
    usable = {tid: members for tid, members in by_tomogram.items() if tid in raw_tomogram_path_by_tomogram}
    missing = set(by_tomogram) - set(usable)
    if missing:
        log.warning(f'No raw tomogram for {len(missing)} tomogram(s), skipping from raw plot: {sorted(missing)}')
    if not usable:
        log.warning('No tomogram has a matching raw tomogram path, skipping raw plot')
        return
    jobs = [
        _RawJob(
            tomogram_id,
            np.array([[p.position[0], p.position[1]] for p in members]),
            raw_tomogram_path_by_tomogram[tomogram_id],
            plot_path(output_dir, f'{stem}_{tomogram_id}', fmt),
        )
        for tomogram_id, members in sorted(usable.items())
    ]
    run_parallel(jobs, _render_raw_job, max_workers=max_workers, label='raw plots')

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
    scatter = ax.scatter([], [], s=6, c=GREEN)
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
            log.debug(f'ffmpeg failed writing {path.name} ({exc}), falling back to .gif')
            have_ffmpeg = False
    if not have_ffmpeg:
        path = path.with_suffix('.gif')
        animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    log.debug(f'Wrote Z-stack movie: {path}')

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

# _ZstackJob: one Z-stack movie's inputs
@dataclass(frozen=True)
class _ZstackJob:
    tomogram_id: str
    positions: np.ndarray
    volume_path: Path
    output_dir: Path
    stem_suffix: str
    render_kind: Literal['segmentation', 'raw']
    fps: int
    slab_voxels: float

_RENDER_PLANE_BY_KIND = {
    'segmentation': _render_segmentation_plane,
    'raw': _render_raw_plane,
}

# plot_zstack_movie: one movie per tomogram (segmentation background, plus raw background if given)
def plot_zstack_movie(
    particles,
    segmentation_path_by_tomogram: dict[str, Path],
    output_dir: Path,
    raw_tomogram_path_by_tomogram: dict[str, Path] | None = None,
    *,
    slab_voxels: float = 3.0,
    fps: int = 8,
    max_workers: int = 1,
) -> None:
    by_tomogram = _group_by_tomogram(particles)
    jobs: list[_ZstackJob] = []
    for tomogram_id, members in sorted(by_tomogram.items()):
        positions = np.array([p.position for p in members])  # (n, 3): x, y, z
        segmentation_path = segmentation_path_by_tomogram.get(tomogram_id)
        if segmentation_path is None:
            log.warning(f'No segmentation.mrc for {tomogram_id}, skipping Z-stack movie')
        else:
            jobs.append(_ZstackJob(tomogram_id, positions, segmentation_path, output_dir, '_segmentation', 'segmentation', fps, slab_voxels))
        raw_path = (raw_tomogram_path_by_tomogram or {}).get(tomogram_id)
        if raw_path is not None:
            jobs.append(_ZstackJob(tomogram_id, positions, raw_path, output_dir, '_raw', 'raw', fps, slab_voxels))
    def _on_error(job: _ZstackJob, exc: Exception) -> None:
        log.warning(f'{job.tomogram_id}: Z-stack movie render failed ({exc})')
    run_parallel(jobs, _render_movie_job, max_workers=max_workers, label='zstack movies', on_error=_on_error)

# _Scatter3DJob: one tomogram's 3D visualisation inputs
@dataclass(frozen=True)
class _Scatter3DJob:
    tomogram_id: str
    positions: np.ndarray  # (n, 3): x, y, z
    segmentation_path: Path | None
    output_path: Path

# _render_3d_job: multiprocessing worker entry point for plot_3d
def _render_3d_job(job: _Scatter3DJob) -> None:
    _ensure_agg_backend()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    if job.segmentation_path is not None:
        with mrcfile.open(str(job.segmentation_path), permissive=True) as mrc:
            segmentation = np.asarray(mrc.data)
        # downsample so marching_cubes/rendering stays fast on full-size tomograms
        step = max(1, max(segmentation.shape) // 128)
        membrane = (segmentation[::step, ::step, ::step] == 1).astype(np.float32)
        if membrane.any() and membrane.min() != membrane.max():
            verts, faces, _normals, _values = measure.marching_cubes(membrane, level=0.5)
            verts *= step
            mesh = Poly3DCollection(verts[faces], alpha=0.15, facecolor=GREY, edgecolor='none')
            ax.add_collection3d(mesh)
        ax.set_xlim(0, segmentation.shape[2])
        ax.set_ylim(0, segmentation.shape[1])
        ax.set_zlim(0, segmentation.shape[0])
    if len(job.positions):
        ax.scatter(job.positions[:, 0], job.positions[:, 1], job.positions[:, 2], s=6, c=GREEN, depthshade=True)
    ax.set_title(job.tomogram_id, fontsize=9)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    finish(fig, job.output_path)

# plot_3d: 3D scatter of picks, with a coarse membrane surface when a segmentation is available
def plot_3d(
    particles,
    output_dir: Path,
    fmt: PlotFormat,
    segmentation_path_by_tomogram: dict[str, Path] | None = None,
    stem: str = 'consensus_picks_3d',
    *,
    max_workers: int = 1,
) -> None:
    by_tomogram = _group_by_tomogram(particles)
    jobs = [
        _Scatter3DJob(
            tomogram_id,
            np.array([p.position for p in members]),
            (segmentation_path_by_tomogram or {}).get(tomogram_id),
            plot_path(output_dir, f'{stem}_{tomogram_id}', fmt),
        )
        for tomogram_id, members in sorted(by_tomogram.items())
    ]
    run_parallel(jobs, _render_3d_job, max_workers=max_workers, label='3D plots')

# plot_positions: raw/3D degrade to nothing when no matching path is given; segmented needs a segmentation path
def plot_positions(
    particles,
    output_dir: Path,
    style: str,  # 'segmented' | 'none' — kept as a style knob for future plot kinds
    fmt: PlotFormat,
    segmentation_path_by_tomogram: dict[str, Path] | None = None,
    raw_tomogram_path_by_tomogram: dict[str, Path] | None = None,
    *,
    zstack_movie: bool = False,
    plot_3d_view: bool = False,
    max_workers: int = 1,
) -> None:
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    if style == 'segmented':
        if not segmentation_path_by_tomogram:
            log.warning(f'--pick-plot-style={style} but no segmentation found, skipping the segmented panel')
        else:
            plot_segmented(particles, segmentation_path_by_tomogram, plots_dir, fmt, max_workers=max_workers)
    if raw_tomogram_path_by_tomogram:
        plot_raw(particles, raw_tomogram_path_by_tomogram, plots_dir, fmt, max_workers=max_workers)
    if zstack_movie:
        if not segmentation_path_by_tomogram and not raw_tomogram_path_by_tomogram:
            log.warning('--pick-zstack-movie but no segmentation or raw tomogram found, skipping')
        else:
            plot_zstack_movie(particles, segmentation_path_by_tomogram or {}, plots_dir, raw_tomogram_path_by_tomogram, max_workers=max_workers)
    if plot_3d_view:
        plot_3d(particles, plots_dir, fmt, segmentation_path_by_tomogram, max_workers=max_workers)
