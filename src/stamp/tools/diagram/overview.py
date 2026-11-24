'''
STAMP: overview of entire pipeline
'''

# Import external dependencies
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# Import internal STAMP objects
from stamp.tools.diagram import classify as classify_panels
from stamp.tools.diagram import identify as identify_panels
from stamp.tools.diagram import pick as pick_panels
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.render import DiagramRun, render_diagram
from stamp.schemas.particles import OrientationPrior
from stamp.utils.plotting.core import BLUE, GREY, bare, finish, style_axis, subgrid_axes, tile_particles

# OverviewRun: the whole mocked pipeline, exposing what the per-stage panel functions read
class OverviewRun(DiagramRun):
    def build(self):
        self.w.run()
        self.half_a, self.half_b = self.w.halves
        self.candidates = self.w.candidates
        self.cluster_ids = self.w.cluster_ids
        self.averages = self.w.averages
        self.scores = self.w.score_matrix
        self.fit_images = self.w.fit_images
        self.calls = self.w.calls

# --- refine panels: stand-ins for `refine prepare` (half-set inputs seeded by the class average, carrying the polish poses) and `refine assess` (FSC of the half-maps from the external RELION or M run) ---
# plot_poses: unaligned members of the target class (top) and the same members at the poses classify estimated (bottom)
def plot_poses(run, ax):
    w = run.w
    half = w.halves[0]
    old = next(old for old, new in w.cluster_map.items() if new == w.target_cluster)
    selected = np.where(half.polish.labels == old)[0][:6]
    ax.imshow(tile_particles(list(w.particles[half.index[selected]]) + list(w.aligned[w.target_cluster][:len(selected)]), w.box, columns=6), cmap=GREY)
    ax.set_title('refine prepare: target members as extracted (top), at classify poses (bottom)', fontsize=8)
    bare(ax)

# plot_refine: half-map FRC with the 0.143 threshold, as `refine assess` reports it
def plot_refine(run, ax):
    w = run.w
    ax.plot(w.frc_freq, w.frc_corr, color=BLUE, label='FRC (half A vs B)')
    ax.axhline(0.143, color='k', ls='--', lw=0.8, label='0.143')
    title = f'refine assess: half-map FRC ~{w.resolution:.0f} A ({w.pixel_size:g} A/px)' if np.isfinite(w.resolution) else 'refine assess: half-map FRC'
    ax.set_ylim(-0.1, 1.05)
    style_axis(ax, title, xlabel='spatial frequency (1/px)', ylabel='FRC', legend=True)

# plot_recovered: recovered map next to the target ground truth
def plot_recovered(run, axes):
    w = run.w
    for ax, image, name in zip(axes, (w.recovered, w.target_truth), ('recovered from STAMP', f'ground truth: {w.target_name}')):
        ax.imshow(image, cmap=GREY)
        ax.set_title(name, fontsize=9)
        bare(ax)

# --- drivers ---
# _individual: (filename stem, plot function, figure size) of each single-axis panel in stage order
def _individual(run) -> list:
    normal = run.w.prior is OrientationPrior.NORMAL
    return [('pick_raw', pick_panels.plot_raw, (5.5, 5.5)), ('pick_orientation', pick_panels.plot_orientation, (5.5, 5.5)), ('pick_filters' if normal else 'pick_consensus', pick_panels.plot_filters if normal else pick_panels.plot_consensus, (5.5, 5.5)), ('pick_decoys', pick_panels.plot_decoys, (5.5, 5.5)),
            ('classify_features', classify_panels.plot_features, (6, 4.5)), ('classify_seeds', lambda r, ax: classify_panels.plot_seeds(r, ax, r.half_a, 'A'), (5.5, 4.5)), ('classify_merge', classify_panels.plot_merge, (5, 4.5)), ('classify_null', classify_panels.plot_null, (6, 4.5)),
            ('identify_registration', identify_panels.plot_registration, (7, 4.5)), ('identify_scores', identify_panels.plot_score_matrix, (5.5, 4)), ('identify_decoy_control', identify_panels.plot_decoy_control, (6, 4)),
            ('refine_frc', plot_refine, (5, 4)), ('refine_poses', plot_poses, (6, 3))]

# save_individual_panels: write each stage panel as a numbered PNG
def save_individual_panels(run, outdir: Path):
    for number, (name, fn, size) in enumerate(_individual(run), start=1):
        fig, ax = plt.subplots(figsize=size)
        fn(run, ax)
        finish(fig, outdir / f'{number:02d}_{name}.png')
    n = max(2, len(run.cluster_ids))
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3))
    classify_panels.plot_class_averages(run, np.atleast_1d(axes))
    finish(fig, outdir / f'{len(_individual(run)) + 1:02d}_class_averages.png')
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    plot_recovered(run, axes)
    finish(fig, outdir / f'{len(_individual(run)) + 2:02d}_recovered.png')

# _stage_label: vertical stage name beside the first panel of a storyboard row
def _stage_label(ax, text: str):
    ax.annotate(text, xy=(-0.22, 0.5), xycoords='axes fraction', rotation=90, va='center', ha='center', fontsize=13, fontweight='bold')

# save_storyboard: lay every stage out on one 00_storyboard.png figure, one row per stage
def save_storyboard(run, outdir: Path):
    normal = run.w.prior is OrientationPrior.NORMAL
    fig = plt.figure(figsize=(22, 24))
    grid = fig.add_gridspec(5, 4, hspace=0.34, wspace=0.28, height_ratios=[1, 1, 0.55, 1, 1])
    first = fig.add_subplot(grid[0, 0])
    pick_panels.plot_raw(run, first)
    _stage_label(first, 'pick')
    pick_panels.plot_orientation(run, fig.add_subplot(grid[0, 1]))
    (pick_panels.plot_filters if normal else pick_panels.plot_consensus)(run, fig.add_subplot(grid[0, 2]))
    pick_panels.plot_decoys(run, fig.add_subplot(grid[0, 3]))

    first = fig.add_subplot(grid[1, 0])
    classify_panels.plot_features(run, first)
    _stage_label(first, 'classify')
    classify_panels.plot_seeds(run, fig.add_subplot(grid[1, 1]), run.half_a, 'A')
    classify_panels.plot_merge(run, fig.add_subplot(grid[1, 2]))
    classify_panels.plot_null(run, fig.add_subplot(grid[1, 3]))
    classify_panels.plot_class_averages(run, subgrid_axes(fig, grid[2, :], max(2, len(run.cluster_ids))))

    first = fig.add_subplot(grid[3, 0])
    identify_panels.plot_registration(run, first)
    _stage_label(first, 'identify')
    identify_panels.plot_score_matrix(run, fig.add_subplot(grid[3, 1]))
    identify_panels.plot_decoy_control(run, fig.add_subplot(grid[3, 2]))
    identify_panels.plot_calls(run, fig.add_subplot(grid[3, 3]))

    first = fig.add_subplot(grid[4, 0])
    plot_refine(run, first)
    _stage_label(first, 'refine')
    plot_poses(run, fig.add_subplot(grid[4, 1]))
    inner = grid[4, 2].subgridspec(2, 1, hspace=0.4)
    plot_recovered(run, [fig.add_subplot(inner[0]), fig.add_subplot(inner[1])])
    classify_panels.plot_halves(run, fig, grid[4, 3])
    finish(fig, outdir / '00_storyboard.png')

# render: run the mocked pipeline and write the overview figures, returning the output directory
def render(config: DiagramConfig) -> Path:
    return render_diagram(config, OverviewRun, save_individual_panels, save_storyboard)
