'''
STAMP: stream workflow diagram
'''

# Import external dependencies
import numpy as np, matplotlib.pyplot as plt
from dataclasses import replace
from pathlib import Path

# Import internal STAMP objects
from stamp.schemas.particles import HalfSet
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.render import DiagramRun, render_diagram
from stamp.tools.diagram.utils.walkthrough import Walkthrough
from stamp.utils.halfset import assign_half_sets
from stamp.utils.plotting.core import BLUE, GREEN, GREY, ORANGE, RED, finish, frame, style_axis

# StreamRun: pick and decoy each tomogram as `stream append` does, then give merged particles global half-sets as `stream finalise` does
class StreamRun(DiagramRun):
    def __init__(self, config: DiagramConfig):
        self.config = config
        self.build()

    def build(self):
        config = self.config
        # small tomograms, each an independent scene (mirrors one `stream append` per tomogram)
        small = replace(config, field_px=round(config.field_px * 0.55), n_vesicles=5, n_filaments=8, n_target=60, n_distractor=50, n_extra=20)
        self.tomograms = []
        for index in range(config.n_tomograms):
            w = Walkthrough(replace(small, seed=config.seed + 100 * index))
            w.build_scene()
            w.pick()
            self.tomograms.append(w)
        self.w = self.tomograms[0]
        self.ids = [f'tomo{index + 1}' for index in range(config.n_tomograms)]
        # finalise: whole tomograms go to one half; real and decoy sets are assigned with different seeds (stream.half_set_seed, stream.seed)
        self.real_half = self._halves([len(w.consensus) for w in self.tomograms], 'p', config.seed)
        self.decoy_half = self._halves([len(w.decoy_points) for w in self.tomograms], 'd', config.seed + 1)

    # _halves: half of each tomogram, from the real assign_half_sets with the tomogram as group
    def _halves(self, counts: list[int], tag: str, seed: int) -> list[HalfSet]:
        ids = [f'{tomogram}-{tag}{row}' for tomogram, count in zip(self.ids, counts) for row in range(count)]
        group_of = {particle_id: particle_id.split('-')[0] for particle_id in ids}
        by_id = assign_half_sets(ids, seed=seed, group_of=group_of)
        return [by_id[next(i for i in ids if i.startswith(tomogram + '-'))] for tomogram in self.ids]

# --- panels ---
# plot_append: one tomogram after `stream append` & its picks and decoys
def plot_append(run, ax, index: int):
    w = run.tomograms[index]
    ax.imshow(w.raw_tomogram, cmap=GREY)
    ax.scatter(w.consensus[:, 0], w.consensus[:, 1], s=14, c=GREEN, label=f'particles (n={len(w.consensus)})')
    ax.scatter(w.decoy_points[:, 0], w.decoy_points[:, 1], s=26, c=RED, marker='x', label=f'decoys (n={len(w.decoy_points)})')
    frame(ax, w.field_px, f'append {run.ids[index]}: pick, decoy, extract')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.4)

# plot_counts: particles and decoys appended per tomogram
def plot_counts(run, ax):
    x = np.arange(len(run.ids))
    ax.bar(x - 0.2, [len(w.consensus) for w in run.tomograms], 0.4, color=GREEN, label='real particles')
    ax.bar(x + 0.2, [len(w.decoy_points) for w in run.tomograms], 0.4, color=RED, label='decoys')
    ax.set_xticks(x, run.ids)
    style_axis(ax, 'appended per tomogram (done.json)', ylabel='count', legend=True)

# plot_halves: global half-set of each tomogram, real and decoy sets assigned with different seeds
def plot_halves(run, ax):
    for row, (name, halves) in enumerate((('real', run.real_half), ('decoy', run.decoy_half))):
        for col, half in enumerate(halves):
            ax.add_patch(plt.Rectangle((col, 1 - row), 0.9, 0.9, color=BLUE if half is HalfSet.A else ORANGE))
            ax.text(col + 0.45, 1.45 - row, half.value, ha='center', va='center', fontsize=9)
        ax.text(-0.1, 1.45 - row, name, ha='right', va='center', fontsize=8)
    ax.set_xlim(-0.9, len(run.ids)); ax.set_ylim(-0.1, 2)
    ax.set_xticks([c + 0.45 for c in range(len(run.ids))], run.ids, fontsize=7)
    ax.set_yticks([])
    ax.set_title('finalise: whole tomograms per half-set (real and decoy seeds differ)', fontsize=9)

# plot_directory: stream directory and what finalise adds to it
def plot_directory(run, ax):
    ax.set_axis_off()
    lines = ['<out>/stamp/stream/', '  stream.json                 settings fixed by the first append', '  tomograms/<id>/', '    particles.json            real particles (consensus, filtered)', '    decoy_particles.json      decoys for those particles', '    done.json                 counts, marks the tomogram appended', '    vesicle_summary.json', '', 'stamp stream finalise  ->', '  particle_set.json           all real particles, global half-sets', '  decoy_particle_set.json     all decoys, checked comparable', '', 'stamp classify <stream dir>   (replaces the two set files)', 'stamp refine prepare --stream-dir ...']
    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left', fontsize=8, family='monospace')

# --- drivers ---
# save_panels: write each panel as its own numbered PNG
def save_panels(run, outdir):
    for index in range(len(run.ids)):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        plot_append(run, ax, index)
        finish(fig, outdir / f'{index + 1:02d}_append_{run.ids[index]}.png')
    for number, (name, fn) in enumerate((('counts', plot_counts), ('halves', plot_halves), ('directory', plot_directory)), start=len(run.ids) + 1):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fn(run, ax)
        finish(fig, outdir / f'{number:02d}_{name}.png')

# save_storyboard: the appends in a row, then finalise below
def save_storyboard(run, outdir):
    n = len(run.ids)
    fig = plt.figure(figsize=(5 * n, 12))
    grid = fig.add_gridspec(2, n, hspace=0.15, wspace=0.15)
    for index in range(n):
        plot_append(run, fig.add_subplot(grid[0, index]), index)
    lower = grid[1, :].subgridspec(1, 3, wspace=0.2)
    plot_counts(run, fig.add_subplot(lower[0, 0]))
    plot_halves(run, fig.add_subplot(lower[0, 1]))
    plot_directory(run, fig.add_subplot(lower[0, 2]))
    finish(fig, outdir / '00_stream_storyboard.png')

# render: run mocked stream append and finalise and write figures, returning the output directory
def render(config: DiagramConfig) -> Path:
    return render_diagram(config, StreamRun, save_panels, save_storyboard)
