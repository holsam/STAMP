'''
STAMP: renderer and running base class
'''

# Import external dependencies
from dataclasses import replace
from pathlib import Path
from typing import Callable

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.walkthrough import Walkthrough

# DiagramRun: class containing config and Walkthrough class
class DiagramRun:
    def __init__(self, config: DiagramConfig):
        self.config = config
        self.w = Walkthrough(config)
        self.build()

    def build(self) -> None:
        self.w.build_scene()
        self.w.pick()

# render_diagram: make output directory, render diagram panels and storyboard
def render_diagram(
    config: DiagramConfig,
    run_factory: Callable[[DiagramConfig], object],
    save_panels: Callable[[object, Path], None],
    save_storyboard: Callable[[object, Path], None],
) -> Path:
    config.outdir.mkdir(parents=True, exist_ok=True)
    run = run_factory(config)
    save_panels(run, config.outdir)
    save_storyboard(run, config.outdir)
    return config.outdir

# render_all_priors: render diagram once per orientation prior, then one comparison figure of panels that differ by prior when a comparer is given
def render_all_priors(
    config: DiagramConfig,
    run_factory: Callable[[DiagramConfig], object],
    save_panels: Callable[[object, Path], None],
    save_storyboard: Callable[[object, Path], None],
    save_comparison: Callable[[dict, Path], None] | None = None,
) -> Path:
    runs = {}
    for prior in OrientationPrior:
        runs[prior] = run_factory(replace(config, orientation_prior=prior, outdir=config.outdir / prior.value))
        (config.outdir / prior.value).mkdir(parents=True, exist_ok=True)
        save_panels(runs[prior], config.outdir / prior.value)
        save_storyboard(runs[prior], config.outdir / prior.value)
    if save_comparison is not None:
        save_comparison(runs, config.outdir)
    return config.outdir
