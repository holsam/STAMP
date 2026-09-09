'''
STAMP: renderer and running base class
'''

# Import external dependencies
from pathlib import Path
from typing import Callable

# Import internal STAMP objects
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
