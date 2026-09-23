'''
STAMP: shared context for filter's GUI modules
'''

# Import external dependencies
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import internal STAMP objects
from stamp.filter.state import FilterState

# GuiContext: shared context class for information that redraw/handlers/actions need
@dataclass
class GuiContext:
    state: FilterState
    output_dir: Path
    fig: Any
    ax_seg: Any
    ax_raw: Any
    root: Any
    view: dict = field(default_factory=dict)
    history: list[frozenset[str]] = field(default_factory=list)
    future: list[frozenset[str]] = field(default_factory=list)
    pan_state: dict = field(default_factory=dict)
    zoomrect_state: dict = field(default_factory=dict)
    widgets: dict = field(default_factory=dict)  # kept referenced as unreferenced mpl/Tk widgets can be garbage-collected
    ui_ready: bool = False

    # current_tomogram_id: the tomogram currently shown
    def current_tomogram_id(self) -> str:
        return self.state.tomogram_ids[self.view['index']]

    # paths_for_axes: (segmentation_path, raw_path) for the current tomogram, either may be None
    def paths_for_axes(self) -> tuple[Path | None, Path | None]:
        tomogram_id = self.current_tomogram_id()
        return (
            self.state.segmentation_path_by_tomogram.get(tomogram_id),
            self.state.raw_tomogram_path_by_tomogram.get(tomogram_id),
        )

    # reference_path: whichever background is available, used for Z slider bounds/loading
    def reference_path(self) -> Path | None:
        seg_path, raw_path = self.paths_for_axes()
        return seg_path if seg_path is not None else raw_path
