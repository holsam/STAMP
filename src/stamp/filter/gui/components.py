'''
STAMP: custom components for filter's GUI
'''

# Import external dependencies
import tkinter as tk
from tkinter import ttk

# Import internal STAMP objects
from stamp.filter.gui.style import BORDER_WHITE, FG, ICON_DIR, MAX_ICON_PX, PANEL_BG

# icon: load a Tk-native PhotoImage for a toolbar button, or None if no icon file exists yet
def icon(name: str) -> tk.PhotoImage | None:
    path = ICON_DIR / f'{name}.png'
    if not path.exists():
        return None
    image = tk.PhotoImage(file=str(path))
    factor = max(1, max(image.width(), image.height()) // MAX_ICON_PX)
    return image.subsample(factor, factor) if factor > 1 else image

# add_tooltip: a small delayed label shown on hover when a button is icon-only
def add_tooltip(widget, text: str) -> None:
    tip = {'window': None}

    def _show(_event=None) -> None:
        if tip['window'] is not None:
            return
        x = widget.winfo_rootx() + 4
        y = widget.winfo_rooty() + widget.winfo_height() + 4
        win = tk.Toplevel(widget)
        win.wm_overrideredirect(True)
        win.wm_geometry(f'+{x}+{y}')
        tk.Label(win, text=text, bg='#111111', fg=FG, bd=1, relief=tk.SOLID, padx=6, pady=2, font=('TkDefaultFont', 9)).pack()
        tip['window'] = win

    def _hide(_event=None) -> None:
        if tip['window'] is not None:
            tip['window'].destroy()
            tip['window'] = None

    widget.bind('<Enter>', _show, add='+')
    widget.bind('<Leave>', _hide, add='+')
    widget.bind('<ButtonPress>', _hide, add='+')

# make_button: a themed toolbar button that uses an icon when one is available
def make_button(parent, text: str, command, *, icon_name: str | None = None, widgets: dict, key: str | None = None):
    image = icon(icon_name) if icon_name else None
    if image is not None:
        button = ttk.Button(parent, image=image, command=command, style='Dark.TButton')
        widgets[f'_icon_{icon_name}'] = image  # kept image referenced to avoid garbage collection
        add_tooltip(button, text)
    else:
        button = ttk.Button(parent, text=text, command=command, style='Dark.TButton')
    if key is not None:
        widgets[key] = button
    return button

# make_live_spinbox: a themed ttk.Spinbox that reports button & typed changes
def make_live_spinbox(parent, var: tk.IntVar, from_: int, to: int, on_change) -> ttk.Spinbox:
    def _report(*_args) -> None:
        try:
            on_change(var.get())
        except tk.TclError:
            pass  # ignore mid-edit (e.g. field temporarily empty)
    var.trace_add('write', _report)
    return ttk.Spinbox(parent, from_=from_, to=to, width=4, textvariable=var, style='Dark.TSpinbox')

# make_swatch: a small clickable circle showing a colour, opening the native colour picker on click
def make_swatch(parent, colour: str, command) -> tuple[tk.Canvas, int]:
    canvas = tk.Canvas(parent, width=22, height=22, bg=PANEL_BG, highlightthickness=0)
    oval_id = canvas.create_oval(3, 3, 19, 19, fill=colour, outline=BORDER_WHITE, width=1)
    canvas.bind('<Button-1>', lambda _event: command())
    return canvas, oval_id
