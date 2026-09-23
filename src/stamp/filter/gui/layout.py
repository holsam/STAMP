'''
STAMP: widget construction for filter's GUI
'''

# Import external dependencies
import tkinter as tk
from tkinter import ttk

# Import internal STAMP objects
from stamp.filter.gui import actions, interactions
from stamp.filter.gui.components import icon, make_button, make_live_spinbox, make_swatch
from stamp.filter.gui.context import GuiContext
from stamp.filter.gui.style import ACCENT, BG, BUTTON_ACTIVE_BG, BUTTON_BG, FG, PANEL_BG
from stamp.filter.render import volume_depth

# apply_style: dark ttk theme for the whole window
def _apply_style(root) -> None:
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', background=PANEL_BG, foreground=FG, bordercolor=BUTTON_BG, darkcolor=BUTTON_BG, lightcolor=BUTTON_BG, troughcolor=BUTTON_BG, relief=tk.FLAT, borderwidth=0)
    style.configure('TCombobox', fieldbackground=BUTTON_BG, background=BUTTON_BG, foreground=FG, arrowcolor=FG)
    style.map('TCombobox', fieldbackground=[('readonly', BUTTON_BG)], foreground=[('readonly', FG)])
    style.configure('TSeparator', background=BG)
    style.configure('Dark.TSpinbox', fieldbackground=BUTTON_BG, background=BUTTON_BG, foreground=FG, arrowcolor=FG)
    style.map('Dark.TSpinbox', fieldbackground=[('readonly', BUTTON_BG)])
    style.configure('Dark.TButton', background=BUTTON_BG, foreground=FG, borderwidth=1)
    style.map('Dark.TButton', background=[('pressed', BUTTON_ACTIVE_BG), ('active', BUTTON_ACTIVE_BG)])
    style.configure('ToolActive.TButton', background=ACCENT, foreground='black', borderwidth=1)
    style.map('ToolActive.TButton', background=[('pressed', ACCENT), ('active', ACCENT)])
    style.configure('Dark.TLabel', background=PANEL_BG, foreground=FG)
    style.configure('DarkZ.TLabel', background=BG, foreground=FG)
    style.configure('DarkHeading.TLabel', background=PANEL_BG, foreground=FG, font=('TkDefaultFont', 11, 'bold'))
    style.configure('Dark.TCheckbutton', background=PANEL_BG, foreground=FG)
    style.map('Dark.TCheckbutton', background=[('active', PANEL_BG)], foreground=[('active', FG)])

# build_window: build a fixed-size dark-theme window
def build_window(ctx: GuiContext) -> tuple:
    root = ctx.root
    root.configure(bg=BG)
    root.wm_title('STAMP: filter')  # set window title
    root.resizable(False, False)  # use fixed-size window
    _apply_style(root)

    canvas_widget = ctx.fig.canvas.get_tk_widget()
    canvas_widget.pack_forget()  # repacked once the surrounding frames claim their space
    top_bar = tk.Frame(root, bg=PANEL_BG, bd=0)
    top_bar.pack(side=tk.TOP, fill=tk.X)
    info_panel = tk.Frame(root, bg=PANEL_BG, bd=0, width=200)
    info_panel.pack(side=tk.LEFT, fill=tk.Y)
    z_bar = tk.Frame(root, bg=BG)
    z_bar.pack(side=tk.LEFT, fill=tk.Y)
    canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    return top_bar, info_panel, z_bar

# build_top_bar: build toolbar containing undo/redo/reset/save, reject-all/lasso-reject/lasso-accept/accept-all, zoom tools, tomogram nav
def build_top_bar(ctx: GuiContext, top_bar) -> None:
    root, widgets, state = ctx.root, ctx.widgets, ctx.state

    group_history = tk.Frame(top_bar, bg=PANEL_BG)
    group_history.pack(side=tk.LEFT, padx=6, pady=4)
    make_button(group_history, 'Undo', lambda: actions.undo(ctx), icon_name='undo', widgets=widgets).pack(side=tk.LEFT)
    make_button(group_history, 'Redo', lambda: actions.redo(ctx), icon_name='redo', widgets=widgets).pack(side=tk.LEFT, padx=(4, 0))
    make_button(group_history, 'Reset tomogram', lambda: actions.reset_tomogram(ctx), icon_name='reset', widgets=widgets).pack(side=tk.LEFT, padx=(8, 0))
    make_button(group_history, 'Save', lambda: actions.autosave(ctx), icon_name='save', widgets=widgets).pack(side=tk.LEFT, padx=(8, 0))

    ttk.Separator(top_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=4)

    group_lasso = tk.Frame(top_bar, bg=PANEL_BG)
    group_lasso.pack(side=tk.LEFT, padx=6, pady=4)
    make_button(group_lasso, 'Reject all', lambda: actions.reject_all(ctx), icon_name='reject_all', widgets=widgets).pack(side=tk.LEFT)
    make_button(group_lasso, 'Lasso: reject', lambda: interactions.set_tool(ctx, 'lasso_reject'), icon_name='lasso_reject', widgets=widgets, key='lasso_reject_btn').pack(side=tk.LEFT, padx=(4, 0))
    make_button(group_lasso, 'Lasso: accept', lambda: interactions.set_tool(ctx, 'lasso_accept'), icon_name='lasso_accept', widgets=widgets, key='lasso_accept_btn').pack(side=tk.LEFT, padx=(4, 0))
    make_button(group_lasso, 'Accept all', lambda: actions.accept_all(ctx), icon_name='accept_all', widgets=widgets).pack(side=tk.LEFT, padx=(4, 0))

    ttk.Separator(top_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=4)

    group_zoom = tk.Frame(top_bar, bg=PANEL_BG)
    group_zoom.pack(side=tk.LEFT, padx=6, pady=4)
    make_button(group_zoom, 'Zoom in', lambda: interactions.zoom(ctx, 0.8), icon_name='zoom_in', widgets=widgets).pack(side=tk.LEFT)
    make_button(group_zoom, 'Zoom out', lambda: interactions.zoom(ctx, 1.25), icon_name='zoom_out', widgets=widgets).pack(side=tk.LEFT, padx=(4, 0))
    make_button(group_zoom, 'Zoom to fit', lambda: interactions.zoom_fit(ctx), icon_name='zoom_fit', widgets=widgets).pack(side=tk.LEFT, padx=(4, 0))
    make_button(group_zoom, 'Zoom to fill', lambda: interactions.zoom_fill(ctx), icon_name='zoom_fill', widgets=widgets).pack(side=tk.LEFT, padx=(4, 0))
    make_button(group_zoom, 'Zoom to selection', lambda: interactions.set_tool(ctx, 'zoom_rect'), icon_name='zoom_rect', widgets=widgets, key='zoom_rect_btn').pack(side=tk.LEFT, padx=(4, 0))

    ttk.Separator(top_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=4)

    group_nav = tk.Frame(top_bar, bg=PANEL_BG)
    group_nav.pack(side=tk.LEFT, padx=6, pady=4)
    widgets['nav_position_var'] = tk.StringVar(master=root)
    ttk.Label(group_nav, textvariable=widgets['nav_position_var'], style='Dark.TLabel').pack(side=tk.TOP)
    nav_row = tk.Frame(group_nav, bg=PANEL_BG)
    nav_row.pack(side=tk.TOP)
    make_button(nav_row, '< Prev', actions.step(ctx, -1), icon_name='prev', widgets=widgets).pack(side=tk.LEFT)
    widgets['tomo_var'] = tk.StringVar(master=root, value=state.tomogram_ids[0])
    tomo_combo = ttk.Combobox(nav_row, values=state.tomogram_ids, textvariable=widgets['tomo_var'], state='readonly', width=24)
    tomo_combo.pack(side=tk.LEFT, padx=4)
    tomo_combo.bind('<<ComboboxSelected>>', lambda _event: actions.jump_to_tomogram(ctx, widgets['tomo_var'].get()))
    make_button(nav_row, 'Next >', actions.step(ctx, 1), icon_name='next', widgets=widgets).pack(side=tk.LEFT)

# build_info_panel: build information panel containing identity/review status, pick styling, Z filter + diagram
def build_info_panel(ctx: GuiContext, info_panel) -> None:
    root, widgets, view = ctx.root, ctx.widgets, ctx.view

    def _info_label(text_var=None, text=None, *, bold=False, pad=(10, 0)):
        style_name = 'DarkHeading.TLabel' if bold else 'Dark.TLabel'
        kwargs = {'textvariable': text_var} if text_var is not None else {'text': text}
        label = ttk.Label(info_panel, style=style_name, anchor='w', justify=tk.LEFT, wraplength=180, **kwargs)
        label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(pad[0], pad[1]))
        return label

    # top section: tomogram identity, review status, pick counts
    widgets['info_id_var'] = tk.StringVar(master=root)
    widgets['info_position_var'] = tk.StringVar(master=root)
    widgets['info_status_var'] = tk.StringVar(master=root)
    _info_label(widgets['info_id_var'], bold=True, pad=(12, 0))
    _info_label(widgets['info_position_var'], pad=(0, 2))
    _info_label(widgets['info_status_var'], pad=(0, 8))  # "Reviewed" once any edit/accept-all/reject-all has touched this tomogram
    widgets['info_total_var'] = tk.StringVar(master=root)
    widgets['info_accepted_var'] = tk.StringVar(master=root)
    widgets['info_rejected_var'] = tk.StringVar(master=root)
    _info_label(widgets['info_total_var'])
    _info_label(widgets['info_accepted_var'])
    _info_label(widgets['info_rejected_var'], pad=(0, 12))
    _info_label(text='Current view', bold=True)
    widgets['info_visible_var'] = tk.StringVar(master=root)
    _info_label(widgets['info_visible_var'], pad=(0, 10))

    ttk.Separator(info_panel, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 14))

    # middle section: pick style
    size_row = tk.Frame(info_panel, bg=PANEL_BG)
    size_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 6))
    ttk.Label(size_row, text='Pick size:', style='Dark.TLabel').pack(side=tk.LEFT)
    size_var = tk.IntVar(master=root, value=view['pick_size'])
    make_live_spinbox(size_row, size_var, 2, 50, lambda value: actions.set_pick_size(ctx, value)).pack(side=tk.LEFT, padx=(4, 0))

    colour_row = tk.Frame(info_panel, bg=PANEL_BG)
    colour_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 6))
    ttk.Label(colour_row, text='Accepted:', style='Dark.TLabel').pack(side=tk.LEFT)
    widgets['accepted_swatch'], widgets['accepted_swatch_id'] = make_swatch(
        colour_row, view['accepted_colour'], lambda: actions.pick_colour(ctx, 'accepted'),
    )
    widgets['accepted_swatch'].pack(side=tk.LEFT, padx=(4, 10))
    ttk.Label(colour_row, text='Rejected:', style='Dark.TLabel').pack(side=tk.LEFT)
    widgets['rejected_swatch'], widgets['rejected_swatch_id'] = make_swatch(
        colour_row, view['rejected_colour'], lambda: actions.pick_colour(ctx, 'rejected'),
    )
    widgets['rejected_swatch'].pack(side=tk.LEFT, padx=(4, 0))

    confidence_var = tk.BooleanVar(master=root, value=view['confidence_mode'])
    confidence_check = ttk.Checkbutton(info_panel, text='Confidence render', variable=confidence_var, style='Dark.TCheckbutton', command=lambda: actions.set_confidence_mode(ctx, confidence_var.get()))
    confidence_check.pack(side=tk.TOP, anchor='w', padx=10, pady=(0, 10))
    confidence_check.state(['!alternate'])

    ttk.Separator(info_panel, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 14))

    # bottom section: Z filter controls + a proportional diagram of filter window
    zfilter_row = tk.Frame(info_panel, bg=PANEL_BG)
    zfilter_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 4))
    zfilter_var = tk.BooleanVar(master=root, value=view['z_filter'])
    zfilter_check = ttk.Checkbutton(zfilter_row, text='Filter picks by Z', variable=zfilter_var, style='Dark.TCheckbutton', command=lambda: actions.set_z_filter(ctx, zfilter_var.get()))
    zfilter_check.pack(side=tk.TOP, anchor='w')
    zfilter_check.state(['!alternate'])  # clear ttk's indeterminate-dash default so it reads as a plain on/off box
    thickness_row = tk.Frame(info_panel, bg=PANEL_BG)
    thickness_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 8))
    ttk.Label(thickness_row, text='± slices:', style='Dark.TLabel').pack(side=tk.LEFT)
    thickness_var = tk.IntVar(master=root, value=view['z_filter_thickness'])
    make_live_spinbox(thickness_row, thickness_var, 0, 200, lambda value: actions.set_z_thickness(ctx, value)).pack(side=tk.LEFT, padx=(4, 0))

    from stamp.filter.gui.rendering import draw_z_diagram
    z_diagram_frame = tk.Frame(info_panel, bg=PANEL_BG)
    z_diagram_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    widgets['z_diagram_canvas'] = tk.Canvas(z_diagram_frame, bg=PANEL_BG, highlightthickness=0)
    widgets['z_diagram_canvas'].pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    # redraw whenever Tk resizes the canvas rather than assuming a fixed size
    widgets['z_diagram_canvas'].bind('<Configure>', lambda _event: draw_z_diagram(ctx))
    info_panel.pack_propagate(False)  # freeze the 200px width now that children have set the panel's natural height

# build_z_bar: build vertical Z slider & colourbar legend axis
def build_z_bar(ctx: GuiContext, z_bar) -> None:
    widgets = ctx.widgets
    initial_path = ctx.reference_path()
    initial_depth = volume_depth(initial_path) if initial_path is not None else 1
    ttk.Label(z_bar, text='Z', style='DarkZ.TLabel').pack(side=tk.TOP, pady=(6, 0))
    widgets['z_scale'] = tk.Scale(z_bar, from_=0, to=max(initial_depth - 1, 0), orient=tk.VERTICAL, command=lambda z: actions.on_zslider(ctx, z), showvalue=True, length=420, bg=BG, fg=FG, troughcolor=BUTTON_BG, highlightthickness=0)
    widgets['z_scale'].set((initial_depth - 1) // 2)
    widgets['z_scale'].pack(side=tk.TOP, fill=tk.Y, expand=True, padx=4)

    # Add colourbar legend as reserved axis that's hidden until confidence render is on
    widgets['cbar_ax'] = ctx.fig.add_axes([0.30, 0.015, 0.4, 0.02])
    widgets['cbar_ax'].set_visible(False)

# build: assemble window given a populated GuiContext.view
def build(ctx: GuiContext) -> None:
    top_bar, info_panel, z_bar = build_window(ctx)
    build_top_bar(ctx, top_bar)
    build_info_panel(ctx, info_panel)
    build_z_bar(ctx, z_bar)
