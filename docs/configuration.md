# STAMP - configuration

This document outlines how to configure a `stamp run` call using a `stamp_run.toml` file. For per-command CLI options, see the [command-line documentation](./cli.md).

<br>

## `stamp_run.toml` structure

```toml
[run]
segmentation_dir = "seg"
raw_tomogram_dir = "tomo"
output_dir = "."                                # root for output files (files will be saved to <output_dir>/stamp/<stage>)
voxel_size_angstrom = 13.48
backend = "local"                               # local | mock | cluster
stop_after = "refine"                           # pick | classify | identify | refine | omit for full run

[decoy]
enabled = true
method = "rejected-surface"                     # rejected-surface | shifted | synthetic-noise
n_decoys_per_tomogram = 50
min_distance_from_real_angstrom = 100.0
min_distance_from_picks_angstrom = 60.0
min_shift_angstrom = 200.0
max_shift_angstrom = 600.0
n_synthetic_tomograms = 3
# synthetic_shape_voxels = [200, 200, 200]      # omit to match the first real tomogram
n_workers = 1

[plots]
enabled = true
format = "tiff"                                 # png | jpg | tiff | svg
pick_style = "both"                             # scatter | segmented | both
pick_zstack_movie = false

[stage.pick]
pickers = ["stamp-native"]
consensus_rule = "intersection"                 # union | intersection
distance_threshold = 15.0
half_set_seed = 0
# backend = "local"                             # omit to inherit [run].backend
n_workers = 1

[stage.classify]
method = "hdbscan"                              # hdbscan | kmeans
min_cluster_size = 20
n_clusters = 5
n_components = 20
box_angstrom = 300.0
n_radial_bins = 12
strict_halfset_independence = true
inplane_alignment = true
inplane_angular_step_degrees = 10.0
inplane_iterations = 3
random_state = 0
n_workers = 1
# backend = "local"                             # omit to inherit [run].backend

[stage.identify]
candidates = "candidates.yaml"                  # required unless stop_after is pick/classify
resolution = 25.0                               # required unless stop_after is pick/classify
fitter = "native"
fetch_missing = false
n_workers = 1
# backend = "local"                             # omit to inherit [run].backend

[stage.refine]
tool = "relion"                                 # relion | m
class_id = "all"
iterations = 5
# mask = "mask.mrc"
# backend = "local"                             # omit to inherit [run].backend
```

All paths (e.g. `segmentation_dir`, `raw_tomogram_dir`, `output_dir`, `stage.identify.candidates`, `stage.refine.mask`) are resolved relative to the `stamp_run.toml` file's directory, not the current working directory.

<br>

## Configuration table reference

### `[run]`

Key | Type | Default
-- | -- | --
`segmentation_dir` | Path | *n/a (required)*
`raw_tomogram_dir` | Path | *n/a (required)*
`output_dir` | Path | *n/a (required)*
`voxel_size_angstrom` | float | *n/a (required)*
`backend` | `local`\|`mock`\|`cluster` | `local` *(default backend for stages without a stage-specific override)*
`stop_after` | `pick`\|`classify`\|`identify`\|`refine`\|`None` | `None` *(runs all four command)*

### `[decoy]`

Key | Type | Default
-- | -- | --
`enabled` | bool | `True`
`method` | `rejected-surface`\|`shifted`\|`synthetic-noise` | `rejected-surface`
`n_decoys_per_tomogram` | int | `50`
`min_distance_from_real_angstrom` | float | `100.0`
`min_distance_from_picks_angstrom` | float | `60.0`
`min_shift_angstrom` | float | `200.0`
`max_shift_angstrom` | float | `600.0`
`n_synthetic_tomograms` | int | `3`
`synthetic_shape_voxels` | `(int,int,int)`\|`None` | `None` *(matches the first real tomogram found under `raw_tomogram_dir`, or `(200,200,200)` if none exist)*
`n_workers` | int | `1`

### `[plots]` — `PlotSettings`

Key | Type | Default
-- | -- | --
`enabled` | bool | `True`
`format` | `png`\|`jpg`\|`tiff`\|`svg` | `tiff`
`pick_style` | `scatter`\|`segmented`\|`both` | `both`
`pick_zstack_movie` | bool | `False`

### `[stage.pick]`

Key | Type | Default
-- | -- | --
`pickers` | `list[str]` | `['stamp-native']`
`consensus_rule` | `union`\|`intersection` | `intersection`
`distance_threshold` | float | `15.0`
`half_set_seed` | int | `0`
`backend` | `local`\|`mock`\|`cluster`\|`None` | `None` (inherits `[run].backend`)
`n_workers` | int | `1`

### `[stage.classify]`

Key | Type | Default
-- | -- | --
`method` | `hdbscan`\|`kmeans` | `hdbscan`
`min_cluster_size` | int | `20`
`n_clusters` | int | `5`
`n_components` | int | `20`
`box_angstrom` | float | `300.0`
`n_radial_bins` | int | `12`
`strict_halfset_independence` | bool | `True`
`inplane_alignment` | bool | `True`
`inplane_angular_step_degrees` | float | `10.0`
`inplane_iterations` | int | `3`
`random_state` | int | `0`
`n_workers` | int | `1`
`backend` | `local`\|`mock`\|`cluster`\|`None` | `None` *(inherit from `[run].backend`)*

### `[stage.identify]`

Key | Type | Default
-- | -- | --
`candidates` | Path | *n/a (required unless `[run].stop_after` is `pick` or `classify`)*
`resolution` | float\|`None` | *n/a (required unless `[run].stop_after` is `pick` or `classify`)*
`fitter` | `native` | `native`
`fetch_missing` | bool | `False`
`n_workers` | int | `1`
`backend` | `local`\|`mock`\|`cluster`\|`None` | `None` *(inherit from `[run].backend`)*

### `[stage.refine]`

Key | Type | Default
-- | -- | --
`tool` | `relion`\|`m` | `relion`
`class_id` | str | `'all'`
`iterations` | int | `5`
`mask` | Path\|`None` | `None`
`backend` | `local`\|`mock`\|`cluster`\|`None` | `None` *(inherit from `[run].backend`)*