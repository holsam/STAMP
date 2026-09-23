# STAMP - CLI

This document outlines the available command-line options for each STAMP command.

**Contents**
- [Global logging options](#global-logging-options)
- [`stamp pick`](#stamp-pick)
- [`stamp decoy`](#stamp-decoy)
    - [`stamp decoy` methods](#stamp-decoy-methods)
- [`stamp classify`](#stamp-classify)
- [`stamp identify`](#stamp-identify)
- [`stamp refine`](#stamp-refine)
- [`stamp run`](#stamp-run)
- [`stamp config`](#stamp-config)
- [`stamp filter`](#stamp-filter)
- [`stamp plot`](#stamp-plot)
- [`stamp tools`](#stamp-tools)
    - [`stamp tools diagram`](#stamp-tools-diagram)
    - [`stamp tools completion`](#stamp-tools-completion)


<br>

## Global logging options

Logging configuration is set using global options before any STAMP command:
```
stamp [-d/--directory PATH] [-m/--mode append|overwrite] [-q/--quiet] [-v/--verbose] <command> ...
```

Available options:
Option | Default | Meaning | Notes
-- | -- | -- | --
`-d, --directory` | Current working directory | Directory to write STAMP log file to |
`-m, --mode` | `append` | If a log file already exists in the directory, `append` or `overwrite` it |
`-q, --quiet` | *n/a* | Decreases verbosity (repeatable, capped at 2) | Mutually exclusive with `-v/--verbose`
`-v, --verbose` | *n/a* | Increases verbosity (repeatable, capped at 2) | Mutually exclusive with `-q/--quiet`

<br>

## `stamp pick`

```
stamp pick <pickers> -s SEG_DIR -r RAW_DIR [options]
```

Argument/Option | Default | Use
-- | -- | --
`pickers` (positional) | *n/a (required)* | Comma-separated names of pickers to use
`-s, --seg-dir` | *n/a (required)* | Directory containing segmented MRC files
`-r, --raw-dir` | *n/a (required)* | Directory of raw tomogram MRCs
`-o, --out-dir` | `.` | Output directory (outputs will be saved to `<dir>/stamp/pick`)
`--voxel-size-a` | `None` *(read from MRC file header)* | Voxel size in Å
`--picker-params` | `'{}'` | JSON of per-picker parameters, e.g. `'{"stamp-native": {"n_mad": 3.5}}'`
`--consensus-rule` | `union` | Rule to use for reconciling picks (`union` or `intersection`)
`--distance-threshold` | `15.0` | Voxel distance within which picks are merged
`--half-set-seed` | `0` | Seed for half-set assignment
`--backend` | `local` | Backend to use for picking (`local` or `mock`)
`--plots/--no-plots` | `--plots` | Write consensus-pick position plots
`--pick-plot-style` | `segmented` | Style of consensus-pick position plots (`segmented` or `none`)
`--plot-format` | `tiff` | Consensus-pick position plot file format (`png` or `jpg` or `tiff` or `svg`)
`--pick-zstack-movie/--no-pick-zstack-movie` | `--pick-zstack-movie` | Write a per-tomogram movie through Z with picks highlighted
`--pick-plot-3d/--no-pick-plot-3d` | `--no-pick-plot-3d` | Write a static 3D pick visualisation per tomogram
`--max-beam-angle-deviation` | `None` | Drop particles whose normal deviates from the beam-orthogonal plane by more than this many degrees
`--vesicle-labels-mrc` | `None` | Path to an MRC file produced by EValuator's `label` command, or a directory to be matched by filename stem
`--normalise-per-vesicle` | `False` | Normalise `stamp-native` scores per vesicle instead of per tomogram

<br>

## `stamp decoy`

```
stamp decoy -o OUT_DIR --voxel-size-a V [options]
```

Option | Default | Use
-- | -- | --
`--voxel-size-a` | *n/a (required)* | Voxel size in Å
`-o, --out-dir` | `.` | Output directory (outputs will be saved to `<dir>/stamp/decoy`)
`--method` | `rejected-surface` | Decoy generation method to use (`rejected-surface` or `shifted` or `synthetic-noise`) *[(see below)](#stamp-decoy-methods)*
`--real-particle-set` | *n/a (required unless `--method synthetic-noise`)* | Path to `particle_set.json` from `stamp pick`
`-s, --seg-dir` | *n/a (required unless `--method synthetic-noise`)* | Directory containing segmented MRC files
`-r, --raw-dir` | *n/a (required unless `--method synthetic-noise`)* | Directory of raw tomogram MRCs
`--picker-params` | `'{}'` | JSON of native-picker parameters (must match the values used for `stamp pick` so decoys are drawn from the same candidate pool)
`--n-decoys-per-tomogram` | `50` | Number of decoy positions to pick per tomogram
`--min-distance-from-real-a` | `100.0` | Minimum distance from a real pick in Å for `--method rejected-surface`
`--min-pick-distance-a` | `60.0` | Minimum distance from a real pick in Å for `--method shifted`
`--min-shift-a` | `200.0` | Minimum displacement in Å for `--method shifted`
`--max-shift-a` | `600.0` | Maximum displacement in Å for `--method shifted`
`--n-synthetic-tomograms` | `3` | Number of volumes to produce for `--method synthetic-noise`
`--synthetic-shape` | `'200,200,200'` | Volume shape (in `z,y,x`) to use for `--method synthetic-noise`
`--seed` | `0` | Seed to use for sampling and half-set assignment
`-backend` | `local` | Backend to use for picking (`local` or `mock`)
`--plots/--no-plots` | `--plots` | Write decoy position plots
`--pick-plot-style` | `segmented` | Style of decoy position plots (`segmented` or `none`)
`--plot-format` | `tiff` | Decoy plot file format (`png` or `jpg` or `tiff` or `svg`)
`--pick-zstack-movie/--no-pick-zstack-movie` | `--pick-zstack-movie` | Write a per-tomogram movie through Z with picks highlighted
`--pick-plot-3d/--no-pick-plot-3d` | `--no-pick-plot-3d` | Write a static 3D pick visualisation per tomogram

### `stamp decoy` methods
`stamp decoy` provides three methods for generating decoy picks:

Method | What it controls for | Notes
-- | -- | --
`rejected-surface` (default) | Structure emerging from real membrane positions with no particle | Reuses pick-stage geometry
`shifted` | Structure emerging from the tomogram at large, independent of the membrane |
`synthetic-noise` | Structure emerging from pure noise |

<br>

## `stamp classify`

```
stamp classify --particles PARTICLES.json -r RAW_DIR -o OUT_DIR --voxel-size-a V [options]
```

Option | Default | Use
-- | -- | --
`--particles` | *n/a (required)* | Path to `particle_set.json` from `stamp pick`, or `decoy_particle_set.json` from `stamp decoy`
`-r, --raw-dir` | *n/a (required)* | Directory of raw tomogram MRCs
`--voxel-size-a` | *n/a (required)* | Voxel size in Å
`-s, --seg-dir` | `None` | Directory containing segmented MRC files
`-o, --out-dir` | `.` | Output directory (outputs will be saved to `<dir>/stamp/classify` or `<dir/stamp/classify_decoy>`)
`--box-length-a` | `300.0` | Extraction box edge length in Å (roughly 2x the largest expected particle)
`--n-bins` | `12` | Radial bins in the rotational average
`--method` | `hbdscan` | Clustering method to use (`hbdscan` or `kmeans`)
`--min-cluster-size` | `20` | HDBSCAN minimum cluster size
`--n-clusters` | `5` | Number of KMeans clusters to use
`--n-components` | `20` | Number of PCA components to use
`--strict-halfset-independence` | `True` | Cluster each halfset separately and match clusters afterwards, rather than clustering the combined set
`--seed` | `0` | Seed to use for PCA/KMeans clustering
`--inplane-alignment/--no-inplane-alignment` | `--inplane-alignment` | Estimate per-particle in-plane angle for averaging
`--inplane-step-deg` | `10.0` | In-plane search step in degrees
`--inplane-iterations` | `3` | Number of iterations to use for reference refinement
`--azimuthal-modes` | `4` | Highest azimuthal Fourier mode retained; `0` = pure rotational averaging, `4` = capture C4 symmetry
`--min-radius-fraction` | `0.25` | Radial bins below this fraction of box radius are excluded from azimuthal features
`--n-azimuthal-samples` | `64` | Number of azimuthal sampling points (must be >= `2*(modes+1)`)
`--plots/--no-plots` | `--plots` | Write embedding and class-average plots
`--plot-format` | `tiff` | Embedding and class average plot file format (`png` or `jpg` or `tiff` or `svg`)


<br>

## `stamp identify`

```
stamp identify --classes CLASSES_DIR --candidates CANDIDATES.yaml -o OUT_DIR -r RESOLUTION [options]
```

Option | Default | Use
-- | -- | --
`--classes` | *n/a (required)* | Directory of class-average MRCs from `stamp classify`
`--candidates` | *n/a (required)* | Path to YAML file containing candidates from proteomics
`-r, --resolution` | *n/a (required)* | Class-average resolution in Å, used for low-pass filtering
`-o, --out-dir` | `.` | Output directory (outputs will be saved to `<dir>/stamp/identify`)
`--decoy-classes` | `None` | Directory containing decoy class averages from `stamp decoy`
`--backend` | `local` | Backend to use for identification (`local` or `mock`)
`--fitter` | `native` | Fitting tool to use
`--fetch-missing` | `False` | Fetch model from AlphaFold when a candidate has no local `structure_path`
`--plots/--no-plots` | `--plots` | Write fit-score heatmap and decoy-control histogram plots
`--plot-format` | `tiff` | Fit-score heatmap and decoy-control histogram plot file format (`png` or `jpg` or `tiff` or `svg`)

<br>

## `stamp refine`

```
stamp refine --class-id ID --identification ID.json --particles P.json --class-assignments CA.json -r RAW_DIR [options]
```

Option | Default | Use
-- | -- | --
`--class-id` | *n/a (required)* | ID for class to refine, or `all` for every identified class
`--identification` | *n/a (required)* | Path to `identification.json` from `stamp identify`
`--particles` | *n/a (required)* | Path to `particle_set.json` from `stamp pick`
`--class-assignments` | *n/a (required)* | Path to `class_assignments.json` from `stamp classify`
`-r, --raw-dir` | *n/a (required)* | Directory of raw tomogram MRCs
`-o, --out-dir` | `.` | Output directory (outputs will be saved to `<dir>/stamp/refine`)
`--voxel-size-a` | `None` | Voxel size in Å
`--tool` | `relion` | Refinement tool to use (`relion` or `m`)
`--mask` | `None` | Path to MRC file to use as mask
`--iterations` | `5` | Number of refinement iterations to use
`--backend` | `local` | Backend to use for refinement (`local` or `mock`)
`--combined-halfset` | `False` | Use a single combined refinement step with an internal split, instead of two independent trees
`--plots/--no-plots` | `--plots` | Write FSC curve plots
`--plot-format` | `tiff` | FSC curve plot file format  (`png` or `jpg` or `tiff` or `svg`)

<br>

## `stamp run`

```
stamp run --config stamp_run.toml [options]
```

Option | Default | Use
-- | -- | --
`--config` | *n/a (required)* | `stamp_run.toml` path
`--force` | `False` | Re-run every stage, ignoring resume state
`--from` | `None` | Re-run from this stage onward (`pick`, `classify`, `identify`, `refine`)
`--no-decoy` | `False` | Skip the decoy control track
`--backend` | `None` | Override configuration files's default backend

See the [configuration documentation](./configuration.md) for information about `stamp_run.toml`.

<br>

## `stamp config`

### `stamp config init`
```
stamp config init --path PATH [options]
```

Create an annotated, blank `stamp_run.toml` file.

Argument/Option | Default | Use
-- | -- | --
`--path` | `.` | Directory to write `stamp_run.toml` into, or path to `stamp_run.toml`
`-f, --force` | `False` | Overwrite an existing `stamp_run.toml`

### `stamp config edit`
```
stamp config edit <path> [options]
```

Edit a `stamp_run.toml` file directly in terminal.

Argument/Option | Default | Use
-- | -- | --
`path` (positional) | `.` | Directory containing `stamp_run.toml`, or path to `stamp_run.toml`
`--editor` | `None` *(falls back to `$EDITOR`, then `vi`)* | Editor to use instead of `$EDITOR`

### `stamp config show`
```
stamp config show <path>
```

Print the contents of a `stamp_run.toml` to the terminal.

Argument/Option | Default | Use
-- | -- | --
`path` (positional) | `.` | Directory containing `stamp_run.toml`, or path to `stamp_run.toml`

<br>

## `stamp filter`
```
stamp filter filter <particle_set.json> -s SEG_DIR -r RAW_DIR
```

Argument/Option | Default | Use
-- | -- | --
`particle_set` (positional) | *n/a (required)* | `particle_set.json` from `stamp pick`
`-s, --seg-dir` | `None` | Directory of segmentation MRCs, matched to particles by `tomogram_id`
`-r, --raw-dir` | `None` | Directory of raw tomogram MRCs, matched to particles by `tomogram_id`

<br>

## `stamp plot`

### `stamp plot pick`
```
stamp plot pick <particle_set.json> [options]
```

Renders position plots for an already-picked particle set, without re-running picking.

Argument/Option | Default | Use
-- | -- | --
`particle_set` (positional) | *n/a (required)* | `particle_set.json` (from `stamp pick`) or `decoy_particle_set.json` (from `stamp decoy`)
`-s, --seg-dir` | `None` | Directory of segmented MRC files
`-r, --raw-dir` | `None` | Directory of raw tomogram MRCs
`-o, --out-dir` | `.` | Directory to write plots to
`--pick-plot-style` | `segmented` | Plot style to use (`segmented` or `none`)
`--plot-format` | `tiff` | Image format for static plots (`png` or `jpg` or `tiff` or `svg`)
`--pick-zstack-movie/--no-pick-zstack-movie` | `--pick-zstack-movie` | Write a per-tomogram movie through Z with picks highlighted
`--pick-plot-3d/--no-pick-plot-3d` | `--no-pick-plot-3d` | Write a static 3D pick visualisation per tomogram
`--tomogram-ids` | `None` | Comma-separated tomogram IDs to plot. Mutually exclusive with `--n-tomograms`
`--n-tomograms` | `None` | Randomly select this many tomograms to plot. Mutually exclusive with `--tomogram-ids`
`--seed` | `0` | Seed for random tomogram selection
`-n, --n-processes` | `1` | Processes to use for plot rendering (1 = sequential)

<br>

## `stamp tools`

### `stamp tools diagram`

Generate diagrams of STAMP workflows. Run `stamp tools diagram --help` for the current subcommand list.

### `stamp tools completion`

Install or print `stamp` shell completion.
