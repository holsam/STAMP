# STAMP - architecture

This document contains information about STAMP's architecture and implementation. For the theory underpinning STAMP, see the [workflow documentation](./workflow.md).

**Contents**
- [Package layout](#package-layout)
    - [Command shape](#command-shape)
- [Adapters & Runners](#adapters--runners)
    - [`ToolAdapter`](#tooladapter)
    - [`Runner`](#runner)
- [Schemas](#schemas)
- [Coordinate conventions](#coordinate-conventions)
- [Picking implementation](#picking-implementation)
    - [Geometric primitives](#geometric-primitives)
    - [`NativePickerConfig`](#nativepickerconfig)
    - [Vesicle labelling](#vesicle-labelling)
    - [Halfset assignment](#halfset-assignment)

<br>

## Package layout

```
src/stamp/
  cli/            CLI Typer definitions (parse args then call commands)
  commands/       per-stage run orchestration via a run_<stage>() function, called by cli/ and run/
  schemas/        pydantic models: config, particles, picks, manifest, provenance, cluster_profile
  adapters/       ToolAdapter implementations: native, relion, m_refine, mock, (base protocol)
  backends/       Runner implementations: local, cluster, mock, (base protocol) + factory
  picking/        pick command scripts: geometry primitives, native picker, vesicle labelling, consensus reconciliation
  decoy/          decoy dataset generation and comparability checks
  classify/       subvolume extraction, feature engineering, clustering, class averaging
  identify/       candidate panel loading, shape fitting, decoy-control statistics
  refine/         FSC computation, half-set guard/tree layout
  run/            stamp_run.toml-driven orchestration, resume state, report generation
  tools/          stamp tools subcommands: diagram generation, shell completion
  utils/          logging, half-set assignment, I/O, plotting, reporting helpers
```

### Command shape
Each stage has the same three-layer shape:

```
cli/<stage>.py        CLI Typer command to parse/validate CLI args and calls commands.<stage>.run_<stage>()
commands/<stage>.py   orchestration: selects adapter + backend, drives the stage end to end
<stage-specific>/     scripts with logic for <stage>
```

`stamp run` is an overall orchestration command, calling each command using parameters defined in a `RunConfig` specified by `stamp_run.toml`.

<br>

## Adapters & Runners
STAMP separates what a tool is from how to run it, as a `ToolAdapter` and `Runner` respectively:
- `ToolAdapter`: builds a `ToolCommand` from `AdapterInputs` and parses a `RunResult` into `AdapterOutput`.
- `Runner`: executes a ToolCommand and returns a `RunResult`

`ToolAdapter`s and `Runner`s do not communicate directly, as the calling command calls the adapter functions and selects a runner independently.

### `ToolAdapter`
```python
class ToolAdapter(Protocol):
    name: str
    stage: str
    mac_compatible: bool        # gates --backend local on MacOS
    requires_gpu: bool          # gates --backend local
    automatable: bool           # whether tool has non-interactive entry point
    batches_natively: bool      # whether one call processes multiple tomograms
    runs_in_process: bool       # whether tool runs in same process or spawns subprocess

    def build_command(self, inputs: AdapterInputs) -> ToolCommand: ...
    def parse_output(self, result: RunResult) -> AdapterOutput: ...
```

Available adapters:
Adapter | File (`src/stamp/adapters/`) | Stage | `mac_compatible` | `requires_gpu` | `runs_in_process`
-- | -- | -- | -- | -- | --
`NativePickerAdapter` (`stamp-native`) | `native.py` | Pick | ✔ | ✘ | ✔
`RelionRefineAdapter` (`relion`) | `relion.py` | Refine | ✘ | ✔ | ✘
`MRefineAdapter` (`m`) | `m_refine.py` | Refine | ✘ | ✔ | ✘
`MockAdapter` | `mock.py` | Any (test adapter) | Configurable | Configurable | Configurable

### `Runner`
```python
class Runner(Protocol):
    def run(self, command: ToolCommand) -> RunResult: ...
```

Available runners:
Runner | `--backend` | File (`src/stamp/backends/`) | Behaviour
-- | -- | -- | --
`LocalRunner` | `local` | `local.py` | Runs `argv` as a real subprocess on current machine
`ClusterRunner` | `cluster` | `cluster.py` | Submits via `sbatch`, using a `ClusterProfile` (partition, time/memory/CPU/GPU defaults, module loads)
`MockRunner` | `mock` | `mock.py` | Returns a mocked successful `RunResult` with no subprocess

Commands will select a `Runner` by mapping to the specified `--backend`. Using `--backend cluster` will raise an error if `sbatch` is not available on `PATH`, and will load the default `ClusterProfile` if none are explicitly passed. Using `--backend local` will raise an error if used with an adapter that requires a GPU or is not Mac-compatible (on MacOS).

<br>

## Schemas
All data which sits at the boundary of a stage (CLI/config input, inter-stage JSONs, provenance sidecars) are implemented using `pydantic`'s `BaseModel` class. Internal data objects use `dataclasses`' `dataclass` instead. 

<br>

## Coordinate conventions
Two conventions for tomogram coordinates are used throughout STAMP:
- **Internal picking/geometry code** works in `(z, y, x)`, matching `mrcfile`/numpy array order.
- **Schema boundaries** work in `(x, y, z)`. Conversion takes place once, during `RawPick` construction.
- **Classification extraction** (`classify/extract.py`) reverses coordinates again at the point of the `map_coordinates` call (tomogram arrays are `(z, y, x)`, particle positions are `(x, y, z)`), and nowhere else.

<br>

## Picking implementation 
### Geometric primitives
`src/stamp/picking/geometry.py` contains several geometric primitives, used by STAMP's native picker:

Function | Role
-- | --
`extract_surface(segmentation, level=0.5)` | Uses marching cubes to convert a binary segmentation into `(vertices, normals)`
`downsample_points(points, normals, spacing_voxels)` | Downsamples a voxel-grid to a target spacing
`sample_along_normals(...)` | Calculates mean tomogram density in a shell offset along each point's normal; `direction` is `+1`/`-1`; out-of-volume samples clamp to the nearest edge (`map_coordinates(mode='nearest')`)
`profile_correlation_scores(...)` | Calculates Pearson correlation of a sign-corrected radial profile against a Gaussian bump at the window's own midpoint
`robust_normalise(...)` | Calculates median/MAD normalisation, global per tomogram
`local_normalise(...)` | Calculates median/MAD normalisation over a k-d tree neighbourhood (`local_radius_angstrom`) if a point has more than `min_local_neighbours` (otherwise falls back to global estimate)
`vesicle_normalise(values, vesicle_ids)` | Calculates per-vesicle median/MAD normalisation if vesicle labels are available (otherwise falls back to pooled/global estimate)
`non_maximum_suppression(...)` | Uses greedy NMS, highest score first, k-d tree exclusion radius
`quaternion_from_reference_to(vector_xyz)` | Constructs a unit quaternion rotating +z onto a given outward vector
`compose_roll_about_normal(...)` | Constructs a quaternion for "roll by angle, then map +z onto the normal"
`exclude_near_boundary(points, shape, margin_voxels)` | Drops points too close to a volume face to sample cleanly
`score_membrane_faces(...)` | Composes above functions to sample both faces across every offset window, scores using both modes, normalise scores, and return per-point results including which window won
`max_order_statistic_offset(n_windows)` | Applies an upward-bias correction for taking the max of `n_windows` independent scores
`beam_angle_deviation_degrees(normal_xyz)` | Derives the angle between an outward normal and the beam-orthogonal (xy) plane; 0 = beam-orthogonal, 90 = beam-aligned (used to flag/drop picks whose orientation the missing wedge makes unreliable)

### `NativePickerConfig`
`picking/native.py` uses a dataclass to store the configuration options for STAMP's native picker. All distances are in Å and are converted to voxels via `config.to_voxels()`.

Field | Default | Meaning
-- | -- | --
`voxel_size_angstrom` | *n/a (required)* | Voxel size of input tomograms in Å
`offset_windows_angstrom` | `((20,50),(40,80),(70,110),(100,160))` | Radial windows searched outward from the surface
`n_samples` | 5 | Number of samples taken per point within each window
`surface_spacing_angstrom` | 20.0 | Target vertex spacing after downsampling
`min_particle_distance_angstrom` | 60.0 | NMS exclusion radius
`n_mad` | 3.0 | Threshold, in robust standard deviations, before the multi-window correction
`density_sign` | -1 | +1 if protein is brighter than background, -1 if darker
`boundary_margin_angstrom` | `None` | Drop points closer than this to a volume face; defaults to the widest window's outer edge
`normalisation` | `'global'` | MAD normalisation method (`'global'` or `'local'`)
`local_radius_angstrom` | 150.0 | Neighbourhood radius for local normalisation
`min_local_neighbours` | 20 | Minimum neighbours before falling back to global
`scoring_mode` | `'mean'` | Scoring mode to use (`'mean'` or `'profile'`)
`profile_width_angstrom` | `None` | Width of the expected profile bump (`None` defaults to a quarter of each window's own range)
`vesicle_labels_mrc` | `None` | Path to EValuator's `label` command output MRC, enabling per-vesicle attribution
`normalise_per_vesicle` | `False` | Normalise scores per vesicle instead of per tomogram (requires `vesicle_labels_mrc`)

### Vesicle labelling
`picking/vesicles.py` adds per-vesicle QC and normalisation when provided with an output MRC from EValuator's `label` command. 

Function | Role
-- | --
`load_vesicle_labels(labels_mrc_path, segmentation_shape)` | Loads the labelled MRC file and checks it has same shape as segmentation file
`vesicle_ids_at(label_volume, points_zyx, tomogram_id)` | Nearest-voxel lookup, returning `'{tomogram_id}:v{label:04d}'` per point (`''` for background)
`vesicle_surface_area_angstrom2(...)` | Per-vesicle membrane surface area, computed from marching-cubes triangle face areas assigned to their nearest labelled voxel (by face centroid)
`summarise_vesicles(picks, areas_by_vesicle, tomogram_id)` | builds a `VesicleSummary` (frozen dataclass: `vesicle_id`, `tomogram_id`, `n_picks`, `surface_area_angstrom2`, `picks_per_1000_angstrom2`) per vesicle

Every kept `RawPick` will contain `vesicle_id`, which is kept through downstream stages.

### Consensus reconciliation
`picking/consensus.py` contains the function `reconcile_picks(...)` for grouping picks across pickers by proximity (complete-linkage agglomerative clustering via a `cKDTree`-backed near-linear implementation). Two modes are available: 
- `'intersection'`: groups survive only if every configured picker contributed to it
- `'union'`: groups survive if at least one picker contributed to it

Each reconciled pick's position is the centroid of its group, and `source_picker` records all contributing pickers (joined with '+').

With the current single picker (STAMP's native picker), `reconcile_picks(...)` is reduced to deduplicating near-identical picks within `distance_threshold`.

### Halfset assignment
`utils/halfset.py` contains functions for implementing the half-set separation:

Function | Role
-- | --
`assign_half_sets()` | Runs once at the end of picking to assign RawPicks to a given half-set; deterministic for a given seed and set of particle IDs; splits picks as close to 50/50 as possible (with remaining particles assigned to halfset A)
`validate_single_half_set()` | Called at the start of any stage that must process halves independently (classify, identify, refine) and raises an error if the input mixes A and B
