# STAMP

A workflow for Sub-Tomogram Averaging Membrane Proteins.

<br>

## Overview
Given a directory of segmented MRC volumes (i.e. as produced by [MemBrain-seg](https://github.com/teamtomo/membrain-seg)), `STAMP` runs the following workflow:
\# | Stage | Description
-- | -- | --
1 | **Picking** | Finds candidate particle positions on the membrane surface, with consensus results returned across multiple pickers
2 | **Classification** | Group similar candidate particles by structural similarity (via unsupervised clustering) and produce class averages
3 | **Identification** | Score each class average against a panel of predicted protein structures from proteomics experiments
4 | **Refinement** | Iteratively refine particle orientations and measure resolution

Half-sets and a decoy control (a matched negative-control set of particles run through the pipeline) are used by default.

<br>

## Getting Started
### Requirements
STAMP's only core requirement is Python >=3.14, and the below installation steps should download the necessary Python version (and any libraries) automatically. Additional functionality (e.g. refinement) requires additional tools to be available via a HPC cluster (RELION-5/M) with GPU nodes, however STAMP only submits these jobs to a SLURM scheduler and so can be run from a login node if refinement is required. automatically.

### Installation

STAMP uses [`uv`](https://docs.astral.sh/uv/) as its package manager and build backend. To install:
```sh
# Install STAMP
uv tool install git+https://github.com/holsam/stamp

# Check STAMP is installed
stamp --help
```

Alternatively, this repo can be cloned directly:
```sh
# Clone repo from GitHub
git clone https://github.com/holsam/stamp && cd stamp

# Set up a virtual environment and install STAMP to it
uv sync

# Check STAMP is installed
source .venv/bin/activate
stamp --help
```

For development, the dev dependency group provides `pytest`:
```sh
uv sync --group dev
```

### Quickstart
Given a directory of membrane segmentations (`seg/`) and matching raw tomograms (`tomo/`):
```sh
# 1. Pick candidate particles using STAMP's native picker
stamp pick stamp-native -s seg/ -r tomo/ -o out/pick --voxel-size-a 5.38

# 2. Group picks and generate class averages
stamp classify --particles out/pick/particle_set.json -r tomo/ -o out/classify --voxel-size-a 13.48

# 3. Compare class averages to a candidate protein panel
stamp identify --classes out/classify/class_averages --candidates candidates.yaml -o out/identify -r 25.0

# 4. Refine an identified class with independent half-sets
stamp refine --class-id all --identification out/identify/identification.json --particles out/pick/particle_set.json --class-assignments out/classify/class_assignments.json -r tomo/ -o out/refine
```

Each step, including decoy control generation, can be run from a single configuration file:
```sh
stamap run --config stamp_run.toml
```

<br>

## Documentation

File | Purpose
-- | --
`docs/architecture.md` | STAMP's module layout and key abstractions
`docs/cli.md` | Full CLI reference
`docs/configuration.md` | `stamp run` configuration reference
`docs/workflow.md` | The theory underpinning STAMP
`tests/TESTS.md` | STAMP's test suite structure

<br>

## Licence
STAMP is available under the terms of the MIT License. For further information, see the accompanying [LICENSE file](./LICENSE).