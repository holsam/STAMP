# STAMP test suite
This document outlines the STAMP test suite: its structure, coverage and known gaps.

**Contents**
- [Running the test suite](#running-the-test-suite)
- [Test suite structure](#test-suite-structure)
    - [Integration tests](#testsintegration)
    - [Unit tests](#testsunit)
    - [Shared fixtures](#testsconftestpy)
- [Known gaps](#known-gaps)

<br>

## Running the test suite
STAMP uses [`pytest`](https://docs.pytest.org) as its testing framework. To run the full suite:

```sh
uv run pytest
```

To run only the unit or integration tests:
```sh
# Unit tests only
uv run pytest tests/unit/

# Integration tests only
uv run pytest tests/integration/
```

<br>

## Test suite structure
STAMP's tests are organised in the following structure:
```
tests/
    ├─ integration/     # end-to-end cli tests
    ├─ unit/            # isolated tests for specific functions
    └─ conftest.py      # shared fixtures
```

### `tests/integration/`
Each STAMP command has an integration test file, which uses `typer`'s `CliRunner` to test the actual CLI against synthetic data.

File | Covers
-- | --
`test_classify_command.py` | `stamp classify` command
`test_decoy_command.py` | `stamp decoy` command (including each decoy method)
`test_identify_command.py` | `stamp identify` command (including decoy-control comparison)
`test_pick_command.py` | `stamp pick` command
`test_plot_command.py` | `stamp plot` command group
`test_refine_command.py` | `stamp refine` command (including half-set tree layout)
`test_run_command.py` | `stamp run` STAMP pipeline from a `stamp_run.toml` configuration file (including resume/`--from` behaviour and `--no-decoy`)

### `tests/unit/`
Each module (directory) within `src/stamp` has a unit test file, containing a test class for each file within the module.

File | Covers
-- | --
`test_adapters.py` | `ToolAdapter` command building, output parsing, `runs_in_process` behaviour
`test_backends.py` | `select_runner`, `check_backend_supports`, `LocalRunner`/`ClusterRunner`/`MockRunner`
`test_classify.py` |  Extraction, coordinate-convention correctness, rotational/azimuthal features, clustering, class averages
`test_cli.py` | Typer app wiring
`test_config.py` | Configuration management functions (`init_config`, `edit_config`, `show_config`), `_resolve_config_path`
`test_decoy.py` | Decoy generation methods (`rejected-surface`, `shifted`, `synthetic-noise`), `assert_comparable`
`test_filter.py` | Particle set filter state functions
`test_identify.py` | Candidate panel loading, fitting, ranking, decoy-control evaluation
`test_picking.py` | Native picker geometry: surface extraction, ray sampling, scoring, normalisation, NMS, quaternion orientation, coordinate-order correctness
`test_refine.py` | Half-set splitting/guards, FSC computation, refine output tree
`test_run.py` | Overall orchestration: stage sequencing, resume/`--force`/`--from`, decoy track wiring
`test_schemas.py` | Pydantic model validation (`Particle`, `RawPick`, `RunConfig`, etc.), including rejection of bad input
`test_utils.py` | Half-set assignment determinism, I/O helpers, logging setup

Anything that would require external tools uses the `--backend mock` to return mocked output for testing purposes.

### `tests/conftest.py`
Three shared fixtures are defined in `conftest.py`:

Fixture | Details
-- | -- 
`make_dataset` | Writes 12 synthetic tomogram/segmentation pairs to the given directory. It also writes two PDB structures (`small.pdb`, `large.pdb`) and `candidates.yaml` for testing `stamp identify`.
`write_config` | Writes a `stamp_run.toml` configuration file, based on outputs of `make_dataset`.
`run_mock_pipeline` | Calls the other two fixtures and `stamp run`

<br>

## Known gaps
- Test data is synthetic. Due to upload size constratints, a deterministic generator is used to create synthetic data for test files which is an 'idealised' version of real tomogram/segmentation data.
- Adapter testing only test command construction, not actual execution.
