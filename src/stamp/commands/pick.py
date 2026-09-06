'''
STAMP: consensus picking logic
'''

# Import external dependencies
import platform
from pathlib import Path

# Import internal STAMP objects
from stamp.adapters.base import AdapterInputs, ToolAdapter
from stamp.adapters.mock import get_mock_adapter
from stamp.adapters.native import NativePickerAdapter
from stamp.backends.base import Runner, ToolCommand
from stamp.backends.local import LocalRunner
from stamp.backends.mock import MockRunner
from stamp.picking.consensus import build_particle_set, reconcile_picks
from stamp.picking.native import PICKER_NAME as NATIVE_PICKER_NAME
from stamp.run.state import stage_dir
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.picks import RawPick
from stamp.utils.io import write_sidecar

# REAL_ADAPTERS: dictionary containing all implemented pickers
REAL_ADAPTERS: dict[str, ToolAdapter] = {
    NATIVE_PICKER_NAME: NativePickerAdapter(),
}

# _execute: build a command, run it, and parse the resulting RawPicks
def _execute(adapter: ToolAdapter, inputs: AdapterInputs, runner: Runner) -> list[RawPick]:
    command = adapter.build_command(inputs)
    result = runner.run(command)
    if not result.succeeded:
        print(f'{adapter.name} failed (exit {result.exit_code}): {result.stderr}')
        raise SystemExit(1)
    output = adapter.parse_output(result)
    return [RawPick(**raw) for raw in output.parsed.get('picks', [])]

# _select_runner: return the Runner for the requested backend
def _select_runner(backend: str) -> Runner:
    if backend == 'mock':
        return MockRunner()
    if backend == 'local':
        return LocalRunner()

# _select_adapter: return the adapter for a picker name, rejecting Mac-incompatible tools on macOS
def _select_adapter(picker_name: str, backend: str) -> ToolAdapter:
    if backend == 'mock':
        return get_mock_adapter(picker_name)

    adapter = REAL_ADAPTERS.get(picker_name)
    if platform.system() == 'Darwin' and not adapter.mac_compatible:
        print(f'{picker_name} cannot run on macOS (needs CUDA). Use {NATIVE_PICKER_NAME}, or run on the cluster.')
        raise SystemExit(1)
    return adapter

# _load_manifests: match segmentations to raw tomograms by filename stem
def _load_manifests(
    segmentation_dir: Path, raw_tomogram_dir: Path, voxel_size_angstrom: float
) -> list[TomogramManifest]:
    '''Unmatched segmentations are skipped with a warning rather than failing the run'''
    raw_by_stem = {path.stem: path for path in sorted(raw_tomogram_dir.glob('*.mrc'))}
    manifests: list[TomogramManifest] = []
    for segmentation_path in sorted(segmentation_dir.glob('*.mrc')):
        raw_path = raw_by_stem.get(segmentation_path.stem)
        if raw_path is None:
            print(f"  warning: no raw tomogram matching '{segmentation_path.stem}', skipping")
            continue
        manifests.append(
            TomogramManifest(
                tomogram_id=segmentation_path.stem,
                segmentation_path=segmentation_path,
                raw_tomogram_path=raw_path,
                voxel_size_angstrom=voxel_size_angstrom,
            )
        )
    return manifests

# _run_picker: dispatch to the in-process, batched, or per-tomogram path
def _run_picker(
    adapter: ToolAdapter,
    manifests: list[TomogramManifest],
    picker_output_dir: Path,
    parameters: dict,
    runner: Runner,
    backend: str,
) -> list[RawPick]:
    if backend != 'mock' and getattr(adapter, 'runs_in_process', False):
        inputs = AdapterInputs(
            input_paths=[m.segmentation_path for m in manifests],
            raw_tomogram_paths=[m.raw_tomogram_path for m in manifests],
            tomogram_ids=[m.tomogram_id for m in manifests],
            output_directory=picker_output_dir,
            parameters=parameters,
        )
        return adapter.run_in_process(inputs)  # type: ignore[attr-defined]

    if backend == 'mock' or adapter.batches_natively:
        inputs = AdapterInputs(
            input_paths=[m.segmentation_path for m in manifests],
            raw_tomogram_paths=[m.raw_tomogram_path for m in manifests],
            tomogram_ids=[m.tomogram_id for m in manifests],
            output_directory=picker_output_dir,
            parameters=parameters,
        )
        return _execute(adapter, inputs, runner)

    collected: list[RawPick] = []
    for manifest in manifests:
        inputs = AdapterInputs(
            input_paths=[manifest.segmentation_path],
            raw_tomogram_paths=[manifest.raw_tomogram_path],
            tomogram_ids=[manifest.tomogram_id],
            output_directory=picker_output_dir / manifest.tomogram_id,
            parameters=parameters,
        )
        collected.extend(_execute(adapter, inputs, runner))
    return collected

# run_pick: pick command orchestration
def run_pick(
    picker_names,
    segmentation_dir,
    raw_tomogram_dir,
    output_dir,
    voxel_size_angstrom,
    extra_params,
    consensus_rule,
    distance_threshold,
    half_set_seed,
    backend,
) -> None:
    # Load manifests to check for matching files
    manifests = _load_manifests(segmentation_dir, raw_tomogram_dir, voxel_size_angstrom)
    if not manifests:
        print(f'No .mrc files in {segmentation_dir} with a matching stem in {raw_tomogram_dir}')
        raise SystemExit(1)
    print(f'Matched {len(manifests)} tomograms/segmentations.')

    runner = _select_runner(backend)
    output_dir.mkdir(parents=True, exist_ok=True)

    picks_by_picker: dict[str, list[RawPick]] = {}
    for picker_name in picker_names:
        adapter = _select_adapter(picker_name, backend)
        picker_output_dir = output_dir / 'raw' / picker_name
        parameters = {
            'voxel_size_angstrom': voxel_size_angstrom,
            **extra_params.get(picker_name, {}),
        }

        print(f'Running {picker_name}...')
        picks_by_picker[picker_name] = _run_picker(adapter, manifests, picker_output_dir, parameters, runner, backend)
        print(f'  {picker_name}: {len(picks_by_picker[picker_name])} raw picks')

    all_reconciled: list[RawPick] = []
    for manifest in manifests:
        all_reconciled.extend(
            reconcile_picks(
                picks_by_picker=picks_by_picker,
                consensus_rule=consensus_rule,  # type: ignore[arg-type]
                distance_threshold=distance_threshold,
                tomogram_id=manifest.tomogram_id,
            )
        )

    if not all_reconciled:
        print('No particles survived reconciliation. If using stamp-native, try lowering n_mad or check density_sign matches your tomograms (-1 for conventional dark-protein contrast).')
        raise SystemExit(1)

    particle_set = build_particle_set(
        reconciled_picks=all_reconciled,
        consensus_rule=consensus_rule,  # type: ignore[arg-type]
        contributing_pickers=picker_names,
        half_set_seed=half_set_seed,
    )

    particle_set_path = output_dir / 'particle_set.json'
    particle_set_path.write_text(particle_set.model_dump_json(indent=2))

    write_sidecar(
        output_dir,
        stage='pick',
        tool='+'.join(picker_names),
        tool_version=None,
        parameters={
            'pickers': picker_names,
            'consensus_rule': consensus_rule,
            'distance_threshold': distance_threshold,
            'half_set_seed': half_set_seed,
            'voxel_size_angstrom': voxel_size_angstrom,
            'backend': backend,
            'picker_params': extra_params,
        },
        inputs=[(f'segmentation:{m.tomogram_id}', m.segmentation_path) for m in manifests]
        + [(f'raw_tomogram:{m.tomogram_id}', m.raw_tomogram_path) for m in manifests],
    )

    print(f'Wrote {len(particle_set.particles)} consensus particles to {particle_set_path}')

# build_pick_commands: the ToolCommand `stamp pick` would run for the real track, without running it
def build_pick_commands(config, output_dir: Path) -> list[ToolCommand]:
    target = stage_dir(output_dir, 'real', 'pick')
    argv = [
        'stamp', 'pick', ','.join(config.stage.pick.pickers),
        '--seg-dir', str(config.run.segmentation_dir),
        '--raw-dir', str(config.run.raw_tomogram_dir),
        '--out-dir', str(target),
        '--voxel-size-a', str(config.run.voxel_size_angstrom),
        '--consensus-rule', config.stage.pick.consensus_rule,
        '--distance-threshold', str(config.stage.pick.distance_threshold),
        '--half-set-seed', str(config.stage.pick.half_set_seed),
        '--backend', 'local',
    ]
    return [ToolCommand(tool='pick', argv=argv, working_directory=target, output_paths=[target / 'particle_set.json'])]