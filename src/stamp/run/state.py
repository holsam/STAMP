'''
STAMP: stage-level resume state for `stamp run`
'''

# Import external dependencies
import json
from pathlib import Path

# Import STAMP objects
from stamp.schemas.config import RunConfig
from stamp.utils.log import log
from stamp.utils.io import resolve_output_dir

# STAGE_ORDER: pipeline stages in order
STAGE_ORDER = ['pick', 'classify', 'identify', 'refine']
_STATE_NAME = 'run_state.json'

# _state_path: return path to run state file
def _state_path(output_dir: Path) -> Path:
    return output_dir / 'stamp' / _STATE_NAME

#  _read_state: read state file
def _read_state(output_dir: Path) -> dict:
    path = _state_path(output_dir)
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning(f'{path} is not valid JSON, treating run state as empty')
            return {}
    return {}

# _write_state: write the state file
def _write_state(output_dir: Path, state: dict) -> None:
    _state_path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    _state_path(output_dir).write_text(json.dumps(state, indent=2))

# stage_dir: output directory for one track's stage
def stage_dir(output_dir: Path, track: str, stage: str) -> Path:
    command = 'decoy' if (stage == 'pick' and track == 'decoy') else stage
    return resolve_output_dir(output_dir, command, track if stage == 'classify' else None)

# mark_complete: record a stage as finished
def mark_complete(output_dir: Path, track: str, stage: str) -> None:
    log.debug(f'Marking {track}/{stage} complete')
    state = _read_state(output_dir)
    state.setdefault(track, {})[stage] = True
    _write_state(output_dir, state)

# is_complete: a stage counts as done when its state flag is set and its sidecar exists
def is_complete(output_dir: Path, track: str, stage: str) -> bool:
    state = _read_state(output_dir)
    if not state.get(track, {}).get(stage):
        return False
    return (stage_dir(output_dir, track, stage) / 'params.toml').is_file()

# stages_to_run: the stages still needing execution
def stages_to_run(
    config: RunConfig,
    output_dir: Path,
    force: bool,
    from_stage: str | None,
) -> list[str]:
    stop_after = config.run.stop_after or STAGE_ORDER[-1]
    planned = STAGE_ORDER[: STAGE_ORDER.index(stop_after) + 1]
    if force:
        log.debug(f'stages_to_run: force={force}, from_stage={from_stage!r} -> {planned}')
        return planned
    if from_stage:
        result = planned[planned.index(from_stage):]
        log.debug(f'stages_to_run: force={force}, from_stage={from_stage!r} -> {result}')
        return result

    decoy_stages = {'pick', 'classify'}

    def incomplete(stage: str) -> bool:
        if not is_complete(output_dir, 'real', stage):
            return True
        if not config.decoy.enabled:
            return False
        if stage in decoy_stages and not is_complete(output_dir, 'decoy', stage):
            return True
        # identify consumes the decoy class averages
        return stage == 'identify' and not is_complete(output_dir, 'decoy', 'classify')

    result = [stage for stage in planned if incomplete(stage)]
    log.debug(f'stages_to_run: force={force}, from_stage={from_stage!r} -> {result}')
    return result
