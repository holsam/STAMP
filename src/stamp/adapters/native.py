'''
STAMP: in-process adapter for the stamp-native picker
'''

# Import external dependencies
import json
from dataclasses import dataclass, replace
from pathlib import Path

# Import internal STAMP objects
from stamp.adapters.base import AdapterInputs, AdapterOutput
from stamp.backends.base import RunResult, ToolCommand
from stamp.picking.native import PICKER_NAME, NativePickerConfig, pick_tomogram
from stamp.schemas.picks import RawPick
from stamp.utils.errors import StampPipelineError, StampValidationError
from stamp.utils.log import log
from stamp.utils.parallel import run_parallel

# _IN_PROCESS_MESSAGE: define a message for if build_command/parse_output are called directly
_IN_PROCESS_MESSAGE = 'stamp-native runs in-process; call run_in_process() rather than build_command()/parse_output()'

# _PickJob: one tomogram's resolved inputs
@dataclass(frozen=True)
class _PickJob:
    segmentation_path: Path
    tomogram_path: Path
    tomogram_id: str
    config: NativePickerConfig

# _pick_one: multiprocessing worker entry point
def _pick_one(job: _PickJob) -> tuple[str, list[RawPick]]:
    picks = pick_tomogram(job.segmentation_path, job.tomogram_path, job.tomogram_id, job.config)
    return job.tomogram_id, picks

# NativePickerAdapter: ToolAdapter surface for the in-process stamp-native picker
class NativePickerAdapter:
    name = PICKER_NAME
    stage = 'pick'
    mac_compatible = True
    requires_gpu = False
    automatable = True
    batches_natively = True
    runs_in_process = True

    def build_command(self, inputs: AdapterInputs) -> ToolCommand:
        raise NotImplementedError(_IN_PROCESS_MESSAGE)

    def parse_output(self, result: RunResult) -> AdapterOutput:
        raise NotImplementedError(_IN_PROCESS_MESSAGE)

    def run_in_process(self, inputs: AdapterInputs) -> list[RawPick]:
        if not inputs.input_paths or not inputs.raw_tomogram_paths:
            raise StampPipelineError('stamp-native requires matched segmentation and raw tomogram paths; pass --raw-tomogram-dir to stamp pick')
        if not (len(inputs.input_paths) == len(inputs.raw_tomogram_paths) == len(inputs.tomogram_ids)):
            raise StampPipelineError('Mismatched input lengths: segmentations, raw tomograms and tomogram IDs must correspond one-to-one')

        voxel_size = inputs.parameters.get('voxel_size_angstrom')
        if voxel_size is None:
            raise StampPipelineError('stamp-native requires voxel_size_angstrom')

        config_fields = {
            key: value
            for key, value in inputs.parameters.items()
            if key in NativePickerConfig.__dataclass_fields__
        }
        config = NativePickerConfig(**config_fields)
        vesicle_labels_mrc_by_tomogram = inputs.parameters.get('vesicle_labels_mrc_by_tomogram', {})
        inputs.output_directory.mkdir(parents=True, exist_ok=True)

        total = len(inputs.input_paths)
        n_workers = int(inputs.parameters.get('n_workers', 1))
        log.progress(f'Running stamp-native over {total} tomogram(s)' + (f' ({n_workers} workers)' if n_workers > 1 else ''))
        jobs: list[_PickJob] = []
        for segmentation_path, tomogram_path, tomogram_id in zip(inputs.input_paths, inputs.raw_tomogram_paths, inputs.tomogram_ids):
            tomogram_config = config
            labels_path = vesicle_labels_mrc_by_tomogram.get(tomogram_id)
            if labels_path is not None:
                tomogram_config = replace(config, vesicle_labels_mrc=Path(labels_path))
            elif config.normalise_per_vesicle:
                log.warning(f'{tomogram_id}: no vesicle labels found, disabling normalise_per_vesicle for this tomogram')
                tomogram_config = replace(config, normalise_per_vesicle=False, vesicle_labels_mrc=None)
            jobs.append(_PickJob(segmentation_path, tomogram_path, tomogram_id, tomogram_config))
    
        picks: list[RawPick] = []
        def _on_success(job: _PickJob, result: tuple[str, list[RawPick]]) -> None:
            tomogram_id, tomogram_picks = result
            log.debug(f'{tomogram_id}: {len(tomogram_picks)} raw picks')
            cache_path = inputs.output_directory / f'{tomogram_id}.json'
            cache_path.write_text(json.dumps([pick.model_dump() for pick in tomogram_picks], indent=2))
            picks.extend(tomogram_picks)
        def _on_error(job: _PickJob, exc: Exception) -> None:
            if isinstance(exc, StampValidationError):
                log.error(f'{job.tomogram_id}: {exc}')
            else:
                raise exc
        run_parallel(jobs, _pick_one, max_workers=n_workers, label='stamp-native', on_success=_on_success, on_error=_on_error)
        return picks
