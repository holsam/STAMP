'''
STAMP: in-process adapter for the stamp-native picker
'''

# Import internal STAMP objects
from stamp.adapters.base import AdapterInputs, AdapterOutput
from stamp.backends.base import RunResult, ToolCommand
from stamp.picking.native import PICKER_NAME, NativePickerConfig, pick_tomogram
from stamp.schemas.picks import RawPick

# _IN_PROCESS_MESSAGE: define a message for if build_command/parse_output are called directly
_IN_PROCESS_MESSAGE = 'stamp-native runs in-process; call run_in_process() rather than build_command()/parse_output()'

# NativePickerAdapter: ToolAdapter surface for the in-process stamp-native picker
class NativePickerAdapter:
    name = PICKER_NAME
    stage = 'C'
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
            raise ValueError('stamp-native requires matched segmentation and raw tomogram paths; pass --raw-tomogram-dir to stamp pick')
        if not (len(inputs.input_paths) == len(inputs.raw_tomogram_paths) == len(inputs.tomogram_ids)):
            raise ValueError('Mismatched input lengths: segmentations, raw tomograms and tomogram IDs must correspond one-to-one')

        voxel_size = inputs.parameters.get('voxel_size_angstrom')
        if voxel_size is None:
            raise ValueError('stamp-native requires voxel_size_angstrom')

        config_fields = {
            key: value
            for key, value in inputs.parameters.items()
            if key in NativePickerConfig.__dataclass_fields__
        }
        config = NativePickerConfig(**config_fields)

        picks: list[RawPick] = []
        for segmentation_path, tomogram_path, tomogram_id in zip(inputs.input_paths, inputs.raw_tomogram_paths, inputs.tomogram_ids):
            picks.extend(pick_tomogram(segmentation_path, tomogram_path, tomogram_id, config))
        return picks
