'''
STAMP: subprocess adapter for an optional M multi-particle refinement pass
'''

# Import external dependencies
from pathlib import Path

# Import STAMP objects
from stamp.adapters.base import AdapterInputs, AdapterOutput
from stamp.backends.base import RunResult, ToolCommand

# _M_ENTRYPOINT: unconfirmed M CLI signature, isolated to one constant
_M_ENTRYPOINT = ('MTools', 'refine')

# MRefineAdapter: command-build-only M pass over one half's RELION output
class MRefineAdapter:
    name = 'm'
    stage = 'refine'
    mac_compatible = False
    requires_gpu = True
    automatable = True
    batches_natively = True
    runs_in_process = False

    # build_command: M multi-particle refinement over the RELION output for one half
    def build_command(self, inputs: AdapterInputs) -> ToolCommand:
        '''M multi-particle refinement over the RELION output for one
        half. Population/species config shape from published docs;
        confirm on the cluster.'''
        if len(inputs.input_paths) != 1:
            raise ValueError('m refine takes one input: this half\'s RELION output directory')
        relion_dir = inputs.input_paths[0]
        population = inputs.output_directory / 'population.settings'
        refined_map = inputs.output_directory / 'm_class001.mrc'
        argv = [
            *_M_ENTRYPOINT,
            '--population', str(population),
            '--source', str(relion_dir),
            '--angpix', str(inputs.parameters['voxel_size_angstrom']),
            '--iterations', str(inputs.parameters.get('iterations', 3)),
            '--refine_particles', '--refine_ctf',
            '--out', str(refined_map),
        ]
        return ToolCommand(
            tool=self.name,
            argv=argv,
            working_directory=inputs.output_directory,
            output_paths=[refined_map],
        )

    # parse_output: the M-refined map for this half
    def parse_output(self, result: RunResult) -> AdapterOutput:
        refined_map = next((p for p in result.output_paths if p.name.endswith('.mrc')), None)
        if refined_map is None or not Path(refined_map).is_file():
            raise ValueError('M produced no refined map')
        return AdapterOutput(
            output_paths=[Path(refined_map)],
            parsed={'final_map': str(refined_map)},
        )
