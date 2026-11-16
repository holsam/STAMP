'''
STAMP: subprocess adapter for a single half-set RELION-5 tomography refinement
'''

# Import external dependencies
import numpy as np, starfile
from pathlib import Path

# Import STAMP objects
from stamp.adapters.base import AdapterInputs, AdapterOutput
from stamp.backends.base import RunResult, ToolCommand
from stamp.refine.relion_export import ParticlePose, write_particle_star
from stamp.schemas.particles import Particle
from stamp.utils.errors import StampAdapterError
from stamp.utils.log import log

# RelionRefineAdapter: one half-set RELION-5 Refine3D job
class RelionRefineAdapter:
    name = 'relion'
    stage = 'refine'
    mac_compatible = False
    requires_gpu = True
    automatable = True
    batches_natively = True
    runs_in_process = False

    # write_particle_star: delegates to refine.relion_export, kept until the adapter is removed
    def write_particle_star(
        self,
        particles: list[Particle],
        path: Path,
        raw_tomogram_paths: dict[str, Path],
        poses: dict[str, ParticlePose] | None = None,
        voxel_size_angstrom: float | None = None,
    ) -> None:
        write_particle_star(particles, path, poses=poses, voxel_size_angstrom=voxel_size_angstrom)

    # build_command: one half-set Refine3D job seeded from this half's Stage D class average (A2)
    def build_command(self, inputs: AdapterInputs) -> ToolCommand:
        if len(inputs.input_paths) != 1:
            raise StampAdapterError('relion refine takes exactly one seed reference (this half\'s class average)')
        reference = inputs.input_paths[0]
        particle_star = inputs.output_directory / 'particles.star'
        output_prefix = inputs.output_directory / 'run'
        parameters = inputs.parameters
        argv = [
            'relion_refine',
            '--i', str(particle_star),
            '--ref', str(reference),
            '--o', str(output_prefix),
            '--angpix', str(parameters['voxel_size_angstrom']),
            '--particle_diameter', str(parameters.get('particle_diameter_angstrom', 300.0)),
            '--iter', str(parameters.get('iterations', 5)),
            '--healpix_order', '2',
            '--offset_range', '5', '--offset_step', '2',
            '--sym', parameters.get('symmetry', 'C1'),
            '--pad', '2', '--flatten_solvent', '--zero_mask',
            '--gpu', '',
        ]
        log.debug(f'{self.name}: {" ".join(argv)}')
        return ToolCommand(
            tool=self.name,
            argv=argv,
            working_directory=inputs.output_directory,
            output_paths=[
                inputs.output_directory / 'run_class001.mrc',
                inputs.output_directory / 'run_model.star',
            ],
        )

    # parse_output: final map path and current resolution from run_model.star
    def parse_output(self, result: RunResult) -> AdapterOutput:
        model_star = next((p for p in result.output_paths if p.name == 'run_model.star'), None)
        final_map = next((p for p in result.output_paths if p.name.endswith('class001.mrc')), None)
        resolution = None
        if model_star is not None and Path(model_star).is_file():
            model = starfile.read(model_star)
            table = model.get('model_general', model) if isinstance(model, dict) else model
            resolution = float(np.asarray(table['rlnCurrentResolution']).ravel()[0])
        else:
            log.warning('run_model.star not found in RELION output; resolution unknown')
        return AdapterOutput(
            output_paths=[p for p in (final_map,) if p is not None],
            parsed={'final_map': str(final_map) if final_map else None, 'resolution_angstrom': resolution},
        )

    # collect_final_map: move the tool's refined map to final_map
    def collect_final_map(self, result: RunResult, reference: Path, final_map: Path) -> None:
        Path(self.parse_output(result).parsed['final_map']).replace(final_map)
