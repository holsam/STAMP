'''
STAMP: unsupervised classification CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated

# Import classify command functions/variables
from stamp.utils.options.cli import AzimuthalModesOption, BinFactorOption, BoxAngstromOption, FeatureNormalisationOption, InplaneAngularStepDegreesOption, KeepRawOption, MergeFractionOption, MethodOption, MinClusterSizeOption, MinRadiusFractionOption, NAzimuthalSamplesOption, NClustersOption, NComponentsOption, NProcessesOption, NRadialBinsOption, NRotationsOption, NullSigmasOption, OutDirOption, PlotFormatOption, PlotsOption, PolishRoundsOption, RandomStateOption, RawDirOption, RejectQuantileOption, SegDirOption, ShiftRangeVoxelsOption, StrictHalfsetIndependenceOption, TiltAxisDegreesOption, TiltMaxDegreesOption, VoxelSizeOption
import stamp.commands.classify as classifyfuncs
from stamp.decoy.validate import is_decoy_particle_set
from stamp.schemas.particles import load_particle_set
from stamp.utils.io import resolve_output_dir
from stamp.utils.log import log_started
from stamp.utils.options.literals import PlotFormat

# Initialise Typer app
classifyCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define classify command
@classifyCli.command()
def classify(
    inputs: Annotated[
        list[Path],
        typer.Argument(exists=True, help='particle_set.json and/or decoy_particle_set.json, classified in turn.'),
    ],
    raw_tomogram_dir: RawDirOption,
    voxel_size_angstrom: VoxelSizeOption = None,
    segmentation_dir: SegDirOption = None,
    output_dir: OutDirOption = Path('.'),
    box_angstrom: BoxAngstromOption = 300.0,
    n_radial_bins: NRadialBinsOption = 12,
    method: MethodOption = 'leiden',
    min_cluster_size: MinClusterSizeOption = 20,
    n_clusters: NClustersOption = 5,
    n_components: NComponentsOption = 20,
    strict_halfset_independence: StrictHalfsetIndependenceOption = True,
    random_state: RandomStateOption = 0,
    inplane_angular_step_degrees: InplaneAngularStepDegreesOption = 10.0,
    azimuthal_modes: AzimuthalModesOption = 4,
    min_radius_fraction: MinRadiusFractionOption = 0.25,
    n_azimuthal_samples: NAzimuthalSamplesOption = 64,
    feature_normalisation: FeatureNormalisationOption = 'both',
    tilt_max_degrees: TiltMaxDegreesOption = None,
    tilt_axis_degrees: TiltAxisDegreesOption = 90.0,
    bin_factor: BinFactorOption = 2,
    n_rotations: NRotationsOption = 3000,
    shift_range_voxels: ShiftRangeVoxelsOption = 3,
    polish_rounds: PolishRoundsOption = 2,
    null_sigmas: NullSigmasOption = 3.0,
    reject_quantile: RejectQuantileOption = 0.95,
    merge_fraction: MergeFractionOption = 1.0,
    n_workers: NProcessesOption = 1,
    make_plots: PlotsOption = True,
    plot_format: PlotFormatOption = 'tiff',
    keep_raw: KeepRawOption = False,
) -> None:
    '''Cluster picked particles by structural similarity.'''
    resolved = classifyfuncs.resolve_inputs(inputs)
    log_started('classification')
    for particles in resolved.particle_sets:
        is_decoy = is_decoy_particle_set(load_particle_set(particles))
        classifyfuncs.run_classify(
            particles=particles,
            raw_tomogram_dir=raw_tomogram_dir,
            segmentation_dir=segmentation_dir,
            output_dir=resolve_output_dir(output_dir, 'classify', 'decoy' if is_decoy else None),
            voxel_size_angstrom=voxel_size_angstrom,
            box_angstrom=box_angstrom,
            n_radial_bins=n_radial_bins,
            method=method,
            min_cluster_size=min_cluster_size,
            n_clusters=n_clusters,
            n_components=n_components,
            strict_halfset_independence=strict_halfset_independence,
            random_state=random_state,
            inplane_angular_step_degrees=inplane_angular_step_degrees,
            n_workers=n_workers,
            azimuthal_modes=azimuthal_modes,
            min_radius_fraction=min_radius_fraction,
            n_azimuthal_samples=n_azimuthal_samples,
            feature_normalisation=feature_normalisation,
            make_plots=make_plots,
            plot_format=plot_format,
            keep_raw=keep_raw,
            tilt_max_degrees=tilt_max_degrees,
            tilt_axis_degrees=tilt_axis_degrees,
            bin_factor=bin_factor,
            n_rotations=n_rotations,
            shift_range_voxels=shift_range_voxels,
            polish_rounds=polish_rounds,
            null_sigmas=null_sigmas,
            reject_quantile=reject_quantile,
            merge_fraction=merge_fraction,
        )
