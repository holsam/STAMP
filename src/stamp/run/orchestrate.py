'''
STAMP: in-process chaining of the real and decoy tracks for `stamp run`
'''

# Import external dependencies
import json, tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Import STAMP objects
from stamp.commands.classify import run_classify
from stamp.commands.decoy import run_decoy
from stamp.commands.identify import run_identify
from stamp.commands.pick import run_pick
from stamp.commands.refine import run_refine
from stamp.run.state import STAGE_ORDER, mark_complete, stage_dir, stages_to_run
from stamp.schemas.config import RunConfig

# RunOutcome: completed run information reported
@dataclass
class RunOutcome:
    output_dir: Path
    stop_after: str
    decoy_enabled: bool
    decoy_control: dict | None = None
    identifications: list[dict] = field(default_factory=list)
    refine_results: list[dict] = field(default_factory=list)
    sidecars: dict[str, dict] = field(default_factory=dict)

# _real_pick: run pick for the real track
def _real_pick(config: RunConfig, output_dir: Path) -> Path:
    target = stage_dir(output_dir, 'real', 'pick')
    run_pick(
        picker_names=config.stage.pick.pickers,
        segmentation_dir=config.run.segmentation_dir,
        raw_tomogram_dir=config.run.raw_tomogram_dir,
        output_dir=target,
        voxel_size_angstrom=config.run.voxel_size_angstrom,
        extra_params={},
        consensus_rule=config.stage.pick.consensus_rule,
        distance_threshold=config.stage.pick.distance_threshold,
        half_set_seed=config.stage.pick.half_set_seed,
        backend=config.stage.pick.backend,
    )
    return target / 'particle_set.json'

# _decoy_pick: run decoy generation
def _decoy_pick(config: RunConfig, output_dir: Path, real_particle_set: Path) -> Path:
    target = stage_dir(output_dir, 'decoy', 'pick')
    run_decoy(
        output_dir=target,
        method=config.decoy.method,
        parameters={},
        real_particle_set=real_particle_set,
        segmentation_dir=config.run.segmentation_dir,
        raw_tomogram_dir=config.run.raw_tomogram_dir,
        voxel_size_angstrom=config.run.voxel_size_angstrom,
        n_decoys_per_tomogram=50,
        min_distance_from_real_angstrom=100.0,
        min_shift_angstrom=200.0,
        max_shift_angstrom=600.0,
        n_synthetic_tomograms=3,
        synthetic_shape='200,200,200',
        seed=config.stage.pick.half_set_seed,
    )
    return target / 'decoy_particle_set.json'

# _classify_track: classify one track's particle set into class_averages/
def _classify_track(config: RunConfig, output_dir: Path, track: str, particles: Path) -> Path:
    target = stage_dir(output_dir, track, 'classify')
    settings = config.stage.classify
    run_classify(
        particles=particles,
        raw_tomogram_dir=config.run.raw_tomogram_dir,
        output_dir=target,
        voxel_size_angstrom=config.run.voxel_size_angstrom,
        box_angstrom=settings.box_angstrom,
        n_radial_bins=settings.n_radial_bins,
        method=settings.method,
        min_cluster_size=settings.min_cluster_size,
        n_clusters=settings.n_clusters,
        n_components=settings.n_components,
        strict_halfset_independence=settings.strict_halfset_independence,
        random_state=settings.random_state,
    )
    return target

# run_pipeline: chain pick -> (decoy) -> classify -> identify (real, with decoy control) -> refine
def run_pipeline(
    config: RunConfig,
    output_dir: Path, *,
    force: bool = False,
    from_stage: str | None = None
) -> RunOutcome:
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = stages_to_run(config, output_dir, force, from_stage)
    stop_after = config.run.stop_after or STAGE_ORDER[-1]
    outcome = RunOutcome(
        output_dir=output_dir,
        stop_after=stop_after,
        decoy_enabled=config.decoy.enabled,
    )

    # --- pick ---
    real_particles = stage_dir(output_dir, 'real', 'pick') / 'particle_set.json'
    if 'pick' in planned:
        real_particles = _real_pick(config, output_dir)
        mark_complete(output_dir, 'real', 'pick')
    decoy_particles = None
    if config.decoy.enabled and 'pick' in planned:
        decoy_particles = _decoy_pick(config, output_dir, real_particles)
        mark_complete(output_dir, 'decoy', 'pick')
    if stop_after == 'pick':
        return _finalise(outcome, output_dir)

    # --- classify ---
    if 'classify' in planned:
        _classify_track(config, output_dir, 'real', real_particles)
        mark_complete(output_dir, 'real', 'classify')
        if config.decoy.enabled and decoy_particles is not None:
            _classify_track(config, output_dir, 'decoy', decoy_particles)
            mark_complete(output_dir, 'decoy', 'classify')
    if stop_after == 'classify':
        return _finalise(outcome, output_dir)

    # --- identify ---
    identify_dir = stage_dir(output_dir, 'real', 'identify')
    real_classes = stage_dir(output_dir, 'real', 'classify') / 'class_averages'
    decoy_classes = stage_dir(output_dir, 'decoy', 'classify') / 'class_averages'
    if 'identify' in planned:
        run_identify(
            classes=real_classes,
            candidates=config.stage.identify.candidates,
            output_dir=identify_dir,
            decoy_classes=decoy_classes if (config.decoy.enabled and decoy_classes.is_dir()) else None,
            resolution=config.stage.identify.resolution,
            backend=config.stage.identify.backend if config.stage.identify.backend != 'cluster' else 'local',
            fitter=config.stage.identify.fitter,
            fetch_missing=config.stage.identify.fetch_missing,
        )
        mark_complete(output_dir, 'real', 'identify')
    outcome.identifications = json.loads((identify_dir / 'identification.json').read_text())
    control_path = identify_dir / 'decoy_control.json'
    if control_path.is_file():
        outcome.decoy_control = json.loads(control_path.read_text())
    if stop_after == 'identify':
        return _finalise(outcome, output_dir)

    # --- refine ---
    refine_dir = stage_dir(output_dir, 'real', 'refine')
    if 'refine' in planned:
        run_refine(
            class_id=config.stage.refine.class_id,
            identification=identify_dir / 'identification.json',
            particles=real_particles,
            class_assignments=stage_dir(output_dir, 'real', 'classify') / 'class_assignments.json',
            raw_tomogram_dir=config.run.raw_tomogram_dir,
            output_dir=refine_dir,
            tool=config.stage.refine.tool,
            mask=config.stage.refine.mask,
            iterations=config.stage.refine.iterations,
            backend=config.stage.refine.backend if config.stage.refine.backend != 'cluster' else 'local',
            voxel_size_angstrom=config.run.voxel_size_angstrom,
            combined_halfset=False,
        )
        mark_complete(output_dir, 'real', 'refine')
    for class_dir in sorted(p for p in refine_dir.glob('*') if (p / 'params.toml').is_file()):
        sidecar = tomllib.loads((class_dir / 'params.toml').read_text())
        outcome.refine_results.append({
            'class_id': class_dir.name,
            'resolution_angstrom': sidecar['parameters'].get('resolution_angstrom'),
        })
    return _finalise(outcome, output_dir)

# _finalise: collect each sidecar into outcome for appendix
def _finalise(outcome: RunOutcome, output_dir: Path) -> RunOutcome:
    for path in sorted(output_dir.rglob('params.toml')):
        outcome.sidecars[str(path.relative_to(output_dir))] = tomllib.loads(path.read_text())
    return outcome
