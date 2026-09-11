'''
STAMP: plans and submits pipeline as SLURM dependency chain
'''

# Import external dependencies
from dataclasses import dataclass, field
from pathlib import Path

# Import STAMP objects
from stamp.backends.base import ToolCommand
from stamp.backends.slurm import render_job_script, submit
from stamp.commands.classify import build_classify_commands
from stamp.commands.decoy import build_decoy_commands
from stamp.commands.identify import build_identify_commands
from stamp.commands.pick import build_pick_commands
from stamp.commands.refine import build_refine_commands
from stamp.schemas.cluster_profile import ClusterProfile

# ClusterJob: one pipeline step (track + stage) and its SLURM dependencies
@dataclass
class ClusterJob:
    step_key: str
    track: str
    stage: str
    commands: list[ToolCommand]
    depends_on: list[str] = field(default_factory=list)
    requires_gpu: bool = False

# _pick_needs_gpu: True if any configured picker adapter wants a GPU
def _pick_needs_gpu(config) -> bool:
    from stamp.commands.pick import REAL_ADAPTERS
    return any(getattr(REAL_ADAPTERS.get(name), 'requires_gpu', False) for name in config.stage.pick.pickers)

# plan_pipeline_jobs: planned pipeline, skipping stages stages_to_run() excludes
def plan_pipeline_jobs(config, output_dir: Path, planned_stages: list[str]) -> list[ClusterJob]:
    jobs = []
    if 'pick' in planned_stages:
        jobs.append(ClusterJob('real.pick', 'real', 'pick', build_pick_commands(config, output_dir), requires_gpu=_pick_needs_gpu(config)))
        if config.decoy.enabled:
            jobs.append(ClusterJob('decoy.pick', 'decoy', 'pick', build_decoy_commands(config, output_dir)))
    if 'classify' in planned_stages:
        jobs.append(ClusterJob('real.classify', 'real', 'classify', build_classify_commands(config, output_dir, track='real'), depends_on=['real.pick']))
        if config.decoy.enabled:
            jobs.append(ClusterJob('decoy.classify', 'decoy', 'classify', build_classify_commands(config, output_dir, track='decoy'), depends_on=['decoy.pick']))
    if 'identify' in planned_stages:
        deps = ['real.classify'] + (['decoy.classify'] if config.decoy.enabled else [])
        jobs.append(ClusterJob('real.identify', 'real', 'identify', build_identify_commands(config, output_dir), depends_on=deps))
    if 'refine' in planned_stages:
        jobs.append(ClusterJob('real.refine', 'real', 'refine', build_refine_commands(config, output_dir), depends_on=['real.identify'], requires_gpu=True))
    return jobs

# submit_pipeline: render job scripts & submit using --dependency=afterok from known job ids; returns step -> job id(s)
def submit_pipeline(
    config,
    output_dir: Path,
    planned_stages: list[str],
    profile: ClusterProfile
) -> dict[str, list[str]]:
    jobs = plan_pipeline_jobs(config, output_dir, planned_stages)
    submitted: dict[str, list[str]] = {}
    for job in jobs:
        dep_ids = [id for key in job.depends_on for id in submitted[key]]
        assert len(job.commands) == 1, f'{job.step_key}: fan-out not supported'
        submitted[job.step_key] = [_submit_one(job.commands[0], job, dep_ids, profile, output_dir)]
    return submitted

# _submit_one: render one job's script (tool command + exit-gated mark-complete), submit it
def _submit_one(
    command: ToolCommand,
    job: ClusterJob,
    dep_ids: list[str],
    profile: ClusterProfile,
    output_dir: Path,
) -> str:
    script = render_job_script(command, requires_gpu=job.requires_gpu, profile=profile, workdir=command.working_directory)
    script += (
        'tool_exit=$?\n'
        f'if [ "$tool_exit" -eq 0 ]; then stamp internal mark-complete '
        f'{output_dir} {job.track} {job.stage}; fi\n'
        'exit "$tool_exit"\n'
    )
    script_path = command.working_directory / f'stamp-{job.step_key}.sbatch'
    script_path.write_text(script)
    return submit(script_path, dependency_ids=dep_ids)
