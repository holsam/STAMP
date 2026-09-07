'''
STAMP: SLURM-specific helpers
'''

# Import external dependencies
import shlex, subprocess
from pathlib import Path

# Import STAMP objects
from stamp.backends.base import ToolCommand
from stamp.schemas.cluster_profile import ClusterProfile

# TERMINAL_STATES: job final states
TERMINAL_STATES = {
    'COMPLETED',
    'FAILED',
    'CANCELLED',
    'TIMEOUT',
    'OUT_OF_MEMORY',
    'NODE_FAIL',
    'PREEMPTED',
    'BOOT_FAIL',
    'DEADLINE'
}

# render_job_script: an sbatch script for one ToolCommand
def render_job_script(
    command: ToolCommand,
    requires_gpu: bool,
    profile: ClusterProfile,
    workdir: Path
) -> str:
    directives = [
        f'#SBATCH --job-name=stamp-{command.tool}',
        f'#SBATCH --partition={profile.partition}',
        f'#SBATCH --cpus-per-task={profile.cpus_per_task}',
        f'#SBATCH --ntasks={profile.ntasks}',
        f'#SBATCH --time={profile.default_time}',
        f'#SBATCH --mem={profile.default_mem}',
        f'#SBATCH --output={workdir}/slurm-%j.out',
        f'#SBATCH --error={workdir}/slurm-%j.err',
    ]
    if requires_gpu:
        directives.append(f'#SBATCH --gpus={profile.gpus}')

    module_lines = [f'module load {name}' for name in profile.module_loads]
    body = ' '.join(shlex.quote(part) for part in command.argv)
    return '\n'.join([
        '#!/bin/bash',
        *directives,
        'set -uo pipefail',
        f'cd {shlex.quote(str(command.working_directory))}',
        *module_lines,
        body,
        '',
    ])

# submit: sbatch --parsable, optionally chained after other jobs via --dependency=afterok
def submit(script_path: Path, dependency_ids: list[str] | None = None) -> str:
    argv = ['sbatch', '--parsable']
    if dependency_ids:
        argv.append('--dependency=afterok:' + ':'.join(dependency_ids))
    argv.append(str(script_path))
    completed = subprocess.run(argv, capture_output=True, text=True, check=True)
    return completed.stdout.strip().split(';')[0]

# job_state: the current sacct state of a job
def job_state(job_id: str) -> str:
    completed = subprocess.run(
        ['sacct', '-j', job_id, '--format=State', '--noheader', '--parsable2'],
        capture_output=True, text=True, check=True,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not rows:
        return 'PENDING'
    return rows[0].split(' ')[0]

# cancel: scancel a job
def cancel(job_id: str) -> None:
    subprocess.run(['scancel', job_id], capture_output=True, text=True, check=False)
