'''
STAMP: runner to dispatches ToolCommand as a SLURM job
'''

# Import external dependencies
import json
from pathlib import Path

# Import STAMP objects
from stamp.backends.base import RunResult, ToolCommand
from stamp.schemas.cluster_profile import ClusterProfile
from stamp.backends.slurm import TERMINAL_STATES, job_state, render_job_script, submit

# _MARKER: record submitted job id
_MARKER = '.stamp_cluster_submitted.json'

# ClusterRunner: submit a job and returns
class ClusterRunner:
    def __init__(self, profile: ClusterProfile, requires_gpu: bool = False) -> None:
        self.profile = profile
        self.requires_gpu = requires_gpu

    def run(self, command: ToolCommand) -> RunResult:
        workdir = command.working_directory
        workdir.mkdir(parents=True, exist_ok=True)
        marker_path = workdir / _MARKER

        if marker_path.is_file():
            record = json.loads(marker_path.read_text())
            state = job_state(record['job_id'])
            if state not in TERMINAL_STATES:
                print(f"stamp: job {record['job_id']} still running - check with "
                      f"sacct -j {record['job_id']}")
                return RunResult(
                    tool=command.tool,
                    exit_code=0,
                    stdout=record['job_id'],
                    stderr='',
                    output_paths=command.output_paths
                )
            marker_path.unlink()
            stdout = self._slurm_stream(workdir, record['job_id'], 'out')
            stderr = self._slurm_stream(workdir, record['job_id'], 'err')
            return RunResult(
                tool=command.tool,
                exit_code=0 if state == 'COMPLETED' else 1,
                stdout=stdout,
                stderr=stderr or f"SLURM job {record['job_id']} ended in state {state}",
                output_paths=command.output_paths,
            )

        script_path = workdir / f'stamp-{command.tool}.sbatch'
        script_path.write_text(render_job_script(command, self.requires_gpu, self.profile, workdir))
        job_id = submit(script_path)
        marker_path.write_text(json.dumps({'job_id': job_id, 'tool': command.tool}))
        print(f'stamp: submitted as job {job_id}')
        return RunResult(
            tool=command.tool,
            exit_code=0,
            stdout=job_id,
            stderr='',
            output_paths=command.output_paths
        )

    # _slurm_stream: contents of slurm-<jobid>.out / .err, empty string if absent
    def _slurm_stream(self, workdir: Path, job_id: str, suffix: str) -> str:
        path = workdir / f'slurm-{job_id}.{suffix}'
        return path.read_text() if path.is_file() else ''
    