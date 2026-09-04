'''
STAMP: full pipeline orchestration
'''

# Import external dependencies
from pathlib import Path

# Import STAMP objects
from stamp.run.orchestrate import run_pipeline
from stamp.run.report import write_report
from stamp.schemas.config import load_run_config

# run_full_pipeline: load the config, run every track, write the combined report
def run_full_pipeline(
    config_path: Path,
    force: bool,
    from_stage: str | None,
    no_decoy: bool,
    backend: str | None
) -> None:
    config = load_run_config(config_path)
    if no_decoy:
        config.decoy.enabled = False
    if backend:
        config.run.backend = backend
    outcome = run_pipeline(config, config.run.output_dir, force=force, from_stage=from_stage)
    md_path, _ = write_report(outcome, config.run.output_dir)
    print(f'Run complete. Report: {md_path}')
