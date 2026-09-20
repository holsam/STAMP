'''
STAMP: full pipeline orchestration
'''

# Import external dependencies
from pathlib import Path

# Import STAMP objects
from stamp.run.orchestrate import run_pipeline
from stamp.run.report import write_report
from stamp.schemas.config import load_run_config
from stamp.utils.io import resolve_output_dir
from stamp.utils.log import log

# run_full_pipeline: load the config, run every track, write the combined report
def run_full_pipeline(
    config_path: Path,
    force: bool,
    from_stage: str | None,
    no_decoy: bool,
    backend: str | None
) -> None:
    log.progress(f'Running STAMP pipeline from {config_path}')
    config = load_run_config(config_path)
    if no_decoy:
        config.decoy.enabled = False
    if backend:
        config.run.backend = backend
        config.stage.pick.backend = backend
        config.stage.identify.backend = backend
        config.stage.refine.backend = backend
    log.debug(f'Overrides applied: no_decoy={no_decoy}, backend={backend!r}')
    output_dir = resolve_output_dir(config.run.output_dir, 'run')
    outcome = run_pipeline(config, output_dir, force=force, from_stage=from_stage)
    md_path, _ = write_report(outcome, output_dir)
    log.info(f'Run complete. Report: {md_path}')
