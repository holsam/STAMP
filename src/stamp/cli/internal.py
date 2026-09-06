'''
STAMP: CLI for `internal command` (internal utilities, not for direct use)
'''

# Import external dependencies
from pathlib import Path
from typing import Annotated
import typer

internalCli = typer.Typer(
    add_completion=False,
    hidden=True
)

# mark-complete: called from a cluster job script after its tool command exits 0
@internalCli.command('mark-complete')
def mark_complete(
    output_dir: Annotated[Path, typer.Argument()],
    track: Annotated[str, typer.Argument()],
    stage: Annotated[str, typer.Argument()],
) -> None:
    from stamp.run.state import mark_complete
    mark_complete(output_dir, track, stage)