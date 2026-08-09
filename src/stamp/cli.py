'''
STAMP: CLI entrypoint
'''

# Import external dependencies
import typer

# Import internal commands
from stamp.commands import classify, decoy, identify, pick, pipeline, refine

# Initialise Typer app 
stamp = typer.Typer(
    name = 'stamp',
    help = 'Sub-Tomogram Averaging Membrane Protein pipeline',
    no_args_is_help = True,
    add_completion = False,
)

# Add command Typer classes
stamp.add_typer(pick.pickCli)
stamp.add_typer(decoy.decoyCli)
stamp.add_typer(classify.classifyCli)
stamp.add_typer(identify.identifyCli)
stamp.add_typer(refine.refineCli)
stamp.add_typer(pipeline.pipelineCli)