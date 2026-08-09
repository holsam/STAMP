'''
STAMP: full pipeline orchestration
'''

# Import external dependencies
import typer

# Initialise Typer app 
pipelineCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define pipeline command
@pipelineCli.command()
def pipeline():
    '''Run the STAMP pipeline'''
    print(f'stamp pipeline: not yet implemented')
    raise SystemExit(0)