'''
STAMP: unsupervised classification
'''

# Import external dependencies
import typer

# Initialise Typer app 
classifyCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define classify command
@classifyCli.command()
def classify():
    '''Cluster a consensus particle set by structural similarity'''
    print(f'stamp classify: not yet implemented')
    raise SystemExit(0)