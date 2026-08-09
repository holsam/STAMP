'''
STAMP: decoy dataset generation
'''

# Import external dependencies
import typer

# Initialise Typer app 
decoyCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define decoy command
@decoyCli.command()
def decoy():
    '''Generate a decoy dataset to run through consensus picking'''
    print(f'stamp decoy: not yet implemented')
    raise SystemExit(0)