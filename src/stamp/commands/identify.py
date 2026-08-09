'''
STAMP: identification against predicted structures
'''

# Import external dependencies
import typer

# Initialise Typer app 
identifyCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define identify command
@identifyCli.command()
def identify():
    '''Fit predicted structures to class averages and score candidates'''
    print(f'stamp identify: not yet implemented')
    raise SystemExit(0)