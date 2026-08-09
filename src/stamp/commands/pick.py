'''
STAMP: consensus picking
'''

# Import external dependencies
import typer

# Initialise Typer app 
pickCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define pick command
@pickCli.command()
def pick():
    '''Run consensus particle picking'''
    print(f'stamp pick: not yet implemented')
    raise SystemExit(0)