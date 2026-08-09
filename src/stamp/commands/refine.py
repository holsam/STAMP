'''
STAMP: refinement with enforced half-set independence
'''

# Import external dependencies
import typer

# Initialise Typer app 
refineCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define refine command
@refineCli.command()
def refine():
    '''Refine identified classes with independent half-sets'''
    print(f'stamp refine: not yet implemented')
    raise SystemExit(0)