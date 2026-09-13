'''
STAMP: CLI for `stamp tools completion` command group
'''

# Import external dependencies
import typer
from typer._completion_shared import Shells
from typing import Annotated

# Import command functions
from stamp.tools.completion.commands import install_completion, show_completion, uninstall_completion

# ShellOpt: shared shell option
ShellOpt = Annotated[
    Shells | None,
    typer.Option('-s', '--shell', help='Target shell (auto-detected from the running shell if omitted).'),
]

# Initialise Typer app
completionCli = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
)

# Define completion install command
@completionCli.command('install')
def install_subcommand(shell: ShellOpt = None) -> None:
    '''Install shell completion for stamp into a shell's config.'''
    install_completion(shell)

# Define completion show command
@completionCli.command('show')
def show_subcommand(shell: ShellOpt = None) -> None:
    '''Show the completion script without editing the shell's config file.'''
    show_completion(shell)

# Define completion uninstall command
@completionCli.command('uninstall')
def uninstall_subcommand(shell: ShellOpt = None) -> None:
    '''Remove shell completion previously added by install subcommand.'''
    uninstall_completion(shell)
