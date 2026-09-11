'''
STAMP: command logic for `stamp tools completion` command group
'''

# Import external dependencies
from rich import print
from typer._completion_shared import Shells, get_completion_script, install as typer_install_completion   

# Import internal completion utilities
from stamp.tools.completion.utils import COMPLETE_VAR, _HOME, PROG_NAME, _completion_path, _detect_shell, _print_script, _strip_line

def install_completion(shell):
    if shell is None:
        shell = _detect_shell()
    try:
        installed_shell, path = typer_install_completion(shell=shell, prog_name=PROG_NAME, complete_var=COMPLETE_VAR)
    except Exception as exc:
        print(f'[bold red]\\[ERROR][/bold red] Could not install completion: {exc}.')
    print(f'[bold green]\\[SUCCESS][/bold green] {installed_shell} completion installed at: [cyan]{path}[/cyan]')
    print('Start a new shell session for it to take effect or use command:')
    print(f'\tsource {path}')

def show_completion(shell):
    if shell is None:
        shell = _detect_shell()
    try:
        script = get_completion_script(prog_name=PROG_NAME, complete_var=COMPLETE_VAR, shell=shell)
    except Exception as exc:
        print(f'[bold red]\\[ERROR][/bold red] Could not create completion script for current shell: {exc}.')
        raise SystemExit(1)
    _print_script(script, shell)

def uninstall_completion(shell):
    if shell is None:
        shell = _detect_shell()
    path = _completion_path(shell)
    if not path.exists():
        print(f'[bold yellow]\\[WARNING][/bold yellow] No {shell} completion script found at: [cyan]{path}[/cyan]')
        return
    if path.is_file():
        path.unlink()
    # remove bash's installer's extra source line in ~/.bashrc
    if shell == Shells.bash:
        _strip_line(_HOME / '.bashrc', f"source '{path}'")
    print(f'[bold green]\\[SUCCESS][/bold green] {shell} completion removed from: [cyan]{path}[/cyan]')
    print('Start a new shell session for the change to take effect.')
