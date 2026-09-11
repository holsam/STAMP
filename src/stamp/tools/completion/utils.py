'''
STAMP: utilities for `stamp tools completion` command group
'''

# Import external dependencies
from pathlib import Path
from rich import print
from rich.console import Console
from rich.syntax import Syntax
from typer._completion_classes import completion_init
from typer._completion_shared import Shells

# register typer's bash/zsh/fish/powershell completion classes into click's shell_completion registry
completion_init()

# PROG_NAME: pin program name as stamp
PROG_NAME = 'stamp'

# COMPLETE_VAR: pin environment variable as _STAMP_COMPLETE
COMPLETE_VAR = f'_{PROG_NAME.upper()}_COMPLETE'

# _HOME: user home directory
_HOME = Path.home()

# _completion_path: resolve the script path a given shell's installer writes to
def _completion_path(shell: Shells) -> Path:
    match shell:
        case Shells.bash:
            return _HOME / '.bash_completions' / f'{PROG_NAME}.sh'
        case Shells.zsh:
            return _HOME / '.zfunc' / f'_{PROG_NAME}'
        case Shells.fish:
            return _HOME / '.config' / 'fish' / 'completions' / f'{PROG_NAME}.fish'
        case _:
            print(f'[bold red]\\[ERROR][/bold red] Shell {shell.value!r} is not supported.')
            raise SystemExit(1)

# _detect_shell: detect the current shell via shellingham
def _detect_shell() -> Shells:
    try:
        import shellingham
        shell_name, _ = shellingham.detect_shell()
        return Shells(shell_name)
    except ValueError:
        print(f'[bold red]\\[ERROR][/bold red] Detected shell {shell_name!r} is not supported. Retry with explicit --shell option.')
        raise SystemExit(1)
    except Exception:
        print('[bold red]\\[ERROR][/bold red] Could not detect the current shell. Retry with explicit --shell option.')
        raise SystemExit(1)

# _print_script: print a completion script and edited files
def _print_script(script, shell: Shells):
    path = _completion_path(shell)
    console = Console()
    console.print(f'\n[cyan]{path}[/cyan]:')
    syntax_block = Syntax(script, 'shell', theme='ansi_dark', line_numbers=True)
    console.print(syntax_block)
    if shell == Shells.bash:
        console.print(f'\n[cyan]{_HOME / '.bashrc'}[/cyan]:')
        console.print(Syntax(f"source '{path}'", shell, theme='ansi_dark', line_numbers=True))
    console.print('\n\n')

# _strip_line: remove every exact-match line from a text file, if present
def _strip_line(path: Path, target: str) -> None:
    if not path.is_file():
        return
    lines = path.read_text().splitlines()
    kept = [line for line in lines if line.strip() != target]
    if len(kept) != len(lines):
        path.write_text('\n'.join(kept) + '\n')
