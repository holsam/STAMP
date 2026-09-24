'''
STAMP: logging set up and handling
'''

# Import external dependencies
import sys
from loguru import logger as logger
from pathlib import Path

# Import internal STAMP objects
import stamp.utils.io as io

# DEFAULT_LOG_NAME: default filename for log file
DEFAULT_LOG_NAME = 'stamp.log'

# LOG_FORMAT: format for log strings
LOG_FORMAT = '<dim>{time:YYYY-MM-DD HH:mm:ss}</> <lvl>[{level}]</> {message}'

# PROGRESS_LEVEL: custom porgress level
PROGRESS_LEVEL = 'PROGRESS'

# Define dictionary to map verbosity values to levels
VERBOSITY_TO_LEVEL: dict[int, str] = {
    0: 'ERROR',
    1: 'WARNING',
    2: 'INFO',
    3: PROGRESS_LEVEL,
    4: 'DEBUG',
}

# _StampLogger: class for logger, proxying standard log methods through opt(colours=True) so messages don't have to include every time
class _StampLogger:
    def _log(self, level: str, message: str, *args, **kwargs) -> None:
        logger.opt(colors=True, depth=2).log(level, message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        self._log('DEBUG', message, *args, **kwargs)

    def progress(self, message: str, *args, **kwargs) -> None:
        self._log(PROGRESS_LEVEL, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        self._log('INFO', message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        self._log('WARNING', message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        self._log('ERROR', message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        self._log('CRITICAL', message, *args, **kwargs)

# register_custom_levels: returns None, but registers custom PROGRESS level and updates default loguru colour scheme
def register_custom_levels() -> None:
    try:
        logger.level(PROGRESS_LEVEL, no=15, color='<cyan><bold>')
    except ValueError:
        # If level is already registered, skip registration instead of raising error
        pass
    for name, colour in (
        ('DEBUG', '<yellow><bold>'),
        ('INFO', '<blue><bold>'),
        ('WARNING', '<fg 178><bold>'),
    ):
        try:
            logger.level(name, color=colour)
        except ValueError:
            pass

# resolve_level: returns string corresponding to resolved log level to use from verbosity and quiet arguments
def resolve_level(verbosity: int, quiet: int) -> str:
    resolved = 2 + verbosity - quiet
    return VERBOSITY_TO_LEVEL[max(0, min(4, resolved))]

# -- resolve_log_directory: returns Path indicating which directory to write the log file within (follows hierarchy: specified directory -> current working directory -> home directory)
def resolve_log_directory(specified_dir: Path | None):
    dirs = [Path.cwd(), Path.home()]
    if specified_dir is not None:
        dirs.insert(0, specified_dir)
    for d in dirs:
        # Get absolute path to directory
        d = io._resolve_abspath(d)
        # If directory doesn't exist, try to create (logging warning if not possible)
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError:
                logger.warning(f'Could not create log file parent directory {d}')
                continue
        # If directory is writable, return otherwise raise warning
        if io._is_writable(d):
            return d
        logger.warning(f'No write permissions for {d} to create log file')
    # If all options exhausted, raise an error
    raise PermissionError('No writable directory found for log file')

# _worker_log_config: (level_name, log_path) as configured by main STAMP process
_worker_log_config: tuple[str, Path | None] | None = None

# configure_logging: returns None, configures the single terminal logging sink
def configure_logging(
    directory: Path | None,
    mode: str,
    quiet: int,
    verbosity: int
) -> None:
    global _worker_log_config
    register_custom_levels()
    level_name = resolve_level(verbosity, quiet)
    logger.remove()
    # Add terminal sink
    logger.add(sys.stderr, format=LOG_FORMAT, level=level_name, colorize=True)
    # Add a temporary buffer sink to catch any warning logs raised during log file directory resolution
    buffer: list[str] = []
    buffer_sink = logger.add(buffer.append, format=LOG_FORMAT, level=level_name, colorize=False)
    # Resolve log file directory/path and remove buffer sink
    log_dir = resolve_log_directory(directory)
    log_path = log_dir / DEFAULT_LOG_NAME
    logger.remove(buffer_sink)
    # Based on mode, open log file as write or append
    file_open_mode = 'w' if mode == 'overwrite' else 'a'
    with open(log_path, file_open_mode, encoding='utf-8') as f:
        f.writelines(buffer)
    # Set up actual log file sink
    logger.add(log_path, format=LOG_FORMAT, level=level_name, colorize=False, mode='a')
    _worker_log_config = (level_name, log_path)
    # Return log file path
    return log_path, level_name

# get_worker_log_config: the main process's currently configured (level_name, log_path)
def get_worker_log_config() -> tuple[str, Path | None]:
    return _worker_log_config or ('WARNING', None)

# init_worker_logging: replicate logging sinks in a worker
def init_worker_logging(level_name: str, log_path: Path | None) -> None:
    register_custom_levels()
    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT, level=level_name, colorize=True)
    if log_path is not None:
        logger.add(log_path, format=LOG_FORMAT, level=level_name, colorize=False, mode='a')

# Register custom levels at import time
register_custom_levels()

# Create logger as instance of _StampLogger
log = _StampLogger()


