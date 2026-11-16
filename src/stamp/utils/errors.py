'''
STAMP: custom exception hierarchy for logging integration
'''

# StampError: base exception class for all STAMP errors, logged once by the CLI entry point
class StampError(Exception):
    level: str = 'error'

    def __init__(self, message: str, *, level: str | None = None) -> None:
        super().__init__(message)
        self.level = level or self.level

# StampConfigError: bad or missing configuration (CLI args, YAML/TOML config, panel/candidate files)
class StampConfigError(StampError):
    level = 'error'

# StampValidationError: bad data reaching a function (shape mismatch, empty set, malformed schema)
class StampValidationError(StampError):
    level = 'error'

# StampPipelineError: unrecoverable failure, pipeline cannot produce a valid result
class StampPipelineError(StampError):
    level = 'critical'
