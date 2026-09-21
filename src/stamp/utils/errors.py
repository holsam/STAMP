'''
STAMP: custom exception hierarchy for logging integration
'''

# Import internal STAMP objects
from stamp.utils.log import log

# StampError: base exception class for all STAMP errors
class StampError(Exception):
    level: str = 'error'

    def __init__(self, message: str, *, level: str | None = None) -> None:
        super().__init__(message)
        log._log((level or self.level).upper(), message)

# StampConfigError: bad or missing configuration (CLI args, YAML/TOML config, panel/candidate files)
class StampConfigError(StampError):
    level = 'error'

# StampValidationError: bad data reaching a function (shape mismatch, empty set, malformed schema)
class StampValidationError(StampError):
    level = 'error'

# StampAdapterError: external tool (RELION, M, stamp-native) failed or produced no usable output
class StampAdapterError(StampError):
    level = 'error'

# StampPipelineError: unrecoverable failure, pipeline cannot produce a valid result
class StampPipelineError(StampError):
    level = 'critical'
