'''
STAMP: field validators shared across schema modules
'''

# Import external dependencies
from typing import Annotated
from pydantic import AfterValidator

# Import internal STAMP objects
from stamp.utils.errors import StampPipelineError

# assert_unit_quaternion: raise unless (w, x, y, z) is unit-length within tolerance, else return it unchanged for AfterValidator
def assert_unit_quaternion(value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = sum(component**2 for component in value) ** 0.5
    if not (0.99 <= norm <= 1.01):
        raise StampPipelineError(f'orientation quaternion must be unit-length, got norm={norm:.4f}')
    return value

# Quaternion: unit-length (w, x, y, z) tuple, validated wherever it is declared
Quaternion = Annotated[tuple[float, float, float, float], AfterValidator(assert_unit_quaternion)]
