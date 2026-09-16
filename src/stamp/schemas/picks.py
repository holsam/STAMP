'''
STAMP: raw pick schema
'''

# Import external dependencies
from pydantic import BaseModel, Field, field_validator

# RawPick: class for a candidate position reported by a picker for a tomogram
class RawPick(BaseModel):
    tomogram_id: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    source_picker: str
    metadata: dict = {}
    beam_angle_deviation_degrees: float | None = Field(default=None, description='Angle in degrees between the outward normal and the beam-orthogonal (xy) plane (0 = beam-orthogonal; 90 = beam-aligned).')
    offset_window_angstrom: tuple[float, float] | None = None
    # score under mean-density scoring
    mean_score: float | None = None
    # score under profile-shape scoring
    profile_score: float | None = None

    @field_validator('orientation')
    @classmethod
    def _quaternion_is_unit_length(cls, value: tuple[float, float, float, float] | None) -> tuple[float, float, float, float] | None:
        if value is None:
            return value
        norm = sum(component**2 for component in value) ** 0.5
        if not (0.99 <= norm <= 1.01):
            raise ValueError(f'orientation quaternion must be unit-length, got norm={norm:.4f}')
        return value
