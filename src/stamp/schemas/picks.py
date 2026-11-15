'''
STAMP: raw pick schema
'''

# Import external dependencies
from pydantic import BaseModel, Field

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior
from stamp.schemas.utils import Quaternion

# RawPick: class for a candidate position reported by a picker for a tomogram
class RawPick(BaseModel):
    tomogram_id: str
    position: tuple[float, float, float]
    orientation: Quaternion | None = None
    prior: OrientationPrior | None = Field(default=None, description='Pick orientation prior')
    confidence: float | None = None
    source_picker: str
    metadata: dict = {}
    beam_angle_deviation_degrees: float | None = Field(default=None, description='Angle in degrees between the outward normal and the beam-orthogonal (xy) plane (0 = beam-orthogonal; 90 = beam-aligned).')
    offset_window_angstrom: tuple[float, float] | None = None
    # score under mean-density scoring
    mean_score: float | None = None
    # score under profile-shape scoring
    profile_score: float | None = None
    vesicle_id: str | None = None
