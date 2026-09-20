'''
STAMP: particle-level schema
'''

# Import external dependencies
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Literal
# Import internal STAMP objects
from stamp.schemas.utils import assert_unit_quaternion

# HalfSet: class for half-set label that gets assigned at the end of consensus picking
class HalfSet(str, Enum):
    A = 'A'
    B = 'B'

# Particle: class for a single consensus particle, located within one tomogram
class Particle(BaseModel):
    particle_id: str
    tomogram_id: str
    position: tuple[float, float, float] = Field(description='Voxel coordinates (x, y, z) within the source tomogram')
    orientation: tuple[float, float, float, float] | None = Field(default=None, description=('Orientation as a unit quaternion (w, x, y, z), if the picker provides one'))
    source_picker: str
    confidence: float | None = None
    half_set: HalfSet
    vesicle_id: str | None = Field(default=None, description='Vesicle ID from an EValuator labelled MRC')
    beam_angle_deviation_degrees: float | None = Field(default=None, description='Angle in degrees between the pick normal and the beam-orthogonal (xy) plane (0 = beam-orthogonal; 90 = beam-aligned)')

    @field_validator('orientation')
    @classmethod
    def _quaternion_is_unit_length(cls, value: tuple[float, float, float, float] | None) -> tuple[float, float, float, float] | None:
        if value is None:
            return value
        assert_unit_quaternion(value)
        return value

# ParticleSet: output of consensus picking for a single run
class ParticleSet(BaseModel):
    particles: list[Particle]
    consensus_rule: Literal['intersection', 'union']
    contributing_pickers: list[str]

    @field_validator('particles')
    @classmethod
    def _not_empty(cls, value: list[Particle]) -> list[Particle]:
        if not value:
            raise ValueError('ParticleSet must contain at least one particle')
        return value

# ClassAssignment: class for a single particles's cluster membership
class ClassAssignment(BaseModel):
    particle_id: str
    cluster_id: str
    classifier: str
    inplane_angle_degrees: float | None = None
    vesicle_id: str | None = None  # denormalised from Particle so class_assignments.json doesn't need to rejoin particle_set.json

# IdentificationResult: candidate protein assignment for a cluster
class IdentificationResult(BaseModel):
    cluster_id: str
    candidate_protein: str
    fit_score: float
    method: str
    score_gap_to_runner_up: float | None = None