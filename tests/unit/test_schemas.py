'''
STAMP: unit tests for schema
'''

# Import external dependencies
import pytest
from datetime import datetime, timezone
from pathlib import Path
from pydantic import ValidationError

# Import schemas
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.particles import (
    ClassAssignment,
    HalfSet,
    IdentificationResult,
    Particle,
    ParticleSet,
)
from stamp.schemas.provenance import ProvenanceSidecar

# TestManifestSchema: class containing unit tests for schemas/manifest.py
class TestManifestSchema:
    def test_tomogram_manifest_defaults_not_decoy(self):
        '''TomogramManifest should accept valid input'''
        manifest = TomogramManifest(
            tomogram_id='tomo001',
            segmentation_path=Path('/data/tomo001_seg.mrc'),
            voxel_size_angstrom=3.4,
        )
        assert manifest.is_decoy is False


# TestParticleSchema: class containing unit tests for schemas/particle.py
class TestParticleSchema:
    def _valid_particle(self, **overrides) -> Particle:
        '''Create a Particle instance'''
        defaults = dict(
            particle_id='p001',
            tomogram_id='tomo001',
            position=(10.0, 20.0, 30.0),
            orientation=None,
            source_picker='example-picker',
            confidence=0.9,
            half_set=HalfSet.A,
        )
        return Particle(**{**defaults, **overrides})

    def test_particle_accepts_valid_unit_quaternion(self):
        '''Particle should accept a valid quaternion when passed'''
        particle = self._valid_particle(orientation=(1.0, 0.0, 0.0, 0.0))
        assert particle.orientation == (1.0, 0.0, 0.0, 0.0)

    def test_particle_rejects_non_unit_quaternion(self):
        '''Particle should reject a quaternion if not unit-length'''
        with pytest.raises(ValidationError):
            self._valid_particle(orientation=(2.0, 0.0, 0.0, 0.0))

    def test_particle_rejects_invalid_half_set(self):
        '''Particle should reject a half_set label that doesn't exist'''
        with pytest.raises(ValidationError):
            self._valid_particle(half_set='C')

    def test_particle_set_rejects_empty_particles(self):
        '''ParticleSet should reject no Particles'''
        with pytest.raises(ValidationError):
            ParticleSet(particles=[], consensus_rule='intersection', contributing_pickers=['example-picker'])

    def test_particle_set_accepts_valid_input(self):
        '''ParticleSet should accept a valid input'''
        particle_set = ParticleSet(
            particles=[self._valid_particle()],
            consensus_rule='intersection',
            contributing_pickers=['example-picker1', 'example-picker2'],
        )
        assert len(particle_set.particles) == 1

    def test_particle_set_rejects_unknown_consensus_rule(self):
        '''ParticleSet should reject an unknown consensus rule'''
        with pytest.raises(ValidationError):
            ParticleSet(
                particles=[self._valid_particle()],
                consensus_rule='unknown-rule',
                contributing_pickers=['pyseg'],
            )

    def test_class_assignment_construction(self):
        '''ClassAssignment should accept a valid input'''
        assignment = ClassAssignment(particle_id='p001', cluster_id='c01', classifier='example-classifier')
        assert assignment.cluster_id == 'c01'

    def test_identification_result_construction(self):
        '''IdentifierResult should accept a valid input'''
        result = IdentificationResult(
            cluster_id='c01',
            candidate_protein='Prot001',
            fit_score=0.82,
            method='example-method',
            score_gap_to_runner_up=0.31,
        )
        assert result.candidate_protein == 'Prot001'


# TestProvenanceSchema: class containing unit tests for schemas/provenance.py
class TestProvenanceSchema:
    def test_provenance_sidecar_construction(self):
        '''ProvenanceSidecar should accept valid input'''
        sidecar = ProvenanceSidecar(
            stage='pick',
            tool='pyseg',
            tool_version='1.0.0',
            parameters={'distance_threshold': 15.0},
            stamp_commit='deadbeef',
            timestamp=datetime.now(timezone.utc),
            input_checksums={'tomo001_seg.mrc': 'abc123'},
        )
        assert sidecar.stage == 'pick'