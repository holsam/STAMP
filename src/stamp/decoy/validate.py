'''
STAMP: guards keeping decoy and real data from ever being processed together
'''

# Import internal STAMP objects
from stamp.decoy.generate import DECOY_SOURCE_PREFIX
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.particles import ParticleSet
from stamp.utils.errors import StampPipelineError

# is_decoy_particle_set: True if every particle is a decoy, False if none are, raising on a mix
def is_decoy_particle_set(particle_set: ParticleSet) -> bool:
    if not particle_set.particles:
        raise StampPipelineError('Cannot determine decoy status of an empty ParticleSet.')

    flags = {
        particle.source_picker.startswith(DECOY_SOURCE_PREFIX)
        for particle in particle_set.particles
    }
    if len(flags) > 1:
        raise StampPipelineError('ParticleSet mixes decoy and real particles, these should be processed separately')
    return flags.pop()

# check_manifest_purity: True if all manifests are decoys, False if none are, raising on a mix
def check_manifest_purity(manifests: list[TomogramManifest]) -> bool:
    if not manifests:
        raise StampPipelineError('Cannot determine decoy status of an empty manifest list.')
    flags = {manifest.is_decoy for manifest in manifests}
    if len(flags) > 1:
        raise StampPipelineError('Manifest list mixes decoy and real tomograms, these should be processed separately')
    return flags.pop()

# assert_comparable: confirm a decoy set is a fair comparator for a real set
def assert_comparable(real: ParticleSet, decoy: ParticleSet) -> None:
    if not is_decoy_particle_set(decoy):
        raise StampPipelineError('The decoy ParticleSet contains real particles')
    if is_decoy_particle_set(real):
        raise StampPipelineError('The real ParticleSet contains decoy particles')

    real_count = len(real.particles)
    decoy_count = len(decoy.particles)
    if decoy_count < 0.5 * real_count:
        raise StampPipelineError(f'Decoy set ({decoy_count}) is less than half the size of the real set ({real_count}), a weaker decoy result is explicable by particle count alone rather than by absence of signal; increase --n-decoys-per-tomogram')
