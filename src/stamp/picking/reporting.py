'''
STAMP: run-level summary reports
'''

# Import external dependencies
import numpy as np

# Import internal STAMP objects
from stamp.schemas.particles import Particle
from stamp.utils.log import log

# _BEAM_ANGLE_BIN_EDGES: histogram bin edges for beam angle deviation reporting, in degrees
_BEAM_ANGLE_BIN_EDGES = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)

# report_beam_angle_distribution: log the angular distribution of a picked population's beam angle deviation
def report_beam_angle_distribution(particles: list[Particle]) -> None:
    deviations = np.array([p.beam_angle_deviation_degrees for p in particles if p.beam_angle_deviation_degrees is not None])
    if deviations.size == 0:
        log.warning('No particles carry a beam angle deviation; skipping angular distribution report.')
        return

    log.info(f'Beam angle deviation (deg): median {np.median(deviations):.1f}, mean {deviations.mean():.1f}')
    counts, edges = np.histogram(deviations, bins=_BEAM_ANGLE_BIN_EDGES)
    total = deviations.size
    for count, low, high in zip(counts, edges[:-1], edges[1:]):
        label = ' (near-orthogonal)' if low == 0.0 else (' (near-aligned)' if high == 90.0 else '')
        log.info(f'  {low:.0f}-{high:.0f}{label:16}: {count} picks ({100 * count / total:.0f}%)')
