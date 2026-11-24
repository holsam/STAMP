'''
STAMP: mixed-prior scene
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass, replace

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior
from stamp.tools.diagram.utils.carriers import Carrier, carrier_distance, make_filaments, make_vesicles, place_on
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.scene import render_scene
from stamp.tools.diagram.utils.walkthrough import Walkthrough

# MIXED_CLASS: species of each prior group in a mixed scene (target on membranes, distractor on filaments, first extra species free)
MIXED_CLASS = {OrientationPrior.NORMAL: 0, OrientationPrior.AXIS: 1, OrientationPrior.NONE: 2}

# SharedScene: one tomogram slice, carriers per prior and every true particle
@dataclass
class SharedScene:
    carriers: dict[OrientationPrior, list[Carrier]]
    particles: list[dict]
    raw_tomogram: np.ndarray
    segmentation: np.ndarray

# build_shared_scene: vesicles, filaments and free particles as shared scene
def build_shared_scene(config: DiagramConfig) -> SharedScene:
    if config.n_synthetic_unknowns + len(config.also) < 1:
        raise ValueError('a mixed scene needs a third species for the none prior: keep --n-synthetic-unknowns at 1 or more')
    walk = Walkthrough(replace(config, shared_scene=None, class_ids=None))
    rng, box, field_px = np.random.default_rng(config.seed + 10), config.box_size, config.field_px
    vesicles = make_vesicles(field_px, max(2, config.n_vesicles * 3 // 5), rng)
    filaments = make_filaments(field_px, config.n_filaments, rng, avoid=vesicles, avoid_gap=0.5 * box)
    carriers = {OrientationPrior.NORMAL: vesicles, OrientationPrior.AXIS: filaments, OrientationPrior.NONE: []}
    counts = {OrientationPrior.NORMAL: config.n_target, OrientationPrior.AXIS: config.n_distractor, OrientationPrior.NONE: config.n_extra}
    particles = []
    for prior, count in counts.items():
        for _ in range(count):
            if carriers[prior]:
                x, y, angle = place_on(carriers[prior][rng.integers(len(carriers[prior]))], rng)
            else:     # free particles sit anywhere clear of the membranes and filaments
                for _try in range(50):
                    x, y = rng.uniform(0.1 * field_px, 0.9 * field_px, size=2)
                    if all(carrier_distance(c, x, y) > 0.8 * box for c in vesicles + filaments):
                        break
                angle = np.nan
            particles.append({'x': x, 'y': y, 'class': MIXED_CLASS[prior], 'angle': angle, 'image': walk.truths[MIXED_CLASS[prior]]})
    rng.shuffle(particles)
    kept: list[dict] = []     # drop particles that fall too close together
    for p in particles:
        if all(np.hypot(p['x'] - q['x'], p['y'] - q['y']) > 0.3 * box for q in kept):
            kept.append(p)
    raw, segmentation = render_scene(field_px, vesicles + filaments, kept, np.random.default_rng(config.seed + 11))
    return SharedScene(carriers, kept, raw, segmentation)
