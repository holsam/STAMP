'''
STAMP: mock scene starting scene with vesicles, raw tomogram slice, membrane segmentation
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import gaussian_filter

# Import internal STAMP objects
from stamp.tools.diagram.utils.carriers import carrier_distance

# _paste_add: add a patch into a canvas centred on (cx, cy), clipped to the canvas bounds
def _paste_add(canvas: np.ndarray, patch: np.ndarray, cx: float, cy: float):
    h, w = patch.shape
    H, W = canvas.shape
    y0, x0 = int(round(cy - h / 2)), int(round(cx - w / 2))
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + h), min(W, x0 + w)
    if ys >= ye or xs >= xe:
        return
    canvas[ys:ye, xs:xe] += patch[ys - y0:ye - y0, xs - x0:xe - x0]

# render_scene: build a raw tomogram slice and its carrier (membrane or filament) segmentation (0 background, 1 membrane, 2 protein density)
def render_scene(field_px, carriers, particles, rng=None, membrane_thickness=2.6, wedge_sigma=(1.6, 0.6), noise_level=0.55):
    # particles: list of dicts with keys x, y, image (the class truth projection)
    rng = rng if rng is not None else np.random.default_rng(0)
    yy, xx = np.mgrid[0:field_px, 0:field_px]
    membrane = np.zeros((field_px, field_px))
    for carrier in carriers:
        membrane += np.exp(-(carrier_distance(carrier, xx, yy) ** 2) / (2 * membrane_thickness ** 2))

    protein = np.zeros((field_px, field_px))
    for p in particles:
        _paste_add(protein, p['image'], p['x'], p['y'])

    signal = 0.7 * membrane + 1.0 * protein
    raw = gaussian_filter(signal, sigma=wedge_sigma)
    raw = raw / (raw.max() or 1.0)
    raw = raw + rng.normal(0.0, noise_level * raw.std(), size=raw.shape)
    # a slow intensity ramp, as tomograms often have
    raw += 0.15 * (xx / field_px - 0.5)

    segmentation = np.zeros((field_px, field_px), dtype=np.uint8)
    segmentation[membrane > 0.35] = 1
    segmentation[protein > 0.38] = 2
    return raw.astype(np.float32), segmentation
