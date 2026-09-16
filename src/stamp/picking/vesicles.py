'''
STAMP: import per-vesicle labels from an EValuator label MRC and attribute surface points to them
'''

# Import external dependencies
import mrcfile, numpy as np
from pathlib import Path

# Import internal STAMP objects
from stamp.utils.log import log

# VESICLE_ID_TEMPLATE: '{tomogram_id}:v{label:04d}', built from EValuator's own integer label
VESICLE_ID_TEMPLATE = '{tomogram_id}:v{label:04d}'

# load_vesicle_labels: read an EValuator labelled MRC and check it matches the segmentation shape
def load_vesicle_labels(labels_mrc_path: Path, segmentation_shape: tuple[int, ...]) -> np.ndarray:
    with mrcfile.open(str(labels_mrc_path), permissive=True) as mrc:
        labels = np.asarray(mrc.data).astype(np.int64)
    if labels.shape != segmentation_shape:
        raise ValueError(
            f'Vesicle labels MRC {labels_mrc_path!r} has shape {labels.shape}, '
            f'expected {segmentation_shape} to match the segmentation'
        )
    log.debug(f'Loaded vesicle labels from {labels_mrc_path.name}, {int(labels.max())} label(s)')
    return labels

# vesicle_ids_at: vesicle_id string for each surface point (nearest-voxel lookup into the label volume), '' if the point lands on background
def vesicle_ids_at(label_volume: np.ndarray, points_zyx: np.ndarray, tomogram_id: str) -> list[str]:
    indices = np.round(points_zyx).astype(np.int64)
    indices = np.clip(indices, 0, np.array(label_volume.shape) - 1)
    voxel_labels = label_volume[indices[:, 0], indices[:, 1], indices[:, 2]]
    return [
        VESICLE_ID_TEMPLATE.format(tomogram_id=tomogram_id, label=int(label)) if label > 0 else ''
        for label in voxel_labels
    ]

# vesicle_surface_area_angstrom2: approximate area per labelled vesicle, from marching-cubes mesh face areas assigned to their nearest labelled voxel
def vesicle_surface_area_angstrom2(
    vertices_zyx: np.ndarray,
    faces: np.ndarray,
    label_volume: np.ndarray,
    voxel_size_angstrom: float,
    tomogram_id: str,
) -> dict[str, float]:
    triangle_vertices = vertices_zyx[faces]  # (n_faces, 3, 3)
    edge_a = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge_b = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    areas_voxel2 = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    centroids = triangle_vertices.mean(axis=1)
    face_ids = vesicle_ids_at(label_volume, centroids, tomogram_id)

    totals: dict[str, float] = {}
    for vesicle_id, area_voxel2 in zip(face_ids, areas_voxel2):
        if not vesicle_id:
            continue
        totals[vesicle_id] = totals.get(vesicle_id, 0.0) + float(area_voxel2) * voxel_size_angstrom ** 2
    return totals
