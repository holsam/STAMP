'''
STAMP: tomogram-level manifest schema
'''

# Import external dependencies
from pathlib import Path
from pydantic import BaseModel

# TomogramManifest: class defining a single tomogram
class TomogramManifest(BaseModel):
    tomogram_id: str
    segmentation_path: Path | None = None   # absent for priors that need no membrane segmentation
    raw_tomogram_path: Path
    voxel_size_angstrom: float
    is_decoy: bool = False
