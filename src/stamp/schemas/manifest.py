'''
STAMP: tomogram-level manifest schema
'''

# Import external dependencies
from pathlib import Path
from pydantic import BaseModel

# TomogramManifest: class defining a single segmented tomogram used by STAMP
class TomogramManifest(BaseModel):
    tomogram_id: str
    segmentation_path: Path
    raw_tomogram_path: Path
    voxel_size_angstrom: float
    is_decoy: bool = False
