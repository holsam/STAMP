'''
STAMP: project.toml and manifest.toml schemas
'''

# Import external dependencies
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, ConfigDict

# Import internal stamp objects
from stamp.schemas.manifest import TomogramManifest

# ManifestRef: [manifest] table in project.toml
class ManifestRef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: Path
    sha256: str

# ConfigRef: [config] table in project.toml
class ConfigRef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: Path
    sha256: str

# ProjectMeta: [project] table in project.toml
class ProjectMeta(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str
    schema_version: int = 1
    created: datetime
    stamp_version: str

# ProjectFile: schema for project.toml
class ProjectFile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    project: ProjectMeta
    manifest: ManifestRef
    config: ConfigRef

# ManifestFile: manifest.toml ([[tomogram]] entries)
class ManifestFile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    tomogram: list[TomogramManifest] = []
