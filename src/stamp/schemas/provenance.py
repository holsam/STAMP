'''
STAMP: per-stage provenance schema
'''

# Import external dependencies
from datetime import datetime
from pydantic import BaseModel

# ProvenanceSidecar: class defining what produced a given output directory for reproducibility
class ProvenanceSidecar(BaseModel):
    stage: str
    tool: str
    tool_version: str | None = None
    parameters: dict
    stamp_commit: str
    timestamp: datetime
    input_checksums: dict[str, str]