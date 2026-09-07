'''
STAMP: site-specific cluster scheduler settings
'''

# Import external dependencies
import tomllib
from pathlib import Path
from pydantic import BaseModel
from typer import get_app_dir

# _DEFAULT_PROFILE_PATH: user-level fallback when the run config has no [cluster] table
_DEFAULT_PROFILE_PATH = Path(get_app_dir('stamp')) / 'cluster.toml'

# ClusterProfile: partition name, resource defaults and module loads for one site
class ClusterProfile(BaseModel):
    partition: str
    default_time: str = '24:00:00'
    default_mem: str = '64G'
    cpus_per_task: int = 1
    ntasks: int = 1
    gpus: int = 1
    module_loads: list[str] = []

# load_cluster_profile: from an explicit dict ([cluster] table), else the user file, else defaults
def load_cluster_profile(table: dict | None = None, path: Path | None = None) -> ClusterProfile:
    if table:
        return ClusterProfile.model_validate(table)
    profile_path = path or _DEFAULT_PROFILE_PATH
    if profile_path.is_file():
        return ClusterProfile.model_validate(tomllib.loads(profile_path.read_text()).get('cluster', {}))
    return ClusterProfile()
