'''
STAMP: backend factory, converting --backend option into Runner
'''

# Import external dependencies
import platform, shutil

# Import STAMP objects
from stamp.adapters.base import ToolAdapter
from stamp.backends.base import Runner
from stamp.backends.cluster import ClusterRunner
from stamp.backends.local import LocalRunner
from stamp.backends.mock import MockRunner
from stamp.schemas.cluster_profile import ClusterProfile, load_cluster_profile

# check_backend_supports: raise if specified backend cannot run adapter
def check_backend_supports(adapter: ToolAdapter, backend: str) -> None:
    if backend != 'local':
        return
    if adapter.requires_gpu:
        raise SystemExit(f'{adapter.name} requires a GPU; --backend local cannot provide one - use --backend cluster')
    if platform.system() == 'Darwin' and not adapter.mac_compatible:
        raise SystemExit(f'{adapter.name} cannot run on macOS - use --backend cluster or a macOS-compatible tool')

# select_runner: map --backend string to Runner
def select_runner(
    backend: str,
    *,
    cluster_profile: ClusterProfile | None = None,
    requires_gpu: bool = False
) -> Runner:
    if backend == 'mock':
        return MockRunner()
    if backend == 'local':
        return LocalRunner()
    if backend == 'cluster':
        if shutil.which('sbatch') is None:
            raise SystemExit('--backend cluster used but sbatch is not on PATH - run STAMP on a cluster login node, or use --backend local/mock')
        return ClusterRunner(profile=cluster_profile or load_cluster_profile(), requires_gpu=requires_gpu)
    raise SystemExit(f'unknown backend {backend!r}')
