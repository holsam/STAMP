'''
STAMP: project directory management
'''

# Import external dependencies
import tomllib, tomli_w
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Import internal stamp objects
from stamp.config.template import TEMPLATE
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.project import ConfigRef, ManifestFile, ManifestRef, ProjectFile, ProjectMeta
from stamp.utils.errors import StampPipelineError
from stamp.utils.io import checksum_file, load_tomogram_manifests, resolve_stamp_commit, toml_none_to_empty
from stamp.utils.log import log

# _PROJECT_NAME: project.toml filename within a project root
_PROJECT_NAME = 'project.toml'
# _MANIFEST_NAME: manifest.toml filename within a project root
_MANIFEST_NAME = 'manifest.toml'
# _CONFIG_NAME: stamp_run.toml filename within a project root
_CONFIG_NAME = 'stamp_run.toml'

# Project: dataclass for an initialised, checksum-verified project directory
@dataclass(frozen=True)
class Project:
    root: Path
    meta: ProjectMeta
    tomograms: list[TomogramManifest]
    config_path: Path
    # active_tomograms: non-excluded tomograms
    @property
    def active_tomograms(self) -> list[TomogramManifest]:
        return [t for t in self.tomograms if not t.excluded]

# _write_manifest_file: write manifest.toml from a list of TomogramManifest
def _write_manifest_file(manifest_path: Path, tomograms: list[TomogramManifest]) -> None:
    manifest_file = ManifestFile(tomogram=tomograms)
    manifest_path.write_text(tomli_w.dumps(toml_none_to_empty(manifest_file.model_dump(mode='json'))))

# _write_project_file: write project.toml with the current manifest/config checksums
def _write_project_file(project_path: Path, meta: ProjectMeta, manifest_path: Path, config_path: Path) -> None:
    project_file = ProjectFile(
        project=meta,
        manifest=ManifestRef(path=manifest_path.name, sha256=checksum_file(manifest_path)),
        config=ConfigRef(path=config_path.name, sha256=checksum_file(config_path)),
    )
    project_path.write_text(tomli_w.dumps(toml_none_to_empty(project_file.model_dump(mode='json'))))

# _validate_project_root: validate if a project.toml file exists at given path
def _validate_project_root(root: Path, exists: bool):
    project_path = root / _PROJECT_NAME
    if not exists and project_path.exists():
        raise StampPipelineError(f'{project_path} already exists, use `stamp project add` to add tomograms')
    if exists and not project_path.is_file():
        raise StampPipelineError(f'No {_PROJECT_NAME} found at {root}, run `stamp project init` first')
    return project_path

# _validate_manifest: validate if a manifest.toml file exists at given path
def _validate_manifest(root: Path, project_file, check: bool):
    manifest_path = root / project_file.manifest.path
    if not manifest_path.is_file():
        raise StampPipelineError(f'Project manifest {manifest_path} is missing')
    if check:
        manifest_checksum = checksum_file(manifest_path)
        if manifest_checksum != project_file.manifest.sha256:
            raise StampPipelineError(f'{manifest_path} has been modified outside STAMP (checksum mismatch). Run `stamp project lock` if this edit was intentional.')
    return manifest_path

# _validate_config: validate if a stamp_run.toml file exists at given path
def _validate_config(root: Path, project_file, check: bool):
    config_path = root / project_file.config.path
    if not config_path.is_file():
        raise StampPipelineError(f'Project config {config_path} is missing')
    if check:
        config_checksum = checksum_file(config_path)
        if config_checksum != project_file.config.sha256:
            raise StampPipelineError(f'{config_path} has been modified outside STAMP (checksum mismatch). Run `stamp project lock` if this edit was intentional.')
    return config_path

# save_manifest: write a new manifest.toml and re-record its checksum in project.toml
def save_manifest(project: Project, tomograms: list[TomogramManifest]) -> Project:
    manifest_path = project.root / _MANIFEST_NAME
    _write_manifest_file(manifest_path, tomograms)
    _write_project_file(project.root / _PROJECT_NAME, project.meta, manifest_path, project.config_path)
    log.info(f'Wrote {len(tomograms)} tomogram(s) to {manifest_path}')
    return Project(root=project.root, meta=project.meta, tomograms=tomograms, config_path=project.config_path)

# record_config_checksum: re-record stamp_run.toml's checksum after `stamp config edit/init` writes to it
def record_config_checksum(root: Path) -> None:
    project_path = root / _PROJECT_NAME
    if not project_path.is_file():
        return
    project_file = ProjectFile.model_validate(tomllib.loads(project_path.read_text()))
    manifest_path = root / project_file.manifest.path
    config_path = root / project_file.config.path
    _write_project_file(project_path, project_file.project, manifest_path, config_path)

# init_project: create a new project directory with an initial manifest and stamp_run.toml
def init_project(
    root: Path,
    seg_dir: Path,
    raw_dir: Path,
    *,
    voxel_size_angstrom: float | None = None,
    name: str | None = None,
) -> Project:
    root.mkdir(parents=True, exist_ok=True)
    project_path = _validate_project_root(root=root, exists=False)
    tomograms = load_tomogram_manifests(seg_dir, raw_dir, voxel_size_angstrom)
    if not tomograms:
        raise StampPipelineError(f'No matched segmentation/tomogram pairs found between {seg_dir} and {raw_dir}')
    manifest_path = root / _MANIFEST_NAME
    _write_manifest_file(manifest_path, tomograms)
    config_path = root / _CONFIG_NAME
    config_path.write_text(TEMPLATE)
    meta = ProjectMeta(
        name=name or root.name,
        created=datetime.now(timezone.utc),
        stamp_version=resolve_stamp_commit(),
    )
    _write_project_file(project_path, meta, manifest_path, config_path)
    log.info(f'Initialised project {meta.name!r} at {root} with {len(tomograms)} tomogram(s)')
    return Project(root=root, meta=meta, tomograms=tomograms, config_path=config_path)

# load_project: load and checksum-verify project.toml, manifest.toml and stamp_run.toml
def load_project(root: Path) -> Project:
    project_path = _validate_project_root(root=root, exists=True)
    project_file = ProjectFile.model_validate(tomllib.loads(project_path.read_text()))
    manifest_path = _validate_manifest(root=root, project_file=project_file, check=True)
    config_path = _validate_config(root=root, project_file=project_file, check=True)
    manifest_file = ManifestFile.model_validate(tomllib.loads(manifest_path.read_text()))
    return Project(root=root, meta=project_file.project, tomograms=manifest_file.tomogram, config_path=config_path)

# update_project: accept the current on-disk manifest.toml/stamp_run.toml as-is, re-recording their checksums
def update_project(root: Path) -> Project:
    project_path = _validate_project_root(root=root, exists=True)
    project_file = ProjectFile.model_validate(tomllib.loads(project_path.read_text()))
    manifest_path = _validate_manifest(root=root, project_file=project_file, check=False)
    config_path = _validate_config(root=root, project_file=project_file, check=False)
    _write_project_file(project_path, project_file.project, manifest_path, config_path)
    manifest_file = ManifestFile.model_validate(tomllib.loads(manifest_path.read_text()))
    log.info(f'Locked {project_path} to the current manifest.toml and stamp_run.toml')
    return Project(root=root, meta=project_file.project, tomograms=manifest_file.tomogram, config_path=config_path)
