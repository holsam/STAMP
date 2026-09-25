'''
STAMP: unit tests for project functions
'''

# Import external dependencies
import pytest

# Import internal STAMP objects
from stamp.project.manage import init_project, load_project, save_manifest, update_project
from stamp.utils.errors import StampPipelineError

# TestManage: class containing unit tests for project/manage.py
class TestManage:
    def test_init_then_load_matches(self, tmp_path, make_project):
        project = make_project(tmp_path)
        reloaded = load_project(project.root)
        assert reloaded.meta.name == project.meta.name
        assert {t.tomogram_id for t in reloaded.tomograms} == {t.tomogram_id for t in project.tomograms}
        assert (project.root / 'project.toml').is_file()
        assert (project.root / 'manifest.toml').is_file()
        assert (project.root / 'stamp_run.toml').is_file()

    def test_active_tomograms_excludes_flagged_entries(self, tmp_path, make_project):
        project = make_project(tmp_path)
        excluded = [t.model_copy(update={'excluded': i == 0}) for i, t in enumerate(project.tomograms)]
        project = save_manifest(project, excluded)
        assert len(project.active_tomograms) == len(project.tomograms) - 1

    def test_hand_edited_manifest_raises_on_load(self, tmp_path, make_project):
        project = make_project(tmp_path)
        manifest_path = project.root / 'manifest.toml'
        manifest_path.write_text(manifest_path.read_text() + '\n# tampered\n')
        with pytest.raises(StampPipelineError, match='checksum mismatch'):
            load_project(project.root)

    def test_update_then_load_succeeds(self, tmp_path, make_project):
        project = make_project(tmp_path)
        manifest_path = project.root / 'manifest.toml'
        manifest_path.write_text(manifest_path.read_text() + '\n# hand edit\n')
        update_project(project.root)
        load_project(project.root)  # no longer raises

    def test_init_twice_raises(self, tmp_path, make_project):
        project = make_project(tmp_path)
        with pytest.raises(StampPipelineError, match='already exists'):
            init_project(tmp_path / 'proj', tmp_path / 'seg', tmp_path / 'tomo', voxel_size_angstrom=13.48)
