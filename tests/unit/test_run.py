'''
STAMP: unit tests for run command functions
'''

# Import external dependencies
import json, pytest

# Import internal STAMP objects
from stamp.run.cluster_orchestrate import plan_pipeline_jobs, submit_pipeline
from stamp.run.orchestrate import RunOutcome, _guard_backends
from stamp.run.report import _decoy_banner, write_report
from stamp.run.state import (
    STAGE_ORDER, is_complete, mark_complete, stage_dir, stages_to_run,
)
from stamp.schemas.cluster_profile import ClusterProfile
from stamp.schemas.config import RunConfig

# _CONFIG_DICT: smallest dict RunConfig.model_validate accepts
_CONFIG_DICT = {
    'run': {
        'segmentation_dir': 'seg',
        'raw_tomogram_dir': 'tomo',
        'output_dir': 'out',
        'voxel_size_angstrom': 13.48,
        'backend': 'local',
    },
    'decoy': {'enabled': True},
    'stage': {
        'identify': {'candidates': 'candidates.yaml', 'resolution': 25.0},
        'refine': {'tool': 'relion', 'backend': 'local'},
    },
}

# _complete: mark a stage done and drop the params.toml is_complete() checks for
def _complete(output_dir, track, stage):
    mark_complete(output_dir, track, stage)
    target = stage_dir(output_dir, track, stage)
    target.mkdir(parents=True, exist_ok=True)
    (target / 'params.toml').write_text('')

# fake_config: a validated RunConfig with no filesystem behind it
@pytest.fixture
def fake_config():
    return RunConfig.model_validate(_CONFIG_DICT)

class TestStagesToRun:
    def test_fresh_output_dir_runs_every_planned_stage(self, fake_config, tmp_path):
        assert stages_to_run(fake_config, tmp_path, force=False, from_stage=None) == STAGE_ORDER

    def test_stop_after_truncates_the_plan(self, fake_config, tmp_path):
        fake_config.run.stop_after = 'classify'
        assert stages_to_run(fake_config, tmp_path, force=False, from_stage=None) == ['pick', 'classify']

    def test_completed_stage_is_skipped(self, fake_config, tmp_path):
        _complete(tmp_path, 'real', 'pick')
        assert stages_to_run(fake_config, tmp_path, force=False, from_stage=None) == STAGE_ORDER[1:]

    def test_force_reruns_completed_stages(self, fake_config, tmp_path):
        _complete(tmp_path, 'real', 'pick')
        assert stages_to_run(fake_config, tmp_path, force=True, from_stage=None) == STAGE_ORDER

    def test_from_stage_slices_from_that_stage(self, fake_config, tmp_path):
        assert stages_to_run(fake_config, tmp_path, force=False, from_stage='identify') == ['identify', 'refine']

    def test_resume_reruns_only_failed_decoy_track(self, fake_config, tmp_path):
        _complete(tmp_path, 'real', 'pick')
        assert stages_to_run(fake_config, tmp_path, force=False, from_stage=None) == STAGE_ORDER

class TestState:
    def test_stage_dir_layout(self, tmp_path):
        assert stage_dir(tmp_path, 'decoy', 'pick') == tmp_path / 'decoy' / 'stage_pick'

    def test_is_complete_needs_both_flag_and_sidecar(self, tmp_path):
        mark_complete(tmp_path, 'real', 'pick')
        assert is_complete(tmp_path, 'real', 'pick') is False  # flag set, no params.toml
        (stage_dir(tmp_path, 'real', 'pick')).mkdir(parents=True, exist_ok=True)
        (stage_dir(tmp_path, 'real', 'pick') / 'params.toml').write_text('')
        assert is_complete(tmp_path, 'real', 'pick') is True

    def test_mark_complete_survives_a_corrupt_state_file(self, tmp_path):
        (tmp_path / 'run_state.json').write_text('{ not json')
        mark_complete(tmp_path, 'real', 'pick')
        assert json.loads((tmp_path / 'run_state.json').read_text()) == {'real': {'pick': True}}

class TestReport:
    def test_decoy_banner_variants(self):
        assert 'NOT RUN (--no-decoy)' in _decoy_banner(RunOutcome(output_dir='.', stop_after='refine', decoy_enabled=False))
        assert 'stopped before identify' in _decoy_banner(RunOutcome(output_dir='.', stop_after='refine', decoy_enabled=True))
        passed = RunOutcome(output_dir='.', stop_after='refine', decoy_enabled=True, decoy_control={'passed': True, 'reason': 'ok'})
        assert _decoy_banner(passed).startswith('DECOY CONTROL: PASS')

    def test_write_report_emits_md_and_json_with_fail_note(self, tmp_path):
        outcome = RunOutcome(
            output_dir=tmp_path, stop_after='identify', decoy_enabled=True,
            decoy_control={'passed': False, 'reason': 'decoy fits'},
            identifications=[{'cluster_id': 'c1', 'candidate_protein': 'small', 'fit_score': 0.9, 'score_gap_to_runner_up': 0.2}],
        )
        md_path, json_path = write_report(outcome, tmp_path)
        md = md_path.read_text()
        assert 'DECOY CONTROL: FAIL' in md
        assert 'should not be treated as trustworthy' in md
        assert json.loads(json_path.read_text())['decoy_control'] == {'passed': False, 'reason': 'decoy fits'}

class TestGuardBackends:
    def test_local_backend_rejects_a_gpu_only_refine_tool(self, fake_config):
        with pytest.raises(SystemExit):
            _guard_backends(fake_config)  # refine tool 'relion' requires a GPU

    def test_mock_backend_passes(self, fake_config):
        fake_config.run.backend = 'mock'
        fake_config.stage.pick.backend = 'mock'
        fake_config.stage.refine.backend = 'mock'
        _guard_backends(fake_config)  # no raise

class TestClusterOrchestrate:
    def test_real_and_decoy_pick_have_no_mutual_dependency(self, fake_config, tmp_path):
        by_key = {j.step_key: j for j in plan_pipeline_jobs(fake_config, tmp_path, ['pick'])}
        assert by_key['real.pick'].depends_on == []
        assert by_key['decoy.pick'].depends_on == []

    def test_classify_depends_on_matching_pick(self, fake_config, tmp_path):
        by_key = {j.step_key: j for j in plan_pipeline_jobs(fake_config, tmp_path, ['pick', 'classify'])}
        assert by_key['real.classify'].depends_on == ['real.pick']
        assert by_key['decoy.classify'].depends_on == ['decoy.pick']

    def test_identify_depends_on_both_classify_tracks(self, fake_config, tmp_path):
        by_key = {j.step_key: j for j in plan_pipeline_jobs(fake_config, tmp_path, ['classify', 'identify'])}
        assert set(by_key['real.identify'].depends_on) == {'real.classify', 'decoy.classify'}

    def test_decoy_jobs_are_dropped_when_decoy_disabled(self, fake_config, tmp_path):
        fake_config.decoy.enabled = False
        keys = {j.step_key for j in plan_pipeline_jobs(fake_config, tmp_path, ['pick', 'classify'])}
        assert keys == {'real.pick', 'real.classify'}

    def test_refine_only_planned_when_in_stage_list(self, fake_config, tmp_path):
        keys = {j.step_key for j in plan_pipeline_jobs(fake_config, tmp_path, ['pick'])}
        assert 'real.refine' not in keys

    def test_submit_pipeline_threads_resolved_dependency_ids(self, monkeypatch, fake_config, tmp_path):
        seen = {}

        def _fake_submit(script_path, dependency_ids=None):
            seen[script_path.name] = list(dependency_ids or [])
            return f'id-{len(seen)}'

        monkeypatch.setattr('stamp.run.cluster_orchestrate.submit', _fake_submit)
        monkeypatch.setattr('stamp.run.cluster_orchestrate.render_job_script', lambda *a, **k: '#!/bin/bash\n')
        jobs = plan_pipeline_jobs(fake_config, tmp_path, ['pick', 'classify'])
        for job in jobs:
            for command in job.commands:
                command.working_directory.mkdir(parents=True, exist_ok=True)

        submitted = submit_pipeline(fake_config, tmp_path, ['pick', 'classify'], ClusterProfile(partition="cpu"))

        real_classify_script = next(k for k in seen if 'real.classify' in k)
        assert seen[real_classify_script] == submitted['real.pick']

    def test_submitted_script_gates_mark_complete_on_exit_zero(self, monkeypatch, fake_config, tmp_path):
        monkeypatch.setattr('stamp.run.cluster_orchestrate.submit', lambda *a, **k: 'job-1')
        monkeypatch.setattr('stamp.run.cluster_orchestrate.render_job_script', lambda *a, **k: '#!/bin/bash\nstamp pick\n')
        jobs = plan_pipeline_jobs(fake_config, tmp_path, ['pick'])
        for job in jobs:
            for command in job.commands:
                command.working_directory.mkdir(parents=True, exist_ok=True)
        submit_pipeline(fake_config, tmp_path, ['pick'], ClusterProfile(partition="cpu"))

        script = (stage_dir(tmp_path, 'real', 'pick') / 'stamp-real.pick.sbatch').read_text()
        assert 'if [ "$tool_exit" -eq 0 ]; then stamp internal mark-complete' in script
        assert 'barrier' not in script
