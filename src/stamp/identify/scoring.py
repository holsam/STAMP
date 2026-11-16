'''
STAMP: identify fit scoring, checkpointed and run in parallel
'''

# Import external dependencies
import json, numpy as np, re
from dataclasses import dataclass
from pathlib import Path

# Import STAMP objects
from stamp.classify.cluster import prior_of_class
from stamp.identify.panel import Candidate
from stamp.identify.register import register_score
from stamp.identify.simulate import simulate_density, to_comparable
from stamp.schemas.particles import OrientationPrior
from stamp.utils.checkpoint import atomic_write_json
from stamp.utils.log import log
from stamp.utils.parallel import run_parallel

# _FitJob: one (class, candidate) pair
@dataclass(frozen=True)
class _FitJob:
    class_id: str
    comparable_average: np.ndarray
    box_voxels: int
    voxel_size: float
    resolution: float
    candidate: Candidate
    aligned_prior: OrientationPrior

# _fit_one: multiprocessing worker entry point
def _fit_one(job: _FitJob) -> tuple[str, str, float]:
    simulated = simulate_density(job.candidate.structure_path, job.box_voxels, job.voxel_size, job.resolution)
    simulated = to_comparable(simulated, job.voxel_size, job.resolution, already_bandlimited=True)
    score = register_score(job.comparable_average, simulated, job.aligned_prior)
    return job.class_id, job.candidate.name, score

# _score_path: checkpoint file for one (class, candidate) score; the name is sanitised for the filesystem, the payload keeps the true names
def _score_path(checkpoint_dir: Path, class_id: str, candidate_name: str) -> Path:
    return checkpoint_dir / f'{class_id}__{re.sub(r"[^\w.-]", "_", candidate_name)}.json'

# _load_score: a checkpointed score, or None when absent, unreadable or for a different (class, candidate)
def _load_score(checkpoint_dir: Path | None, class_id: str, candidate_name: str) -> float | None:
    if checkpoint_dir is None:
        return None
    try:
        saved = json.loads(_score_path(checkpoint_dir, class_id, candidate_name).read_text())
        if saved['class_id'] != class_id or saved['candidate'] != candidate_name:
            return None
        return float(saved['score'])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

# score_panel: fit every candidate to every class average, each finished fit checkpointed in checkpoint_dir when given, returns {class_id: {candidate: score}}
def score_panel(class_averages, panel, resolution, fitter, backend, n_workers: int = 1, checkpoint_dir: Path | None = None):
    all_scores: dict[str, dict[str, float]] = {class_id: {} for class_id in class_averages}
    jobs = []
    for class_id, (average, voxel_size) in class_averages.items():
        for candidate in panel:
            saved = _load_score(checkpoint_dir, class_id, candidate.name)
            if saved is None:
                jobs.append(_FitJob(class_id, to_comparable(average, voxel_size, resolution), average.shape[-1], voxel_size, resolution, candidate, prior_of_class(class_id)))
            else:
                all_scores[class_id][candidate.name] = saved
    n_resumed = sum(len(scores) for scores in all_scores.values())
    if n_resumed:
        log.info(f'Resuming identification: {n_resumed} score(s) loaded from checkpoint')

    def _on_success(job: _FitJob, result: tuple[str, str, float]) -> None:
        class_id, candidate_name, score = result
        all_scores[class_id][candidate_name] = score
        log.debug(f'{class_id}/{candidate_name}: score={score:.3f}')
        if checkpoint_dir is not None:
            atomic_write_json(_score_path(checkpoint_dir, class_id, candidate_name), {'class_id': class_id, 'candidate': candidate_name, 'score': float(score)})

    run_parallel(jobs, _fit_one, max_workers=n_workers, label='identify-fit', on_success=_on_success)
    return all_scores
