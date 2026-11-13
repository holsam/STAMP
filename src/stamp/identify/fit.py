'''
STAMP: ranking of candidate scores for a class
'''

# Import STAMP schema
from stamp.schemas.particles import IdentificationResult
from stamp.utils.log import log

# rank_candidates: ranked IdentificationResult for one class
def rank_candidates(class_id: str, scores: dict[str, float], method: str) -> IdentificationResult:
    if not scores:
        raise ValueError(f'no candidate scores for class {class_id!r}')
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ordered[0]
    gap = float(best_score - ordered[1][1]) if len(ordered) > 1 else None
    log.debug(f'{class_id}: best={best_name} gap={gap}')
    return IdentificationResult(
        cluster_id=class_id,
        candidate_protein=best_name,
        fit_score=float(best_score),
        method=method,
        score_gap_to_runner_up=gap,
    )
