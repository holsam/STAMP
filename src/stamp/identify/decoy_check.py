'''
STAMP: decoy pass/fail check for Stage E identification
'''

# Import external dependencies
from dataclasses import dataclass
from math import comb
from statistics import mean, pstdev
from scipy.stats import mannwhitneyu

# DecoyControlResult: outcome of comparing real and decoy fit-score distributions
@dataclass
class DecoyControlResult:
    passed: bool
    reason: str
    real_best: float
    decoy_best: float
    separation_sigma: float
    mann_whitney_p: float

    def model_dump(self) -> dict:
        return self.__dict__.copy()

# evaluate_decoy_control: fail unless the best real fit stands clear of the decoy fit distribution
def evaluate_decoy_control(
    real_scores: list[float],
    decoy_scores: list[float],
    min_separation_sigma: float = 2.0,
) -> DecoyControlResult:
    if not real_scores or not decoy_scores:
        raise ValueError('need at least one real and one decoy score')
    real_best = max(real_scores)
    decoy_best = max(decoy_scores)
    decoy_spread = pstdev(decoy_scores) if len(decoy_scores) > 1 else 0.0
    separation = (real_best - mean(decoy_scores)) / decoy_spread if decoy_spread > 0 else float('inf')

    if separation < min_separation_sigma:
        return DecoyControlResult(
            passed=False,
            reason=(f'best real fit {real_best:.3f} is only {separation:.1f} sigma above the decoy mean (need {min_separation_sigma:.1f}); best decoy {decoy_best:.3f}'),
            real_best=real_best,
            decoy_best=decoy_best,
            separation_sigma=separation,
            mann_whitney_p=float('nan'),
        )

    p_value, floor_p = float('nan'), float('nan')
    if len(real_scores) >= 2 and len(decoy_scores) >= 2:
        p_value = float(mannwhitneyu(real_scores, decoy_scores, alternative='greater').pvalue)
        floor_p = 1.0 / comb(len(real_scores) + len(decoy_scores), len(real_scores))
    if p_value == p_value and floor_p <= 0.05 and p_value > 0.05:
        return DecoyControlResult(
            passed=False,
            reason=f'Mann-Whitney U cannot separate real from decoy (p={p_value:.3f} > 0.05)',
            real_best=real_best,
            decoy_best=decoy_best,
            separation_sigma=separation,
            mann_whitney_p=p_value,
        )
    return DecoyControlResult(
        passed=True,
        reason=(f'best real fit {real_best:.3f} clears the decoy mean by {separation:.1f} sigma (best decoy {decoy_best:.3f})'),
        real_best=real_best,
        decoy_best=decoy_best,
        separation_sigma=separation,
        mann_whitney_p=p_value,
    )
