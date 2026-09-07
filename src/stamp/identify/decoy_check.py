'''
STAMP: decoy pass/fail check for Stage E identification
'''

# Import external dependencies
from dataclasses import dataclass
from math import comb
from scipy.stats import mannwhitneyu

# DecoyControlResult: outcome of comparing real and decoy fit-score distributions
@dataclass
class DecoyControlResult:
    passed: bool
    reason: str
    real_best: float
    decoy_best: float
    mann_whitney_p: float

    def model_dump(self) -> dict:
        return self.__dict__.copy()

# evaluate_decoy_control: fire on overlap between the real and decoy fit-score distributions (A4)
def evaluate_decoy_control(
    real_scores: list[float],
    decoy_scores: list[float],
    margin: float = 0.05
) -> DecoyControlResult:
    if not real_scores or not decoy_scores:
        raise ValueError('need at least one real and one decoy score')
    real_best = max(real_scores)
    decoy_best = max(decoy_scores)
    if decoy_best >= real_best - margin:
        return DecoyControlResult(
            passed=False,
            reason=f'best decoy fit {decoy_best:.3f} is within {margin} of the best real fit {real_best:.3f}',
            real_best=real_best,
            decoy_best=decoy_best,
            mann_whitney_p=float('nan'),
        )
    # one-sided test (checking if real scores stochastically greater than decoy scores)
    if len(real_scores) >= 2 and len(decoy_scores) >= 2:
        p_value = float(mannwhitneyu(real_scores, decoy_scores, alternative='greater').pvalue)
        floor_p = 1.0 / comb(len(real_scores) + len(decoy_scores), len(real_scores))
    else:
        p_value = float('nan')
        floor_p = float('nan')
    if p_value == p_value and floor_p <= 0.05 and p_value > 0.05:  # test is powered but cannot separate
        return DecoyControlResult(
            passed=False,
            reason=f'Mann-Whitney U cannot separate real from decoy (p={p_value:.3f} > 0.05)',
            real_best=real_best,
            decoy_best=decoy_best,
            mann_whitney_p=p_value,
        )
    return DecoyControlResult(
        passed=True,
        reason=f'best real fit {real_best:.3f} clears the best decoy {decoy_best:.3f} by more than {margin}',
        real_best=real_best,
        decoy_best=decoy_best,
        mann_whitney_p=p_value,
    )
