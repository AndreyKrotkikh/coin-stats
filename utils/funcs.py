from math import comb
from typing import Dict, Literal, Tuple

import numpy as np


def _validate_prob(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")


def _validate_n(n: int) -> None:
    if n < 0:
        raise ValueError("n must be >= 0")


def _validate_k(k: int, n: int) -> None:
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")


def binom_pmf(k: int, n: int, p: float) -> float:
    """Return P(X = k) for a Binomial(n, p) distribution."""
    _validate_n(n)
    _validate_prob(p)
    _validate_k(k, n)
    return comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def binom_cdf(k: int, n: int, p: float) -> float:
    """Return P(X <= k) for a Binomial(n, p) distribution."""
    _validate_n(n)
    _validate_prob(p)
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(binom_pmf(i, n, p) for i in range(0, k + 1))


def binom_sf(k: int, n: int, p: float) -> float:
    """Return P(X > k) for a Binomial(n, p) distribution."""
    _validate_n(n)
    _validate_prob(p)
    if k < 0:
        return 1.0
    if k >= n:
        return 0.0
    return 1.0 - binom_cdf(k, n, p)


def binom_pmf_series(n: int, p: float) -> np.ndarray:
    """Return the PMF values for k=0..n for a Binomial(n, p)."""
    _validate_n(n)
    _validate_prob(p)
    return np.array([binom_pmf(k, n, p) for k in range(n + 1)], dtype=float)


def critical_region_two_sided(
    n: int, p0: float, alpha: float
) -> Tuple[int, int]:
    """Return (lower, upper) rejection cutoffs for a two-sided test."""
    _validate_n(n)
    _validate_prob(p0)
    _validate_prob(alpha)

    half = alpha / 2.0

    lower_candidates = [k for k in range(n + 1) if binom_cdf(k, n, p0) <= half]
    lower = max(lower_candidates) if lower_candidates else -1

    upper_candidates = [
        k for k in range(n + 1) if (1.0 - binom_cdf(k - 1, n, p0)) <= half
    ]
    upper = min(upper_candidates) if upper_candidates else n + 1

    return lower, upper


def reject_two_sided(k_obs: int, n: int, p0: float, alpha: float) -> bool:
    """Check if k_obs falls into the two-sided rejection region."""
    _validate_k(k_obs, n)
    lower, upper = critical_region_two_sided(n, p0, alpha)
    return k_obs <= lower or k_obs >= upper


def p_value_one_sided(
    k_obs: int, n: int, p0: float, alternative: Literal["less", "greater"]
) -> float:
    """Compute a one-sided p-value for the binomial test."""
    _validate_k(k_obs, n)
    _validate_prob(p0)
    if alternative == "less":
        return binom_cdf(k_obs, n, p0)
    return 1.0 - binom_cdf(k_obs - 1, n, p0)


def p_value_two_sided(k_obs: int, n: int, p0: float) -> float:
    """Compute a two-sided p-value for the binomial test."""
    _validate_k(k_obs, n)
    _validate_prob(p0)
    lower_tail = binom_cdf(k_obs, n, p0)
    upper_tail = 1.0 - binom_cdf(k_obs - 1, n, p0)
    return min(1.0, 2.0 * min(lower_tail, upper_tail))


def beta_two_sided(n: int, p0: float, p1: float, alpha: float) -> float:
    """Return beta (Type II error) for a two-sided test at p1."""
    _validate_n(n)
    _validate_prob(p0)
    _validate_prob(p1)
    _validate_prob(alpha)

    lower, upper = critical_region_two_sided(n, p0, alpha)
    accept_low = lower + 1
    accept_high = upper - 1

    if accept_low > accept_high:
        return 0.0

    return sum(binom_pmf(k, n, p1) for k in range(accept_low, accept_high + 1))


def power_two_sided(n: int, p0: float, p1: float, alpha: float) -> float:
    """Return power (1 - beta) for a two-sided test at p1."""
    return 1.0 - beta_two_sided(n, p0, p1, alpha)


def compute_metrics(
    n: int, p0: float, p1: float, alpha: float, k_obs: int
) -> Dict[str, float]:
    """Compute alpha, beta, power, and two-sided p-value for k_obs."""
    _validate_k(k_obs, n)

    beta = beta_two_sided(n, p0, p1, alpha)
    power = 1.0 - beta
    p_value = p_value_two_sided(k_obs, n, p0)

    return {
        "alpha": alpha,
        "beta": beta,
        "power": power,
        "p_value": p_value,
    }
