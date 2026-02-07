from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from bokeh.models import Label, Span
from bokeh.plotting import figure

if TYPE_CHECKING:
    from bokeh.plotting._figure import Figure

from utils.funcs import (
    _validate_k,
    _validate_n,
    _validate_prob,
    binom_pmf_series,
    critical_region_two_sided,
    p_value_two_sided,
)


def _build_distributions(
    n: int, p0: float, p1: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = np.arange(n + 1, dtype=int)
    pmf0 = binom_pmf_series(n, p0)
    pmf1 = binom_pmf_series(n, p1)
    cdf0 = np.cumsum(pmf0)
    sf0 = 1.0 - np.concatenate(([0.0], cdf0[:-1]))
    return k, pmf0, pmf1, cdf0, sf0


def _alpha_mask(n: int, p0: float, alpha: float) -> np.ndarray:
    lower, upper = critical_region_two_sided(n, p0, alpha)
    k = np.arange(n + 1, dtype=int)
    return (k <= lower) | (k >= upper)


def _beta_mask(n: int, p0: float, alpha: float) -> np.ndarray:
    lower, upper = critical_region_two_sided(n, p0, alpha)
    k = np.arange(n + 1, dtype=int)
    return (k >= lower + 1) & (k <= upper - 1)


def _p_value_mask(
    k_obs: int, cdf0: np.ndarray, sf0: np.ndarray
) -> np.ndarray:
    tail_prob = min(cdf0[k_obs], sf0[k_obs])
    return (cdf0 <= tail_prob) | (sf0 <= tail_prob)


def plot_two_coin_distributions(
    n: int,
    p0: float,
    p1: float,
    alpha: float,
    k_obs: Optional[int] = None,
    p_value_threshold: Optional[float] = None,
    title: Optional[str] = None,
) -> "Figure":
    """Plot H0/H1 binomial PMFs with alpha/beta and p-value regions.

    Notes:
    - H0 is Binomial(n, p0), H1 is Binomial(n, p1).
    - alpha region is the two-sided rejection region under H0.
    - beta region is the acceptance region under H1 (Type II error area).
    - p-value tails are shown under H0 for the given k_obs.
    """
    _validate_n(n)
    _validate_prob(p0)
    _validate_prob(p1)
    _validate_prob(alpha)

    if k_obs is not None:
        _validate_k(k_obs, n)
    if p_value_threshold is not None:
        _validate_prob(p_value_threshold)

    k, pmf0, pmf1, cdf0, sf0 = _build_distributions(n, p0, p1)
    alpha_mask = _alpha_mask(n, p0, alpha)
    beta_mask = _beta_mask(n, p0, alpha)
    lower, upper = critical_region_two_sided(n, p0, alpha)

    title_text = title or f"n={n}, p0={p0}, p1={p1}"
    plot = figure(
        title=title_text,
        x_axis_label="k (heads)",
        y_axis_label="Probability",
        width=820,
        height=360,
        toolbar_location=None,
    )

    # "Distribution plot" for discrete PMFs: line + markers.
    plot.line(k, pmf0, line_width=2, color="#4C78A8", alpha=0.9, legend_label="H0 PMF")
    plot.circle(k, pmf0, size=6, color="#4C78A8", alpha=0.9, legend_label="H0 points")

    plot.line(k, pmf1, line_width=2, color="#F58518", alpha=0.7, legend_label="H1 PMF")
    plot.circle(k, pmf1, size=6, color="#F58518", alpha=0.7, legend_label="H1 points")

    # Highlight alpha (under H0) and beta (under H1) regions using colored markers.
    plot.circle(
        k[alpha_mask],
        pmf0[alpha_mask],
        size=9,
        color="#E45756",
        alpha=0.85,
        legend_label="alpha region (reject H0, under H0)",
    )
    plot.circle(
        k[beta_mask],
        pmf1[beta_mask],
        size=9,
        color="#54A24B",
        alpha=0.85,
        legend_label="beta region (keep H0, under H1)",
    )

    # Show rejection cutoffs (discrete boundaries) as dashed vertical lines.
    if lower >= 0:
        plot.add_layout(
            Span(
                location=float(lower),
                dimension="height",
                line_color="#E45756",
                line_width=1,
                line_dash="dashed",
            )
        )
    if upper <= n:
        plot.add_layout(
            Span(
                location=float(upper),
                dimension="height",
                line_color="#E45756",
                line_width=1,
                line_dash="dashed",
            )
        )

    if k_obs is not None:
        p_mask = _p_value_mask(k_obs, cdf0, sf0)
        plot.circle(
            k[p_mask],
            pmf0[p_mask],
            size=10,
            color="#B279A2",
            alpha=0.8,
            legend_label="p-value tails (under H0)",
        )

        line = Span(location=k_obs, dimension="height", line_color="#222222", line_width=2)
        plot.add_layout(line)

        alpha_actual = float(pmf0[alpha_mask].sum())
        beta_actual = float(pmf1[beta_mask].sum())
        p_value = p_value_two_sided(k_obs, n, p0)

        reject_by_alpha = (k_obs <= lower) or (k_obs >= upper)
        if p_value_threshold is not None:
            decision_p = "reject H0" if p_value <= p_value_threshold else "keep H0"
            decision_a = "reject H0" if reject_by_alpha else "keep H0"
            label_text = (
                f"k={k_obs}; alpha≈{alpha_actual:.3f}, beta≈{beta_actual:.3f}; "
                f"p-value={p_value:.4f} <= {p_value_threshold}? {decision_p}; "
                f"alpha-region decision: {decision_a}"
            )
        else:
            decision_a = "reject H0" if reject_by_alpha else "keep H0"
            label_text = (
                f"k={k_obs}; alpha≈{alpha_actual:.3f}, beta≈{beta_actual:.3f}; "
                f"p-value={p_value:.4f}; alpha-region decision: {decision_a}"
            )

        label = Label(x=0, y=max(pmf0.max(), pmf1.max()) * 0.98, text=label_text)
        plot.add_layout(label)

    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"

    return plot
