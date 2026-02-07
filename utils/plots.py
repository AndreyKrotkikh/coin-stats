from typing import Optional, Tuple

import numpy as np
from bokeh.models import Label, Span
from bokeh.plotting import Figure, figure

from utils.funcs import binom_pmf_series, critical_region_two_sided, p_value_two_sided


def _validate_prob(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")


def _validate_n(n: int) -> None:
    if n < 0:
        raise ValueError("n must be >= 0")


def _validate_k(k: int, n: int) -> None:
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")


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
) -> Figure:
    """Plot H0/H1 binomial distributions with alpha/beta and p-value regions."""
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

    title_text = title or f"n={n}, p0={p0}, p1={p1}"
    plot = figure(
        title=title_text,
        x_axis_label="k (heads)",
        y_axis_label="Probability",
        width=820,
        height=360,
        toolbar_location=None,
    )

    plot.vbar(x=k, top=pmf0, width=0.8, color="#4C78A8", alpha=0.7, legend_label="H0")
    plot.vbar(x=k, top=pmf1, width=0.8, color="#F58518", alpha=0.5, legend_label="H1")

    plot.vbar(
        x=k,
        top=np.where(alpha_mask, pmf0, 0.0),
        width=0.8,
        color="#E45756",
        alpha=0.6,
        legend_label="alpha region (H0)",
    )
    plot.vbar(
        x=k,
        top=np.where(beta_mask, pmf1, 0.0),
        width=0.8,
        color="#54A24B",
        alpha=0.6,
        legend_label="beta region (H1)",
    )

    if k_obs is not None:
        p_mask = _p_value_mask(k_obs, cdf0, sf0)
        plot.vbar(
            x=k,
            top=np.where(p_mask, pmf0, 0.0),
            width=0.8,
            color="#B279A2",
            alpha=0.6,
            legend_label="p-value tails (H0)",
        )

        line = Span(location=k_obs, dimension="height", line_color="#222222", line_width=2)
        plot.add_layout(line)

        p_value = p_value_two_sided(k_obs, n, p0)
        if p_value_threshold is not None:
            decision = "reject H0" if p_value <= p_value_threshold else "keep H0"
            label_text = (
                f"k={k_obs}, p-value={p_value:.4f}, threshold={p_value_threshold}, "
                f"{decision}"
            )
        else:
            label_text = f"k={k_obs}, p-value={p_value:.4f}"

        label = Label(x=0, y=max(pmf0.max(), pmf1.max()) * 0.98, text=label_text)
        plot.add_layout(label)

    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"

    return plot
