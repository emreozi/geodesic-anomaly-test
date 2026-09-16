"""Independent algebraic and finite-n checks for Theorem 3.

This script intentionally uses a binomial null for the numerical check because
all probabilities can then be summed, rather than estimated by Monte Carlo.
It verifies (i) the contraction algebra in the proof, (ii) the formula for
``v - 4 b``, and (iii) convergence of exact finite-n cumulants to ``b`` and
``v``.
"""

from __future__ import annotations

from fractions import Fraction as F
from math import lgamma, log

import numpy as np


def add(*terms: tuple[F, F, F, F]) -> tuple[F, F, F, F]:
    """Add coefficients in the basis (P1, K^2, K, 1)."""
    return tuple(sum(values, F(0)) for values in zip(*terms))  # type: ignore[return-value]


def exact_binomial_cumulants(n: int, p: float) -> tuple[float, float]:
    """Return exact mean and variance of the K=2 geodesic statistic."""
    x = np.arange(n + 1, dtype=float)
    logpmf = np.array(
        [
            lgamma(n + 1)
            - lgamma(int(j) + 1)
            - lgamma(n - int(j) + 1)
            + j * log(p)
            + (n - j) * log(1.0 - p)
            for j in x
        ]
    )
    pmf = np.exp(logpmf - np.max(logpmf))
    pmf /= np.sum(pmf)
    phat = x / n
    bc = np.sqrt(phat * p) + np.sqrt((1.0 - phat) * (1.0 - p))
    bc = np.clip(bc, -1.0, 1.0)
    statistic = n * (2.0 * np.arccos(bc)) ** 2
    mean = float(np.sum(pmf * statistic))
    variance = float(np.sum(pmf * (statistic - mean) ** 2))
    return mean, variance


def main() -> None:
    # Expansion of v in the proof, in the basis (P1, K^2, K, 1).
    q_variance = (F(1), F(-1), F(-2), F(2))
    minus_qh = (F(-9), F(3), F(18), F(-12))
    quarter_h_variance = (F(15, 4), F(-9, 4), F(-18, 4), F(3))
    quartic_j = (F(15, 2), F(0), F(-15), F(15, 2))
    quartic_q2 = (F(0), F(1, 6), F(0), F(-1, 6))
    v = add(q_variance, minus_qh, quarter_h_variance, quartic_j, quartic_q2)
    expected_v = (F(13, 4), F(-1, 12), F(-7, 2), F(1, 3))
    assert v == expected_v

    b = (F(7, 16), F(1, 48), F(-3, 8), F(-1, 12))
    residual = add(v, tuple(-4 * value for value in b))
    expected_residual = (F(3, 2), F(-1, 6), F(-2), F(2, 3))
    assert residual == expected_residual

    p = 0.30
    k = 2
    p1 = 1.0 / p + 1.0 / (1.0 - p)
    b_formula = 7.0 * p1 / 16.0 - 3.0 * k / 8.0 + (k * k - 4.0) / 48.0
    v_formula = 13.0 * p1 / 4.0 - k * k / 12.0 - 7.0 * k / 2.0 + 1.0 / 3.0

    ns = np.array([400, 800, 1600, 3200, 6400], dtype=float)
    mean_scaled = []
    variance_scaled = []
    for n_float in ns:
        mean, variance = exact_binomial_cumulants(int(n_float), p)
        mean_scaled.append(n_float * (mean - 1.0))
        variance_scaled.append(n_float * (variance - 2.0))

    design = np.column_stack([np.ones_like(ns), 1.0 / ns, 1.0 / ns**2])
    b_extrapolated = float(np.linalg.lstsq(design, mean_scaled, rcond=None)[0][0])
    v_extrapolated = float(np.linalg.lstsq(design, variance_scaled, rcond=None)[0][0])

    assert abs(b_extrapolated - b_formula) < 2e-5
    assert abs(v_extrapolated - v_formula) < 2e-4

    print("Symbolic contraction algebra: PASS")
    print("v coefficients [P1, K^2, K, 1]:", v)
    print("v - 4b coefficients:", residual)
    print(f"Exact-binomial b: formula={b_formula:.9f}, extrapolated={b_extrapolated:.9f}")
    print(f"Exact-binomial v: formula={v_formula:.9f}, extrapolated={v_extrapolated:.9f}")
    print(f"Exact-binomial v-4b={v_formula - 4.0 * b_formula:.9f} > 0")


if __name__ == "__main__":
    main()
