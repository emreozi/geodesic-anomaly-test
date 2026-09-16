"""Analytic and numerical checks for the scalar non-correctability result.

For P1 = sum_i 1/p_i, the revised manuscript proves

    b = 7 P1/16 - 3 K/8 + (K^2 - 4)/48,
    v = 13 P1/4 - K^2/12 - 7 K/2 + 1/3,
    v - 4 b = (9 P1 - K^2 - 12 K + 4)/6 > 0.

The default run prints these exact coefficients. Pass ``--simulate`` to add a
Monte-Carlo illustration; simulation is evidence for the figure, not the proof.
"""

from __future__ import annotations

import argparse

import numpy as np


def coefficients(p0: np.ndarray) -> dict[str, float]:
    """Return the closed-form first two cumulant coefficients."""
    p0 = np.asarray(p0, dtype=float)
    p0 = p0 / p0.sum()
    if np.any(p0 <= 0):
        raise ValueError("p0 must lie in the open probability simplex")
    k = len(p0)
    if k < 2:
        raise ValueError("at least two categories are required")
    p1 = float(np.sum(1.0 / p0))
    b = 7.0 * p1 / 16.0 - 3.0 * k / 8.0 + (k * k - 4.0) / 48.0
    v = 13.0 * p1 / 4.0 - k * k / 12.0 - 7.0 * k / 2.0 + 1.0 / 3.0
    residual = (9.0 * p1 - k * k - 12.0 * k + 4.0) / 6.0
    lower_bound = 2.0 * (k - 1.0) * (2.0 * k - 1.0) / 3.0
    return {
        "K": float(k),
        "P1": p1,
        "b": b,
        "v": v,
        "v_minus_4b": residual,
        "lower_bound": lower_bound,
    }


def fr_squared(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    bc = np.clip(np.sum(np.sqrt(p * q), axis=-1), -1.0, 1.0)
    return (2.0 * np.arccos(bc)) ** 2


def simulated_residual(
    p0: np.ndarray,
    n: int,
    replicates: int,
    rng: np.random.Generator,
    batch: int = 500_000,
) -> tuple[float, float]:
    """Estimate corrected residuals using Pearson control variates."""
    p0 = np.asarray(p0, dtype=float)
    p0 = p0 / p0.sum()
    values = coefficients(p0)
    k = len(p0)
    df = k - 1
    scale = 1.0 + values["b"] / (df * n)
    sum_delta = sum_score = sum_product = 0.0
    done = 0
    while done < replicates:
        size = min(batch, replicates - done)
        counts = rng.multinomial(n, p0, size=size)
        phat = counts / n
        lam = n * fr_squared(phat, p0) / scale
        pearson = n * np.sum((phat - p0) ** 2 / p0, axis=1)
        delta = lam - pearson
        score = lam + pearson
        sum_delta += float(delta.sum())
        sum_score += float(score.sum())
        sum_product += float((delta * score).sum())
        done += size
    mean_delta = sum_delta / replicates
    mean_score = sum_score / replicates
    variance_difference = sum_product / replicates - mean_delta * mean_score
    variance_pearson = (
        2.0 * df + (values["P1"] - k * k - 2.0 * df) / n
    )
    mean_residual = n * mean_delta
    variance_residual = n * (variance_pearson + variance_difference - 2.0 * df)
    return mean_residual, variance_residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--replicates", type=int, default=2_000_000)
    args = parser.parse_args()

    cases = {
        "uniform-4": np.full(4, 0.25),
        "reference-6": np.array([0.30, 0.24, 0.18, 0.12, 0.09, 0.07]),
    }
    rng = np.random.default_rng(17)
    for name, p0 in cases.items():
        values = coefficients(p0)
        print(f"\n{name}")
        for key in ("P1", "b", "v", "v_minus_4b", "lower_bound"):
            print(f"  {key:12s} = {values[key]:.9f}")
        if args.simulate:
            print("  n       mean residual       variance residual")
            for n in (200, 500, 1000):
                mean_res, variance_res = simulated_residual(
                    p0, n, args.replicates, rng
                )
                print(f"  {n:4d}    {mean_res:12.5f}       {variance_res:12.5f}")


if __name__ == "__main__":
    main()
