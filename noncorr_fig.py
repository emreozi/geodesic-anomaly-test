"""
Figure for the non-correctability theorem (v != 4b).

Companion to noncorr.py: for the reference null (K=6) it plots, for the
MEAN-corrected geodesic statistic  Lam*_n = Lam_n / (1 + b/((K-1) n)):
  (a) first-cumulant residual   n*(E[Lam*_n]  - (K-1)) -> 0   (mean corrected)
  (b) second-cumulant residual  n*(Var[Lam*_n] - 2(K-1)) -> v - 4b != 0
so a single scalar cannot correct both moments. Writes fig_noncorr.pdf.

Panel (a) uses the control variate Lam* - X^2 (Pearson's statistic, whose null
mean is exactly K-1), which removes the leading chi-square fluctuation and
yields a low-variance estimate of the O(1/n) mean bias -- the same
variance-reduction device used in noncorr.py. Deterministic (fixed seed).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["figure.dpi"] = 120

RNG = np.random.default_rng(17)                 # matches noncorr.py
GREEN, RED, PURPLE = "#55A868", "#C44E52", "#8172B3"


def bcoef(p0):
    K = len(p0)
    A3 = np.sum((1 - p0) * (1 - 2 * p0) / p0)
    A4 = np.sum((1 - p0) ** 2 / p0)
    return (15 / 16) * A4 - 0.5 * A3 + (K ** 2 - 1) / 48


def fr2(p, q):
    bc = np.clip(np.sum(np.sqrt(p * q), axis=-1), -1, 1)
    return (2 * np.arccos(bc)) ** 2


def residuals(p0, n, N, b, batch=1_000_000):
    """
    Returns
      mean_res = n*(E[Lam*] - df)     via the control variate Lam* - X^2 (E[X^2]=df exact)
      var_res  = n*(Var[Lam*] - 2df)  direct
    """
    K = len(p0); df = K - 1; c = 1.0 + b / (df * n)
    sd = 0.0            # sum of (Lam* - X2)
    s1 = s2 = 0.0       # sum, sum-of-squares of Lam*
    done = 0
    while done < N:
        m = min(batch, N - done)
        cc = RNG.multinomial(n, p0, size=m); ph = cc / n
        Ls = (n * fr2(ph, p0)) / c
        X2 = n * np.sum((ph - p0) ** 2 / p0, axis=1)   # Pearson (E[X2]=df exact)
        sd += (Ls - X2).sum()
        s1 += Ls.sum(); s2 += (Ls * Ls).sum(); done += m
    mLs = s1 / N; vLs = s2 / N - mLs * mLs
    mean_res = n * (sd / N)                # n*E[Lam*-X2] = n*(E[Lam*]-df)
    var_res = n * (vLs - 2 * df)
    return mean_res, var_res


# reference null of the simulation study (Section on simulation), K = 6
p0 = np.array([0.30, 0.24, 0.18, 0.12, 0.09, 0.07]); p0 = p0 / p0.sum()
K = len(p0); df = K - 1; b = bcoef(p0)
n_grid = [80, 110, 150, 200, 280, 380, 520]
N = 2_000_000

r1, r2 = [], []
print(f"reference null K={K}  b={b:.3f}")
print(" n     n*(E[Lam*]-df)   n*(Var[Lam*]-2df)")
for n in n_grid:
    a, d = residuals(p0, n, N, b)
    r1.append(a); r2.append(d)
    print(f" {n:4d}   {a:8.3f}        {d:8.3f}")
r1, r2 = np.array(r1), np.array(r2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))

ax1.axhline(0.0, color=RED, ls="--", lw=1.4, label="correctable target 0")
ax1.plot(n_grid, r1, "-o", color=GREEN, ms=6)
ax1.set_xlabel("events per session $n$")
ax1.set_ylabel(r"$n\,(\mathbb{E}[\Lambda^\ast_n]-(K-1))$")
ax1.set_title(r"(a) Mean is corrected: $\to 0$")
ax1.set_ylim(-3, 3)
ax1.legend(frameon=False, fontsize=9, loc="lower right")

ax2.plot(n_grid, r2, "-o", color=PURPLE, ms=6,
         label=r"$n\,(\mathrm{Var}[\Lambda^\ast_n]-2(K-1))$")
ax2.set_xlabel("events per session $n$")
ax2.set_ylabel(r"$n\,(\mathrm{Var}[\Lambda^\ast_n]-2(K-1))$")
ax2.set_title(r"(b) Variance is not: $\to v-4b \neq 0$")
ax2.set_ylim(0, max(r2) * 1.12)
ax2.legend(frameon=False, fontsize=9, loc="upper right")

fig.tight_layout()
os.makedirs("figs", exist_ok=True)
fig.savefig("figs/fig_noncorr.pdf", bbox_inches="tight")
fig.savefig("figs/fig_noncorr.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("wrote figs/fig_noncorr.pdf")
