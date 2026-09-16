"""
Composite-null hypothesis test on the categorical manifold.

Null hypothesis: the true law lies on a submanifold M0 of the simplex.
Test statistic:   Lambda_n = n * d_FR(p_hat, M0)^2 ,  the squared Fisher-Rao
geodesic distance from the empirical law to M0 (geodesic projection).

Canonical example (interpretable for a DSS): a 2x3 contingency
  device {mobile, desktop} x action {view, click, purchase}
M0 = independence submanifold  {P_ab = u_a v_b}.
  dim M0 = (2-1)+(3-1) = 3 ,  codim r = (K-1) - 3 = 5 - 3 = 2.
Theory => Lambda_n -> chi^2_2 under H0.

The geodesic projection onto M0 is, to leading order, the product of the
empirical marginals (the m-projection); we use it as the projection point, so
  Lambda_n = n * d_FR(p_hat, p_hat_indep)^2.
"""

from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "DejaVu Sans"
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["figure.dpi"] = 120

RNG = np.random.default_rng(20260628)

A, B = 2, 3                       # device x action
K = A * B                         # = 6 cells
r = (K - 1) - ((A - 1) + (B - 1)) # codim of independence submanifold = 2
n_events = 300
N_null = 200_000

def fisher_rao(p, q):
    bc = np.clip(np.sum(np.sqrt(p * q), axis=-1), -1.0, 1.0)
    return 2.0 * np.arccos(bc)

def indep_projection(p):
    """m-projection of a (..,K) categorical onto the independence submanifold."""
    P = p.reshape(p.shape[:-1] + (A, B))
    row = P.sum(axis=-1, keepdims=True)         # marginal over actions
    col = P.sum(axis=-2, keepdims=True)         # marginal over devices
    Pind = row * col
    return Pind.reshape(p.shape)

# reference independent law P0 = u0 (x) v0
u0 = np.array([0.6, 0.4])                 # mobile, desktop
v0 = np.array([0.5, 0.35, 0.15])          # view, click, purchase
P0 = np.outer(u0, v0).ravel()
print("P0 cells =", np.round(P0, 4), " min*n =", n_events * P0.min())

# ----------------------------------------------------------------------
# 1) Null calibration:  Lambda_n -> chi^2_2
# ----------------------------------------------------------------------
counts = RNG.multinomial(n_events, P0, size=N_null)
phat = counts / n_events
phat_ind = indep_projection(phat)
Lam = n_events * fisher_rao(phat, phat_ind) ** 2

print("\n--- Composite null calibration (independence submanifold) ---")
print(f"codim r = {r}")
print(f"E[Lambda]   = {Lam.mean():.3f}  (chi2_{r} mean = {r})")
print(f"Var[Lambda] = {Lam.var():.3f}  (chi2_{r} var  = {2*r})")
ks = stats.kstest(Lam, "chi2", args=(r,))
print(f"KS vs chi2_{r}: D = {ks.statistic:.4f}")
# classical Pearson independence statistic for cross-check
Eind = n_events * phat_ind
Pearson = np.sum((counts - Eind) ** 2 / Eind, axis=-1)
print(f"corr(Lambda, Pearson-indep) = {np.corrcoef(Lam, Pearson)[0,1]:.4f}")
for a in (0.10, 0.05, 0.01):
    thr = stats.chi2.ppf(1 - a, r)
    print(f"  alpha={a:.2f}  chi2 thr={thr:5.2f}  emp level={np.mean(Lam>thr):.4f}")

# ----------------------------------------------------------------------
# 2) Power vs dependence severity (geodesic vs Euclidean), level 0.05
# ----------------------------------------------------------------------
# pure interaction that preserves marginals: rank-1, mean-zero in each margin
alpha = np.array([-1.0, 1.0])             # device contrast (sum 0)
beta = np.array([-1.0, 0.0, 1.0])         # action contrast (sum 0)
Eint = np.outer(alpha, beta)              # interaction pattern (2x3), margins 0
Eint = Eint / np.abs(Eint).max()

def dependent_law(s):
    P = P0.reshape(A, B) + s * 0.09 * Eint   # add dependence, keep margins
    P = np.clip(P, 1e-3, None)
    P = P / P.sum()
    return P.ravel()

# calibrate each statistic to its own empirical null quantile at 0.05
N_cal = 100_000
c = RNG.multinomial(n_events, P0, size=N_cal); ph = c / n_events
phI = indep_projection(ph)
stat_FR_null = n_events * fisher_rao(ph, phI) ** 2
stat_E_null  = n_events * np.sum((ph - phI) ** 2, axis=-1)
thr_FR = np.quantile(stat_FR_null, 0.95)
thr_E  = np.quantile(stat_E_null, 0.95)

severities = np.linspace(0.0, 1.0, 11)
N_test = 40_000
pw_FR, pw_E = [], []
for s in severities:
    Ps = dependent_law(s)
    c = RNG.multinomial(n_events, Ps, size=N_test); ph = c / n_events
    phI = indep_projection(ph)
    sFR = n_events * fisher_rao(ph, phI) ** 2
    sE  = n_events * np.sum((ph - phI) ** 2, axis=-1)
    pw_FR.append(np.mean(sFR > thr_FR))
    pw_E.append(np.mean(sE > thr_E))
pw_FR, pw_E = np.array(pw_FR), np.array(pw_E)
print("\n--- Power vs dependence severity (level 0.05) ---")
print("  s     power_FR  power_E")
for s, a, b in zip(severities, pw_FR, pw_E):
    print(f"  {s:0.2f}   {a:0.3f}    {b:0.3f}")

# AUC at moderate severity
s_roc = 0.5
N_roc = 60_000
c0 = RNG.multinomial(n_events, P0, size=N_roc); p0h = c0/n_events
c1 = RNG.multinomial(n_events, dependent_law(s_roc), size=N_roc); p1h = c1/n_events
def stat_fr(x): xi=indep_projection(x); return n_events*fisher_rao(x,xi)**2
def stat_e(x):  xi=indep_projection(x); return n_events*np.sum((x-xi)**2,axis=-1)
def auc(neg,pos,g=400):
    lo=min(neg.min(),pos.min()); hi=max(neg.max(),pos.max())
    t=np.linspace(lo,hi,g)
    tpr=np.array([(pos>x).mean() for x in t]); fpr=np.array([(neg>x).mean() for x in t])
    o=np.argsort(fpr); return np.trapezoid(tpr[o],fpr[o])
auc_FR=auc(stat_fr(p0h),stat_fr(p1h)); auc_E=auc(stat_e(p0h),stat_e(p1h))
print(f"\nAUC (s={s_roc}): FR={auc_FR:.4f}, Euclid={auc_E:.4f}")

# ----------------------------------------------------------------------
# FIGURE: composite calibration + power
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.9))
xx = np.linspace(0, 16, 400)
ax1.hist(Lam, bins=120, density=True, color="#55A868", alpha=0.55,
         label=r"$\Lambda_n=n\,d_{FR}(\hat p,M_0)^2$ (simulated)")
ax1.plot(xx, stats.chi2.pdf(xx, r), color="#C44E52", lw=2.2,
         label=fr"$\chi^2_{{{r}}}$ density")
ax1.set_xlim(0, 16); ax1.set_xlabel("statistic value"); ax1.set_ylabel("density")
ax1.set_title(r"(a) Composite null: distance to $M_0$ (independence)")
ax1.legend(frameon=False, fontsize=9)

ax2.plot(severities, pw_FR, "-o", color="#55A868", ms=5,
         label="Fisher–Rao geodesic")
ax2.plot(severities, pw_E, "-s", color="#DD8452", ms=5, label="Euclidean")
ax2.axhline(0.05, color="0.6", ls=":", lw=1)
ax2.set_xlabel("dependence severity $s$")
ax2.set_ylabel("detection power (level 0.05)")
ax2.set_title("(b) Power: detecting induced dependence")
ax2.legend(frameon=False, fontsize=9, loc="lower right")
ax2.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig("figures/fig_composite.pdf", bbox_inches="tight")
fig.savefig("figures/fig_composite.png", bbox_inches="tight", dpi=150)
plt.close(fig)

Path("data/derived").mkdir(parents=True, exist_ok=True)
with open("data/derived/results_composite.txt", "w") as f:
    f.write(f"K={K}, A={A}, B={B}, r={r}, n={n_events}\n")
    f.write(f"E[Lam]={Lam.mean():.3f} (target {r})\n")
    f.write(f"Var[Lam]={Lam.var():.3f} (target {2*r})\n")
    f.write(f"KS D={ks.statistic:.4f}\n")
    f.write(f"corr(Lam,Pearson-indep)={np.corrcoef(Lam,Pearson)[0,1]:.4f}\n")
    f.write(f"AUC_FR={auc_FR:.4f}, AUC_E={auc_E:.4f}\n")
    for a in (0.10,0.05,0.01):
        thr=stats.chi2.ppf(1-a,r); f.write(f"alpha={a}: level={np.mean(Lam>thr):.4f}\n")
    f.write("severity,power_FR,power_E\n")
    for s,a,b in zip(severities,pw_FR,pw_E):
        f.write(f"{s:.2f},{a:.3f},{b:.3f}\n")
print("\nWrote figures/fig_composite.pdf and data/derived/results_composite.txt")
