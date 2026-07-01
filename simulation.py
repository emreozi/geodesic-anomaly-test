"""
Information-geometric hypothesis testing on the categorical statistical manifold.
Produces REAL numbers and REAL figures used in the manuscript.

Model: each user session is summarised as a draw of n events over K action
types -> empirical distribution p_hat in the probability simplex Delta_{K-1}.
The simplex with the Fisher metric is isometric to the radius-2 sphere via
phi(p) = 2*sqrt(p).  Fisher-Rao (geodesic) distance:
    d_FR(p,q) = 2 * arccos( sum_i sqrt(p_i q_i) )      (Bhattacharyya angle x2)

Test statistic for H0: true law = p0 :
    Lambda_FR(p_hat) = n * d_FR(p_hat, p0)^2     -> chi^2_{K-1}  under H0.

Everything below is computed, not assumed.
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["figure.dpi"] = 120

RNG = np.random.default_rng(20260628)
EPS = 1e-12

# ----------------------------------------------------------------------
# Core geometry
# ----------------------------------------------------------------------
def bhattacharyya(p, q):
    return np.sum(np.sqrt(p * q), axis=-1)

def fisher_rao(p, q):
    bc = np.clip(bhattacharyya(p, q), -1.0, 1.0)
    return 2.0 * np.arccos(bc)

def euclid(p, q):
    return np.sqrt(np.sum((p - q) ** 2, axis=-1))

def empirical_dist(counts):
    n = counts.sum(axis=-1, keepdims=True)
    return counts / n

# ----------------------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------------------
K = 6                       # number of action types in the MIS dashboard
n_events = 200              # events per session (adequate-sample regime)
N_null = 200_000            # Monte-Carlo replicates under H0
labels = ["view", "search", "click", "cart", "purchase", "admin"]

# "normal behaviour" reference distribution p0 (min*n = 200*0.07 = 14 -> chi2 ok)
p0 = np.array([0.30, 0.24, 0.18, 0.12, 0.09, 0.07])
p0 = p0 / p0.sum()
print("p0 =", np.round(p0, 4), " sum =", p0.sum(),
      " min(n*p0) =", n_events * p0.min())

# ----------------------------------------------------------------------
# 1) THEOREM VALIDATION: null distribution of Lambda_FR  ->  chi^2_{K-1}
# ----------------------------------------------------------------------
counts_null = RNG.multinomial(n_events, p0, size=N_null)
phat_null = empirical_dist(counts_null)

d_null = fisher_rao(phat_null, p0)
Lam_FR_null = n_events * d_null ** 2                          # geodesic statistic
# Pearson chi-square (the classical score statistic) for reference
Lam_PEARSON = np.sum((counts_null - n_events * p0) ** 2 /
                     (n_events * p0), axis=-1)
# naive Euclidean statistic (NOT chi-square calibrated)
Lam_E_null = n_events * euclid(phat_null, p0) ** 2

df = K - 1
print("\n--- Null calibration (Theorem) ---")
print(f"E[Lambda_FR]      = {Lam_FR_null.mean():.3f}   (chi2_{df} mean = {df})")
print(f"Var[Lambda_FR]    = {Lam_FR_null.var():.3f}   (chi2_{df} var  = {2*df})")
print(f"corr(Lam_FR, Pearson) = "
      f"{np.corrcoef(Lam_FR_null, Lam_PEARSON)[0,1]:.4f}")

# KS test of Lambda_FR against chi^2_{K-1}
ks = stats.kstest(Lam_FR_null, "chi2", args=(df,))
print(f"KS(Lambda_FR, chi2_{df}): D = {ks.statistic:.4f}, p = {ks.pvalue:.3f}")

# empirical level of the chi^2 threshold at nominal alphas
for a in (0.10, 0.05, 0.01):
    thr = stats.chi2.ppf(1 - a, df)
    emp = np.mean(Lam_FR_null > thr)
    print(f"  nominal alpha={a:.2f}  chi2 thr={thr:5.2f}  "
          f"empirical level(FR) = {emp:.4f}")

# show Euclidean statistic is mis-calibrated if one (wrongly) used chi2 thr
print("  [Euclidean stat under chi2_5 thresholds -> badly off:]")
for a in (0.10, 0.05, 0.01):
    thr = stats.chi2.ppf(1 - a, df)
    emp = np.mean(Lam_E_null > thr)
    print(f"    nominal alpha={a:.2f}  empirical level(Eucl) = {emp:.4f}")

# ----------------------------------------------------------------------
# 2) ANOMALY DETECTION: Fisher-Rao vs Euclidean (calibrated per-method)
# ----------------------------------------------------------------------
# Calibrate each statistic to its OWN empirical null quantile (fair comparison)
def emp_threshold(stat_null, alpha):
    return np.quantile(stat_null, 1 - alpha)

thr_FR_05 = emp_threshold(Lam_FR_null, 0.05)
thr_E_05  = emp_threshold(Lam_E_null, 0.05)

# Family of anomalies: shift mass toward the RARE 'admin' channel (index 5),
# a classic security anomaly, parameterised by severity s in [0,1].
def anomaly_dist(s):
    p = p0.copy()
    delta = np.array([-0.10, -0.08, -0.04, 0.0, 0.04, 0.18])  # into 'admin'
    p = p + s * delta
    p = np.clip(p, 1e-3, None)
    return p / p.sum()

severities = np.linspace(0.0, 1.0, 11)
N_test = 40_000

def power_at(stat_fn, thr, p_alt):
    counts = RNG.multinomial(n_events, p_alt, size=N_test)
    ph = empirical_dist(counts)
    stat = stat_fn(ph)
    return np.mean(stat > thr)

stat_FR = lambda ph: n_events * fisher_rao(ph, p0) ** 2
stat_E  = lambda ph: n_events * euclid(ph, p0) ** 2

power_FR, power_E = [], []
for s in severities:
    pa = anomaly_dist(s)
    power_FR.append(power_at(stat_FR, thr_FR_05, pa))
    power_E.append(power_at(stat_E,  thr_E_05,  pa))
power_FR, power_E = np.array(power_FR), np.array(power_E)

print("\n--- Power vs severity (Type-I error fixed at 0.05 each) ---")
print("  s      power_FR   power_E")
for s, a, b in zip(severities, power_FR, power_E):
    print(f"  {s:0.2f}    {a:0.3f}     {b:0.3f}")

# ----------------------------------------------------------------------
# 3) ROC / AUC at a single moderate anomaly level
# ----------------------------------------------------------------------
s_roc = 0.5
p_alt = anomaly_dist(s_roc)
print(f"\nROC anomaly distribution (s={s_roc}):", np.round(p_alt, 4))

# scores under H0 (normal) and H1 (anomalous)
N_roc = 60_000
c0 = RNG.multinomial(n_events, p0,    size=N_roc); ph0 = empirical_dist(c0)
c1 = RNG.multinomial(n_events, p_alt, size=N_roc); ph1 = empirical_dist(c1)

sFR0, sFR1 = stat_FR(ph0), stat_FR(ph1)
sE0,  sE1  = stat_E(ph0),  stat_E(ph1)

def roc_curve(neg, pos, grid=400):
    lo = min(neg.min(), pos.min()); hi = max(neg.max(), pos.max())
    thr = np.linspace(lo, hi, grid)
    tpr = np.array([(pos > t).mean() for t in thr])
    fpr = np.array([(neg > t).mean() for t in thr])
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    auc = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc

fpr_FR, tpr_FR, auc_FR = roc_curve(sFR0, sFR1)
fpr_E,  tpr_E,  auc_E  = roc_curve(sE0,  sE1)
print(f"AUC  Fisher-Rao = {auc_FR:.4f}")
print(f"AUC  Euclidean  = {auc_E:.4f}")

# ----------------------------------------------------------------------
# 4) CONVERGENCE: E[Lambda_FR] -> df and empirical level -> alpha as n grows
# ----------------------------------------------------------------------
n_grid = [25, 50, 75, 100, 150, 200, 300, 500, 800]
N_conv = 60_000
mean_curve, level_curve = [], []
thr05 = stats.chi2.ppf(0.95, df)
for n in n_grid:
    c = RNG.multinomial(n, p0, size=N_conv)
    ph = empirical_dist(c)
    lam = n * fisher_rao(ph, p0) ** 2
    mean_curve.append(lam.mean())
    level_curve.append(np.mean(lam > thr05))
mean_curve = np.array(mean_curve); level_curve = np.array(level_curve)
print("\n--- Convergence of Lambda_FR to chi^2_{K-1} ---")
for n, m, l in zip(n_grid, mean_curve, level_curve):
    print(f"  n={n:4d}  E[Lam]={m:5.2f} (->{df})  level@0.05={l:.3f} (->0.05)")

# ----------------------------------------------------------------------
# FIGURE 2: null calibration histogram vs chi^2 density + QQ + convergence
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 3.8))
xx = np.linspace(0, 25, 400)
ax1.hist(Lam_FR_null, bins=120, density=True, color="#4C72B0",
         alpha=0.55, label=r"$\Lambda_{FR}=n\,d_{FR}^2$ (simulated)")
ax1.plot(xx, stats.chi2.pdf(xx, df), color="#C44E52", lw=2.2,
         label=fr"$\chi^2_{{{df}}}$ density")
ax1.set_xlim(0, 25); ax1.set_xlabel("statistic value")
ax1.set_ylabel("density")
ax1.set_title("(a) Null distribution of the geodesic statistic")
ax1.legend(frameon=False, fontsize=9)

# QQ plot
probs = np.linspace(0.5 / N_null, 1 - 0.5 / N_null, 2000)
q_emp = np.quantile(Lam_FR_null, probs)
q_th = stats.chi2.ppf(probs, df)
ax2.plot(q_th, q_emp, ".", ms=2, color="#4C72B0")
lim = [0, max(q_th.max(), q_emp.max())]
ax2.plot(lim, lim, color="#C44E52", lw=1.6)
ax2.set_xlabel(fr"$\chi^2_{{{df}}}$ theoretical quantile")
ax2.set_ylabel("empirical quantile")
ax2.set_title("(b) Q–Q plot")
ax2.set_xlim(lim); ax2.set_ylim(lim)

# convergence panel
ax3b = ax3.twinx()
l1 = ax3.plot(n_grid, mean_curve, "-o", color="#4C72B0", ms=5,
              label=r"$\mathbb{E}[\Lambda_{FR}]$")
ax3.axhline(df, color="#4C72B0", ls=":", lw=1)
l2 = ax3b.plot(n_grid, level_curve, "-s", color="#DD8452", ms=5,
               label="empirical level @0.05")
ax3b.axhline(0.05, color="#DD8452", ls=":", lw=1)
ax3.set_xlabel("events per session $n$")
ax3.set_ylabel(r"$\mathbb{E}[\Lambda_{FR}]$", color="#4C72B0")
ax3b.set_ylabel("empirical level", color="#DD8452")
ax3.set_title("(c) Convergence to $\\chi^2_{5}$")
ax3.set_ylim(df - 1, mean_curve.max() + 1)
ax3b.set_ylim(0.03, max(level_curve) + 0.02)
lns = l1 + l2
ax3.legend(lns, [x.get_label() for x in lns], frameon=False, fontsize=8.5,
           loc="upper right")
fig.tight_layout()
fig.savefig("figs/fig_null_calibration.pdf", bbox_inches="tight")
fig.savefig("figs/fig_null_calibration.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# FIGURE 3: ROC curves
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 4.4))
ax.plot(fpr_FR, tpr_FR, color="#4C72B0", lw=2.2,
        label=fr"Fisher–Rao geodesic (AUC={auc_FR:.3f})")
ax.plot(fpr_E, tpr_E, color="#DD8452", lw=2.2, ls="--",
        label=fr"Euclidean (AUC={auc_E:.3f})")
ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls=":")
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_title(f"Anomaly detection ROC (severity s={s_roc})")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig("figs/fig_roc.pdf", bbox_inches="tight")
fig.savefig("figs/fig_roc.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# FIGURE 4: power vs severity
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.0, 4.0))
ax.plot(severities, power_FR, "-o", color="#4C72B0", ms=5,
        label="Fisher–Rao geodesic test")
ax.plot(severities, power_E, "-s", color="#DD8452", ms=5,
        label="Euclidean test")
ax.axhline(0.05, color="0.6", ls=":", lw=1)
ax.set_xlabel("anomaly severity $s$")
ax.set_ylabel("detection power (at level 0.05)")
ax.set_title("Power comparison (Type-I error fixed)")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig("figs/fig_power.pdf", bbox_inches="tight")
fig.savefig("figs/fig_power.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# FIGURE 1: geometry illustration (K=3 simplex -> sphere octant)
# geodesic (great-circle) vs Euclidean chord
# ----------------------------------------------------------------------
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kw):
        super().__init__((0, 0), (0, 0), *args, **kw)
        self._xyz = (xs, ys, zs)
    def do_3d_projection(self, renderer=None):
        (x0, x1), (y0, y1), (z0, z1) = self._xyz
        xs, ys, zs = proj_transform((x0, x1), (y0, y1), (z0, z1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

emb = lambda p: 2.0 * np.sqrt(p)        # radius-2 sphere
pA = np.array([0.55, 0.35, 0.10])       # "normal" reference
pB = np.array([0.12, 0.33, 0.55])       # an anomalous session
A, B = emb(pA), emb(pB)

# great-circle geodesic between A and B on the radius-2 sphere
def geodesic(A, B, m=80):
    a = A / np.linalg.norm(A); b = B / np.linalg.norm(B)
    om = np.arccos(np.clip(a @ b, -1, 1))
    t = np.linspace(0, 1, m)[:, None]
    pts = (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om)
    return 2.0 * pts

fig = plt.figure(figsize=(5.6, 5.0))
ax = fig.add_subplot(111, projection="3d")
u = np.linspace(0, np.pi / 2, 60); v = np.linspace(0, np.pi / 2, 60)
U, V = np.meshgrid(u, v)
ax.plot_surface(2*np.sin(U)*np.cos(V), 2*np.sin(U)*np.sin(V), 2*np.cos(U),
                color="#9CC", alpha=0.18, linewidth=0, shade=True)
G = geodesic(A, B)
ax.plot(G[:, 0], G[:, 1], G[:, 2], color="#C44E52", lw=2.6,
        label="Fisher–Rao geodesic")
ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color="#DD8452",
        lw=2.0, ls="--", label="Euclidean chord")
ax.scatter(*A, color="#1f3b73", s=55); ax.text(A[0], A[1], A[2]+0.12,
            r"$p_0$ (normal)", fontsize=10)
ax.scatter(*B, color="#7a1f1f", s=55); ax.text(B[0]-0.1, B[1], B[2]+0.12,
            r"anomalous $\hat p$", fontsize=10)
ax.set_xlabel(r"$2\sqrt{p_1}$"); ax.set_ylabel(r"$2\sqrt{p_2}$")
ax.set_zlabel(r"$2\sqrt{p_3}$")
ax.set_title(r"Simplex $\Delta_2$ embedded on the sphere $\;\phi(p)=2\sqrt{p}$")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.view_init(elev=22, azim=35)
fig.tight_layout()
fig.savefig("figs/fig_geometry.pdf", bbox_inches="tight")
fig.savefig("figs/fig_geometry.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# distances for the illustrative pair (printed for the text)
print(f"\nIllustrative pair  d_FR = {fisher_rao(pA,pB):.4f}, "
      f"d_Eucl(simplex) = {euclid(pA,pB):.4f}")

# ----------------------------------------------------------------------
# Save a compact results table for the manuscript
# ----------------------------------------------------------------------
with open("results_summary.txt", "w") as f:
    f.write("p0 = " + ", ".join(f"{x:.3f}" for x in p0) + "\n")
    f.write(f"n_events={n_events}, K={K}, N_null={N_null}\n")
    f.write(f"E[Lam_FR]={Lam_FR_null.mean():.3f} (target {df})\n")
    f.write(f"Var[Lam_FR]={Lam_FR_null.var():.3f} (target {2*df})\n")
    f.write(f"KS D={ks.statistic:.4f}, p={ks.pvalue:.3f}\n")
    f.write(f"corr(FR,Pearson)={np.corrcoef(Lam_FR_null,Lam_PEARSON)[0,1]:.4f}\n")
    f.write(f"AUC_FR={auc_FR:.4f}, AUC_E={auc_E:.4f}\n")
    for a in (0.10,0.05,0.01):
        thr=stats.chi2.ppf(1-a,df)
        f.write(f"alpha={a}: levelFR={np.mean(Lam_FR_null>thr):.4f}, "
                f"levelEucl={np.mean(Lam_E_null>thr):.4f}\n")
    f.write("severity,power_FR,power_E\n")
    for s,a,b in zip(severities,power_FR,power_E):
        f.write(f"{s:.2f},{a:.3f},{b:.3f}\n")
    f.write("n,E_Lam_FR,level05\n")
    for n,m,l in zip(n_grid,mean_curve,level_curve):
        f.write(f"{n},{m:.3f},{l:.4f}\n")

print("\nAll figures written to figs/. Summary -> results_summary.txt")
