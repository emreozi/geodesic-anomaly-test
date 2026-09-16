"""
Figure for the NSL-KDD real-data application (Section: real-data).

Companion to nslkdd_experiment.py. Produces fig_nslkdd.pdf:
  (a) normal vs attack TCP-flag profiles on a log scale: attacks redistribute
      mass into the rare flag channels (S0 rises from ~0.006 to ~0.59).
  (b) the geodesic statistic Lambda_n on genuine benign sessions is
      overdispersed relative to the asymptotic chi^2_{K-1} law (mean ~7.8 vs 5),
      so the operating threshold (dotted) is set from the empirical 95% quantile.

Data: the NSL-KDD 20% subset (Tavallaee et al. 2009). Save the benchmark file
KDDTrain+_20Percent.txt as data/nslkdd20.txt, exactly as nslkdd_experiment.py
expects. Fixed seed; the loading, split, and statistic match that script.
"""
import csv
import os
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

RNG = np.random.default_rng(20260628)             # matches nslkdd_experiment.py
BLUE, RED, PURPLE = "#4C72B0", "#C44E52", "#8172B3"

# ---- load flag + label (identical to nslkdd_experiment.py) ----
CATS = ["SF", "S0", "REJ", "RSTR", "RSTO"]
K = len(CATS) + 1                                 # +1 "other"
LABELS = CATS + ["other"]


def cat_index(flag):
    return CATS.index(flag) if flag in CATS else K - 1


normal, attack = [], []
with open("data/nslkdd20.txt") as f:
    for row in csv.reader(f):
        ci = cat_index(row[3]); lab = row[41]
        (normal if lab == "normal" else attack).append(ci)
normal = np.array(normal); attack = np.array(attack)

# ---- split normal: estimate p0 on half, evaluate on the other half ----
perm = RNG.permutation(len(normal)); half = len(normal) // 2
train_n = normal[perm[:half]]; eval_n = normal[perm[half:]]
p0 = np.bincount(train_n, minlength=K).astype(float); p0 /= p0.sum()
attack_prof = np.bincount(attack, minlength=K).astype(float); attack_prof /= attack_prof.sum()
print("p0 (normal) =", np.round(p0, 4))
print("attack prof =", np.round(attack_prof, 4))


def sessions(pool, nsess, n):
    idx = RNG.integers(0, len(pool), size=(nsess, n))
    draws = pool[idx]
    C = np.zeros((nsess, K))
    for k in range(K):
        C[:, k] = (draws == k).sum(1)
    return C / n


def fr2(ph):
    bc = np.clip(np.sum(np.sqrt(ph * p0), axis=1), -1, 1)
    return (2 * np.arccos(bc)) ** 2


# ---- benign geodesic statistic + empirical threshold ----
n = 200; Nnull = 40_000; df = K - 1
ph_norm = sessions(eval_n, Nnull, n)
Lfr = n * fr2(ph_norm)
tFR = np.quantile(Lfr, 0.95)
mean_L = Lfr.mean()
print(f"benign Lambda_n: mean={mean_L:.2f} (chi2_{df} mean={df}), "
      f"var={Lfr.var():.2f} (chi2 var={2*df});  empirical 95% thr={tFR:.2f}")

# ======================================================================
# FIGURE
# ======================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))

# --- (a) flag profiles, log scale ---
x = np.arange(K); w = 0.4
ax1.bar(x - w / 2, p0, w, color=BLUE, label=r"normal $p_0$")
ax1.bar(x + w / 2, attack_prof, w, color=RED, label="attack")
ax1.set_yscale("log")
ax1.set_ylim(1e-4, 1.4)
ax1.set_xticks(x); ax1.set_xticklabels(LABELS)
ax1.set_xlabel("TCP flag state")
ax1.set_ylabel("channel probability (log scale)")
ax1.set_title("(a) Flag profile: normal vs attack")
ax1.legend(frameon=False, fontsize=9, loc="upper right")
ax1.annotate("rare channels\nfill under attack",
             xy=(1 + w / 2, attack_prof[1]), xytext=(2.4, 0.4),
             fontsize=9, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color="0.4", lw=1.1))

# --- (b) overdispersed benign null ---
xmax = 25.0
xx = np.linspace(0, xmax, 400)
ax2.hist(Lfr, bins=120, range=(0, xmax), density=True, color=PURPLE,
         alpha=0.55, label=r"real normal $\Lambda_n$")
ax2.plot(xx, stats.chi2.pdf(xx, df), color=RED, lw=2.4,
         label=fr"$\chi^2_{{{df}}}$ (asymptotic null)")
ax2.axvline(df, color=RED, ls="--", lw=1.3)
ax2.axvline(mean_L, color="#3d3d6b", lw=1.3)
ax2.axvline(tFR, color="0.25", ls=":", lw=1.6, label="empirical 95% thr.")
ax2.annotate(fr"mean {mean_L:.1f} vs {df}",
             xy=(mean_L, 0.02), xytext=(mean_L + 3.5, 0.06),
             fontsize=9, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color="0.4", lw=1.1))
ax2.set_xlim(0, xmax)
ax2.set_xlabel(fr"$\Lambda_n$ on normal sessions ($n={n}$)")
ax2.set_ylabel("density")
ax2.set_title("(b) Real null is overdispersed")
ax2.legend(frameon=False, fontsize=8.5, loc="upper right")

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig_nslkdd.pdf", bbox_inches="tight")
fig.savefig("figures/fig_nslkdd.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("wrote figures/fig_nslkdd.pdf")
