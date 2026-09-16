"""
Second-order (curvature) mean correction.
  E[Lambda_n] = (K-1) + b/n + o(1/n),
  b = (15/16) A4 - (1/2) A3 + (K^2-1)/48.
Mean-matched Bartlett-type statistic:
  Lambda*_n = Lambda_n / (1 + b/((K-1) n)).
We (a) confirm the b/n bias curve and (b) show the correction restores the
nominal test level at small n.
"""
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"]="DejaVu Sans"; rcParams["mathtext.fontset"]="dejavusans"
rcParams["axes.spines.top"]=False; rcParams["axes.spines.right"]=False
rcParams["figure.dpi"]=120
RNG=np.random.default_rng(20260628)

p0=np.array([0.30,0.24,0.18,0.12,0.09,0.07]); p0/=p0.sum(); K=len(p0); df=K-1
A3=np.sum((1-p0)*(1-2*p0)/p0); A4=np.sum((1-p0)**2/p0)
b=(15/16)*A4-0.5*A3+(K**2-1)/48
print(f"K={K}, A3={A3:.4f}, A4={A4:.4f}, b={b:.4f}")

def lam(p_hat,n): 
    bc=np.clip(np.sum(np.sqrt(p_hat*p0),axis=1),-1,1)
    return n*(2*np.arccos(bc))**2

ns=np.array([30,40,50,60,75,90,110,140,180,240,320,420])
N=1_500_000
thr=stats.chi2.ppf(0.95,df)

mean_unc=[]; lvl_unc=[]; lvl_cor=[]
for n in ns:
    c=RNG.multinomial(n,p0,size=N); ph=c/n
    L=lam(ph,n)
    Lc=L/(1+b/(df*n))                      # mean-matched scaling
    mean_unc.append(L.mean())
    lvl_unc.append(np.mean(L>thr))
    lvl_cor.append(np.mean(Lc>thr))
mean_unc=np.array(mean_unc); lvl_unc=np.array(lvl_unc); lvl_cor=np.array(lvl_cor)

print("\n n    E[Lam]   K-1+b/n   level_unc level_cor")
for i,n in enumerate(ns):
    print(f" {n:4d}  {mean_unc[i]:.3f}   {df+b/n:.3f}     "
          f"{lvl_unc[i]:.4f}    {lvl_cor[i]:.4f}")

# ---------------- figure ----------------
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.4,3.9))
nn=np.linspace(ns.min(),ns.max(),300)
ax1.plot(ns,mean_unc,"o",color="#4C72B0",ms=6,label=r"simulated $\mathbb{E}[\Lambda_n]$")
ax1.plot(nn,df+b/nn,"-",color="#C44E52",lw=2,
         label=fr"theory $(K\!-\!1)+b/n$, $b={b:.2f}$")
ax1.axhline(df,color="0.5",ls=":",lw=1.2,label=fr"$K-1={df}$")
ax1.set_xlabel("events per session $n$"); ax1.set_ylabel(r"$\mathbb{E}[\Lambda_n]$")
ax1.set_title("(a) Second-order bias law")
ax1.legend(frameon=False,fontsize=9)

ax2.plot(ns,lvl_unc,"-o",color="#4C72B0",ms=5,label=r"uncorrected $\Lambda_n$")
ax2.plot(ns,lvl_cor,"-s",color="#55A868",ms=5,label=r"mean-matched $\Lambda^*_n$")
ax2.axhline(0.05,color="#C44E52",ls="--",lw=1.4,label="nominal 0.05")
ax2.set_xlabel("events per session $n$"); ax2.set_ylabel(r"empirical level ($\alpha=0.05$)")
ax2.set_title("(b) Correction restores calibration")
ax2.legend(frameon=False,fontsize=9); ax2.set_ylim(0.03,None)
fig.tight_layout()
fig.savefig("figures/fig_curvature.pdf",bbox_inches="tight")
fig.savefig("figures/fig_curvature.png",bbox_inches="tight",dpi=150)
print("\nwrote figures/fig_curvature.pdf")

Path("data/derived").mkdir(parents=True, exist_ok=True)
with open("data/derived/results_curvature.txt","w") as f:
    f.write(f"K={K} b={b:.4f} A3={A3:.4f} A4={A4:.4f}\n")
    f.write("n,E_Lam,theory,level_unc,level_cor\n")
    for i,n in enumerate(ns):
        f.write(f"{n},{mean_unc[i]:.4f},{df+b/n:.4f},{lvl_unc[i]:.4f},{lvl_cor[i]:.4f}\n")
