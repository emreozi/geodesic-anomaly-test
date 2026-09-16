"""
Mean correction for the COMPOSITE statistic (independence submanifold).
Reduction:  Lambda_comp - X2_indep = n[-1/2 S3 + 5/16 S4 + 1/48 S2^2] at base = m-projection,
i.e. the SAME geodesic-curvature functional as Theorem 3, evaluated on the interaction residual.
b_comp = b_proj (Pearson-independence bias) + b_geo (geodesic curvature on residual).
We (1) confirm the reduction, (2) fit b_comp, and (3) assess the mean-matched scaling.
"""
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"]="DejaVu Sans"; rcParams["mathtext.fontset"]="dejavusans"
rcParams["axes.spines.top"]=False; rcParams["axes.spines.right"]=False; rcParams["figure.dpi"]=120
RNG=np.random.default_rng(20260628)
A,B=2,3; K=A*B; r=(K-1)-((A-1)+(B-1))
u0=np.array([0.6,0.4]); v0=np.array([0.5,0.35,0.15]); P0=np.outer(u0,v0).ravel()

def proj(p):
    P=p.reshape(p.shape[:-1]+(A,B)); return (P.sum(-1,keepdims=True)*P.sum(-2,keepdims=True)).reshape(p.shape)
def fr2(p,q):
    bc=np.clip(np.sum(np.sqrt(p*q),axis=-1),-1,1); return (2*np.arccos(bc))**2

# (1) confirm reduction at n=300
n=300; m=3_000_000
c=RNG.multinomial(n,P0,size=m); ph=c/n; pi=proj(ph); d=ph-pi
L=n*fr2(ph,pi); E=n*pi; X=np.sum((c-E)**2/np.maximum(E,1e-12),axis=1)
S3=np.sum(d**3/pi**2,axis=1); S4=np.sum(d**4/pi**3,axis=1); S2=np.sum(d**2/pi,axis=1)
recon=n*(-0.5*S3+(5/16)*S4+(1/48)*S2**2)
print(f"reduction check n={n}: mean(Lam-X2)={np.mean(L-X):.5f}  mean(recon)={np.mean(recon):.5f}")

# (2) fit b_comp from precise E[Lam]
ns_fit=[150,250,400,650]; EL=[]
for nn in ns_fit:
    N=6_000_000
    s=0.0; done=0
    while done<N:
        mm=min(2_000_000,N-done); c=RNG.multinomial(nn,P0,size=mm); ph=c/nn
        s+=np.sum(nn*fr2(ph,proj(ph))); done+=mm
    EL.append(s/N)
EL=np.array(EL); nsf=np.array(ns_fit,float)
# simple: fit E[Lam]=r + b/n using last two points (most asymptotic)
b_comp=(EL[-1]-r)*ns_fit[-1]
b_comp2=(EL[-2]-r)*ns_fit[-2]
# Richardson on last two
b_rich = ( (EL[-1]-r)*ns_fit[-1]*ns_fit[-1] - (EL[-2]-r)*ns_fit[-2]*ns_fit[-2] ) / (ns_fit[-1]-ns_fit[-2])
print(f"E[Lam]: {dict(zip(ns_fit,np.round(EL,4)))}")
print(f"n*(E-r) last two: {b_comp2:.3f}, {b_comp:.3f}  Richardson b_comp={b_rich:.3f}")
b_use=b_rich
print(f"USING b_comp={b_use:.3f}")

# (3) calibration improvement
ns=np.array([60,80,110,150,200,280,400]); N=1_500_000; thr=stats.chi2.ppf(0.95,r)
lvlu=[]; lvlc=[]; mean=[]
for nn in ns:
    c=RNG.multinomial(nn,P0,size=N); ph=c/nn; L=nn*fr2(ph,proj(ph))
    Lc=L/(1+b_use/(r*nn))
    mean.append(L.mean()); lvlu.append(np.mean(L>thr)); lvlc.append(np.mean(Lc>thr))
mean=np.array(mean); lvlu=np.array(lvlu); lvlc=np.array(lvlc)
print("\n n   E[Lam]  r+b/n  lvl_unc lvl_cor")
for i,nn in enumerate(ns):
    print(f" {nn:4d} {mean[i]:.3f}  {r+b_use/nn:.3f}  {lvlu[i]:.4f} {lvlc[i]:.4f}")

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.4,3.9))
nn=np.linspace(ns.min(),ns.max(),300)
ax1.plot(ns,mean,"o",color="#8172B3",ms=6,label=r"simulated $\mathbb{E}[\Lambda_n]$")
ax1.plot(nn,r+b_use/nn,"-",color="#C44E52",lw=2,label=fr"theory $r+b_{{\rm comp}}/n$, $b_{{\rm comp}}={b_use:.1f}$")
ax1.axhline(r,color="0.5",ls=":",lw=1.2,label=fr"$r={r}$")
ax1.set_xlabel("events per session $n$"); ax1.set_ylabel(r"$\mathbb{E}[\Lambda_n]$")
ax1.set_title("(a) Composite bias law"); ax1.legend(frameon=False,fontsize=9)
ax2.plot(ns,lvlu,"-o",color="#8172B3",ms=5,label=r"uncorrected $\Lambda_n$")
ax2.plot(ns,lvlc,"-s",color="#55A868",ms=5,label=r"mean-matched $\Lambda^*_n$")
ax2.axhline(0.05,color="#C44E52",ls="--",lw=1.4,label="nominal 0.05")
ax2.set_xlabel("events per session $n$"); ax2.set_ylabel(r"empirical level ($\alpha=0.05$)")
ax2.set_title("(b) Correction restores calibration"); ax2.legend(frameon=False,fontsize=9); ax2.set_ylim(0.03,None)
fig.tight_layout(); fig.savefig("figures/fig_composite_bartlett.pdf",bbox_inches="tight")
fig.savefig("figures/fig_composite_bartlett.png",bbox_inches="tight",dpi=150)
print("\nwrote figures/fig_composite_bartlett.pdf  b_comp=%.3f"%b_use)
