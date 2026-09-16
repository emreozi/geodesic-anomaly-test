"""
(c) Gaussian (location-scale) statistical manifold = hyperbolic plane.
Fisher metric  ds^2 = dmu^2/sigma^2 + 2 dsigma^2/sigma^2  -> in (mu/sqrt2, sigma)
it is twice the Poincare half-plane metric (curvature -1/2). Closed-form distance:
  d_FR(N1,N2) = sqrt(2)*arccosh(1 + ((mu1-mu2)^2/2 + (sig1-sig2)^2)/(2 sig1 sig2)).
Point-null test H0:(mu,sig)=(mu0,sig0):  Lambda_n = n d_FR(MLE,null)^2 -> chi^2_2.
Memory-safe (batched). Fixed seed.
"""
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"]="DejaVu Sans"; rcParams["mathtext.fontset"]="dejavusans"
rcParams["axes.spines.top"]=False; rcParams["axes.spines.right"]=False; rcParams["figure.dpi"]=120
RNG=np.random.default_rng(20260628)

def dFR(mu1,s1,mu0,s0):
    arg=1.0+(((mu1-mu0)**2)/2.0+(s1-s0)**2)/(2.0*s1*s0)
    return np.sqrt(2.0)*np.arccosh(np.clip(arg,1.0,None))

mu0,s0=2.0,1.5

def sample_stats(mu,sig,n,N,batch=40_000):
    fr=[]; eu=[]; done=0
    while done<N:
        m=min(batch,N-done); x=RNG.normal(mu,sig,(m,n)); mh=x.mean(1); sh=x.std(1)
        fr.append(n*dFR(mh,sh,mu0,s0)**2); eu.append(n*((mh-mu0)**2+(sh-s0)**2)); done+=m
    return np.concatenate(fr),np.concatenate(eu)

print("Gaussian point-null calibration (mu0=2, sigma0=1.5):")
for n in [50,100,200,400,800]:
    fr,_=sample_stats(mu0,s0,n,300_000)
    ks=stats.kstest(fr,"chi2",args=(2,)); lvl=np.mean(fr>stats.chi2.ppf(0.95,2))
    print(f"  n={n:4d}: E[Lam]={fr.mean():.3f}(=2) Var={fr.var():.3f}(=4) "
          f"KS D={ks.statistic:.4f} level@.05={lvl:.4f}")

n=200; frn,eun=sample_stats(mu0,s0,n,150_000)
tFR=np.quantile(frn,0.95); tE=np.quantile(eun,0.95)
ds=np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]); pf=[]; pe=[]
for d in ds:
    fr,eu=sample_stats(mu0,s0+d,n,100_000)
    pf.append(np.mean(fr>tFR)); pe.append(np.mean(eu>tE))
pf=np.array(pf); pe=np.array(pe)
print("\nScale-anomaly power (level 0.05, n=200):")
for d,a,b in zip(ds,pf,pe): print(f"  dsigma={d:.2f}: FR={a:.3f} Euclid={b:.3f}")

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.2,3.9))
q=np.linspace(0.5,99.5,200)
ax1.plot(stats.chi2.ppf(q/100,2),np.percentile(frn,q),color="#4C72B0",lw=2)
lim=[0,stats.chi2.ppf(0.999,2)]; ax1.plot(lim,lim,"--",color="#C44E52",lw=1.3)
ax1.set_xlabel(r"$\chi^2_2$ quantile"); ax1.set_ylabel(r"empirical quantile of $\Lambda_n$")
ax1.set_title(r"(a) Gaussian manifold: $\Lambda_n\to\chi^2_2$ ($n=200$)")
ax2.plot(ds,pf,"-o",color="#4C72B0",ms=5,label="Fisher-Rao (hyperbolic)")
ax2.plot(ds,pe,"-s",color="#DD8452",ms=5,label=r"Euclidean on $(\mu,\sigma)$")
ax2.axhline(0.05,color="0.6",ls=":",lw=1)
ax2.set_xlabel(r"scale anomaly $\Delta\sigma$"); ax2.set_ylabel("power (level 0.05)")
ax2.set_title("(b) Detecting a scale anomaly"); ax2.legend(frameon=False,fontsize=9); ax2.set_ylim(0,1.02)
fig.tight_layout(); fig.savefig("figures/fig_gaussian.pdf",bbox_inches="tight")
fig.savefig("figures/fig_gaussian.png",bbox_inches="tight",dpi=150)
print("\nwrote figures/fig_gaussian.pdf")
