"""
Non-correctability witness: a single-scalar (Bartlett) scaling fixes mean & variance
to O(1/n^2) iff v = 4b, where E[Lam]=(K-1)+b/n, Var[Lam]=2(K-1)+v/n.
We measure  n*(Var(Lam*_n) - 2(K-1)) -> (v - 4b)  for the MEAN-corrected statistic
Lam*_n = Lam_n/(1+b/((K-1)n)). If this limit != 0, Lam_n is NOT Bartlett-correctable.
Variance-reduced: also report v and 4b separately.
"""
import numpy as np
RNG=np.random.default_rng(17)
def bcoef(p0):
    K=len(p0);A3=np.sum((1-p0)*(1-2*p0)/p0);A4=np.sum((1-p0)**2/p0)
    return (15/16)*A4-0.5*A3+(K**2-1)/48
def fr2(p,q):
    bc=np.clip(np.sum(np.sqrt(p*q),axis=-1),-1,1);return (2*np.arccos(bc))**2
def stats_at(p0,n,N,b,batch=2_000_000):
    K=len(p0); df=K-1; c=1.0+b/(df*n)
    s1=s2=s1s=s2s=0.0; done=0
    while done<N:
        m=min(batch,N-done); cc=RNG.multinomial(n,p0,size=m); ph=cc/n
        L=n*fr2(ph,p0); Ls=L/c
        s1+=L.sum(); s2+=(L*L).sum(); s1s+=Ls.sum(); s2s+=(Ls*Ls).sum(); done+=m
    mL=s1/N; vL=s2/N-mL*mL; mLs=s1s/N; vLs=s2s/N-mLs*mLs
    return mL,vL,mLs,vLs

for tag,p0 in [("uniform-4",np.array([.25,.25,.25,.25])),
               ("balanced-6",np.array([.30,.24,.18,.12,.09,.07]))]:
    p0=p0/p0.sum(); K=len(p0); df=K-1; b=bcoef(p0)
    print(f"\n=== {tag}  K={K}  b={b:.4f}  4b={4*b:.4f}  2(K-1)={2*df} ===")
    print(" n     n*(Var(Lam)-2df)=v   n*(Var(Lam*)-2df)=v-4b")
    rows=[]
    for n in [100,150,200,300]:
        N=20_000_000
        mL,vL,mLs,vLs=stats_at(p0,n,N,b)
        v_est=n*(vL-2*df); vm4b_est=n*(vLs-2*df)
        rows.append((n,v_est,vm4b_est))
        print(f" {n:4d}   {v_est:8.3f}            {vm4b_est:8.3f}")
    # extrapolate v-4b with linear-in-1/n fit on last two
    (n1,v1,d1),(n2,v2,d2)=rows[-2],rows[-1]
    vm4b_inf=(d2*n2-d1*n1)/(n2-n1)
    v_inf=(v2*n2-v1*n1)/(n2-n1)
    print(f"  -> extrapolated v = {v_inf:.3f},  v-4b = {vm4b_inf:.3f},  4b={4*b:.3f}")
    print(f"  -> Bartlett-correctable (v==4b)? {'~YES' if abs(vm4b_inf)<0.3 else 'NO'}")
