"""
Real-data application: NSL-KDD network-intrusion flag profiles.
Each session = n connections; categorical distribution over K=6 TCP flag states.
Normal traffic defines p0; attacks (DoS/probe) shift mass into rare flag channels.
We compare the geodesic statistic Lambda_n = n d_FR(phat,p0)^2 against the
Euclidean statistic n||phat-p0||^2 at matched Type-I error.
Data: jmnwong/NSL-KDD-Dataset (KDDTrain+_20Percent.txt). Fixed seed.
"""
import csv, numpy as np
from scipy import stats
RNG=np.random.default_rng(20260628)

# ---- load flag + label ----
CATS=["SF","S0","REJ","RSTR","RSTO"]; K=len(CATS)+1  # +1 "other"
def cat_index(flag): return CATS.index(flag) if flag in CATS else K-1
normal=[]; attack=[]; attack_by={}
with open("data/nslkdd20.txt") as f:
    for row in csv.reader(f):
        ci=cat_index(row[3]); lab=row[41]
        if lab=="normal": normal.append(ci)
        else:
            attack.append(ci); attack_by.setdefault(lab,[]).append(ci)
normal=np.array(normal); attack=np.array(attack)
print(f"K={K} cats={CATS+['other']}  normal={len(normal)} attack={len(attack)}")

# ---- split normal: estimate p0 on half, evaluate on the other ----
perm=RNG.permutation(len(normal)); half=len(normal)//2
train_n=normal[perm[:half]]; eval_n=normal[perm[half:]]
p0=np.bincount(train_n,minlength=K).astype(float); p0/=p0.sum()
print("p0 =", np.round(p0,4), " min channel =", p0.min())

def sessions(pool,nsess,n):
    idx=RNG.integers(0,len(pool),size=(nsess,n))
    draws=pool[idx]
    # counts over K per session
    C=np.zeros((nsess,K))
    for k in range(K): C[:,k]=(draws==k).sum(1)
    return C/n

def fr2(ph): 
    bc=np.clip(np.sum(np.sqrt(ph*p0),axis=1),-1,1); return (2*np.arccos(bc))**2
def stat_fr(ph,n): return n*fr2(ph)
def stat_eu(ph,n): return n*np.sum((ph-p0)**2,axis=1)

n=200; Nnull=40000; Natt=40000
ph_norm=sessions(eval_n,Nnull,n)
Lfr_n=stat_fr(ph_norm,n); Leu_n=stat_eu(ph_norm,n)
tFR=np.quantile(Lfr_n,0.95); tEU=np.quantile(Leu_n,0.95)

# chi-square calibration check on real normal sessions
print(f"\n[Null calibration on REAL normal sessions, n={n}]")
print(f"  E[Lambda]={Lfr_n.mean():.2f} (chi2_{K-1} mean={K-1}), Var={Lfr_n.var():.2f} (chi2 var={2*(K-1)})")

# attack sessions (full mix) -> AUC & power
ph_att=sessions(attack,Natt,n)
Lfr_a=stat_fr(ph_att,n); Leu_a=stat_eu(ph_att,n)
def auc(neg,pos,g=500):
    lo=min(neg.min(),pos.min()); hi=max(neg.max(),pos.max()); t=np.linspace(lo,hi,g)
    tpr=np.array([(pos>x).mean() for x in t]); fpr=np.array([(neg>x).mean() for x in t])
    o=np.argsort(fpr); return np.trapezoid(tpr[o],fpr[o])
print(f"\n[Detection, level 0.05]  AUC_FR={auc(Lfr_n,Lfr_a):.4f}  AUC_EU={auc(Leu_n,Leu_a):.4f}")
print(f"  power_FR={np.mean(Lfr_a>tFR):.3f}  power_EU={np.mean(Leu_a>tEU):.3f}")

# per-attack-category power (major categories)
print("\n[Per-category power @0.05]  (FR vs EU)")
for lab in ["neptune","satan","portsweep","ipsweep","smurf","nmap","back","teardrop"]:
    pool=np.array(attack_by[lab])
    if len(pool)<50: continue
    ph=sessions(pool,8000,n)
    pf=np.mean(stat_fr(ph,n)>tFR); pe=np.mean(stat_eu(ph,n)>tEU)
    print(f"  {lab:10s} (m={len(pool):5d})  FR={pf:.3f}  EU={pe:.3f}")

# contamination sweep: session = (1-eps) normal + eps attack
print("\n[Contaminated sessions: power vs attack fraction eps, @0.05]")
eps_list=np.array([0.0,0.02,0.05,0.10,0.15,0.20,0.30]); pf_e=[]; pe_e=[]
for eps in eps_list:
    natt=int(round(eps*n)); nnorm=n-natt
    Ci=np.zeros((20000,K))
    ai=RNG.integers(0,len(attack),size=(20000,max(natt,1)))
    ni=RNG.integers(0,len(eval_n),size=(20000,nnorm))
    draws=np.concatenate([attack[ai][:, :natt], eval_n[ni]],axis=1) if natt>0 else eval_n[ni]
    for k in range(K): Ci[:,k]=(draws==k).sum(1)
    ph=Ci/n
    pf_e.append(np.mean(stat_fr(ph,n)>tFR)); pe_e.append(np.mean(stat_eu(ph,n)>tEU))
    print(f"  eps={eps:.2f}: FR={pf_e[-1]:.3f}  EU={pe_e[-1]:.3f}")

np.savez("data/nsl_results.npz", p0=p0, tFR=tFR, tEU=tEU,
         Lfr_n=Lfr_n, Leu_n=Leu_n, Lfr_a=Lfr_a, Leu_a=Leu_a,
         eps=eps_list, pf_e=np.array(pf_e), pe_e=np.array(pe_e), n=n, K=K)
print("\nsaved data/nsl_results.npz")
