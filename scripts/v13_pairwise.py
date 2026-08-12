"""Which canonical class PAIRS co-cluster, across conditions? Authoritative Wang et al. labels."""
import numpy as np, json, itertools
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
# Wang et al. 2022 Fig.2 + Fig.20, verified from the PDF
ROLE={(2,2):"PTH",(4,11):"PTH",(0,1):"DTH",(3,0):"DTH",(0,10):"DTH",
      (5,5):"IND",(6,9):"IND",(5,8):"IND",(5,9):"IND",
      (7,3):"S-Inh",(7,9):"S-Inh",(8,6):"S-Inh",(8,10):"S-Inh",
      (10,7):"NegNM",(11,10):"NegNM",(9,9):"NM",(9,6):"NM",(10,0):"NM",
      (9,0):"Backup",(9,7):"Backup",(10,1):"Backup",(10,2):"Backup",
      (10,6):"Backup",(10,10):"Backup",(11,2):"Backup",(11,9):"Backup"}
def wht(v):
    v=v.astype(np.float64).copy(); N=len(v); h=1
    while h<N:
        for i in range(0,N,h*2):
            a=v[i:i+h].copy(); b=v[i+h:i+2*h].copy()
            v[i:i+h]=a+b; v[i+h:i+2*h]=a-b
        h*=2
    return v/N
def load(p):
    d=np.load(p,allow_pickle=True); k=list(d.keys())
    ld=d['logit_diff'] if 'logit_diff' in k else (d['target_logits']-d['foil_logits'])
    ld=np.asarray(ld,dtype=np.float64)
    if 'coalition_indices' in k: ld=ld[np.argsort(d['coalition_indices'])]
    if ld.ndim==2: ld=ld.mean(1)
    return int(d['n_players']), ld, [tuple(h) for h in d['circuit_heads'].tolist()]
CONDS=[('c6 zero','data/c6_zero_v2_FRESH.npz'),('c6 mean','data/c6_mean.npz'),
       ('c2 EAP zero','data/c2_zero.npz'),('c5 Walsh zero','data/c5_zero.npz')]
tally={}
for name,path in CONDS:
    n,ld,heads=load(path); lab=[ROLE.get(h) for h in heads]
    present=[l for l in lab if l]; elig=[k for k,l in enumerate(lab) if l and present.count(l)>=2]
    el=[lab[k] for k in elig]
    if len(set(el))<2: print(f"{name}: too few classes"); continue
    w=wht(ld); M=np.zeros((n,n))
    for i,j in itertools.combinations(range(n),2): M[i,j]=M[j,i]=-w[(1<<i)|(1<<j)]
    Ms=M[np.ix_(elig,elig)]; D=Ms.max()-Ms; np.fill_diagonal(D,0)
    Z=linkage(squareform(D,checks=False),'average'); best=None
    for k in range(2,len(elig)):
        cl=fcluster(Z,k,'maxclust')
        if len(set(cl))<2: continue
        s=silhouette_score(D,cl,metric='precomputed')
        if best is None or s>best[0]: best=(s,k,cl)
    cl=best[2]
    classes=sorted(set(el))
    print(f"\n{name}  (k={best[1]}, classes present: {', '.join(classes)})")
    for a,b in itertools.combinations(classes,2):
        ia=[x for x,l in enumerate(el) if l==a]; ib=[x for x,l in enumerate(el) if l==b]
        co=any(cl[x]==cl[y] for x in ia for y in ib)
        tally.setdefault((a,b),[]).append(co)
        print(f"   {a:<7} ~ {b:<7} {'MERGED' if co else 'separate'}")
print("\n=== across conditions ===")
for (a,b),v in sorted(tally.items(), key=lambda kv:-sum(kv[1])):
    print(f"  {a:<7} ~ {b:<8} merged in {sum(v)}/{len(v)} conditions")
json.dump({f"{a}~{b}":v for (a,b),v in tally.items()}, open('results/v13_pairwise.json','w'), indent=1)
