import numpy as np, json, itertools
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
ROLE={(0,1):"DTH",(3,0):"DTH",(0,10):"DTH",(2,2):"PTH",(4,11):"PTH",
      (5,5):"IND",(5,8):"IND",(5,9):"IND",(6,9):"IND",
      (7,3):"S-Inh",(7,9):"S-Inh",(8,6):"S-Inh",(8,10):"S-Inh",
      (9,9):"NM",(9,6):"NM",(10,0):"NM",(10,7):"NegNM",(11,10):"NegNM"}
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
print(f"{'condition':<32}{'elig':>5}{'S/N':>7}{'k':>3}{'ARI':>8}   S-Inh~NegNM?")
res={}
for name,path,link in [('c6 zero  [primary]','data/c6_zero_v2_FRESH.npz','average'),
                       ('c6 mean  [M6 invariance]','data/c6_mean.npz','average'),
                       ('c2 EAP-IG zero  [C3]','data/c2_zero.npz','average'),
                       ('c5 Walsh-o1 zero [C3]','data/c5_zero.npz','average'),
                       ('c6 zero complete-linkage','data/c6_zero_v2_FRESH.npz','complete'),
                       ('c6 zero ward-linkage','data/c6_zero_v2_FRESH.npz','ward')]:
    n,ld,heads=load(path)
    lab=[ROLE.get(h) for h in heads]; allb=[l for l in lab if l]
    elig=[k for k,l in enumerate(lab) if l and allb.count(l)>=2]; el=[lab[k] for k in elig]
    si=[x for x,l in enumerate(el) if l=='S-Inh']; nn=[x for x,l in enumerate(el) if l=='NegNM']
    if not si or not nn:
        print(f"{name:<32}{len(elig):>5}{len(si)}/{len(nn):<5}  n/a (missing class)"); continue
    w=wht(ld); M=np.zeros((n,n))
    for i,j in itertools.combinations(range(n),2): M[i,j]=M[j,i]=-w[(1<<i)|(1<<j)]
    Ms=M[np.ix_(elig,elig)]; D=Ms.max()-Ms; np.fill_diagonal(D,0)
    Z=linkage(squareform(D,checks=False),link); best=None
    for k in range(2,len(elig)):
        cl=fcluster(Z,k,'maxclust')
        if len(set(cl))<2: continue
        s=silhouette_score(D,cl,metric='precomputed')
        if best is None or s>best[0]: best=(s,k,cl)
    k0,cl0=best[1],best[2]; ari=adjusted_rand_score(el,cl0)
    merged=any(cl0[a]==cl0[b] for a in si for b in nn)
    print(f"{name:<32}{len(elig):>5}{len(si)}/{len(nn):<5}{k0:>3}{ari:>8.3f}   {'MERGED' if merged else 'separate'}")
    res[name]=dict(n_elig=len(elig),n_si=len(si),n_nn=len(nn),k=int(k0),ari=float(ari),merged=bool(merged),
                   clusters={f"L{heads[e][0]}H{heads[e][1]}":int(c) for e,c in zip(elig,cl0)},
                   labels={f"L{heads[e][0]}H{heads[e][1]}":lab[e] for e in elig})
json.dump(res,open('results/v13_conditions_point.json','w'),indent=1)
