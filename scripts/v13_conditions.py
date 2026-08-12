"""Does the S-Inh/NegNM merge survive across circuits, ablation types, and linkages?"""
import numpy as np, json, itertools
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ROLE={(0,1):"DTH",(3,0):"DTH",(0,10):"DTH",(2,2):"PTH",(4,11):"PTH",
      (5,5):"IND",(5,8):"IND",(5,9):"IND",(6,9):"IND",
      (7,3):"S-Inh",(7,9):"S-Inh",(8,6):"S-Inh",(8,10):"S-Inh",
      (9,9):"NM",(9,6):"NM",(10,0):"NM",(10,7):"NegNM",(11,10):"NegNM"}
N_BOOT=500
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
    return int(d['n_players']), ld, [tuple(h) for h in d['circuit_heads'].tolist()]

def merge_rate(path, link='average'):
    n,ld,heads=load(path); P=ld.shape[1]
    lab=[ROLE.get(h) for h in heads]; allb=[l for l in lab if l]
    elig=[k for k,l in enumerate(lab) if l and allb.count(l)>=2]
    el=[lab[k] for k in elig]
    si=[x for x,l in enumerate(el) if l=='S-Inh']; nn=[x for x,l in enumerate(el) if l=='NegNM']
    if not si or not nn: return None
    def part(v):
        w=wht(v); M=np.zeros((n,n))
        for i,j in itertools.combinations(range(n),2): M[i,j]=M[j,i]=-w[(1<<i)|(1<<j)]
        Ms=M[np.ix_(elig,elig)]; D=Ms.max()-Ms; np.fill_diagonal(D,0)
        Z=linkage(squareform(D,checks=False),link); best=None
        for k in range(2,len(elig)):
            cl=fcluster(Z,k,'maxclust')
            if len(set(cl))<2: continue
            s=silhouette_score(D,cl,metric='precomputed')
            if best is None or s>best[0]: best=(s,k,cl)
        return best[1],best[2]
    k0,cl0=part(ld.mean(1)); ari0=adjusted_rand_score(el,cl0)
    merged0=any(cl0[a]==cl0[b] for a in si for b in nn)
    rng=np.random.RandomState(42); m=0; aris=[]
    for _ in range(N_BOOT):
        bi=rng.randint(0,P,size=P); k,cl=part(ld[:,bi].mean(1))
        m+=int(any(cl[a]==cl[b] for a in si for b in nn)); aris.append(adjusted_rand_score(el,cl))
    return dict(n_elig=len(elig),n_sinh=len(si),n_negnm=len(nn),k=k0,ari=ari0,
                merged_point=bool(merged0),merge_rate=m/N_BOOT,
                ari_ci=[float(np.percentile(aris,2.5)),float(np.percentile(aris,97.5))])

print(f"{'condition':<34}{'elig':>5}{'S/N':>6}{'k':>3}{'ARI':>7}  {'ARI 95% CI':>18}  merge-rate")
res={}
conds=[('c6 zero  [primary]','data/c6_zero_v2_FRESH.npz','average'),
       ('c6 mean  [M6 invariance]','data/c6_mean.npz','average'),
       ('c2 EAP-IG zero  [C3]','data/c2_zero.npz','average'),
       ('c5 Walsh-o1 zero [C3]','data/c5_zero.npz','average'),
       ('c6 zero, complete linkage','data/c6_zero_v2_FRESH.npz','complete'),
       ('c6 zero, ward linkage','data/c6_zero_v2_FRESH.npz','ward')]
for name,path,link in conds:
    try: r=merge_rate(path,link)
    except Exception as e: print(f"{name:<34} ERROR {type(e).__name__}"); continue
    if r is None: print(f"{name:<34} n/a (no S-Inh or NegNM pair)"); continue
    res[name]=r
    print(f"{name:<34}{r['n_elig']:>5}{r['n_sinh']}/{r['n_negnm']:<4}{r['k']:>3}{r['ari']:>7.3f}  "
          f"[{r['ari_ci'][0]:+.3f},{r['ari_ci'][1]:+.3f}]  {100*r['merge_rate']:.0f}%")
json.dump(res,open('results/v13_conditions.json','w'),indent=1)
