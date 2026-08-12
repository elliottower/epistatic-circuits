import numpy as np, json, itertools
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
N_BOOT=1000
ROLE={(0,1):"DTH",(5,5):"IND",(6,9):"IND",(7,9):"S-Inh",(8,6):"S-Inh",
      (8,10):"S-Inh",(9,9):"NM",(10,0):"NM",(10,7):"NegNM",(11,10):"NegNM"}
def wht(v):
    v=v.astype(np.float64).copy(); N=len(v); h=1
    while h<N:
        for i in range(0,N,h*2):
            a=v[i:i+h].copy(); b=v[i+h:i+2*h].copy()
            v[i:i+h]=a+b; v[i+h:i+2*h]=a-b
        h*=2
    return v/N
d=np.load('data/c6_zero_v2_FRESH.npz',allow_pickle=True)
ld=(d['target_logits']-d['foil_logits'])[np.argsort(d['coalition_indices'])]
heads=[tuple(h) for h in d['circuit_heads'].tolist()]; n=int(d['n_players']); P=ld.shape[1]
lab=[ROLE.get(h) for h in heads]; allb=[l for l in lab if l]
elig=[k for k,l in enumerate(lab) if l and allb.count(l)>=2]; el=[lab[k] for k in elig]

def partition(v):
    w=wht(v); M=np.zeros((n,n))
    for i,j in itertools.combinations(range(n),2):
        M[i,j]=M[j,i]=-w[(1<<i)|(1<<j)]
    Ms=M[np.ix_(elig,elig)]; D=Ms.max()-Ms; np.fill_diagonal(D,0)
    Z=linkage(squareform(D,checks=False),'average')
    best=None
    for k in range(2,len(elig)):
        cl=fcluster(Z,k,'maxclust')
        if len(set(cl))<2: continue
        s=silhouette_score(D,cl,metric='precomputed')
        if best is None or s>best[0]: best=(s,k,cl)
    return best[1],best[2]

k0,cl0=partition(ld.mean(1)); ari0=adjusted_rand_score(el,cl0)
rng=np.random.RandomState(42); aris=[]; ks=[]; merged=0
for _ in range(N_BOOT):
    bi=rng.randint(0,P,size=P)
    k,cl=partition(ld[:,bi].mean(1))
    aris.append(adjusted_rand_score(el,cl)); ks.append(k)
    # does S-Inh co-cluster with NegNM in this draw?
    si=[x for x,l in enumerate(el) if l=='S-Inh']; nn=[x for x,l in enumerate(el) if l=='NegNM']
    merged += int(any(cl[a]==cl[b] for a in si for b in nn))
aris=np.array(aris)
print(f"point ARI (silhouette k={k0}) = {ari0:+.3f}")
print(f"bootstrap ARI mean = {aris.mean():+.3f}   95% CI [{np.percentile(aris,2.5):+.3f}, {np.percentile(aris,97.5):+.3f}]")
print(f"k selected: mode={max(set(ks),key=ks.count)}  distribution={ {v:ks.count(v) for v in sorted(set(ks))} }")
print(f"\nS-Inh co-clusters with NegNM in {merged}/{N_BOOT} = {100*merged/N_BOOT:.1f}% of bootstrap draws")
json.dump({'ari_point':float(ari0),'k_point':int(k0),'ari_mean':float(aris.mean()),
           'ari_ci':[float(np.percentile(aris,2.5)),float(np.percentile(aris,97.5))],
           'k_dist':{str(v):ks.count(v) for v in sorted(set(ks))},
           'sinh_negnm_merge_rate':merged/N_BOOT}, open('results/v13_ari_bootstrap.json','w'),indent=1)
