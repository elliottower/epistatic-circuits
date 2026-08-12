import numpy as np, json
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
ROLE = {(0,1):"DTH",(5,5):"IND",(6,9):"IND",(7,9):"S-Inh",(8,6):"S-Inh",
        (8,10):"S-Inh",(9,9):"NM",(10,0):"NM",(10,7):"NegNM",(11,10):"NegNM"}
C=json.load(open('results/v13_coeffs_c6_zero.json')); heads=[tuple(h) for h in C['heads']]; n=len(heads)
o2={tuple(int(x) for x in k.split('_')):v for k,v in C['order2'].items()}
M=np.zeros((n,n))
for (i,j),v in o2.items(): M[i,j]=M[j,i]=-v
lab=[ROLE.get(h) for h in heads]
elig=[k for k,l in enumerate(lab) if l and [lab[m] for m in range(n) if lab[m]].count(l)>=2]
el=[lab[k] for k in elig]
Ms=M[np.ix_(elig,elig)]; D=Ms.max()-Ms; np.fill_diagonal(D,0)
Z=linkage(squareform(D,checks=False),method='average')
rows=[]
for k in range(2,len(elig)):
    cl=fcluster(Z,k,'maxclust')
    if len(set(cl))<2: continue
    rows.append((k, silhouette_score(D,cl,metric='precomputed'), adjusted_rand_score(el,cl)))
kbest,sbest,ari_at_sil = max(rows,key=lambda r:r[1])
print(" k   silhouette      ARI")
for k,s,a in rows: print(f" {k}   {s:+.4f}    {a:+.4f}{'   <- silhouette-selected' if k==kbest else ''}")
print(f"\nPREREGISTERED P3: k chosen by silhouette = {kbest};  ARI = {ari_at_sil:+.3f}")
verdict = ('ARI>0.4 taxonomy substantially recovered' if ari_at_sil>0.4
           else 'partial (0.1<ARI<=0.4)' if ari_at_sil>0.1 else 'ARI<=0.1 unrelated')
print(f"  -> {verdict}")
print(f"\n[exploratory, NOT prereg] max ARI over k = {max(r[2] for r in rows):+.3f} "
      f"at k={max(rows,key=lambda r:r[2])[0]}  -- selected on outcome, do not report as confirmatory")
json.dump({'k_silhouette':int(kbest),'ari_prereg':float(ari_at_sil),
           'max_ari_exploratory':float(max(r[2] for r in rows)),
           'table':[[int(k),float(s),float(a)] for k,s,a in rows]},
          open('results/v13_p3_silhouette.json','w'),indent=1)
