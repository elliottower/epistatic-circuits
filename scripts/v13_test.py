import numpy as np, json, itertools
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Role labels: Wang et al. 2023 canonical IOI assignment (paper Fig. 2).
# SOURCE CAVEAT: not held as a file in these repos; transcribed from the paper.
ROLE = {(0,1):"DTH", (5,5):"IND", (6,9):"IND", (7,9):"S-Inh", (8,6):"S-Inh",
        (8,10):"S-Inh", (9,9):"NM", (10,0):"NM", (10,7):"NegNM", (11,10):"NegNM"}

C = json.load(open('results/v13_coeffs_c6_zero.json'))
heads = [tuple(h) for h in C['heads']]; n = len(heads)
o2 = {tuple(int(x) for x in k.split('_')): v for k, v in C['order2'].items()}

# --- sign check: verify Walsh order-2 against the unambiguous discrete interaction
d = np.load('data/c6_zero_v2_FRESH.npz', allow_pickle=True)
ld = (d['target_logits'] - d['foil_logits'])[np.argsort(d['coalition_indices'])].mean(1)
def disc_inter(i, j):                      # mean_S [v(S+ij) - v(S+i) - v(S+j) + v(S)]
    bi, bj = 1 << i, 1 << j
    S = np.array([s for s in range(1 << n) if not (s & bi) and not (s & bj)])
    return float(np.mean(ld[S | bi | bj] - ld[S | bi] - ld[S | bj] + ld[S]))
probe = list(o2)[:40]
di = np.array([disc_inter(*p) for p in probe]); w2 = np.array([o2[p] for p in probe])
ratio = np.median(di / w2)
print(f"SIGN CHECK: discrete-interaction / walsh-order2 median ratio = {ratio:+.3f}")
print(f"  r = {np.corrcoef(di, w2)[0,1]:+.4f}  -> masking (sub-additive) is "
      f"{'NEGATIVE' if ratio>0 else 'POSITIVE'} walsh-order2")

# M positive == masking == sub-additive == candidate same-unit
sgn = -1.0 if ratio > 0 else 1.0
M = np.zeros((n, n))
for (i,j), v in o2.items(): M[i,j] = M[j,i] = sgn * v

lab = [ROLE.get(h) for h in heads]
keep = [k for k,l in enumerate(lab) if l is not None]
kl = [lab[k] for k in keep]
print(f"\nlabelled heads: {len(keep)}/{n}  roles: "
      f"{ {r: kl.count(r) for r in sorted(set(kl))} }")

# --- PREDICTION 1: within-role masking > cross-role, label permutation
elig = [k for k in keep if kl.count(lab[k]) >= 2]
el = [lab[k] for k in elig]
def stat(labels):
    w = [M[a,b] for x,a in enumerate(elig) for y,b in enumerate(elig) if x<y and labels[x]==labels[y]]
    c = [M[a,b] for x,a in enumerate(elig) for y,b in enumerate(elig) if x<y and labels[x]!=labels[y]]
    return np.mean(w) - np.mean(c), len(w), len(c)
obs, nw, nc = stat(el)
rng = np.random.default_rng(0)
null = np.array([stat(list(rng.permutation(el)))[0] for _ in range(10000)])
p = float((null >= obs).mean())
print(f"\nPREDICTION 1  (n={len(elig)} heads, {nw} within-pairs, {nc} cross-pairs)")
print(f"  within-role mean masking - cross-role = {obs:+.5f}")
print(f"  permutation p (one-sided, within>cross) = {p:.4f}")
print(f"  -> {'CONFIRMED' if p<0.05 else 'NOT CONFIRMED (registered prediction: disconfirmed)'}")

# --- PREDICTION 3: ARI of masking-clusters vs role labels
Msub = M[np.ix_(elig, elig)]
D = Msub.max() - Msub; np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), method='average')
best = max(((adjusted_rand_score(el, fcluster(Z, k, 'maxclust')), k) for k in range(2, len(elig))))
print(f"\nPREDICTION 3  best ARI over k=2..{len(elig)-1}: {best[0]:+.3f} at k={best[1]}")
print(f"  -> {'ARI<=0.1: taxonomy unrelated to interaction structure' if best[0]<=0.1 else 'partial or better'}")
json.dump({'sign_ratio':ratio,'p1_obs':obs,'p1_p':p,'p3_best_ari':best[0],'p3_k':best[1],
           'labelled':len(keep),'eligible':len(elig)}, open('results/v13_test_results.json','w'), indent=1)
