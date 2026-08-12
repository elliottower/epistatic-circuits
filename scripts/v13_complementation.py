import numpy as np, json, itertools
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

d = np.load('data/c6_zero_v2_FRESH.npz', allow_pickle=True)
tgt, foil = d['target_logits'], d['foil_logits']          # (32768, 512)
idx = d['coalition_indices']; heads = d['circuit_heads']
n = int(d['n_players']); N = 1 << n
assert len(idx) == N and len(np.unique(idx)) == N

ld = tgt - foil                                            # per-coalition, per-prompt logit diff
order = np.argsort(idx)                                    # canonical bitmask order
ld = ld[order]

def wht(v):                                                # fast Walsh-Hadamard, in place
    v = v.astype(np.float64).copy(); h = 1
    while h < len(v):
        for i in range(0, len(v), h*2):
            a = v[i:i+h].copy(); b = v[i+h:i+2*h].copy()
            v[i:i+h] = a + b; v[i+h:i+2*h] = a - b
        h *= 2
    return v / len(v)

popc = np.array([bin(i).count('1') for i in range(N)])
def coeffs(vals):
    w = wht(vals)
    o1 = {i: w[1 << i] for i in range(n)}
    o2 = {(i, j): w[(1 << i) | (1 << j)] for i, j in itertools.combinations(range(n), 2)}
    return w, o1, o2

# --- convention check + full-sample coefficients
v_all = ld.mean(axis=1)
print(f"v(empty bitmask 0) = {v_all[0]:+.4f}   v(full) = {v_all[-1]:+.4f}")
w, o1, o2 = coeffs(v_all)
energy = lambda ks: sum(w[k]**2 for k in ks)
tot = sum(w[k]**2 for k in range(1, N))
print(f"order-1 frac {energy([1<<i for i in range(n)])/tot:.3f}  "
      f"order-2 frac {sum(v**2 for v in o2.values())/tot:.3f}")

# --- RELIABILITY GATE: split prompts in half, refit, correlate order-2
rng = np.random.default_rng(0)
perm = rng.permutation(ld.shape[1]); h1, h2 = perm[:256], perm[256:]
_,_,o2a = coeffs(ld[:, h1].mean(axis=1))
_,_,o2b = coeffs(ld[:, h2].mean(axis=1))
pairs = sorted(o2.keys())
a = np.array([o2a[p] for p in pairs]); b = np.array([o2b[p] for p in pairs])
rel = np.corrcoef(a, b)[0,1]
sb  = 2*rel/(1+rel)                                        # Spearman-Brown to full length
print(f"\nRELIABILITY split-half r = {rel:.3f}   Spearman-Brown = {sb:.3f}   "
      f"{'PASS' if sb >= 0.6 else 'FAIL -> task dropped per prereg'}")

M = np.zeros((n, n))
for (i,j),v in o2.items(): M[i,j] = M[j,i] = -v            # masking = -w_ij
np.save('results/v13_M_c6_zero.npy', M)
json.dump({'heads': heads.tolist(),
           'order2': {f"{i}_{j}": v for (i,j),v in o2.items()},
           'order1': {str(i): v for i,v in o1.items()},
           'split_half_r': float(rel), 'spearman_brown': float(sb)},
          open('results/v13_coeffs_c6_zero.json','w'), indent=1)
print("\nheads:", [f"L{l}H{h}" for l,h in heads.tolist()])
