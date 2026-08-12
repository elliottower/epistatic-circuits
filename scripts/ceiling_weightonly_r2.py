"""Upper bound on R^2 for ANY weight-only predictor of head-pair epistasis.

The weights do not change when the ablation primitive changes. So if W_ij were a
function of the weights alone, W_ij(zero) would equal W_ij(mean) exactly. However
much it moves between ablation types is variance no weight-based measure can
explain. Shared variance across ablation types therefore upper-bounds achievable R^2.

This is the control PREREG_SUBSPACE_EPISTASIS.md names ("bounds how large an R^2
we should expect in P4 even under the 'fully geometric' hypothesis") and does not
compute. Diagnostic, not a new hypothesis test.
"""
import numpy as np, itertools, json
from scipy.stats import spearmanr, pearsonr

def wht(v):
    v = v.astype(np.float64).copy(); h = 1
    while h < len(v):
        for i in range(0, len(v), h*2):
            a = v[i:i+h].copy(); b = v[i+h:i+2*h].copy()
            v[i:i+h] = a + b; v[i+h:i+2*h] = a - b
        h *= 2
    return v / len(v)

def order2(path):
    d = np.load(path, allow_pickle=True); k = list(d.keys())
    ld = d['logit_diff'] if 'logit_diff' in k else (d['target_logits'] - d['foil_logits'])
    ld = np.asarray(ld, dtype=np.float64)
    if 'coalition_indices' in k: ld = ld[np.argsort(d['coalition_indices'])]
    if ld.ndim == 2: ld = ld.mean(1)
    n = int(d['n_players']); w = wht(ld)
    heads = [tuple(h) for h in d['circuit_heads'].tolist()]
    pairs = list(itertools.combinations(range(n), 2))
    return np.array([w[(1 << i) | (1 << j)] for i, j in pairs]), pairs, heads

wz, pz, hz = order2('data/c6_zero_v2_FRESH.npz')
wm, pm, hm = order2('data/c6_mean.npz')
assert pz == pm and hz == hm, "circuits differ; not comparable"

rho = spearmanr(wz, wm).statistic
r   = pearsonr(wz, wm)[0]
print(f"circuit c6 (IOI, order-2 discovered), {len(hz)} heads, {len(pz)} pairs")
print(f"  order-2 coefficients, zero vs mean ablation")
print(f"    Spearman rho = {rho:+.3f}")
print(f"    Pearson  r   = {r:+.3f}")
print(f"    CEILING on weight-only R^2  =  r^2 = {r**2:.3f}")
print()
print(f"  P4 reported pooled R^2 = 0.068, tested against a threshold of 0.30")
if r**2 < 0.15:
    print(f"  -> the threshold of 0.30 was ABOVE the ceiling. P4 could not have passed.")
elif 0.068 > 0.7 * r**2:
    print(f"  -> 0.068 is near the ceiling; little headroom for geometry to have explained more.")
else:
    print(f"  -> ceiling is well above 0.068; the negative result is interpretable.")
# also: magnitude agreement, since sign flips matter for the epistasis reading
same_sign = float(np.mean(np.sign(wz) == np.sign(wm)))
print(f"\n  pairs agreeing in SIGN across ablation types: {same_sign:.1%}")
json.dump({'spearman': float(rho), 'pearson': float(r), 'ceiling_r2': float(r**2),
           'sign_agreement': same_sign, 'n_pairs': len(pz)},
          open('results/ceiling_weightonly_r2.json','w'), indent=1)
