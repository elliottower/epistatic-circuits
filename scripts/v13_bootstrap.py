"""v13 re-analysis using the RTI-paper bootstrap-over-prompts method."""
import numpy as np, json, itertools
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

N_BOOT = 1000
ROLE = {(0,1):"DTH",(5,5):"IND",(6,9):"IND",(7,9):"S-Inh",(8,6):"S-Inh",
        (8,10):"S-Inh",(9,9):"NM",(10,0):"NM",(10,7):"NegNM",(11,10):"NegNM"}

def wht_mat(V):                       # WHT along axis 0, V shape (2^n, B)
    V = V.astype(np.float64).copy(); N = V.shape[0]; h = 1
    while h < N:
        for i in range(0, N, h*2):
            a = V[i:i+h].copy(); b = V[i+h:i+2*h].copy()
            V[i:i+h] = a+b; V[i+h:i+2*h] = a-b
        h *= 2
    return V / N

def load(path):
    d = np.load(path, allow_pickle=True); k = list(d.keys()); n = int(d['n_players'])
    ld = d['logit_diff'] if 'logit_diff' in k else (d['target_logits']-d['foil_logits'])
    ld = np.asarray(ld, dtype=np.float64)
    if 'coalition_indices' in k: ld = ld[np.argsort(d['coalition_indices'])]
    return n, ld, d['circuit_heads']

def rti_epistasis(ld_pp, n):           # per-prompt columns; RTI paper formula
    full = (1 << n) - 1
    grp = ld_pp[full] - ld_pp[0]
    loo = sum(ld_pp[full] - ld_pp[full ^ (1 << i)] for i in range(n))
    return grp, loo

print(f"{'circuit':<24}{'epistasis':>10}  {'95% CI':>18}   {'order2_frac':>11}  {'95% CI':>18}")
out = {}
for name, path in [('c6 IOI (functional)','data/c6_zero_v2_FRESH.npz'),
                   ('gender KNOWN (inert)','data/gender_known_zero.npz'),
                   ('gender RANDOM','data/gender_random_zero.npz')]:
    n, ld, heads = load(path)
    P = ld.shape[1] if ld.ndim == 2 else 1
    rng = np.random.RandomState(42)
    if ld.ndim == 1: ld = ld[:, None]; P = 1
    grp_pp, loo_pp = rti_epistasis(ld, n)
    idxs = rng.randint(0, P, size=(N_BOOT, P))
    ep, o2f = [], []
    for bi in idxs:
        g, l = grp_pp[bi].mean(), loo_pp[bi].mean()
        if abs(g) > 1e-10: ep.append(1.0 - l/g)
        w = wht_mat(ld[:, bi].mean(1)[:, None])[:, 0]
        tot = (w[1:]**2).sum()
        o2f.append(sum(w[(1<<i)|(1<<j)]**2 for i,j in itertools.combinations(range(n),2))/tot)
    ep, o2f = np.array(ep), np.array(o2f)
    pt_g, pt_l = grp_pp.mean(), loo_pp.mean()
    pt = 1.0 - pt_l/pt_g if abs(pt_g) > 1e-10 else float('nan')
    ci = lambda a: (np.percentile(a,2.5), np.percentile(a,97.5))
    e_lo,e_hi = ci(ep); o_lo,o_hi = ci(o2f)
    print(f"{name:<24}{pt:>10.4f}  [{e_lo:+.4f},{e_hi:+.4f}]   {o2f.mean():>11.4f}  [{o_lo:.4f},{o_hi:.4f}]")
    out[name] = {'epistasis_point':float(pt),'epistasis_ci':[float(e_lo),float(e_hi)],
                 'order2_frac_mean':float(o2f.mean()),'order2_frac_ci':[float(o_lo),float(o_hi)],'n_prompts':int(P)}
json.dump(out, open('results/v13_bootstrap_p4.json','w'), indent=1)
