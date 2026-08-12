import numpy as np, json, itertools

def wht(v):
    v=v.astype(np.float64).copy(); h=1
    while h<len(v):
        for i in range(0,len(v),h*2):
            a=v[i:i+h].copy(); b=v[i+h:i+2*h].copy()
            v[i:i+h]=a+b; v[i+h:i+2*h]=a-b
        h*=2
    return v/len(v)

def load(path):
    d=np.load(path,allow_pickle=True); k=list(d.keys())
    n=int(d['n_players'])
    if 'logit_diff' in k:
        ld=d['logit_diff']; idx=np.arange(len(ld)) if 'coalition_indices' not in k else d['coalition_indices']
    else:
        ld=(d['target_logits']-d['foil_logits']); idx=d['coalition_indices']
    ld=np.asarray(ld)
    if ld.ndim==2: ld=ld.mean(1)
    ld=ld[np.argsort(idx)]
    return n, ld, d['circuit_heads']

def o2_energy(n, v):
    w=wht(v)
    tot=sum(w[k]**2 for k in range(1,1<<n))
    o2={(i,j):w[(1<<i)|(1<<j)] for i,j in itertools.combinations(range(n),2)}
    return sum(x**2 for x in o2.values())/tot, o2, tot

rng=np.random.default_rng(0)
print(f"{'circuit':<22}{'n':>3}{'order2_frac':>13}{'null_mean':>11}{'null_sd':>9}{'z':>8}  verdict")
rows={}
for name,path in [('c6 IOI (functional)','data/c6_zero_v2_FRESH.npz'),
                  ('gender KNOWN (inert)','data/gender_known_zero.npz'),
                  ('gender RANDOM','data/gender_random_zero.npz')]:
    n,ld,heads=load(path)
    frac,o2,tot=o2_energy(n,ld)
    # coalition-shuffled null: permute the value vector across coalitions
    nulls=np.array([o2_energy(n, rng.permutation(ld))[0] for _ in range(200)])
    z=(frac-nulls.mean())/nulls.std()
    rows[name]={'n':n,'order2_frac':frac,'null_mean':float(nulls.mean()),
                'null_sd':float(nulls.std()),'z':float(z)}
    v = 'STRUCTURE' if z>3 else ('noise' if abs(z)<=3 else 'below-null')
    print(f"{name:<22}{n:>3}{frac:>13.4f}{nulls.mean():>11.4f}{nulls.std():>9.4f}{z:>8.1f}  {v}")
json.dump(rows,open('results/v13_p4_control.json','w'),indent=1)
