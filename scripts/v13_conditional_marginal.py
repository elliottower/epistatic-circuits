"""Does a head's marginal contribution flip sign conditional on S-Inhibition being ablated?
Exact over the full 2^15 lattice. NOT preregistered - follow-up derived from the pairwise result."""
import numpy as np, json, itertools
ROLE={(2,2):"PTH",(4,11):"PTH",(0,1):"DTH",(3,0):"DTH",(0,10):"DTH",
      (5,5):"IND",(6,9):"IND",(5,8):"IND",(5,9):"IND",
      (7,3):"S-Inh",(7,9):"S-Inh",(8,6):"S-Inh",(8,10):"S-Inh",
      (10,7):"NegNM",(11,10):"NegNM",(9,9):"NM",(9,6):"NM",(10,0):"NM",
      (9,0):"Backup",(9,7):"Backup",(10,1):"Backup",(10,2):"Backup",
      (10,6):"Backup",(10,10):"Backup",(11,2):"Backup",(11,9):"Backup"}
def load(p):
    d=np.load(p,allow_pickle=True); k=list(d.keys())
    ld=d['logit_diff'] if 'logit_diff' in k else (d['target_logits']-d['foil_logits'])
    ld=np.asarray(ld,dtype=np.float64)
    if 'coalition_indices' in k: ld=ld[np.argsort(d['coalition_indices'])]
    return int(d['n_players']), ld, [tuple(h) for h in d['circuit_heads'].tolist()]

n,ld_pp,heads=load('data/c6_zero_v2_FRESH.npz')   # (32768, n_prompts)
lab=[ROLE.get(h) for h in heads]
sinh=[i for i,l in enumerate(lab) if l=="S-Inh"]
print(f"heads: {n}   S-Inh present at bit positions {sinh} -> {[heads[i] for i in sinh]}\n")
idx=np.arange(1<<n)
sinh_mask=np.zeros(1<<n,dtype=bool)          # True where ALL S-Inh heads are KEPT
allb=0
for i in sinh: allb|=(1<<i)
sinh_mask=(idx&allb)==allb
sinh_gone=(idx&allb)==0
v=ld_pp.mean(1)                               # mean over prompts

def marginal(i, subset_mask):
    """E[v(S u {i}) - v(S)] over coalitions S in subset_mask that lack i."""
    bit=1<<i
    lacks=(idx&bit)==0
    sel=lacks&subset_mask
    return float((v[idx[sel]|bit]-v[idx[sel]]).mean()), int(sel.sum())

rows=[]
for i,h in enumerate(heads):
    if lab[i] is None or lab[i]=="S-Inh": continue
    a,na=marginal(i,sinh_mask)     # S-Inh intact
    b,nb=marginal(i,sinh_gone)     # S-Inh fully ablated
    rows.append((lab[i],h,a,b,b-a,np.sign(a)!=np.sign(b)))
rows.sort(key=lambda r:(r[0],r[1]))
print(f"{'role':<8}{'head':<9}{'marg | S-Inh KEPT':>19}{'marg | S-Inh GONE':>19}{'delta':>9}  flip?")
for l,h,a,b,d,f in rows:
    print(f"{l:<8}L{h[0]}H{h[1]:<6}{a:>19.4f}{b:>19.4f}{d:>9.4f}  {'YES' if f else ''}")

# bootstrap over prompts on the flip
P=ld_pp.shape[1]; B=1000; rng=np.random.default_rng()
targets=[(lab[i],h,i) for i,h in enumerate(heads) if lab[i] in ("NegNM","NM")]
flips={f"{l}_L{h[0]}H{h[1]}":0 for l,h,i in targets}
for b in range(B):
    vb=ld_pp[:,rng.integers(0,P,P)].mean(1)
    for l,h,i in targets:
        bit=1<<i; lacks=(idx&bit)==0
        s1=lacks&sinh_mask; s2=lacks&sinh_gone
        m1=(vb[idx[s1]|bit]-vb[idx[s1]]).mean(); m2=(vb[idx[s2]|bit]-vb[idx[s2]]).mean()
        if np.sign(m1)!=np.sign(m2): flips[f"{l}_L{h[0]}H{h[1]}"]+=1
print(f"\nsign-flip rate over {B} prompt-bootstrap draws:")
for k,c in sorted(flips.items()): print(f"   {k:<18} {c}/{B}")
json.dump({'rows':[[l,list(h),a,b,d,bool(f)] for l,h,a,b,d,f in rows],'flip_rate':flips,'B':B},
          open('results/v13_conditional_marginal.json','w'),indent=1)
