"""Möbius and Walsh-Hadamard decomposition of pseudo-Boolean functions.

Pure numpy — no model dependency. Operates on a coalition value table
of shape (2^n,) indexed by coalition bitmask.

Two transforms serve different roles:
- WHT (orthogonal): energy/variance quantities and pre-registered
  thresholds. Parseval holds: sum(wht^2) = 2^n * sum(v^2).
- Möbius (not orthogonal): subset-level identification of which
  specific head combinations carry signal. Descriptive/exploratory only.
"""

import numpy as np


def mobius_transform(values):
    """Exact Möbius (zeta) inversion: recover a(T) from v(S).

    v(S) = sum_{T subseteq S} a(T)  =>  a(T) = sum_{S subseteq T} (-1)^{|T|-|S|} v(S)

    Uses the fast inclusion-exclusion butterfly, O(n * 2^n).
    Input: values of shape (2^n,), indexed by coalition bitmask.
    Output: Möbius coefficients of shape (2^n,), same indexing.
    """
    n = int(np.log2(len(values)))
    assert len(values) == 2 ** n
    a = values.astype(np.float64).copy()
    for i in range(n):
        stride = 1 << (i + 1)
        lo = 1 << i
        a_view = a.reshape(-1, stride)
        a_view[:, lo:] -= a_view[:, :lo]
    return a


def inverse_mobius_transform(coeffs):
    """Reconstruct v(S) from Möbius coefficients: v(S) = sum_{T subseteq S} a(T)."""
    n = int(np.log2(len(coeffs)))
    assert len(coeffs) == 2 ** n
    v = coeffs.astype(np.float64).copy()
    for i in range(n):
        stride = 1 << (i + 1)
        lo = 1 << i
        v_view = v.reshape(-1, stride)
        v_view[:, lo:] += v_view[:, :lo]
    return v


def wht(values):
    """Walsh-Hadamard Transform (unnormalized).

    H_n v, where H_n is the 2^n x 2^n Hadamard matrix with entries
    (-1)^{<x,y>}. O(n * 2^n) vectorized butterfly.
    """
    n = int(np.log2(len(values)))
    assert len(values) == 2 ** n
    a = values.astype(np.float64).copy()
    h = 1
    for _ in range(n):
        a_view = a.reshape(-1, 2 * h)
        lo = a_view[:, :h].copy()
        hi = a_view[:, h:]
        a_view[:, :h] = lo + hi
        a_view[:, h:] = lo - hi
        h *= 2
    return a


def iwht(coeffs):
    """Inverse WHT (normalized by 2^n)."""
    return wht(coeffs) / len(coeffs)


def _popcount_array(n):
    """Precompute popcount for all indices 0..2^n-1."""
    idx = np.arange(2 ** n, dtype=np.uint32)
    pc = np.zeros(2 ** n, dtype=np.int32)
    while idx.any():
        pc += (idx & 1).astype(np.int32)
        idx >>= 1
    return pc


def subset_order(index):
    """Number of set bits (popcount) = interaction order of this subset."""
    return bin(index).count('1')


def energy_by_order(coeffs, n):
    """Squared energy at each interaction order 0..n.

    Returns array of shape (n+1,) where entry k is the sum of squared
    coefficients over all subsets of size k.
    """
    pc = _popcount_array(n)
    c2 = coeffs.astype(np.float64) ** 2
    energy = np.zeros(n + 1)
    for k in range(n + 1):
        energy[k] = c2[pc == k].sum()
    return energy


def energy_by_order_normalized(coeffs, n):
    """Energy spectrum normalized to fraction of total variance."""
    e = energy_by_order(coeffs, n)
    total = e.sum()
    if total == 0:
        return e
    return e / total


def per_head_energy(coeffs, n):
    """Sum of squared coefficients for all subsets containing each head.

    Returns array of shape (n,) — one energy value per head.
    """
    c2 = coeffs.astype(np.float64) ** 2
    idx = np.arange(len(coeffs), dtype=np.int64)
    energies = np.zeros(n)
    for head in range(n):
        mask = idx & (1 << head)
        energies[head] = c2[mask != 0].sum()
    return energies


def restricted_energy_ge3(coeffs, n, head_subset_indices):
    """Order->=3 energy restricted to subsets whose support is within head_subset_indices.

    head_subset_indices: list of head indices (0-based positions in the n-head circuit).
    Returns (restricted_energy, total_energy, fraction).
    """
    subset_mask = 0
    for h in head_subset_indices:
        subset_mask |= (1 << h)

    c2 = coeffs.astype(np.float64) ** 2
    total = c2.sum()

    idx = np.arange(len(coeffs), dtype=np.int64)
    pc = _popcount_array(n)
    within = (idx & ~subset_mask) == 0
    ge3 = pc >= 3
    restricted = c2[within & ge3].sum()

    fraction = float(restricted / total) if total > 0 else 0.0
    return float(restricted), float(total), fraction


def wht_energy_by_order(values, n):
    """WHT energy spectrum: fraction of total variance at each order.

    Uses WHT coefficients (orthogonal, Parseval-valid). This is the
    correct variance partition for pre-registered thresholds.
    """
    w = wht(values)
    return energy_by_order_normalized(w, n)


def wht_per_head_energy(values, n):
    """Per-head WHT energy: sum of squared WHT coefficients containing each head.

    Normalized by total WHT energy so values are comparable across scales.
    """
    w = wht(values)
    raw = per_head_energy(w, n)
    total = np.sum(w ** 2)
    if total == 0:
        return raw
    return raw / total


def wht_restricted_energy_ge3(values, n, head_subset_indices):
    """Order->=3 WHT energy restricted to indices with support within head_subset_indices."""
    w = wht(values)
    return restricted_energy_ge3(w, n, head_subset_indices)


def interaction_fraction(values, n):
    """Fraction of set-function variance due to interactions (order >= 2).

    Uses the full 2^n WHT decomposition. Equivalent to 1 - R^2 of the
    best additive (linear) model fit to all coalition values.

    Returns a scalar in [0, 1]. Zero means purely additive; one means
    no individual-head effects at all.
    """
    spectrum = wht_energy_by_order(values, n)
    nc_energy = spectrum[1:].sum()
    if nc_energy == 0:
        return 0.0
    return float(1.0 - spectrum[1] / nc_energy)


def loo_epistasis(values, n):
    """Leave-one-out epistasis: 1 - sum(LOO marginals) / faithfulness.

    Only uses coalitions of size n and n-1. Unstable when faithfulness
    is near zero. Returns (epistasis, faithfulness, loo_sum) so the
    caller can decide whether to trust it.
    """
    full = (1 << n) - 1
    faith = float(values[full] - values[0])
    loo_sum = 0.0
    for i in range(n):
        complement = full ^ (1 << i)
        loo_sum += values[full] - values[complement]
    loo_sum = float(loo_sum)
    if abs(faith) < 1e-10:
        return 0.0, faith, loo_sum
    return float(1.0 - loo_sum / faith), faith, loo_sum


def shapley_values(values, n):
    """Exact Shapley values from the full 2^n coalition table.

    Returns array of shape (n,) — one value per player.
    """
    from math import factorial
    n_total = 1 << n
    sv = np.zeros(n, dtype=np.float64)
    for j in range(n):
        for S in range(n_total):
            if S & (1 << j):
                continue
            s_size = bin(S).count("1")
            marginal = values[S | (1 << j)] - values[S]
            w = factorial(s_size) * factorial(n - s_size - 1) / factorial(n)
            sv[j] += w * marginal
    return sv


def shapley_interaction_index(values, n):
    """Pairwise Shapley interaction index for all n*(n-1)/2 pairs.

    Returns dict mapping (i, j) -> float. Positive = synergy,
    negative = redundancy. Uses ALL 2^n coalitions.
    """
    from math import factorial
    n_total = 1 << n
    interactions = {}
    for j in range(n):
        for k in range(j + 1, n):
            I_jk = 0.0
            for S in range(n_total):
                if S & (1 << j) or S & (1 << k):
                    continue
                s_size = bin(S).count("1")
                delta = (
                    values[S | (1 << j) | (1 << k)]
                    - values[S | (1 << j)]
                    - values[S | (1 << k)]
                    + values[S]
                )
                w = factorial(s_size) * factorial(n - s_size - 2) / factorial(n - 1)
                I_jk += w * delta
            interactions[(j, k)] = float(I_jk)
    return interactions


def total_shapley_interaction(values, n):
    """Scalar summary of pairwise Shapley interaction: sum of |SII(i,j)|.

    Uses all 2^n coalitions. Higher = more pairwise epistasis.
    """
    sii = shapley_interaction_index(values, n)
    return float(sum(abs(v) for v in sii.values()))


def compute_all_metrics(values, n, head_labels=None):
    """Compute all epistasis metrics from a 2^n coalition value table.

    Returns a dict with every metric we report in the paper.
    """
    full = (1 << n) - 1
    faith = float(values[full] - values[0])

    spectrum = wht_energy_by_order(values, n)
    nc = spectrum[1:]
    nc_total = nc.sum()
    nc_frac = nc / nc_total if nc_total > 0 else nc

    wif = float(1.0 - nc_frac[0]) if nc_total > 0 else 0.0

    loo_ep, _, loo_sum = loo_epistasis(values, n)

    sv = shapley_values(values, n)
    sii = shapley_interaction_index(values, n)
    total_sii = float(sum(abs(v) for v in sii.values()))

    top_interactions = sorted(sii.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    result = {
        "faithfulness": faith,
        "intact_value": float(values[full]),
        "empty_value": float(values[0]),
        "walsh_interaction_fraction": wif,
        "order1_frac": float(nc_frac[0]),
        "order2_frac": float(nc_frac[1]) if len(nc_frac) > 1 else 0.0,
        "order3plus_frac": float(nc_frac[2:].sum()) if len(nc_frac) > 2 else 0.0,
        "loo_epistasis": loo_ep,
        "loo_sum": loo_sum,
        "loo_stable": abs(faith) > 0.1,
        "total_shapley_interaction": total_sii,
    }

    if head_labels:
        result["shapley_values"] = {
            head_labels[j]: float(sv[j]) for j in range(n)
        }
        result["top_interactions"] = [
            {
                "pair": (head_labels[j], head_labels[k]),
                "sii": v,
            }
            for (j, k), v in top_interactions
        ]
    else:
        result["shapley_values"] = {str(j): float(sv[j]) for j in range(n)}
        result["top_interactions"] = [
            {"pair": (j, k), "sii": v} for (j, k), v in top_interactions
        ]

    return result


def complement_values(values, n):
    """Compute w(S) = v(complement(S)) by index flipping.

    v(S) measures sufficiency (output when only S is active).
    w(S) = v(complement(S)) measures necessity (output when S is
    removed and everything else remains).
    """
    full_mask = (1 << n) - 1
    idx = np.arange(len(values), dtype=np.int64)
    return values[full_mask ^ idx]
