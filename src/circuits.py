"""Circuit definitions for all discovery methods evaluated in the paper.

Single source of truth for head lists. Every script imports from here.
"""

# C2: EAP-IG activation patching (top 15 by EAP-IG edge score, collapsed to heads)
C2_EAP = [
    (0, 1), (0, 10), (2, 2), (4, 11), (5, 5), (5, 8), (5, 9), (6, 1),
    (6, 9), (7, 3), (7, 9), (8, 6), (8, 10), (10, 7), (11, 10),
]

# C3: Canonical IOI circuit (Wang et al. 2023, manually identified)
C3_CANONICAL = [
    (0, 1), (3, 0), (2, 2), (4, 11), (5, 5), (6, 9),
    (7, 3), (7, 9), (8, 6), (8, 10), (9, 9), (9, 6), (10, 0),
    (10, 7), (11, 10),
]

# C4: Random baseline (15 heads drawn uniformly, seed=42)
C4_RANDOM = [
    (0, 3), (1, 0), (1, 10), (3, 8), (3, 9), (3, 10), (4, 8),
    (6, 5), (6, 10), (7, 10), (8, 1), (8, 8), (8, 11), (9, 4), (10, 8),
]

# IC-15: Information content discovery (top 15 by mutual information with task label)
IC15 = [
    (10, 7), (4, 0), (0, 1), (4, 11), (9, 4),
    (5, 6), (8, 9), (8, 4), (0, 5), (11, 7),
    (0, 9), (11, 8), (10, 3), (1, 3), (0, 3),
]

# C5: Walsh discovery (LASSO-Walsh over 12K random coalitions, top 15 by |order-1 coefficient|)
C5_WALSH = [
    (5, 5), (10, 7), (11, 1), (8, 6), (8, 10),
    (0, 9), (7, 9), (0, 3), (6, 9), (10, 1),
    (11, 2), (10, 10), (3, 0), (11, 10), (4, 0),
]

# C6: Epistatic discovery (same LASSO-Walsh fit, top 15 by total order-2 energy)
# Selects heads with strongest pairwise interactions, regardless of individual effect
C6_EPISTATIC = [
    (5, 5), (8, 6), (11, 10), (10, 7), (6, 9),
    (0, 10), (0, 1), (10, 0), (5, 9), (8, 10),
    (11, 2), (9, 9), (0, 9), (7, 9), (4, 0),
]

# RTI: Repeated Token Identification circuit (15 heads, manually identified)
RTI = [
    (0, 8), (0, 9), (0, 11),
    (4, 11),
    (4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10),
    (10, 11), (11, 9), (11, 11),
]

IOI_CIRCUITS = {
    "C2_eap": C2_EAP,
    "C3_canonical": C3_CANONICAL,
    "C4_random": C4_RANDOM,
    "IC15": IC15,
    "C5_walsh": C5_WALSH,
    "C6_epistatic": C6_EPISTATIC,
}

RTI_CIRCUITS = {
    "RTI": RTI,
}

ALL_CIRCUITS = {**IOI_CIRCUITS, **RTI_CIRCUITS}

ALL_HEADS = [(layer, head) for layer in range(12) for head in range(12)]
