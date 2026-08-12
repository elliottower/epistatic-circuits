# Verification packet: Subspace decomposition of head-level epistasis

## For Perplexity to check

We are writing a paper on epistasis (non-additive interactions) between
attention heads in transformer circuits. We measure epistasis using Walsh
decomposition of coalition game values (ablate subsets of heads, measure
task performance). We want to test whether head-level epistasis is explained
by residual-stream subspace overlap between heads.

## Specific claims to verify

### 1. Residual stream is the only inter-head communication channel
Claim: In a standard transformer (GPT-2), all communication between attention
heads and MLP layers goes through the residual stream. There is no side channel.
- Is this correct for GPT-2 specifically?
- Are there any exceptions (e.g., shared LayerNorm statistics, attention
  pattern leakage)?
- Does LayerNorm create implicit interaction beyond the residual stream?

### 2. OV circuit composition is standard in mechinterp
Claim: W_OV = W_V @ W_O is the standard "OV circuit" in mechanistic
interpretability, and its subspace overlap between heads is a standard measure.
- Confirm or correct the formula
- Who introduced this? (Elhage et al. 2021? "A Mathematical Framework"?)
- Is Frobenius inner product the standard overlap measure, or is there
  something better (e.g., principal angles, Grassmannian distance)?

### 3. QK composition / virtual attention heads
Claim: The "virtual attention head" concept (Elhage et al. 2021) describes
how head B's attention pattern is influenced by head A's output through
W_Q_B^T @ W_OV_A @ W_K_B (the "QK-OV composition path").
- Confirm or correct this formula
- Is this the same as "Q-composition" and "K-composition" from the
  Anthropic mathematical framework paper?
- Any important subtleties we're missing (e.g., LayerNorm between layers)?

### 4. Walsh decomposition on coalition games
Claim: Walsh-Hadamard decomposition of a set function f: 2^[n] -> R
decomposes it into additive (order-1), pairwise (order-2), and higher-order
interaction terms. The order-2 coefficient W_{ij} measures the interaction
between players i and j.
- Is this the same as "Fourier analysis of Boolean functions" (O'Donnell 2014)?
- Is it equivalent to or distinct from Shapley interaction indices?
- Any prior work applying Walsh decomposition to neural network components
  specifically?

### 5. Connecting epistasis to subspace overlap
Has anyone previously tested whether pairwise interactions between attention
heads (measured by any method — Shapley, Walsh, ablation) correlate with
subspace overlap (OV or QK composition)?
- Search for papers combining "attention head interaction" + "subspace"
- Search for "epistasis" + "transformer" or "neural network"
- Any work on decomposing head-level effects into geometric vs computational?

### 6. LayerNorm as interaction source
Claim: LayerNorm creates implicit interactions because it normalizes across
the full residual stream, so one head's output magnitude affects another
head's input after normalization.
- How significant is this effect in practice?
- Does anyone account for it when computing OV/QK composition?
- Should we use "effective weights" (folding LayerNorm into W_Q, W_K etc.)
  instead of raw weights?

## What is OURS vs BORROWED

**Ours (novel contributions):**
- Full Walsh decomposition of 2^n coalition games on transformer circuits
- Comparing Walsh interaction fraction (WIF), LOO epistasis, TSII as
  construct validity operationalizations
- Testing whether subspace overlap predicts Walsh pairwise coefficients
- The specific pre-registered predictions (P1-P5 in our pre-reg)

**Borrowed/standard (need correct attribution):**
- Walsh-Hadamard decomposition (signal processing / Boolean function analysis)
- OV circuit / W_OV composition (Elhage et al. 2021, "A Mathematical Framework")
- Virtual attention heads / QK composition (same paper)
- Coalition game / Shapley value framework (cooperative game theory)
- Residual stream as communication channel (standard transformer mechinterp)

## Format requested
For each claim (1-6), give:
- Confirmed / Partially correct / Incorrect
- Correct formulation if we got it wrong
- Key citation(s)
- Any important caveats we should mention
