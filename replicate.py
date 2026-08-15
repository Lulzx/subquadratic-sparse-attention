"""Replication check of the numeric claims in the SubQ-1.1-Small technical report.

Each claim is recomputed from first principles and compared against the value
reported in the paper. PASS = replicates within rounding; FAIL = does not.
"""

PASS, FAIL = "PASS", "FAIL"
results = []


def check(claim, reported, computed, tol=0.02):
    """tol is relative tolerance (2% default, ~ one sig fig of rounding)."""
    if isinstance(reported, str) and isinstance(computed, str):
        ok = reported == computed
    else:
        rel = abs(computed - reported) / max(abs(reported), 1e-12)
        ok = rel <= tol
    results.append((PASS if ok else FAIL, claim, reported, computed))


K, M = 1_024, 1_048_576
T, P = 1e12, 1e15  # FLOP units used by the paper (T=10^12, P=10^15)

# ---------------------------------------------------------------- Figure 3
# "At 128K ... attention requires only 8.6 billion operations per layer;
#  at 1M, 549 billion; at 2M, 2.2 trillion, a fourfold increase for a
#  doubling of context."
# Hypothesis: "operations" = number of token-pair comparisons = n^2 / 2.
for n, label, reported in [
    (128 * K, "128K", 8.6e9),
    (1 * M, "1M", 549e9),
    (2 * M, "2M", 2.2e12),
]:
    check(f"Fig3 dense ops/layer @{label} = n²/2", reported, n * n / 2)
check(
    "Fig3 'fourfold increase for a doubling' (2M vs 1M)",
    4.0,
    (2 * M / (1 * M)) ** 2,
)

# ---------------------------------------------------------------- Table 1
# Dense attention per layer (PFLOP), quadratic: fit c*n^2 off the 1M point.
dense_1m = 252 * P
c_dense = dense_1m / (1 * M) ** 2
# SSA per layer (PFLOP), linear: fit a*n off the 1M point.
ssa_1m = 3.9 * P
a_ssa = ssa_1m / (1 * M)

table1 = [
    # (ctx tokens, dense P, SSA P, reduction)
    (32 * K, 0.25, 0.12, 2.1),
    (64 * K, 0.99, 0.25, 4.0),
    (128 * K, 3.9, 0.49, 8.0),
    (256 * K, 15.8, 0.99, 16),
    (512 * K, 63.0, 2.0, 31.5),
    (1 * M, 252, 3.9, 64.5),
]
for n, d_rep, s_rep, r_rep in table1:
    check(f"T1 dense @{n//K}K quadratic from 1M point", d_rep * P, c_dense * n * n)
    check(f"T1 SSA @{n//K}K linear from 1M point", s_rep * P, a_ssa * n)
    check(f"T1 reduction @{n//K}K", r_rep, (d_rep * P) / (s_rep * P))

# Headline: "SSA reduces attention FLOPs by 64.5x at a 1M-token context"
check("Abstract headline 64.5x @1M", 64.5, 252 * P / (3.9 * P))

# ---------------------------------------------------------------- Table 2
# V3.2-style DSA: Lightning Indexer (claimed quadratic) vs main sparse
# attention reading a fixed 2,048 selected tokens (claimed linear).
seqs = [128 * K, 256 * K, 512 * K, 1 * M, 2 * M, 4 * M, 8 * M, 12 * M]
idx_rep = [156.4, 594.2, 2.31e3, 9.13e3, 36.3e3, 144.6e3, 577.5e3, 1298.5e3]
spa_rep = [71.0, 142.1, 284.2, 568.3, 1.14e3, 2.27e3, 4.55e3, 6.82e3]
ratio_rep = [2.2, 4.2, 8.1, 16.1, 31.9, 63.6, 127.0, 190.4]

# Main sparse attention: linear fit through origin off the 128K point.
a_spa = spa_rep[0] * T / seqs[0]
for n, rep in zip(seqs, spa_rep):
    check(f"T2 sparse-attn (top-2048) @{n//K}K linear", rep * T, a_spa * n)

# Indexer: test PURE quadratic first (the paper's stated model).
c_quad = idx_rep[0] * T / seqs[0] ** 2
for n, rep in zip(seqs[1:], idx_rep[1:]):
    check(f"T2 indexer @{n//K}K pure-quadratic from 128K", rep * T, c_quad * n * n)

# Indexer: least-squares fit of c*n^2 + b*n (quadratic + linear overhead).
# Tiny residuals => table consistent with quadratic-plus-overhead model.
def fit_indexer():
    import numpy as np

    A = np.array([[n * n, n] for n in seqs], dtype=float)
    y = np.array([v * T for v in idx_rep])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


try:
    import numpy  # noqa: F401

    (c_idx, b_idx) = fit_indexer()
except ImportError:
    # closed-form 2x2 normal-equation solve, no numpy needed
    s4 = sum(n**4 for n in seqs)
    s3 = sum(n**3 for n in seqs)
    s2 = sum(n * n for n in seqs)
    t4 = sum(n * n * v * T for n, v in zip(seqs, idx_rep))
    t3 = sum(n * v * T for n, v in zip(seqs, idx_rep))
    c_idx = (t4 * s2 - t3 * s3) / (s4 * s2 - s3 * s3)
    b_idx = (t3 - c_idx * s3) / s2

for n, rep in zip(seqs, idx_rep):
    check(f"T2 indexer @{n//K}K fit c·n²+b·n", rep * T, c_idx * n * n + b_idx * n)

# Crossover: indexer cost = sparse-attn cost. Solve c*n^2 + b*n = a*n.
n_cross = (a_spa - b_idx) / c_idx
check("T2/§5.6 indexer-vs-attention crossover ≈52K", 52 * K, n_cross, tol=0.05)

# Reported index/attention ratios.
for n, ir, sr, rr in zip(seqs, idx_rep, spa_rep, ratio_rep):
    check(f"T2 index/attn ratio @{n//K}K", rr, (ir * T) / (sr * T))

# §5.6 text: indexer "16.1x ... at 1M tokens and 190.4x at 12M"
check("§5.6 indexer 16.1x @1M", 16.1, (idx_rep[3] * T) / (spa_rep[3] * T))
check("§5.6 indexer 190.4x @12M", 190.4, (idx_rep[7] * T) / (spa_rep[7] * T))

# ---------------------------------------------------------------- Table 3
# "DSA layer" = indexer + main sparse attention; "DBSA layer" column equals
# the sparse-attention column (matched selected-position budget).
dsa_rep = [227.4, 736.3, 2.60e3, 9.70e3, 37.4e3, 146.9e3, 582.0e3, 1305.4e3]
for n, rep, ir, sr in zip(seqs, dsa_rep, idx_rep, spa_rep):
    check(f"T3 DSA layer @{n//K}K = indexer+sparse", rep * T, ir * T + sr * T)
for n, dr, sr, rr in zip(seqs, dsa_rep, spa_rep, [
    3.2, 5.2, 9.1, 17.1, 32.9, 64.6, 128.0, 191.3
]):
    check(f"T3 DSA/DBSA ratio @{n//K}K", rr, (dr * T) / (sr * T))

# ---------------------------------------------------------------- Sparsity
# "At 12M tokens, the model is attending to only 0.13% of token pairs"
# Model: attended fraction = k/n with fixed selected-position count k.
# From Table 1, SSA/dense @1M = 3.9/252; fraction scales as 1/n (n^2 vs n).
frac_1m = (3.9 * P) / (252 * P)
frac_12m = frac_1m * (1 * M) / (12 * M)
check("§4.1 0.13% of token pairs @12M (derived from Table 1)", 0.0013, frac_12m, tol=0.10)
check(
    "§4.5/abstract 'nearly a 1,000x reduction' (actual 1/0.13%)",
    1000,
    1 / frac_12m,
    tol=0.35,  # 769x vs 'nearly 1000x' — generous tolerance, still flags shape
)

# ---------------------------------------------------------------- Figure 12
check("Fig12 56x @1M (54,164 ms / 966 ms)", 56, 54_164 / 966)

# ---------------------------------------------------------------- Report
width = max(len(r[1]) for r in results)
n_pass = sum(1 for r in results if r[0] == PASS)
print(f"{'':4} { 'CLAIM' :<{width}}  {'REPORTED':>12}  {'COMPUTED':>14}")
for status, claim, reported, computed in results:
    print(f"{status:4} {claim:<{width}}  {reported!r:>12}  {computed!r:>14}")
print(f"\n{n_pass}/{len(results)} checks pass\n")

fails = [r for r in results if r[0] == FAIL]
if fails:
    print("FAILURES:")
    for _, claim, reported, computed in fails:
        print(f"  - {claim}: reported {reported!r}, computed {computed!r}")
