# Architecture

## Goal

The architecture targets three properties:

1. content-dependent routing rather than a fixed positional mask;
2. selection that does not compute all query–key scores;
3. a fixed attention-read budget that works in causal prefill and autoregressive decoding.

It combines two paths:

- a causal sliding window for local syntax and short-range composition;
- a content-routed selected-attention path for distant retrieval.

Learned per-head gates mix the two outputs before the final projection.

## Multi-table SimHash selector

For token state `x_t ∈ R^d`, a learned projection produces `T × b` real-valued hash logits. The measured baseline uses:

- `T = 4` independent tables;
- `b = 16` bits per table;
- `m = 4` prior members read per table;
- two positions per member: the anchor and its successor.

Each table code is the sign pattern of 16 projected values. A query computes the same codes and directly looks up the corresponding causal buckets. There is no query-by-key scoring matrix in the selector.

The selector optionally performs multiprobe lookup by assigning each token to its exact code and `p - 1` one-bit neighbors. Neighbor bits are chosen by the smallest absolute projection logits, which are the sign decisions closest to their boundaries. Multi-assignment lets every probe retain its own causal rank in the sorted index; no all-pairs probe scoring is introduced.

The selected-token budget is therefore:

```text
K = tables × probes × members × block width
  = 4 × 1 × 4 × 2
  = 32 tokens per query
```

The recommended fixed-budget configuration uses eight tables, one probe, two members, and block width two, preserving `K = 32`. A controlled four-table/two-probe/two-member variant also preserves `K = 32` and measures the multiprobe effect without increasing attention work.

## Why include the successor?

Content routing naturally locates a matching key token. In associative recall, however, the desired value often appears immediately after the key. Reading only the matching token forces a later layer to reconstruct a positional pointer without having the value in its candidate set.

The selector therefore returns both:

```text
anchor position p
successor position min(p + 1, current position)
```

The causal restriction ensures the successor never lies in the future. This two-token block read was the critical change that made the selector compatible with induction-style key/value retrieval.

## Causal prefill

The portable implementation flattens `(batch, token, table, probe)` assignments, sorts them by global bucket identifier, and uses the inverse permutation to locate each token inside each bucket. The `m` immediately preceding entries in the sorted bucket are prior causal members.

No future content affects prior outputs. The test suite verifies this by mutating every token after a cutoff and comparing all earlier outputs.

## Autoregressive decoding

The intended decode data structure is simpler than prefill:

1. compute the configured table hashes and low-margin probes for the new token;
2. read the corresponding append-only bucket tails;
3. gather at most the configured `K` key/value positions;
4. compute selected softmax attention;
5. append the new position to its configured exact and probe buckets.

The current repository implements parallel prefill and causal semantics. It does not yet ship a persistent decode cache.

## Attention computation

Queries and gathered keys receive RoPE at their true absolute positions. Attention is exact over the selected set:

```text
P_t = softmax(Q_t K_selected(t)^T / sqrt(d_head))
O_t = P_t V_selected(t)
```

Rows with no valid historical candidate produce zero selected-attention output rather than NaNs. The local sliding-window path remains active for those early positions.

## Router training

Hard hash lookup is discrete. Gradients are explicitly stopped through sort and gather indices. The hash projection is trained through an auxiliary objective:

- **balance** keeps each bit near 50/50 occupancy across a batch;
- **confidence** discourages projections from remaining near the sign boundary.

Q/K/V projections and all gathered values receive ordinary task gradients. A future semantic-routing objective should train hash agreement directly from teacher attention or retrieval labels.

## Complexity

Let sequence length be `n`, model width `d`, tables `T`, probes `p`, bits `b`, selected tokens `K`, and local window `W`.

| Component | Logical implementation | Portable prototype |
|---|---:|---:|
| Hash projection | `O(n d T b)` | `O(n d T b)` |
| Bucket construction | expected `O(n T p)` | `O(n T p log(nTp))` sorting |
| Candidate lookup | expected `O(n T p m)` | `O(n T p m)` after sorting |
| Selected attention | `O(n K d)` | `O(n K d)` |
| Window attention | `O(n W d)` | `O(n W d)` |

With fixed `T`, `p`, `b`, `m`, `K`, and `W`, the intended hash-table design is linear in `n`. The current cross-platform prefill selector is subquadratic but not strictly linear because it uses sorting.

## What is and is not being replicated

The public SubQ report states behavioral requirements and benchmark results but does not disclose its SSA mechanism. This repository independently implements those requirements using learned SimHash routing. No claim is made that SubQ uses hashing, block reads, these gates, or any code in this repository.
