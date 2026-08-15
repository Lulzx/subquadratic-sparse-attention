# Architecture

## Goal

The architecture targets three properties:

1. content-dependent routing rather than a fixed positional mask;
2. selection that does not compute all query–key scores;
3. a fixed attention-read budget that works in causal prefill and autoregressive decoding.

The measured baseline combines two paths:

- a causal sliding window for local syntax and short-range composition;
- a content-routed selected-attention path for distant retrieval.

Learned per-head gates mix the two outputs before the final projection.

An opt-in research variant adds a third path: a fixed number of compressed causal
history slots. Tokens are assigned to interleaved slots, each slot keeps the prefix
mean of its keys and values, and every query attends to the slot states available
strictly before it. With a fixed slot count this path has linear work and memory. It is
a first coarse-global control, not an implementation of SubQ or COBS.

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

## Completed-block index

The LFM2.5 path also implements a true block-index variant. It averages the hidden
states in each fixed-width block, hashes that summary, retrieves a bounded number of
block anchors, and runs exact token attention inside the selected blocks. A block is
searchable only when its final token is strictly before the query's distant-window
cutoff. Computing summaries for later blocks therefore cannot affect an earlier query.

The current three-seed experiment uses eight tables, one probe, two blocks per table, and
four tokens per block:

```text
K = 8 tables × 1 probe × 2 blocks × 4 tokens = 64 distant tokens
```

The portable implementation still sorts bucket entries, and mean pooling is only a
first block-summary baseline. The test suite mutates future blocks and verifies exact
invariance of all earlier selections.

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

With `semantic_router=True`, half of the configured tables remain shared-hash fallback
tables and half own separate query and key hash projections. The total table count and
selected-token budget remain fixed. The auxiliary loss retains balance and confidence
terms and adds a bounded dense-teacher cross-entropy objective: continuous query/key
hash similarity is trained to approximate the attention distribution produced by the
layer's Q/K projections. Only the first configured `router_teacher_tokens` participate,
which keeps the quadratic teacher temporary bounded at longer training lengths. Hard
bucket indices remain stop-gradient. This hybrid fallback is necessary because an
initial all-semantic router lost exact MQAR reachability as soon as its Q/K hashes
diverged.

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

The optional semantic-router loss does form dense teacher and student score matrices,
but only over `r = min(n, router_teacher_tokens)` training positions. Its `O(r^2)` work
is bounded by the configured cap and is absent from inference. See
[Complexity and capacity](complexity-and-capacity.md) for the full proof, path-specific
candidate budgets, causal argument, memory bound, and unresolved recall-scaling problem.

## What is and is not being replicated

The public SubQ report states behavioral requirements and benchmark results but does not disclose its SSA mechanism. This repository independently implements those requirements using learned SimHash routing. No claim is made that SubQ uses hashing, block reads, these gates, or any code in this repository.
