# Complexity and capacity of the selector

This note states the strongest complexity claim supported by the current code and
separates it from the unresolved retrieval-capacity question. It applies to the
portable parallel-prefill path in `ssa/mlx_selector.py`; incremental decoding still
needs a persistent index.

## Parameters and candidate budgets

Let:

- `n` be sequence length and `A` batch size;
- `d` be model width;
- `T` be the number of hash tables;
- `b` be bits per table;
- `p` be query probes per table;
- `m` be members read per bucket;
- `R` be the number of token positions emitted per retrieved member or block; and
- `W` be the local-window width.

The maximum distant token budget is

```text
K = T * p * m * R.
```

`R` is path-specific. The tiny MQAR baseline emits an anchor and its successor, so
`R=2` and `4 * 1 * 4 * 2 = 32`. The current LFM token router emits direct token
positions (`R=1`). Its completed-block path emits every token in a selected block, so
`R=block_size`; the replicated setting uses `8 * 1 * 2 * 4 = 64` distant tokens.
Local tokens and explicit sink tokens are additional fixed candidate sets.

## Portable-prefill theorem

For fixed batch size and architectural parameters, the current portable selector plus
selected attention takes `O(n log n)` work and is therefore strictly subquadratic in
sequence length.

More generally, its work is bounded by

```text
O(A n d T b)                         hash projections
+ O(A n T b log b)                   lowest-margin query probes
+ O(A n T (1 + p) log(A n T (1+p))) portable bucket sorting
+ O(A n T p m)                       bounded predecessor lookup
+ O(A n (K + W + S) d)               gathered exact attention
```

where `S` is the number of explicit sink tokens. Ordinary Q/K/V/O projections add
`O(A n d^2)`, as they do in dense attention. Holding model width and all routing
parameters fixed with respect to `n` gives

```text
O(n) + O(n log n) + O(n) = O(n log n) = o(n^2).
```

The two sorts in the selector do not change the bound: a constant number of
`O(n log n)` operations is still `O(n log n)`. This is a conservative comparison-sort
model for MLX `argsort`; the backend may use a different integer-sort implementation,
but the repository does not rely on a stronger backend-specific bound.

The implementation never constructs a query-by-history routing score matrix during
inference. It hashes each query to a bounded number of bucket addresses and takes only
the `m` preceding members. Selected attention then gathers those K/V states and runs
ordinary softmax over the gathered set. Candidate selection is approximate; attention
arithmetic over the selected set is exact.

Memory is `O(A n T (1+p) + A n(K+W+S))` for the portable prefill temporaries when the
candidate dimension is materialized. Query chunking bounds selected-attention working
memory, but the current selector still constructs a sequence-wide sorted index.

## Causality proof

For a key at position `j`, the token selector uses the within-bucket sort suffix

```text
2*j + 1.
```

For a query at position `i` with distant exclusion `D=min_distance`, it uses

```text
2*max(i-D, 0).
```

Queries sort before keys at the same cutoff. The selector then validates the returned
anchor with `j < i-D`. Consequently, direct token routing cannot select the current or
a future token. If anchor-plus-successor expansion is enabled, `j+1 <= i-D`, so it is
also causal and remains outside the excluded recent region.

The completed-block selector replaces `j` with the block's final token `e`. A block is
valid only if `e < i-D`; expanding the block can therefore expose no token later than
`e`. Future hidden states and incomplete blocks cannot alter an earlier query's result.
These properties are covered by mutation-based causality tests in `mlx_tests.py`.

## Bounded quadratic training teacher

The optional semantic-router objective in `ssa/mlx_model.py` constructs dense teacher
and student score tensors over

```text
r = min(n, router_teacher_tokens).
```

That auxiliary training work is `O(r^2)`. The default cap is 256, so it is bounded with
respect to longer `n`; it is not used by sparse inference. Setting
`router_teacher_tokens=n` would make that training objective quadratic even though the
inference path remained subquadratic.

Other recovery tools can deliberately call dense donor layers or compare dense teacher
outputs during conversion. Those are offline, bounded training/evaluation procedures,
not part of the claimed sparse inference algorithm.

## The unresolved capacity question

Subquadratic computation does not imply context-independent retrieval quality. With a
fixed `b`, each table has `2^b` addresses. Under a roughly balanced hash distribution,
expected bucket occupancy grows like `n/2^b`. Because lookup keeps only a bounded tail
of `m` members, a relevant old item can eventually fall outside that tail.

This yields the central open question:

> How must address width, table count, probing, and candidate budget scale with context
> length to keep retrieval recall high?

Under an idealized balanced and independent hash model, this question has a useful
baseline calculation. For a target with `L` later keys, later collisions in one table
obey

```text
C ~ Binomial(L, 2^-b).
```

The target remains in that table's bounded tail exactly when `C < m`. If query and key
addresses agree with probability `a` per table, idealized recall across `T` independent
tables is

```text
P(recall) = 1 - (1 - a * P[C < m])^T.
```

This is not a learned-router recall theorem: real addresses may be imbalanced,
correlated, or disagree semantically. It is a collision-capacity baseline that isolates
what fixed bucket tails can support even under favorable routing assumptions. The
constant-memory calculator in `capacity_scaling.py` reports this probability and the
minimum integer `b` reaching a requested recall target.

```bash
python3 capacity_scaling.py --self-test
python3 capacity_scaling.py --lengths 1024,4096,16384,65536,1048576
```

For the tiny-model baseline (`b=16`, `T=4`, `m=4`, `a=1`) and the oldest possible
target, the calculator gives:

| Context | Expected later colliders/table | Idealized recall | Minimum bits for 95% |
|---:|---:|---:|---:|
| 1,024 | 0.0156 | 1.00000000 | 9 |
| 4,096 | 0.0625 | 1.00000000 | 11 |
| 16,384 | 0.2500 | 1.00000000 | 13 |
| 65,536 | 1.0000 | 0.99999987 | 15 |
| 1,048,576 | 16.0000 | 0.00037249 | 19 |

These values explain why a fixed 16-bit tail can look excellent at the measured 16K
scale yet cannot be extrapolated indefinitely. Address width only needs to grow slowly
in this ideal model, but learned agreement and bucket balance must be measured rather
than assumed.

A theoretical family could choose `b(n)=Theta(log n)`, giving polynomially many
addresses while retaining near-linear hash and index work. This remains subquadratic
when `T`, `p`, and `m` are constant or sufficiently slow-growing. It is not implemented
by the current code: hash codes are `int32`, and the selector deliberately restricts
`b` to at most 30. Supporting asymptotically growing addresses would require a
multiword, hierarchical, or otherwise wider index representation.

## Precisely supported claim

The repository supports this statement:

> The portable parallel-prefill sparse layer uses a causal content-dependent selector
> with no all-pairs inference scorer. For fixed model and routing parameters, selector
> construction plus exact attention over bounded candidates is `O(n log n)` in sequence
> length. The full converted LFM is not yet wholly subquadratic because four attention
> layers remain dense, and persistent incremental decoding is not implemented.

It does **not** yet support a theorem that fixed-budget retrieval recall remains high as
`n` tends to infinity, an end-to-end speedup claim, or a whole-model subquadratic claim.
