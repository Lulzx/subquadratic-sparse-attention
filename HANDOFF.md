# Handoff: attention-mass hierarchical routing

Date: 2026-08-17

## Active goal status

The attention-mass-aligned adaptive hierarchy is implemented and validated on the
same held-out LFM2.5 layer-14 WikiText/PG-19 segments across seeds 0, 1, and 2.

- At 256, all seeds pass the routing target: 80.16–85.33% of the dense `K=32`
  distant-mass oracle at 2,808 bytes/query.
- At 256 and 512, output recovery keeps all seed/corpus perplexity ratios at or below
  1.0318.
- The MLX reference misses the 350-us target: 421–446 us at 256.
- Routing does not generalize to length: 73.15–76.72% at 512 and 61.37–64.82% at
  1,024. These are retained negative results, not passes.
- A 1,024 output-recovery pilot reached 1,905 MB and was interrupted above the
  1,792 MB limit. Do not rerun it without streaming/recomputation changes.
- `python3 mlx_tests.py`, Python compilation, and `git diff --check` pass.

Seed-0 512-token compressed-reranker follow-up (2026-08-16):

- A jointly trained 32-bit key projection plus RoPE-aware multi-head decoder improves
  the prior seed-0 result to 78.15%/78.06% of the `K=32` oracle on WikiText/PG-19,
  but remains below the 80% gate.
- A zero-initialized nonlinear residual (77.59%/77.95%), pairwise fine-tuning,
  query-conditioned head calibration (78.08%/77.76%), bounded-pool head
  normalization (78.04%/77.85%), and training against that normalized objective
  (78.01%/77.74%) do not close the gap.
- A four-byte supervised categorical VQ initialized to exactly reproduce the binary
  decoder reaches 77.73%/78.07% with 32 training segments per corpus. Scaling to 128
  segments improves held-out cross-entropy by 0.0213 but only reaches
  77.96%/78.26%; it peaks at 1,778.9 MB, leaving only 13 MB under the safety cap.
- All variants preserve 2,808 logical bytes/query. Their generic MLX runtime paths
  remain well above 350 us; the best VQ path is about 838–844 us at 512.
- The retained-pool exact-Q/K diagnostic remains 85.61–87.34%, so the unresolved
  loss is compressed candidate scoring, not absence of relevant candidates.
- A bounded exact-Q/K refinement sweep brackets the missing boundary. Exact reranking
  of the compressed top 34 remains below gate at 79.44%/79.51%; top 36 passes at
  80.40%/80.66%; top 40 reaches 81.84%/82.21%; top 48 reaches 83.60%/84.38%; and
  top 64 reaches 85.17%/86.31%. These are seed-0 WikiText/PG-19 results at 512.
- The top-36 diagnostic reads 36 KiB of exact FP16 K state plus the 2,808-byte index,
  or 39,672 bytes/query total. It therefore does **not** satisfy the fixed traffic
  envelope. Its reported 89–93 us host time covers approximate plus exact scoring
  only and excludes index lookup and KV gather.
- A 128-segment top-36 boundary-distillation run improves its held-out boundary loss
  from 3.3960 to 3.3799 but makes compressed routing worse: 78.28%/77.30% at 512.
  It peaks at 1,778.88 MB. Do not repeat boundary-loss tuning on this formulation.

The fixed-byte allocation/scorer milestone was executed on seed 0 at 512 tokens on
2026-08-16. All rows use the same WikiText/PG-19 held-out segments and remain at or
below the 2,848-byte synthetic-routing envelope:

| Variant | Bytes/query | Wiki oracle-relative | PG-19 oracle-relative | Result |
|---|---:|---:|---:|---|
| 40-bit binary, 3 probes x 12 | 2,712 | 77.96% | 78.65% | fail |
| 48-bit binary, 3 x 11 | 2,760 | 77.97% | 78.62% | fail |
| 40-bit binary, 6 x 6 | 2,832 | 78.81% | 79.62% | best binary, fail |
| 40-bit VQ, 6 x 6 | 2,832 | 79.13% | 79.07% | fail |
| 56-bit VQ, 5 x 6 | 2,840 | 77.74% | 77.86% | fail |
| teacher-top-32 loss, 6 x 6 | 2,832 | 78.48% | 79.50% | fail |
| query-only, lr 1e-4, 6 x 6 | 2,832 | 78.90% | 79.70% | fail |
| joint query/key, 6 x 6 | 2,832 | 78.53% | 79.64% | fail |

Allocation ceilings remain sufficient: exact reranking within 288, 264, and 240
retained candidates gives 85.52%/87.21%, 84.41%/86.19%, and 83.23%/85.01% of the
WikiText/PG-19 `K=32` oracle respectively. The failure is therefore not a byte-budget
impossibility. Reallocating the 288 slots from 3x12 through 6x6 improves routing, but
7x5 regresses. Wider codes, categorical codes, teacher-top-32 listwise loss, and
asymmetric/joint query adaptation do not realize the retained-pool ceiling. Raw
perplexity ratios also remain above gate (typically 1.133-1.169x), and canonical MLX
routing remains about 0.75-1.42 ms/query, above 350 us.

The 512-token address-retraining milestone was executed on seed 0 with 128 training
segments per corpus and 4,000 steps. Address-only checkpoints now preserve the full
initialized scorer checkpoint and replace only the query/key address projections.
The initial unregularized run drives address-candidate mass to 99.95%/99.45%, but
collapses joint codes: eviction rises to 67.8%/66.3% and deployed recall reaches only
76.60%/77.41% of the WikiText/PG-19 oracle. Generic balance/decorrelation weights of
100 reduce eviction to 16.8%/12.9% but over-regularize addressability and do worse.

A new differentiable per-table joint-address entropy penalty directly measures the
eight-bit code distribution. It is covered by tests and preserves the checkpoint
contract. Two bounded full runs reject this formulation:

| Entropy weight | Corpus | Address candidate mass | Retained candidate mass | Eviction | Oracle-relative deployed | Raw PPL ratio |
|---:|---|---:|---:|---:|---:|---:|
| 10 | WikiText | 98.65% | 74.63% | 53.67% | 77.47% | 1.133x |
| 10 | PG-19 | 97.47% | 73.98% | 45.57% | 77.95% | 1.169x |
| 30 | WikiText | 92.61% | 69.14% | 38.78% | 74.20% | 1.206x |
| 30 | PG-19 | 92.83% | 72.01% | 25.65% | 77.53% | 1.133x |

All rows stay at 2,832 logical bytes/query, but the canonical MLX path is about
786-795 us/query. Entropy 10 preserves addressability but not load; entropy 30 trades
away useful address alignment before retention is adequate. Neither improves the
original 78.81%/79.62% router or clears the 80% and 1.05x gates. Do not run more
scalar entropy/balance sweeps or seeds 1/2 on this branch.

A capacity-aware follow-up is implemented and tested. Its straight-through loss uses
the exact deployed primary+adjacent-secondary leaf match in the forward pass and
penalizes only keys in leaves above storage capacity 32. Splitting primary and
secondary match tensors reduces the full-data peak from an unsafe 1,807.8 MB to a
safe 1,774.6 MB under a 1,760 MB allocator/16 MB cache setting. The interrupted
over-cap attempt produced no checkpoint and is not evidence.

| Overflow weight | Corpus | Address mass | Retained mass | Eviction | Oracle-relative deployed | Raw PPL |
|---:|---|---:|---:|---:|---:|---:|
| 0.1 | WikiText | 95.34% | 75.47% | 31.43% | 78.25% | 1.245x |
| 0.1 | PG-19 | 93.31% | 75.06% | 22.68% | 78.57% | 1.133x |
| 0.3 | WikiText | 90.54% | 77.56% | 15.42% | **80.15%** | 1.133x |
| 0.3 | PG-19 | 88.72% | 74.20% | 14.48% | 77.88% | 1.169x |

Both rows use 2,832 bytes/query and about 807-810 us/query. Weight 0.3 proves that
the deployed overflow is differentiably controllable and clears the WikiText routing
gate, but it trades away PG-19 address quality and therefore fails the declared
both-corpus gate. Do not run more scalar overflow weights or seeds 1/2.

Learned attention-aware retention is also implemented and rejected on seed 0. A save
contract bug was caught by smoke testing and fixed: retention-only checkpoints now
preserve all 14 initialized tensors and replace only `retention_projection`. The
generic linear salience objective improves held-out loss from 7.69 to 6.74, but
deployment reaches only 79.42%/79.20% oracle-relative recall on WikiText/PG-19.

A second loss ranks keys only across the actual capacity-32 keep/evict boundary of
each frozen hard leaf. Its held-out leaf loss improves from 1.304 to 0.501, but
deployment reaches only 79.24%/79.36%. Both learned-retention checkpoints keep the
original addresses/scorer, 2,832 bytes/query, and about 781-792 us/query. Neither
beats the original router on both corpora or clears the 80% gate. Do not tune the
linear scalar retention projection further or replicate it across seeds.

The noncausal oracle leaf-retention ceiling is implemented, tested, and explicitly
labeled in evaluator output. It ranks postings by true future distant-attention
salience; its reported 784-790 us lookup timing is a reservoir-equivalent proxy and
excludes oracle score computation. It reaches 79.25% oracle-relative recall on
WikiText and 80.47% on PG-19. Because it fails the both-corpus gate, stop retention
work: even this fixed-score future-information ceiling does not justify a richer
retention model on the current leaves.

A projection-interpolation diagnostic between the original router and overflow-0.3
router also fails. At interpolation weights 0.25, 0.50, and 0.75, WikiText/PG-19
oracle-relative recall is 79.45%/79.17%, 78.17%/77.26%, and 76.31%/77.15%.
Discrete hash geometry is not preserved by checkpoint blending; do not search this
line or choose table mixtures on the canonical held-out segments.

The goal remains active. The next scientific milestone returns upstream to
**domain-balanced discrete address/reranker training**. Design the objective and
model selection on training/evaluation-domain segments, not the canonical held-out
gate, and optimize the deployed address plus compressed scorer jointly enough to
avoid the WikiText/PG-19 tradeoff. Preserve 40-bit 6x6 traffic and first require a
fresh seed-0 checkpoint to clear 80% on both canonical corpora before output recovery
or replication. Metal, append-only decode, latency tuning, and 1,024-token work remain
deferred. The current worktree is uncommitted and must be staged selectively.

That bounded domain-balanced branch is now complete and negative:

- Group-DRO beta 0.25 plus overflow 0.3 reaches 80.66%/77.99% WikiText/PG-19
  oracle-relative recall; refreshing its scorer reaches 80.21%/78.16%.
- `mlx_address_table_mix.py` evaluates 24 declared whole-table masks on reserved
  training-split segments. Its selected cyclic-half mask scores 91.47% on the
  reserved worst corpus but transfers at only 78.78%/78.42% after scorer refresh.
- Evaluation-domain training uses noncanonical segments 1-128, reserves 129-136,
  and leaves canonical segment 0 untouched. It reduces reserved loss from 13.04 to
  10.81, but canonical recall is 79.07%/80.91%. Traffic is 2,832 bytes/query,
  routing is 802-814 us/query, eviction is 21.75%/21.02%, and raw PPL is 1.133x.

The latest checkpoint and report are
`runs/lfm2.5-address512-domain-groupdro0.25-overflow0.3-binary40-p6c6-wide-seed0.safetensors`
and the matching `-eval512.json`. Do not tune on canonical segment 0 or continue
Group-DRO weights, projection interpolation, whole-table masks, or simple domain
adaptation. The next implementation must change the discrete objective/representation
so candidate membership and compressed final ranking are optimized jointly under the
fixed 40-bit 6x6, 2,832-byte envelope. Require a fresh seed-0 pass of at least 80% on
both corpora before output recovery, seeds 1/2, latency work, or 1,024-token expansion.

The first candidate-set-level implementation is complete and rejected. New trainer
flags `--candidate-set-weight`, `--candidate-set-temperature`, and
`--candidate-set-query-stride` add a multiprobe, capacity-aware top-32 mass surrogate.
The accepted run uses weight 10, temperature 16, stride 16, Group-DRO beta 0.25, and
peaks at 1,771.0 MB. An earlier stride-8 attempt was interrupted at 1,780.7 MB, wrote
no checkpoint, and is not evidence.

The accepted checkpoint is
`runs/lfm2.5-address512-candidateset10-groupdro0.25-binary40-p6c6-wide-seed0.safetensors`.
Reserved candidate-set loss improves from 0.2744 to 0.1658, but canonical deployed
recall is only 74.87%/75.29% on WikiText/PG-19. Address mass is 83.59%/83.76%,
retained-top32 ceiling is 82.62%/83.59% oracle-relative, eviction is 14.77%/11.25%,
traffic is 2,832 bytes/query, and routing is 804.3/802.8 us/query. Do not tune this
surrogate's scalar weight on canonical segment 0, refresh its scorer, or replicate it.
The next implementation must expose the exact causal retained-set and final top-32
boundary rather than expected `min(1, capacity/load)` survival.

That exact implementation is now complete and negative on seed 0. It adds
`--exact-boundary-weight`, `--exact-boundary-negative-weight`, and
`--exact-boundary-query-stride`, stores mined causal masks bit-packed on the host, and
materializes one example at a time. Two 1,000-step phases remine between checkpoints.
Without overflow control, exact positive mining collapses to 62.40%/56.05% eviction
and 76.15%/77.48% WikiText/PG-19 oracle-relative recall.

The accepted coupled diagnostic uses the established overflow weight 0.3, boundary
stride 8, cache limit 8 MB, and peaks at 1,776.1 MB. Its phase-2 checkpoint is
`runs/lfm2.5-address512-exactboundary-overflow0.3-phase2-binary40-p6c6-wide-seed0.safetensors`.
Canonical recall is 77.48%/78.14%; retained-top32 ceilings are 85.41%/86.94%, eviction
is 14.97%/13.45%, raw PPL is 1.169x/1.133x, routing is 797.1/793.3 us/query, and traffic
is 2,832 bytes/query.

One justified scorer refresh produces
`runs/lfm2.5-exactboundary-overflow0.3-joint-binary40-p6c6-512-wide-seed0.safetensors`.
It peaks at 1,778.9 MB and reaches only 77.47%/77.98%. Do not replicate, run output
recovery, or tune scalar boundary/overflow weights. Several combined memory probes
were interrupted before checkpoint save; they are not evidence. The next milestone
requires joint optimization against exact retained membership and final rank swaps,
not another sequential address/scorer pass.

That unified milestone is complete and negative. `mlx_joint_binary_attention_train.py
--joint-address` now interleaves compressed-scorer and exact-boundary/overflow address
updates and saves the two trainable components together. Host-backed examples reduce
the accepted full-run peak to 1,196.6 MB; an earlier 1,810.5 MB attempt was interrupted
without a checkpoint and is not evidence.

The no-pairwise phase-2 checkpoint reaches 78.45%/79.49% oracle-relative recall on
the canonical WikiText/PG-19 segments. A separately predeclared pairwise top-32 run
improves reserved pairwise loss from 0.3561 to 0.3503. Its selected checkpoint is
`runs/lfm2.5-joint-exact-pairwise-phase2-binary40-p6c6-512-wide-seed0.safetensors`,
and its one-shot canonical report is the matching `-eval512.json`.

The pairwise checkpoint reaches 78.51%/79.61% oracle-relative recall. Retained
ceilings are 87.10%/88.47%, eviction is 14.82%/13.33%, raw PPL is 1.169x/1.098x,
routing is 794.7/793.0 us/query, and traffic is 2,832 bytes/query. It fails both
corpus gates, so output recovery, seeds 1/2, a third phase, and 1,024-token expansion
are not warranted. The next scientific milestone must change the representation or
discrete optimization mechanism, not retune the completed surrogate/exact/pairwise
objectives against canonical segment 0.

One such representation check is now complete. `--joint-address-thresholds` adds
checkpointed asymmetric query/key sign biases and freezes the 64 projection
directions. Exact causal masks are remined with those biases between two 1,000-step
phases. The selected phase-2 run improves its reserved address loss from 13.131 to
12.944 and peaks at 1,165.7 MB (phase 1 peaks at 1,194.4 MB).

Canonical performance regresses to 78.31%/78.37% WikiText/PG-19 oracle-relative
recall. Retained ceilings are 86.58%/87.15%, address-candidate mass is 87.47%/88.19%,
eviction is 14.45%/12.82%, raw PPL is 1.169x/1.133x, routing is 818.1/816.5 us/query,
and traffic remains 2,832 bytes/query. Do not sweep thresholds. The next bounded
implementation should use direct categorical byte assignments (or an equivalently
different balanced discrete partitioner), then select on reserved segments before
one canonical seed-0 check.

That direct categorical check is now implemented, tested, and rejected. Each of the
eight address tables predicts one of 256 categories directly; initialization exactly
reproduces the source binary byte, and the query probes the six highest-scoring
secondary categories. The index, 288-slot candidate budget, 40-bit reranker, and
2,832-byte accounting remain unchanged. Two 1,000-step phases peak at 1,075.1 MB.

Phase 2 improves freshly remined reserved address loss from 10.334 to 10.074 and its
overflow surrogate from 0.081 to 0.030. Canonical transfer nevertheless collapses:

- WikiText: 47.71% oracle-relative recall, 34.94% address-candidate mass, 49.23%
  retained ceiling, 0.366% eviction, 1.367x raw PPL, 794.1 us/query.
- PG-19: 48.93% oracle-relative recall, 36.26% address-candidate mass, 51.15%
  retained ceiling, 0.708% eviction, 1.206x raw PPL, 781.8 us/query.

The corrected timing path uses the categorical index itself: p99 leaf occupancy is
13/7 and maximum occupancy 39/48. An earlier report accidentally timed/reported the
binary index while using categorical codes for correctness; it was overwritten after
the timing path was fixed and is not evidence. Do not sweep categorical temperature,
rate, phases, or balance weights. The next architecture should freeze the proven
binary primary address and learn only a residual categorical secondary partition
inside each primary region, so load can be split without relearning global semantic
colocation.

That follow-up is now complete and negative. A new primary-only evaluator diagnostic
shows that the frozen binary primary pools preserve 99.09%/98.53% of distant mass and
99.42%/99.41% of the K=32 oracle on canonical WikiText/PG-19. This exonerates primary
discovery and localizes the remaining problem to secondary partitioning, retention,
and compressed reranking.

`ResidualCategoricalSecondaryRouter` freezes the primary projections, learns a
256-way secondary category per table, restricts positives/hard negatives to matching
primary regions, and penalizes exact local overflow. Its initialization reproduces
the old adjacent binary secondary code. Phase 1 lowers reserved loss 3.670 to 2.459;
phase 2 remaps and lowers its fresh reserved loss 2.517 to 2.444. Both peak at
1,075.1 MB. The selected artifacts are:

```text
runs/lfm2.5-joint-residual-secondary-phase2-binary40-p6c6-512-wide-seed0.safetensors
runs/lfm2.5-joint-residual-secondary-phase2-binary40-p6c6-512-wide-seed0-eval512.json
```

Canonical deployed/oracle recall is only 72.00%/73.04%. The decomposition is:

- primary K32/oracle: 99.42%/99.41%;
- secondary-address K32/oracle: 81.17%/82.36%;
- retained K32/oracle: 78.16%/79.64%;
- deployed/oracle: 72.00%/73.04%.

Eviction is 1.880%/1.245%, raw PPL is 1.169x on both single-segment rows, traffic is
2,832 bytes/query, routing is 765.8/802.1 us/query, and evaluation peaks at
1,153.5 MB. The residual partition is much better than unconstrained categorical
addresses but worse than the original binary router. Do not run phase 3, a scalar
sweep, output recovery, or seeds 1/2. The next implementation should condition the
secondary partition on the frozen primary region instead of sharing one global local
classifier across all primary regions, while jointly aligning final compressed
ranking. Preserve the canonical holdout and fixed 40-bit 6x6 envelope.

That primary-conditioned branch is now implemented, tested, and closed. The new
`--joint-address-primary-conditioned-secondary` mode freezes primary and shared
secondary projections, then learns `(table, primary byte, secondary category)`
query/key biases. Its zero state exactly reproduces the adjacent binary secondary
code, and both correctness and timed lookup use the conditioned index.

Phase 1 improves reserved loss 3.670 to 3.250. Phase 2 remaps the actual leaves and
improves fresh loss 3.322 to 3.147 and overflow 1.061 to 0.684. Both peak at
1,079.1 MB. The selected router is:

```text
runs/lfm2.5-joint-primary-conditioned-secondary-phase2-binary40-p6c6-512-wide-seed0.safetensors
runs/lfm2.5-joint-primary-conditioned-secondary-phase2-binary40-p6c6-512-wide-seed0-eval512.json
```

Its canonical WikiText/PG-19 ladder is primary K32 99.42%/99.41%, conditioned
secondary K32 89.12%/93.14%, retained K32 84.37%/87.88%, and deployed
76.54%/78.93% of oracle. Eviction is 6.445%/6.812%, raw PPL is 1.206x/1.133x,
traffic is 2,832 bytes/query, and the generic conditioned lookup is about 952 us/query.

One fixed-address 16,000-step scorer refresh marginally improves aggregate reserved
loss from 4.70330 to 4.70285. The first 64 MB-cache attempt reached 1,795.9 MB and
was interrupted at step 6,000 without a checkpoint. The accepted 1,760 MB allocator,
8 MB-cache rerun peaks at 1,764.7 MB. Its artifacts are:

```text
runs/lfm2.5-primary-conditioned-secondary-refreshed-binary40-p6c6-512-wide-seed0.safetensors
runs/lfm2.5-primary-conditioned-secondary-refreshed-binary40-p6c6-512-wide-seed0-eval512.json
```

The refresh reaches only 76.72%/79.09%, with the same 1.206x/1.133x PPL ratios. Do
not add a third address phase, tune bias rates, refresh the scorer again, run output
recovery, or replicate seeds. The remaining bounded problem is to allocate the fixed
288 posting reads according to local leaf pressure/relevance and train the final
compressed ranking against that exact deployed allocation. Metal, 1,024 tokens, and
output recovery remain gated.

Current verification before the primary-conditioned full run passed `python3
mlx_tests.py`. Re-run it, `python3 tests.py`, compilation of all edited Python entry
points, and `git diff --check` after these documentation edits. No new checkpoint in
this section is promoted as a passing router.

## Legacy handoff: fixed-budget bucket-tail eviction

Date: 2026-08-15

## Goal

Reduce learned-router bucket-tail eviction at a fixed 32-candidate budget, validate
the intervention across three LFM2.5-350M seeds and retrieval distances, document the
evidence, and publish the completed change to `main`.

The goal is still active. The work below is uncommitted and has not been pushed.
The branch started clean at `main` commit `4396d07` (`Align learned routing with hard
retrieval`).

## Diagnosis

The previous three-seed audit found that 42.92% of queries had an exact query/key
address agreement but lost the teacher target before selection. The selector retained
only the four newest members of each bucket. This makes eviction deterministically
hostile to older relevant keys.

The intervention keeps the slot count unchanged. For each table with four members:

- slots 0-2 retain the three newest eligible matching keys;
- slot 3 retains the matching key at 50% of the causal bucket history;
- the resulting tensor remains `8 tables * 4 members = 32` slots per query.

The policy is named `hybrid`; `recent` remains the default so existing baselines and
checkpoints do not silently change behavior. `history_fraction=0.5` is the winning
setting.

## Three-seed evidence

Checkpoints:

- `runs/lfm2.5-layer14-hard-hamming-top32-pos10-seed0.safetensors`
- `runs/lfm2.5-layer14-hard-hamming-top32-pos10-seed1.safetensors`
- `runs/lfm2.5-layer14-hard-hamming-top32-pos10-seed2.safetensors`

Matched audits:

- recent baseline: `runs/lfm2.5-layer14-research-aligned-audit.json`
- winning hybrid: `runs/lfm2.5-layer14-history-0.5-audit.json`

Both audits use four held-out 256-token segments per seed, 876 distant queries per
seed, eight tables, four members, one probe, and exactly 32 selected slots.

| Metric | Recent | Hybrid 0.5 | Delta |
|---|---:|---:|---:|
| Retained teacher contribution | 0.309743 | 0.319065 | +0.009322 |
| Teacher top-1 recall | 0.374049 | 0.386225 | +0.012177 |
| Agreement followed by eviction | 0.429224 | 0.417047 | -0.012177 |
| Mean unique candidates | 15.957 | 16.483 | +0.526 |

Every seed improves:

| Seed | Contribution delta | Top-1 delta | Eviction delta |
|---:|---:|---:|---:|
| 0 | +0.009655 | +0.019406 | -0.019406 |
| 1 | +0.009921 | +0.007991 | -0.007991 |
| 2 | +0.008389 | +0.009132 | -0.009132 |

Distance-conditioned result:

| Distance | Recent recall | Hybrid recall | Recent contribution | Hybrid contribution |
|---|---:|---:|---:|---:|
| 1-64 | 0.597368 | 0.587719 | 0.400138 | 0.405241 |
| 65-128 | 0.276249 | 0.305810 | 0.266027 | 0.282480 |
| 129-256 | 0.061144 | 0.088757 | 0.191077 | 0.196084 |

The tradeoff is explicit: near-distance top-1 falls 0.97 percentage points, while
near-distance retained contribution still rises. Aggregate recall, aggregate
contribution, medium distance, long distance, and all three seeds improve.

## Rejected variants

Do not repeat these unless changing the formulation:

- Two recent plus midpoint and oldest (`[0, 1, midpoint, oldest]`): long-distance
  top-1 rose from 6.11% to 24.46%, but aggregate retained contribution fell from
  0.309743 to 0.297509 and near recall fell sharply.
- Three recent plus oldest (`[0, 1, 2, oldest]`): aggregate contribution fell to
  0.289720 and recall to 0.356545.
- Single history slot at fractions 0.25 and 0.75 both underperformed fraction 0.5
  on the aggregate objective. Their reports are under
  `runs/lfm2.5-layer14-history-{0.25,0.75}-audit.json`.

## Selector benchmark

The benchmark command used eight 8-bit tables, four members, one probe,
`--anchors-only`, and nine repeats. Both policies report exactly 32 slots. Hybrid peak
memory at 16K was 57.02 MB versus 53.27 MB for recent. Exact-needle recall under high
8-bit collision pressure was 0.37 versus 0.23 at 4K, 0.17 versus 0.00 at 8K, and 0.07
versus 0.00 at 16K. Latencies were noisy and should not be used as a speed claim.

Outputs:

- `runs/selector-recent-k32.jsonl`
- `runs/selector-hybrid-k32.jsonl`

MLX requires host GPU access in the managed sandbox. The `python3 mlx_selector.py ...`
benchmark command was approved for escalation after an in-sandbox run failed with
`No Metal device available`.

## Current edits

- `ssa/mlx_selector.py`: adds `member_policy` and `history_fraction`; implements the
  causal midpoint-history slot using sorted bucket starts.
- `mlx_selector.py`: exposes policy, fraction, and `--anchors-only` for K=32
  benchmarking.
- `mlx_donor_router.py`: forwards the policy into hard metrics.
- `mlx_router_audit.py`: exposes and records the policy/fraction.
- `mlx_lfm_replacement.py`: wires the policy into actual sparse candidate selection.
- `mlx_lfm_quality_eval.py`, `mlx_lfm_behavior_eval.py`,
  `mlx_lfm_multilayer_eval.py`, `mlx_lfm_joint_recovery.py`,
  `mlx_lfm_retrieval_router.py`, and `mlx_lfm_retrieval_recovery.py`: expose the
  opt-in CLI flags.
- `mlx_tests.py`: adds simple history/causality coverage and a NumPy reference test.

## Remaining work

1. Run `python3 mlx_tests.py`. The suite passed before the final NumPy reference test
   was added; that new test has not yet been executed.
2. Run `python3 tests.py`, `python3 -m py_compile` for all edited Python files, and
   `git diff --check`.
3. Review the production CLI wiring. In particular, confirm every script using
   `install_replacements` receives `--member-policy hybrid --history-fraction 0.5`
   when requested and that block-index mode remains on its existing policy.
4. Add a concise experiment section to `docs/experiments.md` and reproduction commands
   to `docs/reproduction.md`. Do not edit `docs/limitations.md`: it is part of the
   held-out audit corpus and would change the reported evaluation examples.
5. Update the README only if the result can be stated without displacing the stronger
   inference-aligned objective result already there.
6. Re-run the final hybrid audit if any evaluation-corpus file changes. Documentation
   outside `docs/limitations.md` and `docs/model-card-audit.md` does not change this
   audit corpus.
7. Stage only the owned files, commit, push `main`, verify remote HEAD, then mark the
   active goal complete.

Suggested final audit command:

```bash
python3 mlx_router_audit.py \
  runs/lfm2.5-layer14-hard-hamming-top32-pos10-seed0.safetensors \
  runs/lfm2.5-layer14-hard-hamming-top32-pos10-seed1.safetensors \
  runs/lfm2.5-layer14-hard-hamming-top32-pos10-seed2.safetensors \
  --eval-segments 4 \
  --member-policy hybrid \
  --history-fraction 0.5 \
  --output runs/lfm2.5-layer14-history-0.5-audit.json \
  --markdown-output runs/lfm2.5-layer14-history-0.5-audit.md
```

Suggested fixed-budget selector benchmark:

```bash
python3 mlx_selector.py \
  --lengths 1024,2048,4096,8192,16384 \
  --tables 8 --bits 8 --members 4 --probes 1 \
  --anchors-only --repeats 9 \
  --member-policy hybrid --history-fraction 0.5
```
