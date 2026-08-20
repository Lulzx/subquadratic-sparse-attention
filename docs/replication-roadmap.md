# SubQ replication ledger and roadmap

## Scope and accounting

This project distinguishes three kinds of evidence:

1. **Full replication** reruns the same disclosed method, task, scale, and metric.
2. **Partial behavioral replication** demonstrates the same property with a different,
   smaller, or synthetic system.
3. **Arithmetic validation** reconstructs a reported number without rerunning the model.

The SubQ-1.1-Small report does not release its weights, donor identity, SSA algorithm,
exact layer dimensions, training corpus, benchmark harness, or production kernels.
Consequently, literal reproduction is not currently possible. The practical target is
an independent, inspectable reproduction of every publicly testable behavioral claim
on Apple silicon, followed by matched accelerator measurements when suitable hardware
is available.

At the 2026-08-15 checkpoint, the claim-family count is:

- **0 of 14 fully replicated**;
- **6 of 14 partially reproduced**;
- **3 of 14 arithmetically validated only**;
- **5 of 14 not reproduced**.

Separately, `python3 replicate.py` passes 67 of 76 table-level consistency checks.
That is an audit result, not 67 replicated model claims.

## Claim ledger

| # | Public claim family | Status | Evidence or missing work |
|---:|---|---|---|
| 1 | Arbitrary-position content-dependent sparse retrieval | Partial | The expanded three-seed 1K gate uses three unseen values, two templates, and three positions per task. Recent K=32 preserves 132/189 dense-pass trials, hybrid K=32 preserves 127/189, and block K=64 preserves 111/189. All preserve 54/54 exact trials, but task-specific failures remain. |
| 2 | End-to-end linear selection and attention | Partial | Attention reads fixed `K=32`; portable bucket construction still sorts in `O(n log n)`. |
| 3 | Linear memory scaling | Partial | Selected attention has bounded reads and measured memory through 16K, not millions of tokens. |
| 4 | Full-context training and ordinary autoregressive operation | Partial | Parallel causal prefill works; a persistent incremental decode cache is absent. |
| 5 | Conversion of a dense pretrained donor without losing language quality | Partial | LFM2.5 layers 12 and 14 are replaced and pass the 65K-token raw-text gate. The expanded retrieval gate preserves only 59–70% of dense-pass trials depending on the selector, so conversion is paused before layer 10. |
| 6 | 64.5x FLOP reduction at 1M | Arithmetic only | `252 / 3.9 = 64.6`; undisclosed dimensions prevent an absolute rerun. |
| 7 | 56x attention-layer wall-clock speedup at 1M | Arithmetic only | `54,164 / 966 = 56.07`; no matched H100 run exists. |
| 8 | RULER average 99.12 at 128K | Not reproduced | The public benchmark has not yet been integrated. |
| 9 | NIAH retrieval through 12M and extrapolation beyond training length | Partial | MQAR reaches 95.22% at 16x training length. On 54 local NIAH-style dense-pass trials at 1K, hybrid K=32 preserves 44, recent K=32 preserves 41, and block K=64 preserves 8. This is not the public benchmark or multi-million-token evidence. |
| 10 | 0.13% of token pairs at 12M, described as nearly 1,000x fewer | Arithmetic only | The table-derived fraction implies about 775x fewer pairs. |
| 11 | GPQA Diamond 85.4 | Not reproduced | Requires a capable converted language model and matched evaluation. |
| 12 | LiveCodeBench 89.7 pass@4 | Not reproduced | Requires a capable converted language model and matched evaluation. |
| 13 | AutomationBench Finance 13% | Not reproduced | Requires the benchmark harness and a capable converted model. |
| 14 | Long-context training volume and curriculum dominate quality | Partial | Controlled MQAR runs also find curriculum dominant, but do not establish the language-model claim. |

The report contains one material NIAH inconsistency: its body text and plotted bars
show 98% at 6M and 12M, while the figure caption says 100%. This project treats 98%
as the reproducible target. The detailed numerical reconstruction lives in
[the model-card audit](model-card-audit.md).

## What has actually been established locally

The current tiny MLX model establishes a bounded but useful result:

- exact selector recall is 100/100 seeded needles at 16K;
- selected attention reads a fixed token budget and matches the NumPy reference;
- causal future-mutation tests pass;
- held-out MQAR reaches 95.22% at 2K after training only at length 128;
- an eight-table curriculum reaches 97.03% mean accuracy at 4K over three seeds;
- a seed-0 model trained through 8K reaches 97.38% at 16K and 95.32% at 32K.

MQAR repeats exact key tokens and caps the number of associations. These results test
sparse associative retrieval and positional extrapolation, not general language
modeling, semantic matching, or increasing information density at the longest lengths.

## Research synthesis

The exact newest-first arXiv search for
[subquadratic attention](https://arxiv.org/search/?query=subquadratic+attention&searchtype=all&abstracts=show&order=-announced_date_first&size=25)
contains several useful directions, mixed with papers that use the term in unrelated
settings. The most actionable findings are below.

### Train routing from a dense teacher

[HashAttention](https://arxiv.org/abs/2412.14468) frames sparse attention as learned
query/key retrieval in a compact Hamming space. [SpotAttention](https://arxiv.org/abs/2606.22874)
uses a lightweight selector, KL distillation, calibrated probabilities, and adaptive
budgets on frozen pretrained transformers. [StreamKL](https://arxiv.org/abs/2606.20005)
reduces the memory cost of long-context attention distillation.

Implication for this repository:

- use separate query and key hash projections;
- supervise their continuous similarities from dense attention distributions;
- measure retained teacher-attention mass, not merely bucket balance;
- ultimately allow the selected budget to vary with query difficulty.

The first opt-in implementation of separate query/key hashing and a bounded dense
router-distillation loss is now exposed by `--semantic-router`. Half the configured
tables remain shared-hash fallback tables and half use separate semantic Q/K hashes;
the total table count and selected-token budget do not change. This hybrid was adopted
after an all-semantic first attempt destroyed exact-key recall as soon as the Q/K codes
diverged. It is an experimental training mechanism, not yet evidence of improved
semantic retrieval.

### Combine local, selected, and compressed-global paths

[Native Sparse Attention](https://arxiv.org/abs/2502.11089) combines local attention,
fine-grained selected tokens, and coarse compressed context. [COBS](https://arxiv.org/abs/2607.09052)
shows that richer block summaries can recover attention mass missed by a single
representative token. [MoSA](https://arxiv.org/abs/2505.00315) provides another
content-routed mixture design from the exact search results.

The local prototype previously had only local and selected paths. The opt-in
`--global-slots N` path adds `N` fixed interleaved causal history summaries. It has
`O(nN)` work and memory for fixed `N`, never reads future tokens, and begins behind a
low learned gate. It is deliberately simpler than COBS: second-order block statistics
and learned block selection remain future experiments.

### Convert a pretrained donor gradually

[SeerAttention](https://arxiv.org/abs/2410.13276) trains a block-sparsity gate against
a pretrained model. [Taylor-Calibrate](https://arxiv.org/abs/2606.16429) uses teacher
statistics and layer-output alignment to initialize efficient replacements with much
less distillation data. [Lizard](https://arxiv.org/abs/2507.09025), found in the exact
search, studies conversion of pretrained transformers to efficient memory mechanisms.
[LongRoPE2](https://arxiv.org/abs/2502.20082) provides a relevant position-extension
and mixed-context training recipe.

The Mac-feasible conversion sequence should be:

1. select a small open-weight donor already supported by a local MLX stack;
2. preserve its embeddings, MLPs, normalization, and output head;
3. initialize sparse Q/K/V/O projections from each dense attention layer;
4. freeze the donor and train only selectors and output gates against teacher attention;
5. align sparse and dense layer outputs;
6. replace layers gradually rather than simultaneously;
7. recover capability on mixed short documents and staged long contexts;
8. record perplexity and short-task regressions at every replacement stage.

The historical first donor was
[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), which
established the three-seed routing baseline. The active causal donor is now the much
more recent [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M), with
[LFM2.5-350M-Base](https://huggingface.co/LiquidAI/LFM2.5-350M-Base) retained as a
base-model ablation. Its hybrid 16-layer decoder has six full-attention layers and ten
convolution layers, making it both Mac-feasible and relevant to hybrid efficient
architectures. [LFM2.5-Embedding-350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M)
is a separate bidirectional semantic teacher, not a causal-LM replacement.

The first donor-router experiment distills layer 15's dense Q/K distribution without
modifying donor weights. It excludes the first four attention-sink tokens and the local
32-token window, then trains eight 8-bit query/key hash tables with a fixed maximum of
32 bucket reads. Straight-through signs make the training forward pass use the same
binary values as Hamming lookup. Across three seeds and 876 held-out queries per seed:

| Metric | Random hash | Distilled hash |
|---|---:|---:|
| Retained distant teacher-attention mass | 27.32% | **33.36%** |
| Teacher top-1 recall | 37.14% | **55.78%** |
| Mean unique hard candidates | 16.95 | **11.70** |
| Continuous top-32 retained mass | 42.48% | **68.25%** |
| Continuous top-32 teacher top-1 recall | 44.37% | **85.58%** |

This is the first natural-language partial result in the repository, but it is a router
metric rather than an end-to-end language-model replication. The 34.89-point gap
between continuous and hard retained mass identifies binary indexing as the next
bottleneck. It also shows why continuous router scores must not be reported as sparse
lookup quality.

### Make the system claim genuinely linear

The portable sort is appropriate for architecture work but cannot establish an
end-to-end linear selector. The systems milestone requires:

- append-only per-table bucket tails for expected `O(n)` construction;
- persistent hash and KV state for token-by-token decoding;
- exact prefill/decode parity tests;
- a fused Metal block-gather and selected-attention kernel;
- latency and peak-memory slopes rather than isolated endpoints.

At contexts larger than local GPU memory, [HiSparse](https://arxiv.org/abs/2608.07009)
and [OasisKV](https://arxiv.org/abs/2608.08097) suggest keeping full state in host memory
while predicting and prefetching a bounded GPU working set. That is a later systems
stage, not a prerequisite for learning a good router.

### Do not target exact unrestricted softmax equivalence

The conditional-hardness results in [arXiv:2505.14840](https://arxiv.org/abs/2505.14840)
argue against expecting an exact general replacement for dense softmax attention in
the unrestricted case. The appropriate target is distributional: retain the attention
mass and downstream behavior needed by natural language while doing bounded work.

## Mac-local execution plan

### Stage A: router and global-path ablations

Run matched seeds and fixed `K=32` for:

| Variant | Semantic Q/K | Global slots | Purpose |
|---|---:|---:|---|
| Published local baseline | No | 0 | Existing control |
| Semantic router only | Yes | 0 | Isolate teacher-distilled hashing |
| Global summaries only | No | 4 or 8 | Isolate compressed history |
| Combined | Yes | 4 or 8 | Test complementarity |

Acceptance requires at least three seeds, identical token budgets and curricula, no
causality failures, and improvement in held-out answers rather than current-batch loss.
The semantic loss uses at most `--router-teacher-tokens` tokens, so its temporary dense
matrix remains bounded during long curriculum stages.

#### Initial seed-0 probe

A 300-step, length-128 probe used width 64, two layers, eight tables, two members,
one probe, `K=32`, batch 16, and 1,280 held-out length-128 answers. Extrapolation used
eight batch-1 evaluations containing 712 answers at 2K and 1,432 answers at 4K.

| Variant | 128 | 2K | 4K |
|---|---:|---:|---:|
| Baseline | 99.84% | 97.47% | **80.73%** |
| Four compressed global slots | **100.00%** | 96.21% | 79.68% |
| Hybrid semantic router | 99.61% | 94.80% | 80.03% |
| Hybrid semantic router + four global slots | 99.77% | 94.38% | 77.93% |

This is a single-seed exploratory result, not a comparative claim. Neither new path
improved extrapolation. The direct all-semantic design was substantially worse: its
128-token held-out accuracy was 0.16%. Reducing its semantic loss weight from 1.0 to
0.1, 0.05, or 0.01 did not recover learning. The fixed-budget hybrid restored learning,
but its semantic half still lacked a trustworthy teacher and did not beat the baseline.

The conclusion is architectural rather than hyperparameter-specific: a Q/K projection
from a sparse model being trained from scratch is not a useful dense teacher, and exact
MQAR supplies no pressure for lexical mismatch. Further semantic-router work should use
a pretrained dense donor or explicit retrieval labels. The global slots average large
amounts of filler on MQAR; learned block summaries or second-order statistics should
replace them before a larger ablation.

### Stage B: harder retrieval data

Add tasks in increasing order of semantic difficulty:

1. MQAR with distractor keys and non-adjacent values;
2. transformed keys where query and stored surface forms differ;
3. variable-length passkeys and paraphrased needles;
4. NoLiMa-style lexical-mismatch retrieval;
5. public RULER tasks at 4K, 8K, 16K, and 32K.

Track answer accuracy, selector recall, retained teacher-attention mass, candidate
budget, latency, peak memory, and failure by retrieval distance.

The first expanded gate now covers the first four task families at 1K with a fixed
288-case manifest spanning four target lengths. Only 72 cases are approved on the
current laptop: dense passes 63, and each sparse variant is evaluated on those same
cases across three seeds. Recent K=32 is best overall at 132/189 dense-pass trials;
hybrid K=32 is best on NIAH-style retrieval at 44/54; block K=64 is best on
multi-token retrieval at 22/54. No variant passes the gate uniformly. A dense 4K probe
peaks at 2.82 GB and a hybrid 4K probe at 2.48 GB, so longer-context rows remain
unexecuted under the 1.792 GB operating limit rather than being reported as failures.

### Stage C: pretrained language-model conversion

LFM2.5-350M is now the primary donor. Layers 12 and 14 are independently converted
while every convolution, MLP, normalization, embedding, and output component remains
frozen. A paired 65,536-token audit invalidated the first final-hidden recovery result,
finding 1.1100x dense perplexity on WikiText and 0.9945x on PG-19. Mixed-corpus top-64
teacher-distribution recovery then passes the gate across all three seeds: geometric-
mean ratios are 0.9675 on WikiText and 0.8712 on PG-19, with every paired 95% interval
below parity. It peaks at 1.61 GB, and the zero gate still exactly reproduces donor
loss. Run language-model retrieval and instruction-behavior gates before converting
layer 10, then repeat every gate after each new layer.
GPQA, LiveCodeBench, and AutomationBench belong at the end of this sequence; they cannot
diagnose router quality in a tiny synthetic model.

The expanded retrieval gate does not authorize layer 10 conversion. Exact retrieval
is stable, but lexical, multi-token, and NIAH-style preservation trade off across
recent, hybrid, and block selectors, with the best aggregate retaining only 69.84% of
dense-pass trials. The next model-quality step is a selector that combines the token
router's NIAH behavior with the block path's multi-token behavior, followed by this
same gate; adding another sparse layer would confound that diagnosis.

### Stage D: selector and decode systems work

A routing-only persistent direct-address prototype now separates the indexing
bottleneck from attention. From 16K to 2M, one-query FP16 and 64-bit scans grow
14.58x and 11.35x, while bounded addressed lookup changes by 0.94x and reads a fixed
1.5 KiB of historical index payload. At 2M it is 9.21x faster than FP routing, but
capacity-16 bucket eviction reduces planted-needle recall to 53.12% despite 96.88%
address agreement. Its overlap with the FP top 32 is only 2.10%, and index
construction is still an offline rebuild rather than a causal
append-only update.

A sparse hierarchical index now clears that gate. It uses an 8-bit secondary address,
four secondary probes, capacity-seven compact leaves, and a split start/count
directory. At 2M it reaches 95.31% planted-needle recall at 2.78 KiB/query and
294.4 us/query; conditional retention is 100% for address-matched queries. Peak MLX
memory is 768 MB. Dense 2-bit and 4-bit hierarchical variants are retained as negative
results because their eviction/address tradeoffs miss the recall target.

The learned-address gate on real LFM2.5 layer-14 states now fails. Across three seeds
on one shared WikiText held-out segment, the hierarchy recovers only 3.31–4.69% of
distant attention mass and raises perplexity to 1.325–1.367× dense. A separate PG-19
segment reproduces 4.36–8.11% distant-mass recall. An oracle at 1,024 tokens finds
92.69% of distant mass inside the best 224 candidates, but only 55.49% survives the
`K=32` ceiling and 28.31% survives learned-code Hamming reranking. Address discovery
is the largest loss; reranking is independently misaligned, and retention eviction
rises to 7.51% at 1,024.

The attention-mass-aligned hierarchy now passes the 256-token routing gate across
three seeds. On the identical held-out WikiText and PG-19 segments it reaches
80.16–85.33% of the dense `K=32` distant-mass oracle at 2,808 bytes/query. Trained
sparse output projections keep all six seed/corpus perplexity ratios at or below
1.0318. The MLX reference remains above the 350-us target at 421–446 us/query.

The next gate is length generalization, not another 256-token reranker sweep. At 512,
oracle-relative routing falls to 73.15–76.72%; at 1,024 it falls to 61.37–64.82%.
Output recovery restores all 512 perplexity ratios to at most 1.0318, but 1,024-token
recovery training exceeded the 1,792 MB safety ceiling at 1,905 MB and was stopped.
Append-only decode integration, Metal fusion, and end-to-end speed claims remain
gated on long-context routing recall and a memory-safe 1,024 quality protocol.

Seed-0 compressed-scoring follow-ups do not clear the 512-token gate. The strongest
joint binary decoder reaches 78.15%/78.06% of the oracle on WikiText/PG-19; nonlinear,
head-calibrated, normalized-head, and supervised four-byte VQ variants remain below
78.3%. The 128-segment VQ run peaks at 1,778.9 MB. The next gate is therefore a
48/64-candidate compressed shortlist followed by exact reranking with explicit exact-K
traffic accounting. It is not valid to count only the 2,808 index bytes for that test.

The shortlist sweep brackets the seed-0 boundary: 34 exact finalists remain below
gate at 79.44%/79.51%, while 36 reach 80.40%/80.66%. The latter costs 39,672 total
bytes/query (2,808 index plus 36 KiB exact K), so it is diagnostic rather than a
fixed-envelope success. The next training milestone is compressed-only distillation
of the exact swaps among approximate ranks 25–36, followed by the same 512-token
WikiText/PG-19 gate before any additional seed or length expansion.

That boundary-distillation attempt fails: its held-out objective improves, but the
compressed selector reaches only 78.28%/77.30% at 512. The next step is therefore a
fixed-byte allocation ceiling study, not another loss variant: 40/48-bit codes with
candidate counts reduced to keep total traffic at 2,808 bytes, and no training unless
the resulting retained pool has an at-least-80% exact-rerank ceiling.

The fixed-byte study confirms adequate pool ceilings but no deployed pass. At 512,
288 retained candidates contain 85.52%/87.21% of the WikiText/PG-19 oracle; 264 and
240 candidates also remain above 80%. The best deployed point is 40-bit 6x6 routing
at 2,832 bytes/query and 78.81%/79.62%. Binary 48-bit, categorical 40/56-bit,
teacher-top-32, query-only, and joint query/key variants all miss on seed 0.

The length-matched address stage has now been run on seed 0 and does not pass. Direct
attention-mass training makes 97-99% of distant mass addressable but collapses joint
codes into hot leaves. A new joint-address entropy loss at weights 10 and 30 exposes
the tradeoff but does not beat the original 78.81%/79.62% router: its best deployed
row is 77.47%/77.95%, with 45-54% eviction, while stronger entropy reduces eviction
at the cost of address alignment. All rows preserve 2,832 bytes/query.

The next gate is therefore capacity-aware optimization, not another scalar
regularization sweep. Train directly against overflow in the deployed 6x6 leaves or
learn bounded local retention, then repeat the identical seed-0 decomposition.
Require both corpora to improve over the original router and reach at least 80% of
the oracle before seeds 1/2, output recovery, latency work, or 1,024-token expansion.

The capacity-aware branch is now implemented and bounded. A straight-through leaf
overflow loss reduces eviction to 15.42%/14.48% at weight 0.3 and lifts WikiText to
80.15% oracle-relative recall, but PG-19 reaches only 77.88%. Weight 0.1 reaches
78.25%/78.57%. The both-corpus gate therefore remains closed despite a real load
improvement. The next gate is learned attention-aware retention within crowded
leaves; do not continue scalar overflow sweeps or replicate these seed-0 failures.

Linear learned retention does not clear that gate. Global attention-salience training
reaches 79.42%/79.20%, and an objective restricted to each hard leaf's capacity-32
boundary reaches 79.24%/79.36%. Both preserve the original routing/scorer tensors and
fixed traffic. Before implementing a larger retention model, measure the oracle
future-attention retention ceiling on the same leaves. Continue only if that ceiling
passes both corpora; otherwise return upstream to address/rerank representation.

The future-attention oracle retention ceiling reaches 79.25% on WikiText and 80.47%
on PG-19, so it does not pass both corpora and closes retention work for the current
leaves. Interpolating original and capacity-aware address projections also fails:
the best intermediate is only 79.45%/79.17%. The roadmap therefore returns upstream
to domain-balanced discrete address/reranker training, with model selection performed
on training/evaluation-domain segments before one canonical held-out gate check.

That branch has now completed its bounded seed-0 checks. Group-DRO address training
reaches 80.66%/77.99%, and refreshing its scorer reaches 80.21%/78.16%. A 24-mask
whole-table mixture selected on reserved training-split segments reaches only
78.78%/78.42% after canonical transfer. Evaluation-domain training on noncanonical
segments reaches 79.07%/80.91%. Each result is WikiText/PG-19 oracle-relative recall
at 512 tokens and 2,832 bytes/query; none passes both corpora.

Do not replicate these seed-0 checkpoints, run output recovery for them, tune against
canonical segment 0, or continue scalar Group-DRO/domain/table-mixture sweeps. The next
implementation gate is a materially different discrete set objective: optimize the
actual bounded candidate membership and compressed final ranking jointly across both
training domains. First require one untouched seed-0 canonical pass at at least 80%
on each corpus. Only then run output recovery, seeds 1/2, latency work, and 1,024-token
expansion.

The first candidate-set surrogate does not clear that implementation gate. It models
all six secondary probes and expected capacity-32 reservoir survival, reduces its
reserved loss from 0.274 to 0.166, and lowers canonical eviction. Nevertheless,
canonical WikiText/PG-19 recall falls to 74.87%/75.29% because address-candidate mass
falls to about 83.6%. The checkpoint is a rejected seed-0 diagnostic, not a candidate
for scorer refresh or replication.

The next objective must represent the exact causal retained membership and final
top-32 boundary more faithfully than `min(1, capacity/load)`. Use training-domain
segments for any model selection and preserve canonical segment 0 as a one-shot gate.
Do not run a scalar candidate-set-weight sweep on the canonical results.

Exact causal boundary mining is now implemented and also remains below gate. Without
load control it collapses to 56-62% eviction. Coupled with the pre-existing overflow
weight 0.3, it holds eviction to 13-15% and preserves retained-top32 ceilings of
85.41%/86.94%, but deployed WikiText/PG-19 recall is only 77.48%/78.14%. A refreshed
40-bit attention scorer reaches 77.47%/77.98%.

Do not replicate or run output recovery for this seed-0 branch. The next architecture
must jointly expose exact retained membership and final rank swaps during optimization;
another sequential address phase, linear scorer refresh, boundary stride, or scalar
weight sweep is not supported. Preserve the 2,848-byte envelope and canonical holdout.

That joint architecture is now implemented in `mlx_joint_binary_attention_train.py`.
It interleaves scorer updates with exact-boundary and overflow-aware address updates
on the same examples, saves both components in one checkpoint, and supports an
explicit pairwise top-32 rank-swap objective. Host backing keeps the accepted full
run at 1,196.6 MB; a prior 1,810.5 MB attempt was interrupted before checkpoint save.

The two-phase no-pairwise checkpoint reaches 78.45%/79.49% oracle-relative recall.
The predeclared pairwise version improves its reserved scorer objective, but its
single canonical evaluation reaches only 78.51%/79.61% on WikiText/PG-19, with
14.82%/13.33% eviction, 1.169x/1.098x raw perplexity, 2,832 bytes/query, and about
794 us/query. It therefore remains a rejected seed-0 diagnostic.

Do not add a third phase, tune pairwise weights against canonical segment 0, run
output recovery, or replicate this branch. The present formulation has now tested
expected survival, exact causal boundary mining, sequential scorer refresh, and
unified exact membership plus rank swaps without clearing the shared 512-token gate.
The next milestone must change the representation or discrete optimizer rather than
repeat these objectives; the goal remains at least 80% on both corpora before any
quality recovery or seed expansion.

Learned sign thresholds provide one bounded representation check and are also
negative. `--joint-address-thresholds` freezes the address projections, learns only
per-table query/key biases, persists them in the checkpoint, and applies them during
both exact mining and deployment. The selected two-phase seed-0 checkpoint reaches
78.31%/78.37% oracle-relative recall, versus 78.51%/79.61% for the zero-threshold
pairwise checkpoint. Eviction changes to 14.45%/12.82%, but address-candidate mass
falls to 87.47%/88.19%.

Do not sweep threshold learning rates, phases, or the same overflow/boundary weights.
The next implementation should replace correlated sign-bit byte construction with
direct categorical address assignment or another genuinely discrete balanced
partitioner, while keeping the same directory/probe/leaf byte accounting.

That unconstrained categorical experiment is complete and negative. The new
checkpoint format stores four assignment tensors for direct 256-way query/key byte
prediction. Binary-derived initialization, categorical top-P probing, checkpoint
reload, causal lookup, and the real timing path are covered by tests. The two-phase
run stays at 2,832 bytes/query and peaks at 1,075.1 MB during training.

Although phase-2 reserved address loss improves from 10.334 to 10.074, canonical
WikiText/PG-19 oracle-relative recall collapses to 47.71%/48.93%. Address-candidate
mass is only 34.94%/36.26%, while eviction falls to 0.366%/0.708%. This is strong
evidence that globally balanced categorical partitions solve load by discarding the
attention geometry the binary primary address had preserved.

Do not tune the categorical temperature, learning rate, balance weights, or add a
third phase. The next bounded implementation should freeze binary primary discovery
and learn a separate categorical secondary code inside each primary region. It must
first demonstrate a sufficient primary-only addressability ceiling and then improve
reserved exact membership without changing the six-probe, six-posting, 2,832-byte
contract.

That residual-secondary milestone is complete and negative. The frozen primary pool
has ample capacity: its exact K=32 ceiling is 99.42%/99.41% of oracle on canonical
WikiText/PG-19. A shared learned 256-way secondary classifier per table reduces
eviction to 1.880%/1.245%, but secondary K=32 ceilings fall to 81.17%/82.36%; after
retention they are 78.16%/79.64%, and deployed recall is only 72.00%/73.04%.

Do not add another phase, tune this branch against canonical segment 0, run output
recovery, or replicate it. The next bounded representation should make the secondary
partition explicitly conditional on the frozen primary region (for example, a
low-rank or per-primary categorical bias) and train its compressed reranker against
the same retained boundary. It must preserve six probes, six postings, and 2,832
bytes/query, improve on training-domain reserved metrics, then clear 80% on both
canonical corpora before any seed or length expansion.

That primary-conditioned representation is also complete and negative. Per-primary
query/key secondary biases raise the address K=32 ceilings to 89.12%/93.14% of oracle
and retained ceilings to 84.37%/87.88%, but deployed recall is 76.54%/78.93%.
A single safe 16,000-step scorer refresh reaches only 76.72%/79.09%. Address training
peaks at 1,079.1 MB and the accepted refresh at 1,764.7 MB. A first refresh attempt
exceeded the cap at 1,795.9 MB, was interrupted, and wrote no checkpoint.

Do not tune the primary biases, add another remine phase, or try another fixed-pool
linear scorer refresh. The next milestone must allocate the fixed 288 posting reads
according to leaf pressure/relevance and align the compressed final scorer to that
deployed allocation. Require reserved-domain improvement before a single seed-0
canonical check; output recovery, seeds, 1,024 tokens, and Metal remain gated.

The [indexed-memory article analysis](indexed-memory-article-analysis.md) adds two
explicit gates before any end-to-end sub-quadratic claim: compare token routing with a
fixed-budget block/span index, and report index build, lookup, exact attention, and
full-model costs separately.

## Reporting rules

Every result added to this repository must state:

- whether it is arithmetic, synthetic, natural-language, or matched replication;
- model, seed, training lengths, token count, and selected-token budget;
- hardware and software versions;
- exact evaluation examples and uncertainty across seeds;
- whether timing includes routing, attention, and cache movement;
- whether memory means active MLX memory, peak allocator memory, or host memory.

No synthetic result should be described as a replicated RULER, NIAH, language-model,
or production-serving result.
