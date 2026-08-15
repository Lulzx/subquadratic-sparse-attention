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
| 1 | Arbitrary-position content-dependent sparse retrieval | Partial | SimHash retrieves exact MQAR keys, and a donor-distilled router improves held-out natural-language teacher-attention recall; downstream semantic retrieval remains untested. |
| 2 | End-to-end linear selection and attention | Partial | Attention reads fixed `K=32`; portable bucket construction still sorts in `O(n log n)`. |
| 3 | Linear memory scaling | Partial | Selected attention has bounded reads and measured memory through 16K, not millions of tokens. |
| 4 | Full-context training and ordinary autoregressive operation | Partial | Parallel causal prefill works; a persistent incremental decode cache is absent. |
| 5 | Conversion of a dense pretrained donor without losing language quality | Partial | LFM2.5 layers 12 and 14 are replaced; three-seed joint recovery has a 0.9935 mean WikiText perplexity ratio on the current test slice. Four attention layers and broader evaluations remain. |
| 6 | 64.5x FLOP reduction at 1M | Arithmetic only | `252 / 3.9 = 64.6`; undisclosed dimensions prevent an absolute rerun. |
| 7 | 56x attention-layer wall-clock speedup at 1M | Arithmetic only | `54,164 / 966 = 56.07`; no matched H100 run exists. |
| 8 | RULER average 99.12 at 128K | Not reproduced | The public benchmark has not yet been integrated. |
| 9 | NIAH retrieval through 12M and extrapolation beyond training length | Partial | MQAR reaches 95.22% at 16x its original training length and 95.32% at 32K after staged training, but is not NIAH. |
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

### Stage C: pretrained language-model conversion

LFM2.5-350M is now the primary donor. Layers 12 and 14 are independently converted
while every convolution, MLP, normalization, embedding, and output component remains
frozen. Joint final-hidden recovery trains only those two sparse branches. Across three
seeds it moves their combined mean WikiText perplexity ratio from 1.0119 to 0.9935 on
4,096 disjoint test tokens and peaks at 1.39 GB of MLX memory. The zero gate exactly
reproduces donor loss. This is no measured loss on a small slice, not evidence that
sparsity improves general quality. Convert layer 10 next, then repeat combined recovery
and broader evaluation before progressing to layers 8, 5, and 2.
Afterward, run language-model retrieval benchmarks, then general capability benchmarks.
GPQA, LiveCodeBench, and AutomationBench belong at the end of this sequence; they cannot
diagnose router quality in a tiny synthetic model.

### Stage D: selector and decode systems work

Replace sorting with append-only tables, implement persistent decoding, then fuse the
gather/attention path in Metal. Only after correctness parity should the project compare
against dense MLX attention at increasing lengths.

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
