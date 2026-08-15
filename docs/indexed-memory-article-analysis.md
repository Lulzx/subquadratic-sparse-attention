# Indexed-memory article analysis

This note maps Jian's May 6, 2026 article, *How Sub-Quadratic Attention Could Actually
Work*, to this repository. The article was supplied directly in the project discussion.
Its central thesis is a useful standard for judging this work:

> Keep the past addressable, use a genuinely non-quadratic selector to retrieve a small
> candidate set, then run exact attention over those candidates.

The comparison below summarizes and analyzes the article rather than reproducing it.

## Where the article and this implementation agree

### Dense attention is brute-force retrieval

Dense causal attention compares each query with every eligible earlier key. Its
attention comparison grid therefore grows as `O(n²)`. The benefit is arbitrary
addressability: no early compression decision has to predict which detail a later
query will need.

This repository keeps that addressability. It stores token states and retrieves
candidate positions; it does not compress the entire past into one fixed recurrent
state.

### Sparse attention is easy; sparse selection is the claim

Once candidate indices exist, gathering their K/V states and applying ordinary
attention is straightforward. The difficult question is how those indices are found.
An all-pairs learned scorer merely moves quadratic work from attention into routing.

The repository therefore reports selector construction, lookup behavior, candidate
counts, recall, and gathered-attention cost separately. Continuous all-pairs router
scores are treated only as a diagnostic ceiling, never as sparse lookup performance.

### The plausible architecture is hybrid indexed memory

The current LFM replacement attends over:

- 32 recent local tokens;
- four explicit attention-sink tokens;
- at most 32 distant positions retrieved from learned binary hash tables;
- exact softmax attention over that combined candidate set.

The portable prefill selector sorts hash entries, giving an intended complexity of
`O(n log n + nK)` for a fixed candidate budget K. That is genuinely sub-quadratic at
the selector/attention-layer level. It is not strictly linear, and the current full
LFM model remains quadratic because four of its six attention layers are still dense.

### Conversion should preserve a pretrained model

The article argues for starting from open weights, training the selector against dense
attention, and then recovering the sparse model rather than pretraining an unfamiliar
architecture from scratch. That is exactly the local sequence:

1. load and freeze LFM2.5-350M;
2. train a binary router against dense attention targets;
3. copy the donor layer's Q/K/V/O, normalization, GQA, and RoPE behavior;
4. align one sparse layer to cached dense layer outputs;
5. compose converted layers;
6. jointly recover only the sparse branches against cached dense final hidden states;
7. retain an exact zero-valued dense fallback gate throughout.

This staged method is why conversion fits on the Mac. The two-layer joint run peaks at
about 1.39 GB of MLX allocator memory.

## Verified DeepSeek-V3.2 comparison

The primary [DeepSeek-V3.2 paper](https://arxiv.org/abs/2512.02556) supports the
article's main factual comparison:

- DSA contains a lightning indexer followed by top-k token selection and exact sparse
  attention.
- DeepSeek starts from DeepSeek-V3.1-Terminus and introduces DSA through continued
  training.
- Its dense warm-up freezes the model and trains only the indexer against an aggregated
  dense-attention distribution for 1,000 steps.
- Its sparse stage selects 2,048 K/V tokens per query, optimizes the main model with LM
  loss, and continues indexer alignment for 15,000 steps.
- The paper explicitly characterizes core attention as changing from `O(L²)` to
  `O(Lk)`, while the lightning indexer itself remains `O(L²)` with a smaller constant.
- Post-training then uses specialist distillation and mixed reinforcement learning.

Our method is far smaller and more conservative. It uses 32 distant candidates, trains
copied attention branches instead of the full model, and uses an `O(n log n)` hash
index rather than an all-pairs indexer. It has nowhere near DeepSeek's token budget,
benchmark coverage, or post-training. The useful similarity is the conversion recipe,
not model capability.

## What the article exposes as our remaining risks

### Token-level indexing may be too noisy

The article expects realistic systems to combine local attention with retrieved
blocks, spans, summaries, entities, or symbols. This repository still indexes
individual token states. LFM2.5-Embedding experiments demonstrate a multiprobe
recall/candidate tradeoff, but they do not yet implement block-level routing.

Required comparison: hold the total attended-token budget fixed and evaluate token
hashing against block summaries plus exact within-block attention.

### Phonebook success is necessary but insufficient

MQAR proves that arbitrary exact details can remain retrievable, but the generator caps
associations at 512 and rewards exact-key matching. It does not establish lexical
mismatch, semantic routing, entity resolution, or millions of independently useful
memories. NoLiMa-style retrieval, public NIAH, and RULER remain required.

### Long context is not free

Even with sub-quadratic attention, the system must read input tokens, retain addressable
state, construct and move index data, retrieve candidates, and execute exact attention.
Every report must therefore include:

- candidate budget and recall;
- index construction and lookup time;
- exact attention time;
- index, K/V, and allocator memory;
- full-model latency rather than attention-only latency;
- whether routing cost is included.

### Prefill is not decoding

The current implementation constructs a parallel prefill index. It has no persistent
append-only index or K/V cache for token-by-token generation. Demonstrating causal
prefill correctness does not establish practical autoregressive serving.

### Benchmark routing can overfit

The embedding projection overfit a 40-section documentation corpus, and the project
records that failure. Router training must use disjoint natural-language sources,
mixed lengths, and unseen retrieval formats. A small WikiText perplexity slice cannot
substitute for long-context retrieval and broad capability evaluations.

## Roadmap changes implied by the article

1. Continue layerwise LFM conversion through layers 10, 8, 5, and 2, with combined
   recovery and zero-gate checks after every addition.
2. Add a fixed-budget block/span router and compare it with token routing.
3. Train routers on mixed-length natural-language attention and explicit retrieval
   labels rather than repository prose alone.
4. Implement an append-only persistent decode index that does not rebuild or sort the
   full history per generated token.
5. Report selector build, lookup, gathered attention, and full-model costs separately.
6. Run public NIAH, NoLiMa-style lexical mismatch, and RULER before describing the
   conversion as a long-context language model.
7. Only after all attention layers are sparse should the project claim end-to-end
   sub-quadratic model execution or compare speed against dense MLX attention.

## Epistemic boundary

The article's launch-drama claims about exposed endpoints, system prompts, or an
underlying model identity are not required to evaluate the architecture and are not
treated as verified evidence here. The relevant scientific questions are selector
complexity, retrieval recall, information preservation, conversion quality, and
end-to-end measured cost.
