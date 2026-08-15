# Limitations and roadmap

## Relationship to SubQ

This repository does not reproduce SubQ's unpublished SSA mechanism or its model weights. Similarity is limited to public behavioral goals: content-dependent sparse retrieval, subquadratic selection, causal operation, and long-context extrapolation.

## Current limitations

### Synthetic task only

The end-to-end result is MQAR, not natural language. Exact repeated key tokens are favorable to hashing. Semantic paraphrases, fuzzy matches, compositional reasoning, code, and multi-hop retrieval have not been tested.

### Portable prefill is `O(n log n)`

The current MLX selector sorts hash assignments. The intended append-only hash-table implementation has expected linear construction and constant expected lookup, but that kernel has not been implemented.

### No persistent decode cache

Causality is validated, but production autoregressive state management—bucket storage, KV cache ownership, eviction, batching, and beam behavior—is not implemented.

### Discrete routing objective is weak

The router currently receives balance and confidence losses. It is not trained to reproduce dense-attention mass or to maximize semantic retrieval recall. Hard routing has no task gradient through its indices.

### Collision frontier

The 128-token baseline falls from about 95% at 2K to 78% at 4K. A joint eight-table, two-probe, multi-length configuration raises 4K accuracy to 95.67%, but falls to 75% at 16K in a smaller frontier check. Because selector capacity, read budget, training steps, and curriculum changed together, the experiment does not isolate the dominant cause.

### Tiny model

The model has width 64 and two layers. Results do not establish that the mechanism transfers cleanly into a pretrained language model or remains stable during attention surgery.

### Performance is not yet kernel-optimal

MLX standard operations are fast enough to validate the design, but hash, gather, RoPE, softmax, and value reduction are not fused. Timings should not be interpreted as a production ceiling.

## Highest-value next experiments

1. **Run the collision ablation**: tables, bits, members, and lowest-margin multiprobe count are now configurable; compare them at fixed and increasing `K`.
2. **Run the length curriculum**: staged 128–1,024-token training is now implemented; train full checkpoints and retest 4K–16K.
3. **Semantic routing loss**: distill bucket agreement from fixed-window dense attention or labeled retrieval positions.
4. **Custom Metal selector**: replace sorting with append-only bucket tails and benchmark true expected-linear prefill/decode behavior.
5. **Persistent decode cache**: verify token-by-token outputs against parallel causal prefill.
6. **Natural-language retrieval**: evaluate passkey, variable-length needle, and RULER-style tasks.
7. **Pretrained-model surgery**: replace attention in a genuinely small open MLX model and measure perplexity recovery.
8. **Dense matched control**: train a dense MLX model with the same initialization, untied output, curriculum, and optimizer.

## Evidence standard

Future claims should always identify:

- the exact checkpoint and seed;
- training and evaluation context lengths;
- number of evaluated examples;
- whether routing matches exact or semantic content;
- selected-token budget;
- peak memory and hardware;
- whether timing includes selection, attention, projections, or the entire layer;
- whether the result is synthetic or natural-language.
