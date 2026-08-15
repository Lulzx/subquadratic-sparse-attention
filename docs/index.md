# Documentation

This directory is the technical record for the subquadratic sparse-attention prototype. It documents both the working MLX architecture and the experiments that failed on the way to it.

## Suggested reading paths

### Understand the idea

1. [Architecture](architecture.md)
2. [Design history](design-history.md)
3. [Limitations](limitations.md)

### Reproduce the results

1. [Reproduction guide](reproduction.md)
2. [Memory safety](memory-safety.md)
3. [Experiments and results](experiments.md)

### Audit the relationship to SubQ

1. [Model-card claim audit](model-card-audit.md)
2. [Replication ledger and roadmap](replication-roadmap.md)
3. [Architecture](architecture.md#what-is-and-is-not-being-replicated)
4. [Limitations](limitations.md#relationship-to-subq)

## Document map

| File | Purpose |
|---|---|
| [`architecture.md`](architecture.md) | Defines the selector, attention layer, causal rules, and complexity model. |
| [`mlx-implementation.md`](mlx-implementation.md) | Maps the design to MLX code and explains the differentiation boundary. |
| [`experiments.md`](experiments.md) | Records hardware, versions, protocols, measurements, and conclusions. |
| [`model-card-audit.md`](model-card-audit.md) | Separates reproducible arithmetic from inaccessible empirical claims. |
| [`replication-roadmap.md`](replication-roadmap.md) | Tracks claim status, recent research, and the staged Mac-local replication plan. |
| [`reproduction.md`](reproduction.md) | Provides clean-environment commands and expected outputs. |
| [`memory-safety.md`](memory-safety.md) | Explains default limits and how to avoid unsafe dense allocations. |
| [`design-history.md`](design-history.md) | Records why the original product-codebook router was rejected. |
| [`limitations.md`](limitations.md) | States what has not been established and prioritizes next work. |

## Terminology

- **Selector**: the content-routing stage that produces key/value positions for each query.
- **Selected attention**: exact softmax attention over the selector's fixed candidate set.
- **Anchor**: a prior token whose content hash matches the current query.
- **Successor**: the token immediately after an anchor; included to support associative and induction-style retrieval.
- **MQAR**: multi-query associative recall, the synthetic task used for end-to-end tests.
- **Prefill**: processing a complete prompt in parallel.
- **Decode**: incrementally processing one autoregressive token at a time.
- **Logical complexity**: the algorithm expected with an actual hash table.
- **Prototype complexity**: the complexity of the portable sorting implementation currently in the repository.
