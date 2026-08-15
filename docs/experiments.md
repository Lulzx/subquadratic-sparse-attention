# Experiments and results

## Environment

Measurements in this repository were produced on an Apple-silicon Mac in August 2026 with:

| Dependency | Version |
|---|---:|
| Python | 3.13 |
| MLX | 0.30.0 |
| PyTorch reference | 2.7.1 |
| NumPy | 2.2.6 |

Timings are local single-machine measurements. They are not portable performance guarantees.

## Correctness properties

| Test | Result |
|---|---:|
| Output shapes and finite values | Pass |
| Causality under future-token mutation | Pass; maximum earlier difference `0.0` |
| Identical-content hash colocation | Pass |
| Exact selected position included | Pass |
| 16K exact-needle selector recall | `100 / 100` |
| MLX selected attention vs NumPy | maximum absolute error `8.77e-7` |
| Chunked vs unchunked MLX attention | maximum absolute error `0.0` in test configuration |

## Selector benchmark

Configuration: batch 1, width 64, four 16-bit tables, four prior members per table, anchor plus successor, `K=32`, five timed repetitions.

| Context | Median latency | Peak memory | Exact recall |
|---:|---:|---:|---:|
| 1,024 | 0.58 ms | 1.70 MB | 100% |
| 2,048 | 0.64 ms | 3.39 MB | 100% |
| 4,096 | 0.70 ms | 6.77 MB | 100% |
| 8,192 | 0.99 ms | 13.52 MB | 100% |
| 16,384 | 1.42 ms | 27.02 MB | 100% |

Later regression runs vary slightly—approximately 1.45 ms at 16K—but preserve the same recall and memory scale.

## Selected-attention benchmark

Configuration: batch 1, width 64, four heads, float16 projections, `K=32`, 1,024-query chunks, three timed repetitions.

| Context | Median latency | Peak memory |
|---:|---:|---:|
| 1,024 | 3.08 ms | 43.96 MB |
| 2,048 | 5.30 ms | 44.83 MB |
| 4,096 | 9.70 ms | 46.58 MB |
| 8,192 | 14.94 ms | 50.08 MB |
| 16,384 | 17.54 ms | 57.08 MB |

The nearly flat memory curve is the result of query chunking. This table measures selected attention only, not a complete multi-layer language model.

## End-to-end MQAR training

Training configuration:

| Parameter | Value |
|---|---:|
| Context | 128 tokens |
| Batch | 16 |
| Width | 64 |
| Layers | 2 |
| Heads | 4 |
| Local window | 32 |
| Steps | 300 |
| Learning rate | `1e-3` |
| Selected budget | 32 tokens/query |

Training reached 100% current-batch accuracy by step 250. Evaluation on 1,280 independently seeded associations produced 99.92% accuracy.

## Length extrapolation

The same checkpoint was evaluated without fine-tuning. The number of stored key/value records scales with context length.

| Context | Multiple of training context | Correct / evaluated | Accuracy |
|---:|---:|---:|---:|
| 128 | 1× | 320 / 320 | 100.00% |
| 256 | 2× | 697 / 704 | 99.01% |
| 512 | 4× | 1,395 / 1,408 | 99.08% |
| 1,024 | 8× | 2,686 / 2,816 | 95.38% |
| 2,048 | 16× | 678 / 712 | 95.22% |
| 4,096 | 32× | 1,116 / 1,432 | 77.93% |

The clear result is strong extrapolation through 16× training length and material degradation at 32×. Likely contributors include hash collisions, finite per-bucket history, the tiny model's induction circuit, and training on a single context length.

## Multiprobe curriculum follow-up

A seeded follow-up run jointly changed selector capacity and training lengths:

| Parameter | Value |
|---|---:|
| Seed | 0 |
| Training steps | 1,200 |
| Curriculum | 128, 256, 512, 1,024 tokens; 300 steps each |
| Batch by stage | 16, 8, 4, 2 |
| Width / layers / heads | 64 / 2 / 4 |
| Tables / bits | 8 / 16 |
| Probes / members / block width | 2 / 2 / 2 |
| Selected budget | `K = 64` |

The checkpoint reached 100% held-out accuracy at every curriculum length. A 16-batch, batch-4 extrapolation run produced:

| Context | Correct / evaluated | Accuracy |
|---:|---:|---:|
| 128 | 320 / 320 | 100.00% |
| 256 | 704 / 704 | 100.00% |
| 512 | 1,408 / 1,408 | 100.00% |
| 1,024 | 2,816 / 2,816 | 100.00% |
| 2,048 | 5,690 / 5,696 | 99.89% |
| 4,096 | 10,960 / 11,456 | 95.67% |

This improves the documented 4K baseline from 77.93% to 95.67%, a gain of 17.74 percentage points. The experiment changes tables, probes, members, selected budget, step count, and curriculum together, so it does not identify which change caused the gain.

A smaller batch-1, four-batch frontier check measured 96.09% at 8K (688 / 716) and 75.00% at 16K (537 / 716). MQAR caps stored pairs at 512, so beyond 4K these cases increase retrieval distance and filler rather than the number of associations.

## PyTorch reference results

The initial PyTorch/MPS language-model curriculum did not learn general associative retrieval in the tested configurations, although it could overfit a fixed batch to 100%. This was useful evidence that optimization worked but the setup was not producing a general circuit.

The MLX model differs in implementation details, including an untied output projection, so its success should not be attributed solely to the runtime. The runtime did make iteration substantially faster and enabled the current memory-bounded implementation.
