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

## Controlled 4K ablation

The joint follow-up changed too many variables to identify the source of the gain. A controlled ablation held width, layers, heads, optimizer, 1,200 training steps, curriculum, evaluation seed, and selected budget `K=32` fixed. Each reported selector configuration was trained with seeds 0, 1, and 2 and evaluated on 11,456 answers per seed at 4K.

| Configuration | Seed 0 | Seed 1 | Seed 2 | Aggregate / evaluated | Mean accuracy |
|---|---:|---:|---:|---:|---:|
| 4 tables, 1 probe, 4 members | 95.38% | 96.29% | 95.16% | 32,860 / 34,368 | 95.61% |
| 8 tables, 1 probe, 2 members | 96.99% | 97.38% | 96.72% | 33,347 / 34,368 | **97.03%** |
| 4 tables, 2 probes, 2 members | 97.01% | 96.60% | 96.82% | 33,272 / 34,368 | 96.81% |

Both extra tables and multiprobe routing improve every tested seed at fixed read budget. Eight tables lead by 0.22 percentage points on the three-seed mean, which is too small a sample to treat as a statistically established difference. It is the recommended configuration because it is slightly better here and avoids multiprobe index expansion.

Two additional seed-0 controls isolate curriculum from training duration under the original four-table selector:

| Training schedule | Steps | 4K accuracy |
|---|---:|---:|
| 128 tokens only | 300 | 78.99% |
| 128 tokens only | 1,200 | 82.30% |
| 128, 256, 512, 1,024 curriculum | 1,200 | 95.38% |

On this seed and matched evaluation protocol, additional optimizer steps account for 3.31 percentage points, while changing from 128-only training to the length curriculum at 1,200 steps accounts for another 13.08 points. The curriculum is therefore the dominant measured cause of the recovered 4K accuracy.

## Extended-curriculum frontier

Because curriculum was the dominant 4K intervention, a seed-0 exploratory series extended the same eight-table, one-probe, `K=32` model to longer training stages. Every stage received 300 steps. Evaluation used batch 1 and 16 batches at each reported extrapolation length.

| Maximum training length | Training steps | 8K accuracy | 16K accuracy | 32K accuracy |
|---:|---:|---:|---:|---:|
| 1,024 | 1,200 | 95.81% | 74.23% | — |
| 4,096 | 1,800 | 99.97% | 89.70% | — |
| 8,192 | 2,100 | 100.00% | **95.67%** | 91.41% |

The 16K rows each evaluate 2,864 answers. The 32K result also evaluates 2,864 answers after a one-batch memory-safety probe completed successfully; peak evaluation memory was 499 MB. The 8K training stage raised observed peak training memory to 1.64 GB and reduced average throughput for the full run to 15.4 steps/s.

The movement of the failure frontier strongly tracks maximum curriculum length on this seed. It does not establish the same result across training seeds, and MQAR stops increasing its association count after 4K because the task has only 512 distinct keys. Beyond that point, longer sequences add filler and retrieval distance.

### Stagewise 8K fine-tuning

The 4K checkpoint was also continued for 300 steps at 8K using the new `--resume` path. Model weights were restored, while AdamW state intentionally started fresh.

| 8K training strategy | 16K accuracy | 32K accuracy |
|---|---:|---:|
| One optimizer across the full 2,100-step curriculum | 95.67% | 91.41% |
| Resume 4K checkpoint; fresh optimizer for 300 steps at 8K | **97.38%** | **95.32%** |

Both rows use seed 0 and evaluate 2,864 answers per length. The stagewise run indicates that carrying optimizer moments across large length transitions can hurt the final frontier. It also avoids replaying earlier stages when a shorter-context checkpoint already exists. This is one seed and does not yet establish the best reset schedule or learning rate.

## PyTorch reference results

The initial PyTorch/MPS language-model curriculum did not learn general associative retrieval in the tested configurations, although it could overfit a fixed batch to 100%. This was useful evidence that optimization worked but the setup was not producing a general circuit.

The MLX model differs in implementation details, including an untied output projection, so its success should not be attributed solely to the runtime. The runtime did make iteration substantially faster and enabled the current memory-bounded implementation.
## Semantic-router and compressed-global seed-0 probe

This 2026-08-15 experiment is exploratory and single-seed. It tests whether two ideas
from the replication roadmap can be trained without changing the selected-token budget.

Protocol:

- Apple M4 Pro with 24 GB unified memory;
- width 64, two layers, four heads, window 32;
- eight total tables, 16 bits, two members, one probe, `K=32`;
- 300 steps at length 128, batch 16, seed 0;
- 16 held-out batches at length 128;
- eight batch-1 held-out batches at 2K and 4K.

| Variant | 128 | 2K | 4K |
|---|---:|---:|---:|
| Baseline | 99.84% | 97.47% | **80.73%** |
| Four compressed global slots | **100.00%** | 96.21% | 79.68% |
| Hybrid semantic router | 99.61% | 94.80% | 80.03% |
| Hybrid semantic router + four global slots | 99.77% | 94.38% | 77.93% |

The original all-semantic selector replaced every shared hash table with separate Q/K
hashes. It collapsed to 0.16% held-out accuracy at length 128. Semantic loss weights
of 0.1, 0.05, and 0.01 also remained at or below 0.23%. A fixed-budget hybrid that
retains four shared fallback tables restored learning, but neither it nor the global
summary path improved extrapolation in this seed.

This is evidence against the immediate mechanisms, not against semantic routing or
global compression generally. Exact MQAR rewards identical token hashes, the from-scratch
sparse Q/K projections are not a trained dense teacher, and slot means blend mostly
filler at long lengths. The next semantic experiment must use a pretrained dense donor
or explicit query-to-source labels; the next compressed path should learn block
summaries instead of using fixed means.

## SmolLM2 natural-language donor-router distillation

This 2026-08-15 experiment uses the frozen BF16
[SmolLM2-135M base model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) as a real
dense-attention teacher. It trains only a standalone hash router and does not replace
the donor's attention layer or measure language-model output quality.

Protocol:

- donor layer 15 of 30;
- 256-token sequences from repository prose;
- 32 training segments and four held-out segments from disjoint documentation files;
- first four key positions excluded as attention sinks;
- local 32-token window excluded from the routing target;
- eight tables, eight bits, four members, one probe, maximum hard budget 32;
- straight-through binary codes, 2,000 optimizer steps;
- seeds 0, 1, and 2;
- 876 held-out distant queries per seed.

| Metric | Seed 0 before / after | Seed 1 before / after | Seed 2 before / after | Mean before / after |
|---|---:|---:|---:|---:|
| Hard retained teacher mass | 29.65 / 33.16% | 26.81 / 33.54% | 25.51 / 33.38% | **27.32 / 33.36%** |
| Hard teacher top-1 recall | 40.07 / 57.42% | 36.19 / 54.45% | 35.16 / 55.48% | **37.14 / 55.78%** |
| Mean unique hard candidates | 17.48 / 11.53 | 16.90 / 12.05 | 16.47 / 11.52 | **16.95 / 11.70** |
| Continuous top-32 mass | 41.04 / 68.18% | 47.30 / 67.66% | 39.12 / 68.92% | **42.48 / 68.25%** |
| Continuous top-32 top-1 recall | 41.89 / 88.70% | 54.79 / 83.68% | 36.42 / 84.36% | **44.37 / 85.58%** |

Two failed intermediate designs were necessary to make the measurement honest:

1. Including the first tokens produced 100% top-1 recall with one candidate because
   the router learned an attention sink. Sink exclusion yields 143 distinct teacher
   top-1 positions in the held-out set, with the most common only 4.22% of queries.
2. Looking up the most recent bucket members and discarding local tokens afterward
   allowed local collisions to consume the sparse budget. `min_distance=window` now
   seeks directly before the local cutoff.

The hard router improves all three seeds despite returning fewer unique candidates.
However, it retains only about half the mass available to its own continuous top-32
scores. The next router experiment should focus on quantization/indexing fidelity:
learned multi-probe policies, smaller independently calibrated code groups, or a
two-stage hash-plus-rerank selector. It should not increase the hard budget silently.

This is partial natural-language evidence for learned content routing. It is not yet
evidence for language-model quality, subquadratic end-to-end execution, RULER, or NIAH.
