# Design history

## 1. Public-claim audit

The work began by decomposing the SubQ model card into arithmetic claims and empirical claims. Published tables were internally consistent in most places, but model-quality results could not be rerun because weights and the SSA implementation were unavailable.

The implementation goal therefore became behavioral: build a content-dependent selector whose own cost is subquadratic, then test retrieval and scaling independently.

## 2. Fixed product-code buckets

The first selector projected tokens into a fixed set of product-code buckets. Queries scored a constant number of codebook entries, selected several buckets, and read recent members.

This passed basic shape and causality tests, but it contained a fundamental scaling flaw:

- fixed bucket count;
- fixed members read per bucket;
- bucket occupancy grows with sequence length;
- the probability that an old needle remains among the recent members therefore falls toward zero.

A fixed-capacity variant only moved the problem: total addressable memory became `bucket_count × capacity`, independent of context length.

That architecture was rejected before treating short-context success as evidence of long-context viability.

## 3. Multi-table SimHash pivot

The replacement uses a much larger implicit address space—`2^16` buckets per table—and direct hash lookup. Four independent tables reduce the probability that all routes suffer harmful collisions.

For exact repeated content, query and key hashes match by construction. At 16K tokens, expected occupancy remains small enough that reading four prior entries per table retained 100/100 seeded needles.

## 4. MLX migration

The PyTorch/MPS implementation established correctness but had high dynamic-operation overhead. MLX was already installed and supports native Metal execution, lazy graphs, explicit memory counters, and custom kernels.

The initial MLX selector was roughly 2–3× faster than the equivalent PyTorch selector in local tests. That justified porting selected attention and then the tiny model.

## 5. Broken numerical test

The first NumPy comparison reported zero error but emitted suspicious warnings. Investigation showed that 64 random tokens produced no hash collisions, so every selected-attention output was zero. The test was not exercising attention at all.

The corrected test repeats 32 records, guarantees nonempty retrieval, requires nonzero reference output, and measures maximum error below `9e-7`.

## 6. Memory-bounded chunking

Unchunked MLX selected attention reached roughly 720 MB peak memory at 16K. Processing 1,024 queries at a time reduced the peak to about 57 MB, with identical output in the regression test.

## 7. End-to-end result

The MLX tiny model learned fresh-random MQAR associations and achieved 99.92% held-out accuracy after 300 steps. It retained about 95% accuracy at 16× its training context, establishing the first complete selector-to-model result in the project.
