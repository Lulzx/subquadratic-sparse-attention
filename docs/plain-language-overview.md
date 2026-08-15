# Plain-language overview

The simplest way to think about it is: we are teaching a normal language model to
"look up the few useful earlier words" instead of rereading every earlier word.

Normal attention compares every token with every other token. If the context doubles,
that attention work grows roughly fourfold. That is why very long contexts become
expensive.

## How our replacement works

1. Keep the recent 32 tokens, because nearby context is usually useful.
2. Keep four beginning "sink" tokens that the model commonly relies on.
3. Use small learned hashes to retrieve at most 32 relevant distant tokens.
4. Run real attention only over that small set.

The learned hash router is like a very fast index at the back of a book. It tries to
send related token states to the same buckets.

## Why this is useful

- Long documents and codebases could use much less attention memory and computation.
- We can convert an existing model instead of training a language model from scratch.
- Only one small component is trained at a time, so experiments fit on this laptop.
- The dense model stays frozen as a teacher, giving us an exact target and a safe
  fallback gate.

More concretely, a small router is trained first, then one or two attention layers are
repaired at a time while every other model component stays frozen. Dense teacher
activations are cached once and reused. The current one-layer conversion peaks near
1.15 GB of MLX memory; two-layer joint recovery peaks near 1.39 GB.

## What we have completed

- Built a causal sparse selector and selected-attention implementation.
- Demonstrated synthetic retrieval 16 times beyond training context.
- Measured bounded sparse attention through 16K tokens.
- Corrected multiprobe routing so future tokens cannot influence past routing.
- Trained LFM2.5 routers to imitate real natural-language attention.
- Replaced LFM2.5 attention layer 14. Its mean WikiText perplexity penalty before
  multi-layer recovery is only 1.64%.
- Replaced layer 12 independently; it has no measured penalty on this test slice.
- Loaded layers 12 and 14 together. Before joint recovery, their combined mean penalty
  is 1.19%.
- Completed three-seed joint recovery of both layers. Their mean perplexity ratio is
  0.9935 relative to dense, using only 1.39 GB peak MLX memory. All three seeds are at
  or slightly below dense loss on this small slice. That apparent improvement may be
  test noise or regularization, so it is reported as no measured quality loss rather
  than a real quality improvement.

## What this does not prove yet

- We have converted only two of six attention layers.
- Evaluation is still small; we have not run RULER, NIAH, coding, or broad language
  benchmarks.
- Incremental token-by-token decoding is not implemented.
- Bucket construction still contains a sort, so the whole system is not yet strictly
  linear-time.
- We have not demonstrated an end-to-end speedup over dense attention.

## What comes next

The next conversion target is attention layer 10. After every new layer, individual
and combined quality must be measured before proceeding to layers 8, 5, and 2. Only
after all six attention layers survive conversion should the project focus on a
persistent decode cache, strict linear-time indexing, broad benchmarks, and real
end-to-end speedups.
