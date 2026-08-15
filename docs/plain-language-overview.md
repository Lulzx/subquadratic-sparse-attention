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
- Completed three-seed joint recovery of both layers using only 1.39 GB peak MLX
  memory. An initial 4,096-token slice appeared equal to dense, but a stronger paired
  audit over 65,536 tokens per corpus finds 1.1100 times dense perplexity on WikiText
  and 0.9945 times on PG-19. That invalidated the original recovery objective.
- Recovered the two-layer model again by teaching it to match the dense model's output
  probabilities on mixed WikiText and PG-19 data. Across three seeds, the larger audit
  now measures 0.9675 times dense perplexity on WikiText and 0.8712 times on PG-19.
  Every confidence interval passes the 2% gate, at 1.61 GB peak MLX memory.

## What this does not prove yet

- We have converted only two of six attention layers.
- Evaluation is still small; we have not run RULER, NIAH, coding, or broad language
  benchmarks.
- Incremental token-by-token decoding is not implemented.
- Bucket construction still contains a sort, so the whole system is not yet strictly
  linear-time.
- We have not demonstrated an end-to-end speedup over dense attention.

## What comes next

The immediate target is proving that the recovered model still follows instructions
and retrieves distant information. The raw-text quality gate has passed, but it does
not establish those behaviors. Layer 10 can proceed only after the retrieval and
instruction checks pass. After every new layer, individual and combined quality must
be measured before proceeding to layers 8, 5, and 2. Only
after all six attention layers survive conversion should the project focus on a
persistent decode cache, strict linear-time indexing, broad benchmarks, and real
end-to-end speedups.
