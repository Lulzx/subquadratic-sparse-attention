# MLX implementation

## Why MLX

The project began with a PyTorch/MPS reference. That version was useful for correctness, but the dynamic sort/gather path carried substantial dispatch overhead. MLX reduced selector latency by roughly 2–3× in local measurements and exposes both graph compilation and custom Metal kernels for future fusion.

The MLX implementation is now authoritative. PyTorch remains a readable cross-check.

## Module map

| Module | Responsibility |
|---|---|
| `ssa/mlx_selector.py` | Hash codes, causal bucket ordering, fixed-budget position selection |
| `ssa/mlx_attention.py` | RoPE, gathered softmax attention, chunked inference path |
| `ssa/mlx_model.py` | Sliding-window path, learned gates, transformer block, tiny LM |
| `ssa/tasks.py` | NumPy-first MQAR generator plus optional PyTorch wrapper |
| `mlx_train.py` | MLX autograd and AdamW training loop |
| `mlx_evaluate.py` | Held-out and length-extrapolation evaluation |

## Differentiation boundary

MLX correctly rejects attempts to differentiate through scatter indices. The implementation makes the boundary explicit:

```python
selected = select_indices(
    mx.stop_gradient(x),
    mx.stop_gradient(hash_projection),
)
```

This does not stop gradients through the selected Q/K/V computation. It only prevents autograd from treating discrete bucket membership as differentiable. The router projection receives gradients from `router_loss`.

## Memory-bounded selected attention

Naively materializing gathered keys and values for all queries at once caused peak memory to grow to about 720 MB at 16K. The inference path now processes 1,024 queries at a time:

1. project and realize Q/K/V once;
2. compute and realize selected indices once;
3. gather and attend one query chunk;
4. realize the chunk output so temporary gathers can be released;
5. concatenate only the compact outputs.

At 16K and width 64, this reduced observed peak memory from about 720 MB to 57 MB while preserving bit-identical output in the test configuration.

## Window attention

The local path uses MLX's fused scaled-dot-product attention over blocks of one window of queries and at most two windows of keys. A boolean causal/window mask restricts every query to prior positions within `W`.

The causality test mutates future tokens and observes a maximum prior-output difference of exactly zero in the tested configuration.

## Numerical validation

`mlx_tests.py` constructs repeated records so selected attention is nonzero, then compares the complete projected, RoPE-rotated, masked attention output against an independent NumPy implementation. Maximum absolute error is below `9e-7` in float32.

The same test compares chunked and unchunked MLX attention and requires identical selected indices and output error below `2e-5`.

## Future Metal fusion

MLX supports Python-authored custom Metal kernels. The highest-value fusion target is:

```text
hash → bucket-tail lookup → gather K/V → RoPE → dot products → masked softmax → weighted sum
```

The current implementation deliberately uses standard MLX operations until the architecture stabilizes. A fused kernel should be evaluated only with correctness parity, causal tests, and measured memory/latency wins.
