# Memory safety

## Why the limits exist

During development, a 32K dense-attention benchmark on MPS attempted an additional 12 GB allocation after the process and system had already accumulated substantial GPU memory. The laptop became unresponsive.

That run produced useful evidence—dense attention exhausted memory while the sparse path completed—but it was not an acceptable operating procedure. The repository now treats memory limits as part of correctness.

## Default limits

| Command | Default maximum |
|---|---:|
| `bench.py` dense attention | 8,192 tokens |
| `bench.py` PyTorch SSA | 16,384 tokens |
| `mlx_selector.py` | 16,384 tokens |
| `mlx_attention_bench.py` | 16,384 tokens |
| `mlx_evaluate.py` | 4,096 tokens |

Larger requested lengths are skipped with an explicit message instead of being attempted.

## Cache handling

- PyTorch/MPS benchmark outputs are deleted after each repetition.
- `torch.mps.empty_cache()` runs after every benchmark length.
- MLX benchmarks call `mx.clear_cache()` after each length.
- MLX peak-memory counters are reset for each row.
- Selected attention is chunked in groups of 1,024 queries at inference time.

## Recommended long-context settings

Use batch size 1 beyond 1K for end-to-end model evaluation:

```bash
python3 mlx_evaluate.py checkpoint.safetensors \
  --lengths 2048,4096 \
  --batch 1 \
  --batches 8
```

## Before increasing a cap

1. Estimate the largest gathered tensor: approximately `batch × queries × K × width × element_size` for each gathered K/V buffer.
2. Run one length at a time.
3. Use one repetition first.
4. Monitor Activity Monitor or MLX memory counters.
5. Never use dense 32K attention on this laptop.

The command-line override is an escape hatch for different hardware, not a recommendation.
