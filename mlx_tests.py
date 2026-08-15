import math

import mlx.core as mx
import numpy as np

from ssa.mlx_attention import random_weights, sparse_attention, sparse_attention_chunked


def numpy_rope(x, positions, base=50000.0):
    dim = x.shape[-1]
    inv = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    angles = positions.astype(np.float32)[..., None] * inv
    while angles.ndim < x.ndim:
        angles = np.expand_dims(angles, -2)
    even, odd = x[..., 0::2], x[..., 1::2]
    return np.stack([
        even * np.cos(angles) - odd * np.sin(angles),
        even * np.sin(angles) + odd * np.cos(angles),
    ], axis=-1).reshape(x.shape)


def numpy_reference(x, weights, selected, heads):
    wq, wk, wv, wo, _ = weights
    batch, length, width = x.shape
    dim = width // heads
    with np.errstate(all="ignore"):
        q = (x @ wq).reshape(batch, length, heads, dim)
        k = (x @ wk).reshape(batch, length, heads, dim)
        v = (x @ wv).reshape(batch, length, heads, dim)
    safe = np.maximum(selected, 0)
    gathered_k = np.stack([k[b][safe[b]] for b in range(batch)])
    gathered_v = np.stack([v[b][safe[b]] for b in range(batch)])
    q = numpy_rope(q, np.arange(length)[None, :])
    gathered_k = numpy_rope(gathered_k, safe)
    scores = np.sum(q[:, :, None] * gathered_k, axis=-1) / math.sqrt(dim)
    valid = selected >= 0
    scores = np.where(valid[..., None], scores, -1e9)
    scores = scores - np.max(scores, axis=2, keepdims=True)
    probability = np.exp(scores) * valid[..., None]
    probability /= np.maximum(probability.sum(axis=2, keepdims=True), 1e-9)
    output = np.sum(probability[..., None] * gathered_v, axis=2).reshape(batch, length, width)
    with np.errstate(all="ignore"):
        return output @ wo


def test_reference():
    mx.random.seed(0)
    base = mx.random.normal((2, 32, 32)).astype(mx.float32)
    x = mx.concatenate([base, base], axis=1)
    weights = tuple(w.astype(mx.float32) for w in random_weights(32, seed=1))
    output, selected = sparse_attention(x, *weights, heads=4)
    mx.eval(x, output, selected, *weights)
    reference = numpy_reference(
        np.array(x.tolist(), dtype=np.float64),
        tuple(np.array(w.tolist(), dtype=np.float64) for w in weights),
        np.array(selected),
        heads=4,
    )
    error = float(np.max(np.abs(np.array(output) - reference)))
    assert np.abs(reference).max() > 0.01
    assert error < 2e-5, error
    assert np.isfinite(np.array(output)).all()
    print("PASS MLX sparse attention vs NumPy, max error", error)

    chunked, chunked_selected = sparse_attention_chunked(x, *weights, heads=4, chunk_q=17)
    mx.eval(chunked, chunked_selected)
    chunk_error = float(np.max(np.abs(np.array(chunked) - np.array(output))))
    assert np.array_equal(np.array(chunked_selected), np.array(selected))
    assert chunk_error < 2e-5, chunk_error
    print("PASS chunked MLX attention, max error", chunk_error)


if __name__ == "__main__":
    test_reference()
