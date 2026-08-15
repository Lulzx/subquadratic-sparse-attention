import math

import mlx.core as mx

from .mlx_selector import select_indices


def apply_rope(x, positions, base=50000.0):
    head_dim = x.shape[-1]
    inv = 1.0 / (base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    angles = positions.astype(mx.float32)[..., None] * inv
    while angles.ndim < x.ndim:
        angles = mx.expand_dims(angles, axis=-2)
    cosine, sine = mx.cos(angles), mx.sin(angles)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = mx.stack([even * cosine - odd * sine, even * sine + odd * cosine], axis=-1)
    return rotated.reshape(x.shape).astype(x.dtype)


def attend_selected(q, k, v, selected, query_start=0, rope_base=50000.0):
    batch, query_length, heads, head_dim = q.shape
    length = k.shape[1]
    valid = selected >= 0
    selected_safe = mx.maximum(selected, 0)
    offsets = mx.arange(batch).reshape(batch, 1, 1) * length
    global_indices = selected_safe + offsets
    gathered_k = k.reshape(batch * length, heads, head_dim)[global_indices]
    gathered_v = v.reshape(batch * length, heads, head_dim)[global_indices]
    q_positions = mx.arange(query_start, query_start + query_length).reshape(1, query_length)
    q = apply_rope(q, q_positions, rope_base)
    gathered_k = apply_rope(gathered_k, selected_safe, rope_base)
    scores = mx.sum(q[:, :, None, :, :] * gathered_k, axis=-1) / math.sqrt(head_dim)
    scores = mx.where(valid[..., None], scores, mx.array(-1e9, dtype=scores.dtype))
    probabilities = mx.softmax(scores.astype(mx.float32), axis=2).astype(q.dtype)
    probabilities = mx.where(valid[..., None], probabilities, mx.zeros_like(probabilities))
    probabilities = probabilities / mx.maximum(
        mx.sum(probabilities, axis=2, keepdims=True), mx.array(1e-9, dtype=probabilities.dtype)
    )
    return mx.sum(probabilities[..., None] * gathered_v, axis=2)


def sparse_attention(x, wq, wk, wv, wo, hash_projection, heads=4, tables=4, bits=16,
                     members=4, probes=1, rope_base=50000.0):
    batch, length, width = x.shape
    if width % heads:
        raise ValueError("width must be divisible by heads")
    head_dim = width // heads
    q = (x @ wq).reshape(batch, length, heads, head_dim)
    k = (x @ wk).reshape(batch, length, heads, head_dim)
    v = (x @ wv).reshape(batch, length, heads, head_dim)
    selected = select_indices(
        x, hash_projection, tables=tables, bits=bits, members=members, probes=probes
    )
    output = attend_selected(q, k, v, selected, rope_base=rope_base)
    return output.reshape(batch, length, width) @ wo, selected


def sparse_attention_chunked(x, wq, wk, wv, wo, hash_projection, heads=4, tables=4, bits=16,
                             members=4, probes=1, rope_base=50000.0, chunk_q=1024):
    batch, length, width = x.shape
    if width % heads:
        raise ValueError("width must be divisible by heads")
    head_dim = width // heads
    q = (x @ wq).reshape(batch, length, heads, head_dim)
    k = (x @ wk).reshape(batch, length, heads, head_dim)
    v = (x @ wv).reshape(batch, length, heads, head_dim)
    selected = select_indices(
        x, hash_projection, tables=tables, bits=bits, members=members, probes=probes
    )
    mx.eval(q, k, v, selected)
    chunks = []
    for start in range(0, length, chunk_q):
        end = min(start + chunk_q, length)
        output = attend_selected(
            q[:, start:end], k, v, selected[:, start:end], query_start=start, rope_base=rope_base
        )
        output = output.reshape(batch, end - start, width) @ wo
        mx.eval(output)
        chunks.append(output)
    return mx.concatenate(chunks, axis=1), selected


def random_weights(width, tables=4, bits=16, dtype=mx.float16, seed=0):
    mx.random.seed(seed)
    scale = 1.0 / math.sqrt(width)
    matrices = [mx.random.normal((width, width)).astype(dtype) * scale for _ in range(4)]
    projection = mx.random.normal((width, tables * bits)).astype(dtype) * scale
    return (*matrices, projection)
