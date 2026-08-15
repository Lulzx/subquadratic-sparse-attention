import mlx.core as mx
import numpy as np


def hash_codes(x, projection, tables, bits):
    logits = (x @ projection).reshape(*x.shape[:-1], tables, bits)
    powers = mx.array(1 << np.arange(bits), dtype=mx.int32)
    return mx.sum((logits >= 0).astype(mx.int32) * powers, axis=-1)


def select_indices(x, projection, tables=4, bits=16, members=4, block=True):
    batch, length, _ = x.shape
    codes = hash_codes(x, projection, tables, bits)
    table = mx.arange(tables).reshape(1, 1, tables)
    sample = mx.arange(batch).reshape(batch, 1, 1)
    global_codes = (sample * tables + table) * (1 << bits) + codes
    flat_codes = global_codes.reshape(-1)
    flat_pos = (mx.arange(batch * length * tables) // tables) % length
    order = mx.argsort(flat_codes)
    inverse = mx.put_along_axis(mx.zeros_like(order), order, mx.arange(order.size), axis=0)
    current = mx.arange(batch * length * tables).reshape(batch, length, tables)
    rank = inverse[current]
    offsets = mx.arange(members).reshape(1, 1, 1, members)
    take = rank[..., None] - 1 - offsets
    valid = take >= 0
    candidate_entry = order[mx.maximum(take, 0)]
    anchor = flat_pos[candidate_entry]
    candidate_code = flat_codes[candidate_entry]
    valid = valid & (candidate_code == global_codes[..., None])
    anchor = mx.where(valid, anchor, -1)
    if not block:
        return anchor.reshape(batch, length, -1)
    successor = mx.where(anchor >= 0, mx.minimum(anchor + 1, length - 1), -1)
    return mx.stack([anchor, successor], axis=-1).reshape(batch, length, -1)
