import mlx.core as mx
import numpy as np


def hash_logits(x, projection, tables, bits):
    return (x @ projection).reshape(*x.shape[:-1], tables, bits)


def hash_codes(x, projection, tables, bits):
    logits = hash_logits(x, projection, tables, bits)
    powers = mx.array(1 << np.arange(bits), dtype=mx.int32)
    return mx.sum((logits >= 0).astype(mx.int32) * powers, axis=-1)


def probe_codes(x, projection, tables, bits, probes=1):
    """Return the exact SimHash code and its lowest-margin one-bit neighbors."""
    if bits < 1 or bits > 30:
        raise ValueError("bits must be between 1 and 30 for int32 hash codes")
    if probes < 1 or probes > bits + 1:
        raise ValueError("probes must be between 1 and bits + 1")
    logits = hash_logits(x, projection, tables, bits)
    powers = mx.array(1 << np.arange(bits), dtype=mx.int32)
    exact = mx.sum((logits >= 0).astype(mx.int32) * powers, axis=-1)
    if probes == 1:
        return exact[..., None]
    uncertain_bits = mx.argsort(mx.abs(logits), axis=-1)[..., : probes - 1]
    flip_masks = powers[uncertain_bits]
    neighbors = mx.bitwise_xor(exact[..., None], flip_masks)
    return mx.concatenate([exact[..., None], neighbors], axis=-1)


def select_indices(x, projection, tables=4, bits=16, members=4, probes=1, block=True):
    batch, length, _ = x.shape
    codes = probe_codes(x, projection, tables, bits, probes)
    table = mx.arange(tables).reshape(1, 1, tables, 1)
    sample = mx.arange(batch).reshape(batch, 1, 1)
    global_codes = (sample[..., None] * tables + table) * (1 << bits) + codes
    flat_codes = global_codes.reshape(-1)
    entries = batch * length * tables * probes
    flat_pos = (mx.arange(entries) // (tables * probes)) % length
    order = mx.argsort(flat_codes)
    inverse = mx.put_along_axis(mx.zeros_like(order), order, mx.arange(order.size), axis=0)
    current = mx.arange(entries).reshape(batch, length, tables, probes)
    rank = inverse[current]
    offsets = mx.arange(members).reshape(1, 1, 1, 1, members)
    take = rank[..., None] - 1 - offsets
    valid = take >= 0
    candidate_entry = order[mx.maximum(take, 0)]
    anchor = flat_pos[candidate_entry]
    candidate_code = flat_codes[candidate_entry]
    valid = valid & (candidate_code == global_codes[..., None])
    query_pos = mx.arange(length).reshape(1, length, 1, 1, 1)
    valid = valid & (anchor < query_pos)
    anchor = mx.where(valid, anchor, -1)
    if not block:
        return anchor.reshape(batch, length, -1)
    successor = mx.where(anchor >= 0, mx.minimum(anchor + 1, length - 1), -1)
    return mx.stack([anchor, successor], axis=-1).reshape(batch, length, -1)
