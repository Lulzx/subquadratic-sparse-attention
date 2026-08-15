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
    # Store each key once under its exact code and multiprobe only the query.
    # Besides matching conventional LSH lookup, the merged query/key index uses
    # position in its sort key, so future tokens cannot displace earlier members.
    return select_indices_qk(
        x, x, projection, projection, tables=tables, bits=bits,
        members=members, probes=probes, block=block,
    )


def select_indices_qk(query, key, query_projection, key_projection, tables=4, bits=16,
                      members=4, probes=1, block=True, min_distance=0):
    """Select causal keys with independently trainable query and key hashes.

    Query entries are inserted before key entries at the same position in the
    temporary sorted index. The prefix key count at a query therefore points to
    strictly earlier keys in the same bucket. This preserves causal semantics
    without materializing a query-by-key score matrix.
    """
    if query.shape != key.shape:
        raise ValueError("query and key router inputs must have the same shape")
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative")
    batch, length, _ = query.shape
    query_codes = probe_codes(query, query_projection, tables, bits, probes)
    key_codes = hash_codes(key, key_projection, tables, bits)
    table = mx.arange(tables).reshape(1, 1, tables)
    sample = mx.arange(batch).reshape(batch, 1, 1)
    query_buckets = (sample[..., None] * tables + table[..., None]) * (1 << bits) + query_codes
    key_buckets = (sample * tables + table) * (1 << bits) + key_codes

    key_entries = batch * length * tables
    query_entries = batch * length * tables * probes
    key_pos = (mx.arange(key_entries) // tables) % length
    query_pos = (mx.arange(query_entries) // (tables * probes)) % length
    stride = 2 * length + 1
    # Queries sort before keys at equal positions so a token cannot retrieve itself.
    key_sort_values = key_buckets.reshape(-1) * stride + 2 * key_pos + 1
    query_cutoff = mx.maximum(query_pos - min_distance, 0)
    query_sort_values = query_buckets.reshape(-1) * stride + 2 * query_cutoff
    combined = mx.concatenate([key_sort_values, query_sort_values])
    order = mx.argsort(combined)
    inverse = mx.put_along_axis(mx.zeros_like(order), order, mx.arange(order.size), axis=0)
    is_key = order < key_entries
    key_count = mx.cumsum(is_key.astype(mx.int32), axis=0)
    query_rank = inverse[key_entries:]  # location of each query in the combined ordering
    keys_before = key_count[query_rank]
    key_order = mx.argsort(key_sort_values)

    offsets = mx.arange(members).reshape(1, members)
    take = keys_before.reshape(-1, 1) - 1 - offsets
    valid = take >= 0
    candidate_entry = key_order[mx.maximum(take, 0)]
    anchor = key_pos[candidate_entry]
    candidate_bucket = key_buckets.reshape(-1)[candidate_entry]
    valid = valid & (candidate_bucket == query_buckets.reshape(-1, 1))
    valid = valid & (anchor < (query_pos - min_distance).reshape(-1, 1))
    anchor = mx.where(valid, anchor, -1).reshape(batch, length, tables, probes, members)
    if not block:
        return anchor.reshape(batch, length, -1)
    successor = mx.where(anchor >= 0, mx.minimum(anchor + 1, length - 1), -1)
    return mx.stack([anchor, successor], axis=-1).reshape(batch, length, -1)


def select_block_indices_qk(
    query,
    block_key,
    query_projection,
    key_projection,
    block_size,
    context_length,
    tables=4,
    bits=16,
    members=1,
    probes=1,
    min_distance=0,
):
    """Select completed causal blocks without query-by-block scoring."""
    if query.shape[0] != block_key.shape[0] or query.shape[2] != block_key.shape[2]:
        raise ValueError("query and block keys must share batch and hidden dimensions")
    if block_size < 1 or context_length < 1:
        raise ValueError("block_size and context_length must be positive")
    batch, length, _ = query.shape
    block_count = block_key.shape[1]
    query_codes = probe_codes(query, query_projection, tables, bits, probes)
    key_codes = hash_codes(block_key, key_projection, tables, bits)
    table = mx.arange(tables).reshape(1, 1, tables)
    sample = mx.arange(batch).reshape(batch, 1, 1)
    query_buckets = (
        (sample[..., None] * tables + table[..., None]) * (1 << bits)
        + query_codes
    )
    key_buckets = (sample * tables + table) * (1 << bits) + key_codes

    key_entries = batch * block_count * tables
    query_entries = batch * length * tables * probes
    key_block = (mx.arange(key_entries) // tables) % block_count
    key_end = mx.minimum((key_block + 1) * block_size, context_length) - 1
    query_pos = (mx.arange(query_entries) // (tables * probes)) % length
    stride = 2 * context_length + 1
    key_sort_values = key_buckets.reshape(-1) * stride + 2 * key_end + 1
    query_cutoff = mx.maximum(query_pos - min_distance, 0)
    query_sort_values = query_buckets.reshape(-1) * stride + 2 * query_cutoff
    combined = mx.concatenate([key_sort_values, query_sort_values])
    order = mx.argsort(combined)
    inverse = mx.put_along_axis(
        mx.zeros_like(order), order, mx.arange(order.size), axis=0
    )
    is_key = order < key_entries
    key_count = mx.cumsum(is_key.astype(mx.int32), axis=0)
    query_rank = inverse[key_entries:]
    keys_before = key_count[query_rank]
    key_order = mx.argsort(key_sort_values)

    offsets = mx.arange(members).reshape(1, members)
    take = keys_before.reshape(-1, 1) - 1 - offsets
    valid = take >= 0
    candidate_entry = key_order[mx.maximum(take, 0)]
    candidate_block = key_block[candidate_entry]
    candidate_bucket = key_buckets.reshape(-1)[candidate_entry]
    candidate_end = key_end[candidate_entry]
    valid = valid & (candidate_bucket == query_buckets.reshape(-1, 1))
    valid = valid & (candidate_end < (query_pos - min_distance).reshape(-1, 1))
    anchor = candidate_block * block_size
    anchor = mx.where(valid, anchor, -1)
    return anchor.reshape(batch, length, tables * probes * members)
