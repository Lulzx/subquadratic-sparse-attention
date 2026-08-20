"""Compare linear routing scans with persistent fixed-budget bucket lookup."""

import argparse
import json
import math
import pathlib
import time

import mlx.core as mx
import numpy as np

from ssa.mlx_selector import probe_codes


def parse_lengths(spec):
    lengths = [int(value.strip()) for value in spec.split(",") if value.strip()]
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("lengths must be positive")
    return lengths


def per_length_query_rng(seed, length, queries):
    """Return a query RNG whose stream does not depend on other sweep lengths."""
    return np.random.default_rng(np.random.SeedSequence([seed, length, queries]))


def packed_codes(vectors, projection, chunk_size=16384):
    """Encode vectors as eight bytes and four 16-bit table addresses."""
    if projection.shape[1] != 64:
        raise ValueError("projection must produce exactly 64 bits")
    projection = np.asarray(projection, dtype=np.float32)
    powers = (1 << np.arange(8, dtype=np.uint16)).reshape(1, 1, 8)
    byte_codes = np.empty((len(vectors), 8), dtype=np.uint8)
    for start in range(0, len(vectors), chunk_size):
        end = min(start + chunk_size, len(vectors))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logits = vectors[start:end].astype(np.float32) @ projection
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("non-finite routing projection")
        bits = (logits >= 0).reshape(-1, 8, 8)
        byte_codes[start:end] = np.sum(
            bits * powers, axis=-1, dtype=np.uint16
        ).astype(np.uint8)
    table_codes = (
        byte_codes[:, 0::2].astype(np.uint16)
        | (byte_codes[:, 1::2].astype(np.uint16) << np.uint16(8))
    )
    return byte_codes, table_codes


def occupancy_summary(table_codes, capacity, address_bits):
    bucket_count = 1 << address_bits
    occupancies = []
    evictions = 0
    for table in range(table_codes.shape[1]):
        counts = np.bincount(
            table_codes[:, table].astype(np.int64), minlength=bucket_count
        )
        occupancies.append(counts[counts > 0])
        evictions += int(np.sum(np.maximum(counts - capacity, 0)))
    occupied = np.concatenate(occupancies)
    return {
        "nonempty_buckets": int(len(occupied)),
        "mean_occupancy": float(np.mean(occupied)),
        "p50_occupancy": float(np.quantile(occupied, 0.50)),
        "p90_occupancy": float(np.quantile(occupied, 0.90)),
        "p95_occupancy": float(np.quantile(occupied, 0.95)),
        "p99_occupancy": float(np.quantile(occupied, 0.99)),
        "max_occupancy": int(np.max(occupied)),
        "eviction_count": evictions,
        "evicted_fraction": evictions / table_codes.size,
    }


def secondary_table_codes(table_codes, secondary_bits):
    """Use low bits from the next independent table as a secondary address."""
    if secondary_bits < 1 or secondary_bits > 16:
        raise ValueError("secondary_bits must be between 1 and 16")
    mask = np.uint16((1 << secondary_bits) - 1)
    next_tables = np.roll(np.arange(table_codes.shape[1]), -1)
    return table_codes[:, next_tables] & mask


def hierarchical_occupancy_summary(
    table_codes, secondary_codes, capacity, address_bits, secondary_bits
):
    bucket_count = 1 << address_bits
    leaf_count = 1 << secondary_bits
    primary_occupancies = []
    leaf_occupancies = []
    evictions = 0
    for table in range(table_codes.shape[1]):
        primary = table_codes[:, table].astype(np.int64)
        secondary = secondary_codes[:, table].astype(np.int64)
        primary_counts = np.bincount(primary, minlength=bucket_count)
        leaf_counts = np.bincount(
            primary * leaf_count + secondary,
            minlength=bucket_count * leaf_count,
        )
        primary_occupancies.append(primary_counts[primary_counts > 0])
        leaf_occupancies.append(leaf_counts[leaf_counts > 0])
        evictions += int(np.sum(np.maximum(leaf_counts - capacity, 0)))
    primary = np.concatenate(primary_occupancies)
    leaves = np.concatenate(leaf_occupancies)
    return {
        "nonempty_buckets": int(len(leaves)),
        "mean_occupancy": float(np.mean(leaves)),
        "p50_occupancy": float(np.quantile(leaves, 0.50)),
        "p90_occupancy": float(np.quantile(leaves, 0.90)),
        "p95_occupancy": float(np.quantile(leaves, 0.95)),
        "p99_occupancy": float(np.quantile(leaves, 0.99)),
        "max_occupancy": int(np.max(leaves)),
        "nonempty_primary_buckets": int(len(primary)),
        "mean_primary_occupancy": float(np.mean(primary)),
        "p99_primary_occupancy": float(np.quantile(primary, 0.99)),
        "max_primary_occupancy": int(np.max(primary)),
        "eviction_count": evictions,
        "evicted_fraction": evictions / table_codes.size,
    }


def mix_uint64(values):
    values = values.astype(np.uint64, copy=True)
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def reservoir_priorities(positions, codes, table):
    values = (
        positions.astype(np.uint64)
        ^ (codes.astype(np.uint64) << np.uint64(32))
        ^ np.uint64((table + 1) * 0x9E3779B1)
    )
    return mix_uint64(values)


def fingerprint_slots(table_codes, table, capacity):
    if capacity & (capacity - 1):
        raise ValueError("fingerprint retention requires power-of-two capacity")
    full_code = np.zeros((len(table_codes),), dtype=np.uint64)
    for component in range(table_codes.shape[1]):
        full_code |= (
            table_codes[:, component].astype(np.uint64)
            << np.uint64(16 * component)
        )
    mixed = mix_uint64(
        full_code ^ np.uint64(
            ((table + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        )
    )
    return (mixed & np.uint64(capacity - 1)).astype(np.int32)


def build_bucket_index(
    table_codes, capacity, address_bits=16, retention_policy="tail"
):
    """Build a bounded direct-address index and return retention diagnostics."""
    if table_codes.ndim != 2:
        raise ValueError("table_codes must have shape [keys, tables]")
    if capacity < 1:
        raise ValueError("capacity must be positive")
    if retention_policy not in ("tail", "reservoir", "fingerprint"):
        raise ValueError(
            "retention_policy must be 'tail', 'reservoir', or 'fingerprint'"
        )
    bucket_count = 1 << address_bits
    if np.any(table_codes >= bucket_count):
        raise ValueError("table code exceeds address space")
    length, tables = table_codes.shape
    tails = np.full((tables, bucket_count, capacity), -1, dtype=np.int32)
    positions = np.arange(length, dtype=np.int32)
    for table in range(tables):
        codes = table_codes[:, table]
        if retention_policy == "tail":
            order = np.argsort(codes, kind="stable")[::-1]
        elif retention_policy == "reservoir":
            priorities = reservoir_priorities(positions, codes, table)
            order = np.lexsort((priorities, codes))
        else:
            slots = fingerprint_slots(table_codes, table, capacity)
            composite = codes.astype(np.int64) * capacity + slots
            order = np.argsort(composite, kind="stable")[::-1]
            ordered_composite = composite[order]
            first = np.concatenate((
                np.ones((1,), dtype=np.bool_),
                ordered_composite[1:] != ordered_composite[:-1],
            ))
            selected = order[first]
            tails[table, codes[selected], slots[selected]] = positions[selected]
            continue
        ordered_codes = codes[order]
        ordered_positions = positions[order]
        starts = np.concatenate((
            np.ones((1,), dtype=np.bool_),
            ordered_codes[1:] != ordered_codes[:-1],
        ))
        group_starts = np.maximum.accumulate(np.where(
            starts, np.arange(length, dtype=np.int32), 0
        ))
        ranks = np.arange(length, dtype=np.int32) - group_starts
        keep = ranks < capacity
        tails[table, ordered_codes[keep], ranks[keep]] = ordered_positions[keep]
    summary = occupancy_summary(table_codes, capacity, address_bits)
    retained = int(np.sum(tails >= 0))
    summary["eviction_count"] = table_codes.size - retained
    summary["evicted_fraction"] = summary["eviction_count"] / table_codes.size
    return tails, summary


def build_hierarchical_index(
    table_codes, capacity, address_bits=16, secondary_bits=4,
    retention_policy="reservoir",
):
    """Build primary buckets split into query-addressable secondary leaves."""
    if table_codes.ndim != 2:
        raise ValueError("table_codes must have shape [keys, tables]")
    if capacity < 1:
        raise ValueError("capacity must be positive")
    if retention_policy not in ("tail", "reservoir"):
        raise ValueError("hierarchical retention must be 'tail' or 'reservoir'")
    bucket_count = 1 << address_bits
    leaf_count = 1 << secondary_bits
    if np.any(table_codes >= bucket_count):
        raise ValueError("table code exceeds address space")
    length, tables = table_codes.shape
    secondary_codes = secondary_table_codes(table_codes, secondary_bits)
    leaves = np.full(
        (tables, bucket_count, leaf_count, capacity), -1, dtype=np.int32
    )
    positions = np.arange(length, dtype=np.int32)
    for table in range(tables):
        primary = table_codes[:, table]
        secondary = secondary_codes[:, table]
        composite = primary.astype(np.int64) * leaf_count + secondary
        if retention_policy == "tail":
            order = np.argsort(composite, kind="stable")[::-1]
        else:
            priorities = reservoir_priorities(positions, composite, table)
            order = np.lexsort((priorities, composite))
        ordered_composite = composite[order]
        ordered_positions = positions[order]
        starts = np.concatenate((
            np.ones((1,), dtype=np.bool_),
            ordered_composite[1:] != ordered_composite[:-1],
        ))
        group_starts = np.maximum.accumulate(np.where(
            starts, np.arange(length, dtype=np.int32), 0
        ))
        ranks = np.arange(length, dtype=np.int32) - group_starts
        keep = ranks < capacity
        selected = order[keep]
        leaves[
            table, primary[selected], secondary[selected], ranks[keep]
        ] = ordered_positions[keep]
    summary = hierarchical_occupancy_summary(
        table_codes, secondary_codes, capacity, address_bits, secondary_bits
    )
    retained = int(np.sum(leaves >= 0))
    summary["eviction_count"] = table_codes.size - retained
    summary["evicted_fraction"] = summary["eviction_count"] / table_codes.size
    return leaves, summary


def build_sparse_hierarchical_index(
    table_codes, capacity, address_bits=16, secondary_bits=8,
    retention_policy="reservoir", fingerprints=None, retention_scores=None,
    secondary_codes_override=None,
):
    """Build a compact posting array with a packed direct-address leaf directory."""
    if table_codes.ndim != 2:
        raise ValueError("table_codes must have shape [keys, tables]")
    if capacity < 1 or capacity > 255:
        raise ValueError("capacity must be between 1 and 255")
    if retention_policy not in ("tail", "reservoir", "diversity", "score"):
        raise ValueError(
            "sparse hierarchical retention must be tail, reservoir, or diversity"
        )
    if retention_policy == "diversity":
        if fingerprints is None or fingerprints.shape[0] != table_codes.shape[0]:
            raise ValueError("diversity retention requires one fingerprint per key")
    if retention_policy == "score":
        if retention_scores is None or len(retention_scores) != table_codes.shape[0]:
            raise ValueError("score retention requires one score per key")
    bucket_count = 1 << address_bits
    leaf_count = 1 << secondary_bits
    if np.any(table_codes >= bucket_count):
        raise ValueError("table code exceeds address space")
    length, tables = table_codes.shape
    secondary_codes = (
        secondary_table_codes(table_codes, secondary_bits)
        if secondary_codes_override is None
        else np.asarray(secondary_codes_override)
    )
    if secondary_codes.shape != table_codes.shape:
        raise ValueError("secondary code override must match table_codes shape")
    if np.any(secondary_codes >= leaf_count):
        raise ValueError("secondary code exceeds secondary address space")
    starts_directory = np.zeros(
        (tables, bucket_count * leaf_count), dtype=np.int32
    )
    counts_directory = np.zeros(
        (tables, bucket_count * leaf_count), dtype=np.uint8
    )
    positions = np.arange(length, dtype=np.int32)
    posting_parts = []
    posting_base = 0
    for table in range(tables):
        primary = table_codes[:, table]
        secondary = secondary_codes[:, table]
        composite = primary.astype(np.int64) * leaf_count + secondary
        if retention_policy == "diversity":
            leaves = {}
            for position, address in enumerate(composite):
                leaf = leaves.setdefault(int(address), [])
                leaf.append(position)
                if len(leaf) > capacity:
                    priorities = reservoir_priorities(
                        np.asarray(leaf, dtype=np.int32),
                        np.full(len(leaf), address, dtype=np.int64),
                        table,
                    )
                    drop = diversity_eviction_index(
                        leaf, fingerprints, priorities
                    )
                    leaf.pop(drop)
            kept_composite = []
            kept_positions = []
            for address in sorted(leaves):
                values = sorted(leaves[address])
                kept_composite.extend([address] * len(values))
                kept_positions.extend(values)
            kept_composite = np.asarray(kept_composite, dtype=np.int64)
            kept_positions = np.asarray(kept_positions, dtype=np.int32)
            unique, first, counts = np.unique(
                kept_composite, return_index=True, return_counts=True
            )
            starts_directory[table, unique] = posting_base + first.astype(np.int32)
            counts_directory[table, unique] = counts.astype(np.uint8)
            posting_parts.append(kept_positions)
            posting_base += len(kept_positions)
            continue
        if retention_policy == "tail":
            order = np.argsort(composite, kind="stable")[::-1]
        elif retention_policy == "score":
            order = np.lexsort((-np.asarray(retention_scores), composite))
        else:
            priorities = reservoir_priorities(positions, composite, table)
            order = np.lexsort((priorities, composite))
        ordered_composite = composite[order]
        ordered_positions = positions[order]
        starts = np.concatenate((
            np.ones((1,), dtype=np.bool_),
            ordered_composite[1:] != ordered_composite[:-1],
        ))
        group_starts = np.maximum.accumulate(np.where(
            starts, np.arange(length, dtype=np.int32), 0
        ))
        ranks = np.arange(length, dtype=np.int32) - group_starts
        keep = ranks < capacity
        kept_composite = ordered_composite[keep]
        kept_positions = ordered_positions[keep]
        unique, first, counts = np.unique(
            kept_composite, return_index=True, return_counts=True
        )
        starts_directory[table, unique] = posting_base + first.astype(np.int32)
        counts_directory[table, unique] = counts.astype(np.uint8)
        posting_parts.append(kept_positions)
        posting_base += len(kept_positions)
    postings = np.concatenate(posting_parts).astype(np.int32, copy=False)
    summary = hierarchical_occupancy_summary(
        table_codes, secondary_codes, capacity, address_bits, secondary_bits
    )
    summary["eviction_count"] = table_codes.size - len(postings)
    summary["evicted_fraction"] = summary["eviction_count"] / table_codes.size
    summary["retained_postings"] = int(len(postings))
    return starts_directory, counts_directory, postings, summary


def diversity_eviction_index(positions, fingerprints, priorities):
    """Choose the most redundant posting, breaking ties by reservoir priority."""
    positions = np.asarray(positions, dtype=np.int32)
    codes = np.asarray(fingerprints, dtype=np.uint8)[positions]
    popcount = np.array(
        [int(value).bit_count() for value in range(256)], dtype=np.uint8
    )
    distance = np.sum(
        popcount[np.bitwise_xor(codes[:, None, :], codes[None, :, :])],
        axis=-1,
    ).astype(np.int16)
    np.fill_diagonal(distance, 1_000)
    nearest = np.min(distance, axis=-1)
    redundant = np.flatnonzero(nearest == np.min(nearest))
    return int(redundant[np.argmax(np.asarray(priorities)[redundant])])


def build_bucket_tails(table_codes, members, address_bits=16):
    tails, _ = build_bucket_index(
        table_codes, members, address_bits=address_bits, retention_policy="tail"
    )
    return tails


def logical_history_bytes(
    length, dim, tables, bucket_capacity, probes=1, secondary_probes=1,
    fp_bytes=2, directory_entry_bytes=0, fingerprint_bytes=8,
):
    """Logical historical index payload read per query, excluding query encoding."""
    return {
        "fp_scan": length * dim * fp_bytes,
        "binary64_scan": length * 8,
        "bucket_lookup": (
            tables * probes * secondary_probes * bucket_capacity
            * (4 + fingerprint_bytes)
            + tables * probes * secondary_probes * directory_entry_bytes
        ),
    }


def candidate_recall(selected, reference, k):
    recalls = []
    for selected_row, reference_row in zip(selected, reference):
        selected_set = {int(value) for value in selected_row if value >= 0}
        reference_set = {int(value) for value in reference_row if value >= 0}
        recalls.append(len(selected_set & reference_set) / k)
    return float(np.mean(recalls))


def query_byte_codes(query, projection):
    return query_byte_codes_from_logits(query @ projection)


def query_byte_codes_from_logits(logits):
    if logits.shape[-1] % 8:
        raise ValueError("binary fingerprint width must be divisible by eight")
    byte_count = logits.shape[-1] // 8
    bits = (logits >= 0).reshape(logits.shape[0], byte_count, 8).astype(mx.int32)
    powers = mx.array(1 << np.arange(8), dtype=mx.int32).reshape(1, 1, 8)
    return mx.sum(bits * powers, axis=-1).astype(mx.uint8)


def fp_scan(query, keys, k):
    scores = query @ keys.T
    return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]


def binary64_scan(query, projection, key_bytes, popcount_lut, k):
    query_bytes = query_byte_codes(query, projection)
    xor = mx.bitwise_xor(query_bytes[:, None, :], key_bytes[None, :, :])
    distances = mx.sum(popcount_lut[xor.astype(mx.int32)], axis=-1)
    return mx.argpartition(distances, kth=k - 1, axis=-1)[..., :k]


def bucket_lookup(
    query, projection, tails, key_bytes, popcount_lut, tables, bits, probes, k
):
    table_codes = probe_codes(
        query, projection, tables=tables, bits=bits, probes=probes
    )
    pool = mx.stack([
        tails[table, table_codes[:, table]]
        for table in range(tables)
    ], axis=1).reshape(query.shape[0], -1)
    valid = pool >= 0
    safe_pool = mx.maximum(pool, 0)
    candidate_bytes = key_bytes[safe_pool]
    query_bytes = query_byte_codes(query, projection)
    xor = mx.bitwise_xor(query_bytes[:, None, :], candidate_bytes)
    distances = mx.sum(popcount_lut[xor.astype(mx.int32)], axis=-1)
    distances = mx.where(valid, distances, mx.array(1_000_000, distances.dtype))
    keep = mx.argpartition(distances, kth=k - 1, axis=-1)[..., :k]
    return mx.take_along_axis(pool, keep, axis=-1)


def hierarchical_query_codes(
    query, projection, tables, bits, probes, secondary_bits, secondary_probes
):
    logits = (query @ projection).reshape(query.shape[0], tables, bits)
    return hierarchical_query_codes_from_logits(
        logits, tables, bits, probes, secondary_bits, secondary_probes
    )


def hierarchical_query_codes_from_logits(
    logits, tables, bits, probes, secondary_bits, secondary_probes
):
    powers = mx.array(1 << np.arange(bits), dtype=mx.int32)
    exact = mx.sum((logits >= 0).astype(mx.int32) * powers, axis=-1)
    if probes == 1:
        primary = exact[..., None]
    else:
        uncertain = mx.argsort(mx.abs(logits), axis=-1)[..., :probes - 1]
        primary = mx.concatenate([
            exact[..., None],
            mx.bitwise_xor(exact[..., None], powers[uncertain]),
        ], axis=-1)
    secondary_logits = mx.concatenate(
        [logits[:, 1:, :secondary_bits], logits[:, :1, :secondary_bits]],
        axis=1,
    )
    secondary_powers = mx.array(
        1 << np.arange(secondary_bits), dtype=mx.int32
    )
    secondary_exact = mx.sum(
        (secondary_logits >= 0).astype(mx.int32) * secondary_powers, axis=-1
    )
    if secondary_probes == 1:
        secondary = secondary_exact[..., None]
    elif secondary_probes == secondary_bits + 1:
        secondary = mx.concatenate([
            secondary_exact[..., None],
            mx.bitwise_xor(
                secondary_exact[..., None],
                secondary_powers.reshape(1, 1, secondary_bits),
            ),
        ], axis=-1)
    else:
        secondary_uncertain = mx.argsort(
            mx.abs(secondary_logits), axis=-1
        )[..., :secondary_probes - 1]
        secondary = mx.concatenate([
            secondary_exact[..., None],
            mx.bitwise_xor(
                secondary_exact[..., None],
                secondary_powers[secondary_uncertain],
            ),
        ], axis=-1)
    return primary, secondary


def hierarchical_bucket_lookup(
    query, projection, leaves, key_bytes, popcount_lut, tables, bits, probes,
    secondary_bits, secondary_probes, k,
):
    primary, secondary = hierarchical_query_codes(
        query, projection, tables, bits, probes,
        secondary_bits, secondary_probes,
    )
    leaf_count = 1 << secondary_bits
    table_stride = leaves.shape[1] * leaf_count
    table_offsets = (
        mx.arange(tables, dtype=mx.int32).reshape(1, tables, 1, 1)
        * table_stride
    )
    addresses = (
        table_offsets
        + primary[..., :, None] * leaf_count
        + secondary[..., None, :]
    )
    pool = leaves.reshape(-1, leaves.shape[-1])[addresses].reshape(
        query.shape[0], -1
    )
    valid = pool >= 0
    safe_pool = mx.maximum(pool, 0)
    candidate_bytes = key_bytes[safe_pool]
    query_bytes = query_byte_codes(query, projection)
    xor = mx.bitwise_xor(query_bytes[:, None, :], candidate_bytes)
    distances = mx.sum(popcount_lut[xor.astype(mx.int32)], axis=-1)
    distances = mx.where(valid, distances, mx.array(1_000_000, distances.dtype))
    keep = mx.argpartition(distances, kth=k - 1, axis=-1)[..., :k]
    return mx.take_along_axis(pool, keep, axis=-1)


def sparse_hierarchical_bucket_lookup(
    query, projection, starts_directory, counts_directory, postings, key_bytes,
    popcount_lut, tables, bits, probes, secondary_bits, secondary_probes,
    capacity, k, query_positions=None, min_distance=0, sink_tokens=0,
    reranker="full-hamming", rerank_projection=None,
    rerank_confidence_power=1.0, rerank_confidence_mix=1.0,
    pq_codebook=None,
    rerank_global_weights=None,
    candidate_budget=None,
    probe_capacities=None,
    rerank_bilinear=None,
    rerank_lookup=None,
    rerank_decoder_query=None,
    rerank_decoder_keys=None,
    rerank_distance_bias=None,
    attention_query_weight=None,
    attention_query_norm=None,
    attention_key_decoder=None,
    joint_binary_attention_decoder=None,
    joint_binary_attention_decoder_hidden_weight=None,
    joint_binary_attention_decoder_hidden_bias=None,
    joint_binary_attention_decoder_output_weight=None,
    joint_binary_attention_decoder_output_bias=None,
    joint_binary_attention_head_bias_weight=None,
    joint_binary_attention_head_bias=None,
    joint_vq_attention_decoder=None,
    attention_rope_base=1_000_000.0,
    attention_scale=0.125,
    query_lookup_weight=None,
    query_lookup_bias=None,
    address_query_assignment_weight=None,
    address_query_assignment_bias=None,
    secondary_query_assignment_weight=None,
    secondary_query_assignment_bias=None,
    secondary_query_primary_bias=None,
):
    flat_query_logits = query @ projection
    if address_query_assignment_weight is None:
        primary, secondary = hierarchical_query_codes_from_logits(
            flat_query_logits.reshape(query.shape[0], tables, bits), tables,
            bits, probes, secondary_bits, secondary_probes,
        )
    else:
        categorical_logits = mx.einsum(
            "nd,dtc->ntc", query, address_query_assignment_weight
        ) + address_query_assignment_bias
        category_order = mx.argsort(-categorical_logits, axis=-1)
        primary = category_order[..., :probes]
        secondary_order = mx.concatenate(
            [category_order[:, 1:], category_order[:, :1]], axis=1
        )
        secondary = secondary_order[..., :secondary_probes]
    if secondary_query_assignment_weight is not None:
        secondary_logits = mx.einsum(
            "nd,dtc->ntc", query, secondary_query_assignment_weight
        ) + secondary_query_assignment_bias
        if secondary_query_primary_bias is not None:
            if probes != 1:
                raise ValueError(
                    "primary-conditioned secondary lookup requires one "
                    "primary probe"
                )
            primary_codes = primary[..., 0].astype(mx.int32)
            table_offsets = (
                mx.arange(tables, dtype=mx.int32).reshape(1, tables) * 256
            )
            secondary_logits = secondary_logits + mx.take(
                secondary_query_primary_bias.reshape(tables * 256, 256),
                table_offsets + primary_codes,
                axis=0,
            )
        secondary = mx.argsort(-secondary_logits, axis=-1)[
            ..., :secondary_probes
        ]
    leaf_count = 1 << secondary_bits
    table_stride = starts_directory.shape[1]
    table_offsets = (
        mx.arange(tables, dtype=mx.int32).reshape(1, tables, 1, 1)
        * table_stride
    )
    addresses = (
        table_offsets
        + primary[..., :, None] * leaf_count
        + secondary[..., None, :]
    )
    if probe_capacities is not None:
        pool, valid = sparse_static_probe_posting_pool(
            addresses, starts_directory, counts_directory, postings,
            capacity, probe_capacities,
        )
    elif candidate_budget is None:
        pool, valid = sparse_posting_pool(
            addresses, starts_directory, counts_directory, postings, capacity
        )
        pool = pool.reshape(query.shape[0], -1)
        valid = valid.reshape(query.shape[0], -1)
    else:
        pool, valid = sparse_adaptive_posting_pool(
            addresses, starts_directory, counts_directory, postings,
            capacity, candidate_budget,
        )
    if query_positions is not None:
        positions = query_positions.reshape(-1, 1)
        valid = valid & (pool < positions - min_distance) & (pool >= sink_tokens)
        pool = mx.where(valid, pool, mx.array(-1, dtype=pool.dtype))
    safe_pool = mx.maximum(pool, 0)
    candidate_bytes = key_bytes[safe_pool]
    rerank_logits = (
        flat_query_logits
        if rerank_projection is None
        else query @ rerank_projection
    )
    query_bytes = query_byte_codes_from_logits(rerank_logits)
    xor = mx.bitwise_xor(query_bytes[:, None, :], candidate_bytes)
    distance_by_table = popcount_lut[xor.astype(mx.int32)]
    if reranker == "path-hamming":
        secondary_distance = mx.concatenate(
            [distance_by_table[..., 1:], distance_by_table[..., :1]], axis=-1
        )
        distances = mx.min(distance_by_table + secondary_distance, axis=-1)
    elif reranker == "full-hamming":
        distances = mx.sum(distance_by_table, axis=-1)
    elif reranker == "confidence-hamming":
        bit_shifts = mx.arange(8, dtype=mx.uint8)
        mismatch_bits = (
            mx.right_shift(xor[..., None], bit_shifts) & mx.array(1, mx.uint8)
        ).astype(rerank_logits.dtype)
        bit_weights = mx.power(
            mx.abs(rerank_logits), rerank_confidence_power
        ).reshape(
            query.shape[0], query_bytes.shape[-1], 8
        )
        if rerank_global_weights is not None:
            bit_weights = bit_weights * rerank_global_weights.reshape(
                1, query_bytes.shape[-1], 8
            )
        bit_weights = bit_weights / mx.maximum(
            mx.mean(bit_weights, axis=(1, 2), keepdims=True),
            mx.array(1e-6, bit_weights.dtype),
        )
        bit_weights = (
            (1.0 - rerank_confidence_mix)
            + rerank_confidence_mix * bit_weights
        )
        distances = mx.sum(
            mismatch_bits * bit_weights[:, None, :, :], axis=(-1, -2)
        )
    elif reranker == "product-quantized":
        if pq_codebook is None:
            raise ValueError("product-quantized reranking requires a codebook")
        subquantizers = pq_codebook.shape[0]
        query_vectors = rerank_logits.reshape(
            query.shape[0], subquantizers, pq_codebook.shape[-1]
        )
        query_vectors = query_vectors / mx.maximum(
            mx.sqrt(mx.sum(mx.square(query_vectors), axis=-1, keepdims=True)),
            mx.array(1e-6, query_vectors.dtype),
        )
        subspaces = mx.arange(subquantizers, dtype=mx.int32).reshape(
            1, 1, subquantizers
        )
        centroids = pq_codebook[
            subspaces, candidate_bytes.astype(mx.int32)
        ]
        distances = -mx.sum(
            centroids * query_vectors[:, None, :, :], axis=(-1, -2)
        )
    elif reranker == "bilinear-code":
        if rerank_bilinear is None:
            raise ValueError("bilinear-code requires a matrix")
        bit_shifts = mx.arange(8, dtype=mx.uint8)
        query_bits = (
            (mx.right_shift(query_bytes[..., None], bit_shifts)
             & mx.array(1, mx.uint8)).astype(rerank_logits.dtype) * 2.0 - 1.0
        ).reshape(query.shape[0], -1)
        key_bits = (
            (mx.right_shift(candidate_bytes[..., None], bit_shifts)
             & mx.array(1, mx.uint8)).astype(rerank_logits.dtype) * 2.0 - 1.0
        ).reshape(query.shape[0], candidate_bytes.shape[1], -1)
        bit_weights = mx.power(
            mx.abs(rerank_logits), rerank_confidence_power
        )
        if rerank_global_weights is not None:
            bit_weights = bit_weights * rerank_global_weights.reshape(1, -1)
        bit_weights = bit_weights / mx.maximum(
            mx.mean(bit_weights, axis=-1, keepdims=True),
            mx.array(1e-6, bit_weights.dtype),
        )
        bit_weights = (
            (1.0 - rerank_confidence_mix)
            + rerank_confidence_mix * bit_weights
        )
        transformed = (query_bits * bit_weights) @ rerank_bilinear
        distances = -mx.sum(transformed[:, None, :] * key_bits, axis=-1)
    elif reranker == "lookup-code":
        if rerank_lookup is None:
            raise ValueError("lookup-code requires lookup tables")
        table_scores = []
        for table in range(query_bytes.shape[-1]):
            table_scores.append(rerank_lookup[
                table,
                query_bytes[:, None, table].astype(mx.int32),
                candidate_bytes[:, :, table].astype(mx.int32),
            ])
        distances = -mx.sum(mx.stack(table_scores, axis=-1), axis=-1)
    elif reranker == "decoder-code":
        if rerank_decoder_query is None or rerank_decoder_keys is None:
            raise ValueError("decoder-code requires decoder parameters")
        decoded_query = query @ rerank_decoder_query
        decoded_keys = mx.sum(mx.stack([
            rerank_decoder_keys[
                table, candidate_bytes[:, :, table].astype(mx.int32)
            ]
            for table in range(candidate_bytes.shape[-1])
        ], axis=-2), axis=-2)
        distances = -mx.sum(decoded_query[:, None, :] * decoded_keys, axis=-1)
    elif reranker in (
        "attention-key-decoder", "joint-binary-attention",
        "joint-binary-attention-normalized", "joint-vq-attention",
    ):
        required_decoder = (
            joint_vq_attention_decoder
            if reranker == "joint-vq-attention"
            else attention_key_decoder
            if reranker == "attention-key-decoder"
            else joint_binary_attention_decoder
        )
        if any(value is None for value in (
            attention_query_weight, attention_query_norm, required_decoder
        )):
            raise ValueError("attention-key-decoder requires decoder parameters")
        if query_positions is None:
            raise ValueError("attention-key-decoder requires query positions")

        def apply_rope(values, positions):
            dimensions = values.shape[-1]
            frequencies = mx.exp(
                -mx.arange(0, dimensions, 2, dtype=mx.float32)
                * (math.log(attention_rope_base) / dimensions)
            )
            angles = positions.astype(mx.float32)[..., None] * frequencies
            cosine = mx.cos(angles)[..., None, :]
            sine = mx.sin(angles)[..., None, :]
            first, second = mx.split(values, 2, axis=-1)
            return mx.concatenate([
                first * cosine - second * sine,
                first * sine + second * cosine,
            ], axis=-1)

        query_heads = (query @ attention_query_weight.T).reshape(
            query.shape[0], -1, attention_query_norm.shape[0]
        )
        query_heads = query_heads * mx.rsqrt(
            mx.mean(mx.square(query_heads), axis=-1, keepdims=True) + 1e-5
        ) * attention_query_norm.reshape(1, 1, -1)
        query_heads = apply_rope(query_heads, query_positions)
        if reranker in ("attention-key-decoder", "joint-vq-attention"):
            current_decoder = (
                joint_vq_attention_decoder
                if reranker == "joint-vq-attention"
                else attention_key_decoder
            )
            decoded_keys = mx.sum(mx.stack([
                current_decoder[
                    table, candidate_bytes[:, :, table].astype(mx.int32)
                ]
                for table in range(candidate_bytes.shape[-1])
            ], axis=-3), axis=-3)
            if decoded_keys.ndim == 3:
                decoded_keys = decoded_keys.reshape(
                    candidate_bytes.shape[0], candidate_bytes.shape[1],
                    -1, attention_query_norm.shape[0],
                )
        else:
            bit_shifts = mx.arange(8, dtype=mx.uint8)
            key_bits = (
                mx.right_shift(candidate_bytes[..., None], bit_shifts)
                & mx.array(1, mx.uint8)
            ).astype(query.dtype) * 2.0 - 1.0
            key_bits = key_bits.reshape(
                candidate_bytes.shape[0], candidate_bytes.shape[1], -1
            )
            decoded_flat = key_bits @ joint_binary_attention_decoder
            if joint_binary_attention_decoder_hidden_weight is not None:
                nonlinear_hidden = mx.tanh(
                    key_bits @ joint_binary_attention_decoder_hidden_weight
                    + joint_binary_attention_decoder_hidden_bias
                )
                decoded_flat = (
                    decoded_flat
                    + nonlinear_hidden
                    @ joint_binary_attention_decoder_output_weight
                    + joint_binary_attention_decoder_output_bias
                )
            decoded_keys = decoded_flat.reshape(
                candidate_bytes.shape[0], candidate_bytes.shape[1],
                joint_binary_attention_decoder.shape[-1]
                // attention_query_norm.shape[0],
                attention_query_norm.shape[0],
            )
        decoded_keys = apply_rope(decoded_keys, safe_pool)
        repeats = query_heads.shape[-2] // decoded_keys.shape[-2]
        decoded_keys = mx.repeat(decoded_keys, repeats, axis=-2)
        logits = mx.sum(
            query_heads[:, None, :, :] * decoded_keys, axis=-1
        ) * attention_scale
        if joint_binary_attention_head_bias_weight is not None:
            logits = logits + (
                query @ joint_binary_attention_head_bias_weight
                + joint_binary_attention_head_bias
            )[:, None, :]
        if reranker == "joint-binary-attention-normalized":
            masked_logits = mx.where(
                valid[..., None], logits,
                mx.array(-1e9, logits.dtype),
            )
            logits = logits - mx.logsumexp(
                masked_logits, axis=1, keepdims=True
            )
        distances = -mx.logsumexp(logits, axis=-1)
    elif reranker == "query-lookup":
        if query_lookup_weight is None or query_lookup_bias is None:
            raise ValueError("query-lookup requires query score parameters")
        lookup = (query @ query_lookup_weight + query_lookup_bias).reshape(
            query.shape[0], query_bytes.shape[-1], 256
        )
        table_scores = []
        for table in range(query_bytes.shape[-1]):
            table_scores.append(mx.take_along_axis(
                lookup[:, table, :],
                candidate_bytes[:, :, table].astype(mx.int32),
                axis=1,
            ))
        distances = -mx.sum(mx.stack(table_scores, axis=-1), axis=-1)
    else:
        raise ValueError("unsupported sparse hierarchical reranker")
    if rerank_distance_bias is not None:
        if query_positions is None:
            raise ValueError("distance bias requires query positions")
        relative = mx.maximum(query_positions.reshape(-1, 1) - safe_pool, 1)
        buckets = mx.minimum(
            mx.floor(mx.log2(relative.astype(mx.float32))).astype(mx.int32),
            rerank_distance_bias.shape[0] - 1,
        )
        distances = distances - rerank_distance_bias[buckets]
    distances = mx.where(valid, distances, mx.array(1_000_000, distances.dtype))
    keep = mx.argpartition(distances, kth=k - 1, axis=-1)[..., :k]
    return mx.take_along_axis(pool, keep, axis=-1)


def sparse_posting_pool(
    addresses, starts_directory, counts_directory, postings, capacity
):
    """Gather bounded posting lists and preserve empty slots as -1."""
    starts = starts_directory.reshape(-1)[addresses]
    counts = counts_directory.reshape(-1)[addresses].astype(mx.int32)
    slots = mx.arange(capacity, dtype=mx.int32).reshape(1, 1, 1, 1, capacity)
    posting_offsets = starts[..., None] + slots
    valid = slots < counts[..., None]
    safe_offsets = mx.minimum(posting_offsets, postings.shape[0] - 1)
    pool = postings[safe_offsets]
    pool = mx.where(valid, pool, mx.array(-1, dtype=pool.dtype))
    return pool, valid


def sparse_adaptive_posting_pool(
    addresses, starts_directory, counts_directory, postings,
    storage_capacity, candidate_budget,
):
    """Water-fill posting prefixes while gathering at most candidate_budget."""
    batch = addresses.shape[0]
    starts = starts_directory.reshape(-1)[addresses].reshape(batch, -1)
    counts = counts_directory.reshape(-1)[addresses].astype(mx.int32).reshape(
        batch, -1
    )
    rank_values = mx.arange(storage_capacity + 1, dtype=mx.int32).reshape(1, -1)
    totals = mx.sum(
        mx.minimum(counts[..., None], rank_values[:, None, :]), axis=1
    )
    base = mx.max(
        mx.where(totals <= candidate_budget, rank_values, mx.zeros_like(rank_values)),
        axis=-1,
    )
    allocation = mx.minimum(counts, base[:, None])
    remaining = candidate_budget - mx.sum(allocation, axis=-1)
    can_extend = counts > base[:, None]
    extension_rank = mx.cumsum(can_extend.astype(mx.int32), axis=-1)
    allocation = allocation + (
        can_extend & (extension_rank <= remaining[:, None])
    ).astype(mx.int32)
    slots = mx.arange(storage_capacity, dtype=mx.int32).reshape(1, 1, -1)
    valid = slots < allocation[..., None]
    offsets = starts[..., None] + slots
    flat_valid = valid.reshape(batch, -1)
    flat_offsets = offsets.reshape(batch, -1)
    packed_index = mx.cumsum(flat_valid.astype(mx.int32), axis=-1) - 1
    keep = flat_valid & (packed_index < candidate_budget)
    safe_index = mx.where(keep, packed_index, mx.zeros_like(packed_index))
    batch_index = mx.broadcast_to(
        mx.arange(batch, dtype=mx.int32).reshape(-1, 1), safe_index.shape
    )
    selected_offsets = mx.zeros(
        (batch, candidate_budget), dtype=mx.int32
    ).at[batch_index, safe_index].add(
        mx.where(keep, flat_offsets, mx.zeros_like(flat_offsets))
    )
    selected_valid = mx.zeros(
        (batch, candidate_budget), dtype=mx.int32
    ).at[batch_index, safe_index].add(keep.astype(mx.int32)).astype(mx.bool_)
    safe_offsets = mx.minimum(selected_offsets, postings.shape[0] - 1)
    pool = postings[safe_offsets]
    pool = mx.where(selected_valid, pool, mx.array(-1, dtype=pool.dtype))
    return pool, selected_valid


def sparse_static_probe_posting_pool(
    addresses, starts_directory, counts_directory, postings,
    storage_capacity, probe_capacities,
):
    """Gather fixed asymmetric prefixes for each secondary probe."""
    starts = starts_directory.reshape(-1)[addresses]
    counts = counts_directory.reshape(-1)[addresses].astype(mx.int32)
    slots = mx.arange(storage_capacity, dtype=mx.int32).reshape(1, 1, 1, 1, -1)
    valid = slots < counts[..., None]
    offsets = starts[..., None] + slots
    flat_offsets = offsets.reshape(addresses.shape[0], -1)
    flat_valid = valid.reshape(addresses.shape[0], -1)
    selected = []
    leaves = addresses.shape[1] * addresses.shape[2] * addresses.shape[3]
    secondary_probes = len(probe_capacities)
    for leaf in range(leaves):
        probe = leaf % secondary_probes
        base = leaf * storage_capacity
        selected.extend(range(base, base + probe_capacities[probe]))
    selected = mx.array(selected, dtype=mx.int32)
    selected_offsets = flat_offsets[:, selected]
    selected_valid = flat_valid[:, selected]
    safe_offsets = mx.minimum(selected_offsets, postings.shape[0] - 1)
    pool = postings[safe_offsets]
    return mx.where(
        selected_valid, pool, mx.array(-1, dtype=pool.dtype)
    ), selected_valid


def host_probe_codes(vectors, projection, tables, bits, probes):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        logits = vectors.astype(np.float32) @ np.asarray(projection, dtype=np.float32)
    if not np.all(np.isfinite(logits)):
        raise FloatingPointError("non-finite query routing projection")
    logits = logits.reshape(len(vectors), tables, bits)
    powers = (1 << np.arange(bits, dtype=np.uint32)).reshape(1, 1, bits)
    exact = np.sum((logits >= 0) * powers, axis=-1, dtype=np.uint32)
    if probes == 1:
        return exact[..., None]
    uncertain = np.argsort(np.abs(logits), axis=-1)[..., :probes - 1]
    flips = np.take_along_axis(
        np.broadcast_to(powers, logits.shape), uncertain, axis=-1
    )
    return np.concatenate([exact[..., None], exact[..., None] ^ flips], axis=-1)


def host_hierarchical_probe_codes(
    vectors, projection, tables, bits, probes, secondary_bits, secondary_probes
):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        logits = vectors.astype(np.float32) @ np.asarray(projection, dtype=np.float32)
    if not np.all(np.isfinite(logits)):
        raise FloatingPointError("non-finite query routing projection")
    logits = logits.reshape(len(vectors), tables, bits)
    powers = (1 << np.arange(bits, dtype=np.uint32)).reshape(1, 1, bits)
    exact = np.sum((logits >= 0) * powers, axis=-1, dtype=np.uint32)
    if probes == 1:
        primary = exact[..., None]
    else:
        uncertain = np.argsort(np.abs(logits), axis=-1)[..., :probes - 1]
        flips = np.take_along_axis(
            np.broadcast_to(powers, logits.shape), uncertain, axis=-1
        )
        primary = np.concatenate([exact[..., None], exact[..., None] ^ flips], axis=-1)
    next_tables = np.roll(np.arange(tables), -1)
    secondary_logits = logits[:, next_tables, :secondary_bits]
    secondary_powers = (
        1 << np.arange(secondary_bits, dtype=np.uint32)
    ).reshape(1, 1, secondary_bits)
    secondary_exact = np.sum(
        (secondary_logits >= 0) * secondary_powers, axis=-1, dtype=np.uint32
    )
    if secondary_probes == 1:
        secondary = secondary_exact[..., None]
    elif secondary_probes == secondary_bits + 1:
        secondary = np.concatenate([
            secondary_exact[..., None],
            secondary_exact[..., None] ^ secondary_powers,
        ], axis=-1)
    else:
        uncertain = np.argsort(
            np.abs(secondary_logits), axis=-1
        )[..., :secondary_probes - 1]
        flips = np.take_along_axis(
            np.broadcast_to(secondary_powers, secondary_logits.shape),
            uncertain,
            axis=-1,
        )
        secondary = np.concatenate([
            secondary_exact[..., None], secondary_exact[..., None] ^ flips,
        ], axis=-1)
    return primary, secondary


def timed_queries(call, query, warmups, repeats):
    for index in range(warmups):
        output = call(query[index % query.shape[0]:index % query.shape[0] + 1])
        mx.eval(output)
        mx.synchronize()
    samples_us = []
    for _ in range(repeats):
        for index in range(query.shape[0]):
            started = time.perf_counter()
            output = call(query[index:index + 1])
            mx.eval(output)
            mx.synchronize()
            samples_us.append((time.perf_counter() - started) * 1_000_000.0)
    return {
        "median_routing_us_per_query": float(np.median(samples_us)),
        "p10_routing_us_per_query": float(np.quantile(samples_us, 0.1)),
        "p90_routing_us_per_query": float(np.quantile(samples_us, 0.9)),
        "timed_queries": len(samples_us),
        "samples_us": samples_us,
    }


def batched_output(call, query, batch_size):
    outputs = []
    for start in range(0, query.shape[0], batch_size):
        output = call(query[start:start + batch_size])
        mx.eval(output)
        outputs.append(np.array(output).astype(np.int64))
    return np.concatenate(outputs, axis=0)


def normalized_rows(values):
    values = np.asarray(values, dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values


def make_normalized_keys(length, dim, rng, chunk_size=65536):
    keys = np.empty((length, dim), dtype=np.float16)
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        values = rng.standard_normal((end - start, dim), dtype=np.float32)
        keys[start:end] = normalized_rows(values).astype(np.float16)
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="16384,32768,65536,131072,262144")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--tables", type=int, default=4)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--probes", type=int, default=2)
    parser.add_argument("--bucket-capacity", type=int, default=16)
    parser.add_argument(
        "--index-kind",
        choices=("flat", "hierarchical", "sparse-hierarchical"),
        default="flat",
    )
    parser.add_argument("--secondary-bits", type=int, default=4)
    parser.add_argument("--secondary-probes", type=int, default=4)
    parser.add_argument(
        "--retention-policy",
        choices=("tail", "reservoir", "fingerprint"),
        default="tail",
    )
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--recall-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--memory-limit-mb", type=int, default=1792)
    parser.add_argument("--cache-limit-mb", type=int, default=256)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    lengths = parse_lengths(args.lengths)
    if args.dim < 1 or args.queries < 1 or args.k < 1 or args.recall_batch < 1:
        parser.error("--dim, --queries, --k, and --recall-batch must be positive")
    if args.tables * args.bits != 64:
        parser.error("--tables * --bits must equal 64")
    if args.probes < 1 or args.probes > args.bits + 1:
        parser.error("--probes must be between 1 and bits + 1")
    if args.secondary_bits < 1 or args.secondary_bits > args.bits:
        parser.error("--secondary-bits must be between 1 and --bits")
    if args.secondary_probes < 1 or args.secondary_probes > args.secondary_bits + 1:
        parser.error("--secondary-probes must be between 1 and secondary bits + 1")
    effective_secondary_probes = (
        args.secondary_probes if args.index_kind != "flat" else 1
    )
    if (
        args.tables * args.probes * effective_secondary_probes
        * args.bucket_capacity < args.k
    ):
        parser.error("addressed candidate pool must contain at least K slots")
    if args.index_kind != "flat" and args.retention_policy == "fingerprint":
        parser.error("hierarchical index does not support fingerprint retention")
    if max(lengths) < args.k:
        parser.error("every length must be at least K")

    mx.set_memory_limit(args.memory_limit_mb * 2**20)
    mx.set_cache_limit(args.cache_limit_mb * 2**20)
    key_rng = np.random.default_rng(args.seed)
    projection_rng = np.random.default_rng(args.seed + 1)
    max_length = max(lengths)
    keys_np = make_normalized_keys(max_length, args.dim, key_rng)
    projection_np = (
        projection_rng.standard_normal((args.dim, 64), dtype=np.float32)
        / np.float32(np.sqrt(args.dim))
    ).astype(np.float32)
    # Match the float16 representation used by Metal while keeping deterministic
    # host-side index construction outside the timed lookup path.
    routing_projection_np = projection_np.astype(np.float16).astype(np.float32)
    key_bytes_np, table_codes_np = packed_codes(
        keys_np, routing_projection_np
    )
    popcount = np.array([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    projection = mx.array(routing_projection_np.astype(np.float16))
    popcount_lut = mx.array(popcount)
    mx.eval(projection, popcount_lut)

    report = {
        "format_version": 1,
        "methodology": {
            "timed_scope": "query encoding plus index lookup/scan and candidate selection",
            "excluded_scope": "index build, exact KV gather, attention, and model execution",
            "bytes_definition": "logical historical index payload read per query",
            "fp_dtype": "float16",
            "binary_code_bits": 64,
            "bucket_index": (
                "persistent hierarchical direct-address leaves followed by bounded "
                "Hamming rerank"
                if args.index_kind != "flat"
                else "persistent direct-address bounded postings followed by Hamming rerank"
            ),
        },
        "config": vars(args) | {"output": str(args.output) if args.output else ""},
        "results": [],
    }
    for length in lengths:
        query_rng = per_length_query_rng(args.seed, length, args.queries)
        targets = query_rng.choice(length, size=args.queries, replace=False)
        noise = query_rng.standard_normal(
            (args.queries, args.dim), dtype=np.float32
        )
        queries_np = normalized_rows(keys_np[targets] + args.noise * noise)
        if args.index_kind != "flat":
            query_primary_np, query_secondary_np = host_hierarchical_probe_codes(
                queries_np.astype(np.float16), routing_projection_np,
                args.tables, args.bits, args.probes,
                args.secondary_bits, args.secondary_probes,
            )
            target_secondary_np = secondary_table_codes(
                table_codes_np[targets], args.secondary_bits
            )
            primary_match = (
                query_primary_np == table_codes_np[targets, :, None]
            )
            secondary_match = (
                query_secondary_np == target_secondary_np[:, :, None]
            )
            target_probe_address_match = np.any(
                primary_match[..., :, None] & secondary_match[..., None, :],
                axis=(1, 2, 3),
            )
        else:
            query_probe_codes_np = host_probe_codes(
                queries_np.astype(np.float16), routing_projection_np,
                args.tables, args.bits, args.probes,
            )
            target_probe_address_match = np.any(
                query_probe_codes_np == table_codes_np[targets, :, None], axis=(1, 2)
            )
        build_started = time.perf_counter()
        if args.index_kind == "sparse-hierarchical":
            starts_np, counts_np, postings_np, occupancy = (
                build_sparse_hierarchical_index(
                table_codes_np[:length], args.bucket_capacity,
                address_bits=args.bits, secondary_bits=args.secondary_bits,
                retention_policy=args.retention_policy,
                )
            )
            tails_np = None
        elif args.index_kind == "hierarchical":
            tails_np, occupancy = build_hierarchical_index(
                table_codes_np[:length], args.bucket_capacity,
                address_bits=args.bits, secondary_bits=args.secondary_bits,
                retention_policy=args.retention_policy,
            )
        else:
            tails_np, occupancy = build_bucket_index(
                table_codes_np[:length], args.bucket_capacity,
                address_bits=args.bits,
                retention_policy=args.retention_policy,
            )
        build_ms = (time.perf_counter() - build_started) * 1000.0

        keys = mx.array(keys_np[:length])
        query = mx.array(queries_np.astype(np.float16))
        key_bytes = mx.array(key_bytes_np[:length])
        if args.index_kind == "sparse-hierarchical":
            starts_directory = mx.array(starts_np)
            counts_directory = mx.array(counts_np)
            postings = mx.array(postings_np)
            mx.eval(
                keys, query, key_bytes, starts_directory, counts_directory, postings
            )
        else:
            tails = mx.array(tails_np)
            mx.eval(keys, query, key_bytes, tails)

        if args.index_kind == "sparse-hierarchical":
            addressed_call = lambda current: sparse_hierarchical_bucket_lookup(
                current, projection, starts_directory, counts_directory, postings,
                key_bytes, popcount_lut, args.tables, args.bits, args.probes,
                args.secondary_bits, args.secondary_probes,
                args.bucket_capacity, args.k,
            )
        elif args.index_kind == "hierarchical":
            addressed_call = lambda current: hierarchical_bucket_lookup(
                current, projection, tails, key_bytes, popcount_lut,
                args.tables, args.bits, args.probes, args.secondary_bits,
                args.secondary_probes, args.k,
            )
        else:
            addressed_call = lambda current: bucket_lookup(
                current, projection, tails, key_bytes, popcount_lut,
                args.tables, args.bits, args.probes, args.k,
            )
        methods = (
            ("fp_scan", lambda current: fp_scan(current, keys, args.k)),
            ("binary64_scan", lambda current: binary64_scan(
                current, projection, key_bytes, popcount_lut, args.k
            )),
            ("bucket_lookup", addressed_call),
        )
        outputs = {}
        timing = {}
        for name, call in methods:
            outputs[name] = batched_output(call, query, args.recall_batch)
            timing[name] = timed_queries(call, query, args.warmups, args.repeats)
        reference = outputs["fp_scan"]
        byte_counts = logical_history_bytes(
            length, args.dim, args.tables, args.bucket_capacity,
            probes=args.probes, secondary_probes=effective_secondary_probes,
            directory_entry_bytes=(
                5 if args.index_kind == "sparse-hierarchical" else 0
            ),
        )
        if args.index_kind == "sparse-hierarchical":
            resident_bytes = int(
                starts_np.nbytes + counts_np.nbytes + postings_np.nbytes
            )
        else:
            resident_bytes = int(tails_np.nbytes)
        row = {
            "length": length,
            "queries": args.queries,
            "k": args.k,
            "bucket_index_build_ms": build_ms,
            "bucket_index_resident_bytes": resident_bytes,
            "bucket_occupancy": occupancy,
            "query_target_probe_address_match": float(
                np.mean(target_probe_address_match)
            ),
            "methods": {},
        }
        for name, _ in methods:
            selected = outputs[name]
            row["methods"][name] = timing[name] | {
                "logical_history_bytes_per_query": byte_counts[name],
                "candidate_recall_at_k": candidate_recall(selected, reference, args.k),
                "needle_recall": float(np.mean([
                    target in candidates
                    for target, candidates in zip(targets, selected)
                ])),
                "mean_unique_candidates": float(np.mean([
                    len({int(value) for value in candidates if value >= 0})
                    for candidates in selected
                ])),
            }
        report["results"].append(row)
        print(json.dumps(row), flush=True)
        del keys, query, key_bytes
        if args.index_kind == "sparse-hierarchical":
            del starts_directory, counts_directory, postings
            del starts_np, counts_np, postings_np
        else:
            del tails
        mx.clear_cache()

    report["peak_memory_mb"] = mx.get_peak_memory() / 2**20
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output) if args.output else None,
        "peak_memory_mb": report["peak_memory_mb"],
    }), flush=True)


if __name__ == "__main__":
    main()
