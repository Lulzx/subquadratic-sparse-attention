"""Evaluate sparse hierarchical routing on real LFM2.5 attention states."""

import argparse
import hashlib
import json
import math
import pathlib
import platform
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from mlx_donor_router import (
    DonorHashRouter,
    HierarchicalAttentionRouter,
    language_body,
)
from mlx_lfm_quality_eval import pg19_tokens
from mlx_lfm_replacement import wikitext_tokens
from mlx_routing_scan_bench import (
    batched_output,
    build_sparse_hierarchical_index,
    diversity_eviction_index,
    logical_history_bytes,
    reservoir_priorities,
    sparse_hierarchical_bucket_lookup,
)


def parse_ints(spec):
    values = [int(value.strip()) for value in spec.split(",") if value.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("integer list must contain positive values")
    return values


def parse_paths(spec):
    paths = [pathlib.Path(value.strip()) for value in spec.split(",") if value.strip()]
    if not paths:
        raise ValueError("at least one router checkpoint is required")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"router checkpoints do not exist: {missing}")
    return paths


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attention_qkv(attention, x):
    batch, length, _ = x.shape
    heads = getattr(
        attention, "n_heads", getattr(attention, "num_attention_heads", None)
    )
    kv_heads = getattr(
        attention, "n_kv_heads",
        getattr(attention, "num_key_value_heads", None),
    )
    head_dim = getattr(attention, "head_dim", x.shape[-1] // heads)
    projected_query = attention.q_proj(x)
    if projected_query.shape[-1] == heads * head_dim * 2:
        queries, _ = mx.split(
            projected_query.reshape(batch, length, heads, -1), 2, axis=-1
        )
    else:
        queries = projected_query.reshape(batch, length, heads, head_dim)
    keys = attention.k_proj(x).reshape(batch, length, kv_heads, head_dim)
    values = attention.v_proj(x).reshape(batch, length, kv_heads, head_dim)
    query_norm = getattr(
        attention, "q_norm", getattr(attention, "q_layernorm", None)
    )
    key_norm = getattr(
        attention, "k_norm", getattr(attention, "k_layernorm", None)
    )
    if query_norm is not None:
        queries = query_norm(queries)
    if key_norm is not None:
        keys = key_norm(keys)
    queries = attention.rope(queries.transpose(0, 2, 1, 3))
    keys = attention.rope(keys.transpose(0, 2, 1, 3))
    values = values.transpose(0, 2, 1, 3)
    if heads != kv_heads:
        repeats = heads // kv_heads
        keys = mx.repeat(keys, repeats, axis=1)
        values = mx.repeat(values, repeats, axis=1)
    return queries, keys, values


def capture_layer(model, tokens, layer_index):
    body = language_body(model)
    h = body.embed_tokens(tokens)
    attention_mask = create_attention_mask(h)
    state_mask = create_ssm_mask(h)
    for layer in body.layers[:layer_index]:
        mask = attention_mask if hasattr(layer, "self_attn") else state_mask
        h = layer(h, mask)
    layer = body.layers[layer_index]
    if not hasattr(layer, "self_attn"):
        raise ValueError(f"layer {layer_index} is not a full-attention layer")
    norm = getattr(layer, "operator_norm", getattr(layer, "input_layernorm", None))
    if norm is None:
        raise ValueError(f"layer {layer_index} has no supported input norm")
    x = norm(h)
    queries, keys, values = attention_qkv(layer.self_attn, x)
    dense_attention = layer.self_attn(x, mask=attention_mask)
    mx.eval(h, x, queries, keys, values, dense_attention)
    return {
        "body": body,
        "layer": layer,
        "h": h,
        "x": x,
        "queries": queries,
        "keys": keys,
        "values": values,
        "dense_attention": dense_attention,
        "attention_mask": attention_mask,
        "state_mask": state_mask,
    }


def router_codes(hidden, projection, tables, bits, bias=None):
    hidden_mx = mx.array(hidden.astype(np.float16))
    projection_mx = mx.array(projection.astype(np.float16))
    logits_mx = hidden_mx @ projection_mx
    mx.eval(logits_mx)
    logits = np.array(logits_mx.astype(mx.float32))
    logits = logits.reshape(len(hidden), tables, bits)
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float32)
        if bias.shape != (tables, bits):
            raise ValueError("address bias must have shape [tables, bits]")
        logits = logits + bias[None]
    powers = (1 << np.arange(bits, dtype=np.uint16)).reshape(1, 1, bits)
    codes = np.sum((logits >= 0) * powers, axis=-1, dtype=np.uint16)
    if tables * bits != 64 or bits != 8:
        raise ValueError("the current fingerprint path requires eight 8-bit tables")
    return logits, codes, codes.astype(np.uint8)


def categorical_address_codes(hidden, weight, bias):
    hidden_mx = mx.array(hidden.astype(np.float16))
    weight_mx = mx.array(weight.astype(np.float16))
    bias_mx = mx.array(bias.astype(np.float16))
    logits_mx = mx.einsum("nd,dtc->ntc", hidden_mx, weight_mx) + bias_mx
    mx.eval(logits_mx)
    logits = np.array(logits_mx.astype(mx.float32))
    if logits.ndim != 3 or logits.shape[-1] != 256:
        raise ValueError(
            "categorical address logits must have shape [tokens, tables, 256]"
        )
    codes = np.argmax(logits, axis=-1).astype(np.uint8)
    return logits, codes.astype(np.uint16), codes


def primary_conditioned_categorical_address_codes(
    hidden, weight, bias, primary_projection, primary_bias,
):
    """Categorical secondary codes with a per-primary-region logit bias."""
    logits, _, _ = categorical_address_codes(hidden, weight, bias)
    _, primary_codes, _ = router_codes(
        hidden, primary_projection, logits.shape[1], 8
    )
    primary_bias = np.asarray(primary_bias, dtype=np.float32)
    expected = (logits.shape[1], 256, 256)
    if primary_bias.shape != expected:
        raise ValueError(
            f"primary-conditioned secondary bias must have shape {expected}"
        )
    offsets = (
        np.arange(logits.shape[1], dtype=np.int64)[None, :] * 256
        + primary_codes.astype(np.int64)
    )
    logits = logits + np.take(
        primary_bias.reshape(logits.shape[1] * 256, 256), offsets, axis=0
    )
    codes = np.argmax(logits, axis=-1).astype(np.uint8)
    return logits, codes.astype(np.uint16), codes


def binary_fingerprint_bytes(hidden, projection):
    hidden_mx = mx.array(hidden.astype(np.float16))
    projection_mx = mx.array(projection.astype(np.float16))
    logits_mx = hidden_mx @ projection_mx
    mx.eval(logits_mx)
    logits = np.array(logits_mx.astype(mx.float32))
    if logits.shape[-1] % 8:
        raise ValueError("fingerprint projection width must be divisible by eight")
    bits = logits.reshape(len(hidden), -1, 8) >= 0
    powers = (1 << np.arange(8, dtype=np.uint8)).reshape(1, 1, 8)
    return np.sum(bits * powers, axis=-1, dtype=np.uint8)


def pq_query_vectors(hidden, projection, codebook):
    subquantizers, _, subspace_width = codebook.shape
    logits = np.einsum(
        "nd,df->nf", hidden.astype(np.float64),
        projection.astype(np.float64), optimize=False,
    ).astype(np.float32).reshape(len(hidden), subquantizers, subspace_width)
    return logits / np.maximum(
        np.linalg.norm(logits, axis=-1, keepdims=True), 1e-6
    )


def pq_key_codes(hidden, projection, codebook):
    vectors = pq_query_vectors(hidden, projection, codebook)
    scores = np.einsum("nmd,mcd->nmc", vectors, codebook, optimize=True)
    return np.argmax(scores, axis=-1).astype(np.uint8)


def retention_score_values(hidden, projection):
    scores = np.sum(
        np.asarray(hidden, dtype=np.float64)
        * np.asarray(projection, dtype=np.float64)[None, :],
        axis=-1,
    )
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("non-finite learned retention score")
    return scores


def secondary_probe_codes(logits, table, probes):
    next_table = (table + 1) % logits.shape[1]
    values = logits[:, next_table]
    if values.shape[-1] == 256:
        return np.argsort(-values, axis=-1)[..., :probes].astype(np.uint16)
    powers = 1 << np.arange(values.shape[-1], dtype=np.uint16)
    exact = np.sum((values >= 0) * powers, axis=-1, dtype=np.uint16)
    if probes == 1:
        return exact[:, None]
    uncertain = np.argsort(np.abs(values), axis=-1)[..., :probes - 1]
    flips = powers[uncertain]
    return np.concatenate([exact[:, None], exact[:, None] ^ flips], axis=-1)


def numpy_rope(values, positions, base=1_000_000.0):
    """Apply MLX non-traditional RoPE to [tokens, heads, head_dim]."""
    dimensions = values.shape[-1]
    frequencies = 1.0 / (
        base ** (np.arange(0, dimensions, 2, dtype=np.float32) / dimensions)
    )
    angles = np.asarray(positions, dtype=np.float32)[:, None] * frequencies[None, :]
    cosine = np.cos(angles)[:, None, :]
    sine = np.sin(angles)[:, None, :]
    first, second = np.split(values.astype(np.float32), 2, axis=-1)
    return np.concatenate([
        first * cosine - second * sine,
        first * sine + second * cosine,
    ], axis=-1)


def causal_hierarchical_candidates(
    query_logits, query_codes, query_bytes, key_codes, key_bytes, window,
    sink_tokens, secondary_probes, capacity, k, reranker="full-hamming",
    retention_policy="reservoir", retention_scores=None,
    query_bit_weights=None,
    query_pq_vectors=None, pq_codebook=None,
    storage_capacity=None, candidate_budget=None,
    probe_capacities=None,
    bilinear_matrix=None,
    lookup_tables=None,
    query_decoder_vectors=None, decoder_key_embeddings=None,
    distance_bias=None,
    attention_queries=None, decoded_attention_keys=None, attention_scale=1.0,
    query_lookup_scores=None,
    secondary_query_logits=None,
    secondary_key_codes=None,
):
    if probe_capacities is not None:
        storage_capacity = max(probe_capacities)
    storage_capacity = storage_capacity or capacity
    candidate_budget = candidate_budget or tables * secondary_probes * capacity
    length, tables = key_codes.shape
    popcount = np.array([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    secondary_by_table = (
        [
            secondary_probe_codes(query_logits, table, secondary_probes)
            for table in range(tables)
        ]
        if secondary_query_logits is None else
        [
            np.argsort(-secondary_query_logits[:, table], axis=-1)[
                ..., :secondary_probes
            ].astype(np.uint16)
            for table in range(tables)
        ]
    )
    indexed_secondary_codes = (
        None if secondary_key_codes is None
        else np.asarray(secondary_key_codes)
    )
    if indexed_secondary_codes is not None and (
        indexed_secondary_codes.shape != key_codes.shape
    ):
        raise ValueError("secondary key codes must match primary key codes")
    leaves = [dict() for _ in range(tables)]
    distant = np.full((length, k), -1, dtype=np.int32)
    full_rows = []
    address_rows = []
    retained_rows = []
    eviction_count = 0
    for query_position in range(length):
        historical = np.arange(
            sink_tokens, max(query_position - window, sink_tokens),
            dtype=np.int32,
        )
        address_match = np.zeros(len(historical), dtype=bool)
        for table in range(tables):
            primary_match = (
                key_codes[historical, table]
                == query_codes[query_position, table]
            )
            secondary_values = (
                key_codes[historical, (table + 1) % tables]
                if indexed_secondary_codes is None else
                indexed_secondary_codes[historical, table]
            )
            secondary_match = np.any(
                secondary_values[:, None]
                == secondary_by_table[table][query_position][None, :],
                axis=-1,
            )
            address_match |= primary_match & secondary_match
        address_rows.append(historical[address_match].tolist())
        pool = []
        queried_leaves = []
        for table in range(tables):
            primary = int(query_codes[query_position, table])
            for probe_index, secondary in enumerate(
                secondary_by_table[table][query_position]
            ):
                leaf = leaves[table].get((primary, int(secondary)), ())
                ordered_leaf = sorted([
                    (priority, position) for priority, position in leaf
                    if sink_tokens <= position < query_position - window
                ])
                if probe_capacities is not None:
                    ordered_leaf = ordered_leaf[:probe_capacities[probe_index]]
                pool.extend(ordered_leaf)
                queried_leaves.append(ordered_leaf)
        if storage_capacity > capacity and probe_capacities is None:
            counts = np.asarray([len(leaf) for leaf in queried_leaves])
            base = 0
            while np.sum(np.minimum(counts, base + 1)) <= candidate_budget:
                base += 1
                if base == storage_capacity:
                    break
            allocation = np.minimum(counts, base)
            remaining = candidate_budget - int(np.sum(allocation))
            for index in np.flatnonzero(counts > base)[:remaining]:
                allocation[index] += 1
            pool = [
                item
                for leaf, count in zip(queried_leaves, allocation)
                for item in leaf[:count]
            ]
        eligible_by_position = {}
        for priority, position in pool:
            if sink_tokens <= position < query_position - window:
                eligible_by_position[position] = min(
                    priority, eligible_by_position.get(position, priority)
                )
        eligible = sorted(eligible_by_position)
        if storage_capacity <= capacity and len(eligible) > candidate_budget:
            eligible = sorted(
                eligible,
                key=lambda position: (eligible_by_position[position], position),
            )[:candidate_budget]
            eligible.sort()
        retained_rows.append(eligible)
        if eligible:
            candidates = np.asarray(eligible, dtype=np.int32)
            xor = np.bitwise_xor(
                query_bytes[query_position][None, :], key_bytes[candidates]
            )
            distance_by_table = popcount[xor]
            if reranker == "path-hamming":
                distances = np.min(
                    distance_by_table + np.roll(distance_by_table, -1, axis=-1),
                    axis=-1,
                )
            elif reranker == "full-hamming":
                distances = np.sum(distance_by_table, axis=-1)
            elif reranker == "confidence-hamming":
                if query_bit_weights is None:
                    raise ValueError(
                        "confidence-hamming requires query bit weights"
                    )
                mismatch_bits = np.unpackbits(
                    xor, axis=-1, bitorder="little"
                ).astype(np.float32)
                weights = query_bit_weights[query_position].astype(np.float32)
                weights = weights / max(float(np.mean(weights)), 1e-6)
                distances = np.sum(mismatch_bits * weights[None, :], axis=-1)
            elif reranker == "product-quantized":
                if query_pq_vectors is None or pq_codebook is None:
                    raise ValueError(
                        "product-quantized reranking requires query vectors and a codebook"
                    )
                centroids = pq_codebook[
                    np.arange(pq_codebook.shape[0])[None, :],
                    key_bytes[candidates],
                ]
                distances = -np.sum(
                    centroids * query_pq_vectors[query_position][None, :, :],
                    axis=(-1, -2),
                )
            elif reranker == "bilinear-code":
                if bilinear_matrix is None:
                    raise ValueError("bilinear-code requires a matrix")
                query_bits = (
                    np.unpackbits(
                        query_bytes[query_position], bitorder="little"
                    ).astype(np.float32) * 2.0 - 1.0
                )
                if query_bit_weights is not None:
                    query_bits *= query_bit_weights[query_position]
                key_bits = (
                    np.unpackbits(
                        key_bytes[candidates], axis=-1, bitorder="little"
                    ).astype(np.float32) * 2.0 - 1.0
                )
                distances = -np.einsum(
                    "d,df,nf->n", query_bits, bilinear_matrix, key_bits,
                    optimize=True,
                )
            elif reranker == "lookup-code":
                if lookup_tables is None:
                    raise ValueError("lookup-code requires lookup tables")
                distances = -np.sum([
                    lookup_tables[
                        table, query_bytes[query_position, table],
                        key_bytes[candidates, table],
                    ]
                    for table in range(query_bytes.shape[-1])
                ], axis=0)
            elif reranker == "decoder-code":
                if query_decoder_vectors is None or decoder_key_embeddings is None:
                    raise ValueError("decoder-code requires decoder parameters")
                decoded_keys = np.sum([
                    decoder_key_embeddings[
                        table, key_bytes[candidates, table]
                    ]
                    for table in range(query_bytes.shape[-1])
                ], axis=0)
                distances = -np.sum(
                    query_decoder_vectors[query_position][None, :] * decoded_keys,
                    axis=-1,
                )
            elif reranker in (
                "attention-key-decoder", "joint-binary-attention",
                "joint-binary-attention-normalized", "joint-vq-attention",
            ):
                if attention_queries is None or decoded_attention_keys is None:
                    raise ValueError("attention-key-decoder requires Q/K heads")
                logits = np.einsum(
                    "hd,nhd->nh", attention_queries[query_position],
                    decoded_attention_keys[candidates], optimize=True,
                ) * attention_scale
                if reranker == "joint-binary-attention-normalized":
                    maxima = np.max(logits, axis=0, keepdims=True)
                    logits = logits - (
                        maxima + np.log(np.sum(
                            np.exp(logits - maxima), axis=0, keepdims=True
                        ))
                    )
                maxima = np.max(logits, axis=-1)
                distances = -(
                    maxima + np.log(np.sum(
                        np.exp(logits - maxima[:, None]), axis=-1
                    ))
                )
            elif reranker == "query-lookup":
                if query_lookup_scores is None:
                    raise ValueError("query-lookup requires query score tables")
                distances = -np.sum([
                    query_lookup_scores[
                        query_position, table, key_bytes[candidates, table]
                    ]
                    for table in range(query_bytes.shape[-1])
                ], axis=0)
            else:
                raise ValueError(
                    "unsupported hierarchical reranker"
                )
            if distance_bias is not None:
                relative = max(query_position, 1) - candidates
                buckets = np.minimum(
                    np.floor(np.log2(np.maximum(relative, 1))).astype(np.int32),
                    len(distance_bias) - 1,
                )
                distances = distances - distance_bias[buckets]
            order = np.lexsort((candidates, distances))[:k]
            selected = candidates[order]
            distant[query_position, :len(selected)] = selected
        local_start = max(sink_tokens, query_position - window + 1)
        local = range(local_start, query_position + 1)
        sinks = range(0, min(sink_tokens, query_position + 1))
        combined = sorted(set(sinks) | set(local) | {
            int(value) for value in distant[query_position] if value >= 0
        })
        full_rows.append(combined)

        position_array = np.asarray([query_position], dtype=np.int32)
        for table in range(tables):
            primary = int(key_codes[query_position, table])
            secondary = int(
                key_codes[query_position, (table + 1) % tables]
                if indexed_secondary_codes is None else
                indexed_secondary_codes[query_position, table]
            )
            composite = np.asarray(
                [primary * (1 << 8) + secondary], dtype=np.int64
            )
            priority = int(reservoir_priorities(
                position_array, composite, table
            )[0])
            leaf = leaves[table].setdefault((primary, secondary), [])
            if len(leaf) < storage_capacity:
                leaf.append((priority, query_position))
            else:
                if retention_policy == "reservoir":
                    worst = max(range(len(leaf)), key=lambda index: leaf[index][0])
                    if priority < leaf[worst][0]:
                        leaf[worst] = (priority, query_position)
                elif retention_policy == "tail":
                    leaf.pop(0)
                    leaf.append((priority, query_position))
                elif retention_policy == "diversity":
                    leaf.append((priority, query_position))
                    drop = diversity_eviction_index(
                        [position for _, position in leaf],
                        key_bytes,
                        [value for value, _ in leaf],
                    )
                    leaf.pop(drop)
                elif retention_policy in ("norm", "learned", "oracle"):
                    if retention_scores is None:
                        raise ValueError(
                            f"{retention_policy} retention requires retention scores"
                        )
                    candidates = [position for _, position in leaf] + [query_position]
                    scores = np.asarray(retention_scores)[candidates]
                    drop = int(np.argmin(scores))
                    if drop < len(leaf):
                        leaf[drop] = (priority, query_position)
                else:
                    raise ValueError(
                        "retention_policy must be reservoir, tail, diversity, norm, "
                        "learned, or oracle"
                    )
                eviction_count += 1
    def padded(rows):
        max_candidates = max(max(map(len, rows)), 1)
        result = np.full((length, max_candidates), -1, dtype=np.int32)
        for index, row in enumerate(rows):
            result[index, :len(row)] = row
        return result

    full = padded(full_rows)
    address_candidates = padded(address_rows)
    retained_candidates = padded(retained_rows)
    retained = sum(len(leaf) for table in leaves for leaf in table.values())
    return distant, full, address_candidates, retained_candidates, {
        "retained_postings": retained,
        "eviction_count": eviction_count,
        "evicted_fraction": eviction_count / max(length * tables, 1),
        "nonempty_leaves": sum(len(table) for table in leaves),
    }


def primary_address_candidates(query_codes, key_codes, window, sink_tokens):
    """All causal history sharing at least one frozen primary table code."""
    length, tables = key_codes.shape
    if query_codes.shape != (length, tables):
        raise ValueError("query/key primary codes must have matching shapes")
    rows = []
    for position in range(length):
        historical = np.arange(
            sink_tokens, max(position - window, sink_tokens), dtype=np.int32
        )
        if len(historical):
            match = np.any(
                key_codes[historical] == query_codes[position][None, :],
                axis=-1,
            )
            rows.append(historical[match])
        else:
            rows.append(np.empty((0,), dtype=np.int32))
    width = max(max(map(len, rows)), 1)
    result = np.full((length, width), -1, dtype=np.int32)
    for position, row in enumerate(rows):
        result[position, :len(row)] = row
    return result


def exact_shortlist_rerank(
    retained_candidates, approximate_queries, decoded_attention_keys,
    exact_queries,
    exact_attention_keys, scale, shortlist_size, k, window, sink_tokens,
):
    """Approximate-shortlist then exact-Q/K rerank without a global scan."""
    length = len(retained_candidates)
    distant = np.full((length, k), -1, dtype=np.int32)
    rows = []
    started = time.perf_counter()
    scored_queries = 0
    for position in range(length):
        pool = retained_candidates[position]
        pool = pool[pool >= 0]
        if len(pool):
            approximate_logits = np.einsum(
                "hd,nhd->nh", approximate_queries[position],
                decoded_attention_keys[pool], optimize=True,
            ) * scale
            approximate_scores = np.logaddexp.reduce(
                approximate_logits, axis=-1
            )
            keep = min(shortlist_size, len(pool))
            shortlist = pool[np.argpartition(-approximate_scores, keep - 1)[:keep]]
            exact_logits = np.einsum(
                "hd,nhd->nh", exact_queries[position],
                exact_attention_keys[shortlist], optimize=True,
            ) * scale
            exact_scores = np.logaddexp.reduce(exact_logits, axis=-1)
            selected_count = min(k, keep)
            order = np.lexsort((shortlist, -exact_scores))[:selected_count]
            distant[position, :selected_count] = shortlist[order]
            scored_queries += 1
        local_start = max(sink_tokens, position - window + 1)
        combined = sorted(
            set(range(0, min(sink_tokens, position + 1)))
            | set(range(local_start, position + 1))
            | {int(value) for value in distant[position] if value >= 0}
        )
        rows.append(combined)
    elapsed_us = (time.perf_counter() - started) * 1_000_000
    max_candidates = max(max(map(len, rows)), 1)
    full = np.full((length, max_candidates), -1, dtype=np.int32)
    for index, row in enumerate(rows):
        full[index, :len(row)] = row
    return distant, full, {
        "shortlist_size": shortlist_size,
        "host_approximate_plus_exact_us_per_scored_query": (
            elapsed_us / max(scored_queries, 1)
        ),
        "scored_queries": scored_queries,
    }


def streaming_teacher_metrics(
    queries, keys, full_candidates, distant_candidates, scale, topk, window,
    sink_tokens, query_chunk, key_chunk, query_bytes, key_bytes, candidate_slots,
    reranker="full-hamming",
    address_candidates=None, retained_candidates=None,
    query_bit_weights=None,
    query_pq_vectors=None, pq_codebook=None,
    bilinear_matrix=None,
    lookup_tables=None,
    query_decoder_vectors=None, decoder_key_embeddings=None,
    distance_bias=None,
    attention_queries=None, decoded_attention_keys=None, attention_scale=1.0,
    query_lookup_scores=None,
    primary_candidates=None,
):
    heads, length, _ = queries.shape
    query_positions = np.arange(window + sink_tokens + 1, length, dtype=np.int32)
    metrics = {
        "topk_recall": [],
        "attention_mass_recall": [],
        "distant_topk_recall": [],
        "distant_attention_mass_recall": [],
        "distant_attention_mass": [],
        "oracle_candidate_distant_mass_recall": [],
        "oracle_topk_distant_mass_recall": [],
        "oracle_hamming_distant_mass_recall": [],
    }
    if address_candidates is not None:
        metrics["address_candidate_distant_mass_recall"] = []
        metrics["address_oracle_topk_distant_mass_recall"] = []
    if retained_candidates is not None:
        metrics["retained_candidate_distant_mass_recall"] = []
        metrics["retained_oracle_topk_distant_mass_recall"] = []
    if primary_candidates is not None:
        metrics["primary_candidate_distant_mass_recall"] = []
        metrics["primary_oracle_topk_distant_mass_recall"] = []
    popcount = np.array(
        [int(value).bit_count() for value in range(256)], dtype=np.uint8
    )
    for query_start in range(0, len(query_positions), query_chunk):
        positions = query_positions[query_start:query_start + query_chunk]
        query_block = queries[:, positions].astype(np.float32)
        maxima = np.full((heads, len(positions)), -np.inf, dtype=np.float32)
        sums = np.zeros((heads, len(positions)), dtype=np.float64)
        for key_start in range(0, length, key_chunk):
            key_end = min(key_start + key_chunk, length)
            scores = np.einsum(
                "hqd,hkd->hqk", query_block,
                keys[:, key_start:key_end].astype(np.float32), optimize=True,
            ) * np.float32(scale)
            key_positions = np.arange(key_start, key_end)
            scores = np.where(
                key_positions[None, None, :] <= positions[None, :, None],
                scores,
                -np.inf,
            )
            chunk_max = np.max(scores, axis=-1)
            new_max = np.maximum(maxima, chunk_max)
            sums = (
                sums * np.exp(maxima - new_max)
                + np.sum(np.exp(scores - new_max[..., None]), axis=-1)
            )
            maxima = new_max
        mass = np.zeros((len(positions), length), dtype=np.float32)
        for key_start in range(0, length, key_chunk):
            key_end = min(key_start + key_chunk, length)
            scores = np.einsum(
                "hqd,hkd->hqk", query_block,
                keys[:, key_start:key_end].astype(np.float32), optimize=True,
            ) * np.float32(scale)
            key_positions = np.arange(key_start, key_end)
            scores = np.where(
                key_positions[None, None, :] <= positions[None, :, None],
                scores,
                -np.inf,
            )
            probability = np.exp(scores - maxima[..., None]) / sums[..., None]
            mass[:, key_start:key_end] = np.mean(probability, axis=0)
        for local_index, position in enumerate(positions):
            probability = mass[local_index]
            eligible_count = int(position) + 1
            label_count = min(topk, eligible_count)
            teacher_top = np.argpartition(
                -probability[:eligible_count], label_count - 1
            )[:label_count]
            selected = full_candidates[position]
            selected = selected[selected >= 0]
            selected_set = set(map(int, selected))
            metrics["topk_recall"].append(
                len(selected_set & set(map(int, teacher_top))) / label_count
            )
            metrics["attention_mass_recall"].append(
                float(np.sum(probability[selected])) if len(selected) else 0.0
            )
            distant_end = int(position) - window
            distant_keys = np.arange(sink_tokens, max(distant_end, sink_tokens))
            distant_mass = float(np.sum(probability[distant_keys]))
            routed = distant_candidates[position]
            routed = routed[routed >= 0]
            routed_mass = float(np.sum(probability[routed])) if len(routed) else 0.0
            metrics["distant_attention_mass"].append(distant_mass)
            metrics["distant_attention_mass_recall"].append(
                routed_mass / distant_mass if distant_mass > 0 else 1.0
            )
            denominator = distant_mass if distant_mass > 0 else 1.0
            if primary_candidates is not None:
                primary = primary_candidates[position]
                primary = primary[primary >= 0]
                metrics["primary_candidate_distant_mass_recall"].append(
                    float(np.sum(probability[primary])) / denominator
                    if len(primary) else 0.0
                )
                keep = min(topk, len(primary))
                primary_top = primary[np.argpartition(
                    -probability[primary], keep - 1
                )[:keep]] if keep else primary
                metrics["primary_oracle_topk_distant_mass_recall"].append(
                    float(np.sum(probability[primary_top])) / denominator
                    if keep else 0.0
                )
            if address_candidates is not None:
                addressed = address_candidates[position]
                addressed = addressed[addressed >= 0]
                metrics["address_candidate_distant_mass_recall"].append(
                    float(np.sum(probability[addressed])) / denominator
                    if len(addressed) else 0.0
                )
                keep = min(topk, len(addressed))
                addressed_top = addressed[np.argpartition(
                    -probability[addressed], keep - 1
                )[:keep]] if keep else addressed
                metrics["address_oracle_topk_distant_mass_recall"].append(
                    float(np.sum(probability[addressed_top])) / denominator
                    if keep else 0.0
                )
            if retained_candidates is not None:
                retained = retained_candidates[position]
                retained = retained[retained >= 0]
                metrics["retained_candidate_distant_mass_recall"].append(
                    float(np.sum(probability[retained])) / denominator
                    if len(retained) else 0.0
                )
                keep = min(topk, len(retained))
                retained_top = retained[np.argpartition(
                    -probability[retained], keep - 1
                )[:keep]] if keep else retained
                metrics["retained_oracle_topk_distant_mass_recall"].append(
                    float(np.sum(probability[retained_top])) / denominator
                    if keep else 0.0
                )
            oracle_pool_count = min(candidate_slots, len(distant_keys))
            if oracle_pool_count:
                distant_scores = probability[distant_keys]
                oracle_pool = distant_keys[np.argpartition(
                    -distant_scores, oracle_pool_count - 1
                )[:oracle_pool_count]]
                oracle_candidate_mass = float(np.sum(probability[oracle_pool]))
                oracle_keep = min(topk, len(oracle_pool))
                oracle_top = oracle_pool[np.argpartition(
                    -probability[oracle_pool], oracle_keep - 1
                )[:oracle_keep]]
                oracle_top_mass = float(np.sum(probability[oracle_top]))
                xor = np.bitwise_xor(
                    query_bytes[position][None, :], key_bytes[oracle_pool]
                )
                distance_by_table = popcount[xor]
                if reranker == "path-hamming":
                    hamming_distance = np.min(
                        distance_by_table
                        + np.roll(distance_by_table, -1, axis=-1),
                        axis=-1,
                    )
                elif reranker == "full-hamming":
                    hamming_distance = np.sum(distance_by_table, axis=-1)
                elif reranker == "confidence-hamming":
                    if query_bit_weights is None:
                        raise ValueError(
                            "confidence-hamming requires query bit weights"
                        )
                    mismatch_bits = np.unpackbits(
                        xor, axis=-1, bitorder="little"
                    ).astype(np.float32)
                    weights = query_bit_weights[position].astype(np.float32)
                    weights = weights / max(float(np.mean(weights)), 1e-6)
                    hamming_distance = np.sum(
                        mismatch_bits * weights[None, :], axis=-1
                    )
                elif reranker == "product-quantized":
                    if query_pq_vectors is None or pq_codebook is None:
                        raise ValueError(
                            "product-quantized reranking requires query vectors and a codebook"
                        )
                    centroids = pq_codebook[
                        np.arange(pq_codebook.shape[0])[None, :],
                        key_bytes[oracle_pool],
                    ]
                    hamming_distance = -np.sum(
                        centroids * query_pq_vectors[position][None, :, :],
                        axis=(-1, -2),
                    )
                elif reranker == "bilinear-code":
                    if bilinear_matrix is None:
                        raise ValueError("bilinear-code requires a matrix")
                    query_bits = (
                        np.unpackbits(
                            query_bytes[position], bitorder="little"
                        ).astype(np.float32) * 2.0 - 1.0
                    )
                    if query_bit_weights is not None:
                        query_bits *= query_bit_weights[position]
                    key_bits = (
                        np.unpackbits(
                            key_bytes[oracle_pool], axis=-1, bitorder="little"
                        ).astype(np.float32) * 2.0 - 1.0
                    )
                    hamming_distance = -np.einsum(
                        "d,df,nf->n", query_bits, bilinear_matrix, key_bits,
                        optimize=True,
                    )
                elif reranker == "lookup-code":
                    if lookup_tables is None:
                        raise ValueError("lookup-code requires lookup tables")
                    hamming_distance = -np.sum([
                        lookup_tables[
                            table, query_bytes[position, table],
                            key_bytes[oracle_pool, table],
                        ]
                        for table in range(query_bytes.shape[-1])
                    ], axis=0)
                elif reranker == "decoder-code":
                    if query_decoder_vectors is None or decoder_key_embeddings is None:
                        raise ValueError("decoder-code requires decoder parameters")
                    decoded_keys = np.sum([
                        decoder_key_embeddings[
                            table, key_bytes[oracle_pool, table]
                        ]
                        for table in range(query_bytes.shape[-1])
                    ], axis=0)
                    hamming_distance = -np.sum(
                        query_decoder_vectors[position][None, :] * decoded_keys,
                        axis=-1,
                    )
                elif reranker in (
                    "attention-key-decoder", "joint-binary-attention",
                    "joint-binary-attention-normalized", "joint-vq-attention",
                ):
                    if attention_queries is None or decoded_attention_keys is None:
                        raise ValueError("attention-key-decoder requires Q/K heads")
                    logits = np.einsum(
                        "hd,nhd->nh", attention_queries[position],
                        decoded_attention_keys[oracle_pool], optimize=True,
                    ) * attention_scale
                    if reranker == "joint-binary-attention-normalized":
                        maxima = np.max(logits, axis=0, keepdims=True)
                        logits = logits - (
                            maxima + np.log(np.sum(
                                np.exp(logits - maxima), axis=0, keepdims=True
                            ))
                        )
                    maxima = np.max(logits, axis=-1)
                    hamming_distance = -(
                        maxima + np.log(np.sum(
                            np.exp(logits - maxima[:, None]), axis=-1
                        ))
                    )
                elif reranker == "query-lookup":
                    if query_lookup_scores is None:
                        raise ValueError("query-lookup requires query score tables")
                    hamming_distance = -np.sum([
                        query_lookup_scores[
                            position, table, key_bytes[oracle_pool, table]
                        ]
                        for table in range(query_bytes.shape[-1])
                    ], axis=0)
                else:
                    raise ValueError("unsupported teacher reranker")
                if distance_bias is not None:
                    relative = position - oracle_pool
                    buckets = np.minimum(
                        np.floor(np.log2(np.maximum(relative, 1))).astype(np.int32),
                        len(distance_bias) - 1,
                    )
                    hamming_distance = hamming_distance - distance_bias[buckets]
                hamming_top = oracle_pool[np.lexsort(
                    (oracle_pool, hamming_distance)
                )[:oracle_keep]]
                hamming_mass = float(np.sum(probability[hamming_top]))
                metrics["oracle_candidate_distant_mass_recall"].append(
                    oracle_candidate_mass / denominator
                )
                metrics["oracle_topk_distant_mass_recall"].append(
                    oracle_top_mass / denominator
                )
                metrics["oracle_hamming_distant_mass_recall"].append(
                    hamming_mass / denominator
                )
            distant_label_count = min(topk, len(distant_keys))
            if distant_label_count:
                distant_scores = probability[distant_keys]
                distant_top = distant_keys[np.argpartition(
                    -distant_scores, distant_label_count - 1
                )[:distant_label_count]]
                metrics["distant_topk_recall"].append(
                    len(set(map(int, routed)) & set(map(int, distant_top)))
                    / distant_label_count
                )
    return {
        key: {
            "mean": float(np.mean(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "count": len(values),
        }
        for key, values in metrics.items()
    }


def oracle_future_distant_salience(
    queries, keys, scale, window, sink_tokens, query_chunk=8,
):
    """Noncausal diagnostic score from all future distant-attention queries.

    Each eligible future query contributes one unit of mass distributed across
    its distant keys.  This intentionally uses future teacher information and
    must never be presented as a deployable retention policy.
    """
    heads, length, _ = queries.shape
    result = np.zeros(length, dtype=np.float64)
    positions = np.arange(window + sink_tokens + 1, length, dtype=np.int32)
    key_positions = np.arange(length, dtype=np.int32)
    for start in range(0, len(positions), query_chunk):
        block_positions = positions[start:start + query_chunk]
        scores = np.einsum(
            "hqd,hkd->hqk",
            queries[:, block_positions].astype(np.float32),
            keys.astype(np.float32),
            optimize=True,
        ) * np.float32(scale)
        eligible = (
            (key_positions[None, :] < block_positions[:, None] - window)
            & (key_positions[None, :] >= sink_tokens)
        )
        scores = np.where(eligible[None, :, :], scores, -np.inf)
        maxima = np.max(scores, axis=-1, keepdims=True)
        probability = np.exp(scores - maxima)
        probability /= np.maximum(
            np.sum(probability, axis=-1, keepdims=True), 1e-30
        )
        result += np.sum(np.mean(probability, axis=0), axis=0)
    return result.astype(np.float32)


def dense_oracle_candidates(
    queries, keys, scale, topk, window, sink_tokens, query_chunk,
    values=None, mode="mass",
):
    heads, length, _ = queries.shape
    rows = []
    for start in range(0, length, query_chunk):
        positions = np.arange(start, min(start + query_chunk, length))
        scores = np.einsum(
            "hqd,hkd->hqk",
            queries[:, positions].astype(np.float32),
            keys.astype(np.float32),
            optimize=True,
        ) * np.float32(scale)
        key_positions = np.arange(length)
        scores = np.where(
            key_positions[None, None, :] <= positions[None, :, None],
            scores,
            -np.inf,
        )
        maxima = np.max(scores, axis=-1, keepdims=True)
        probability = np.exp(scores - maxima)
        probability /= np.sum(probability, axis=-1, keepdims=True)
        if mode == "mass":
            selection_signal = np.mean(probability, axis=0)
        elif mode == "contribution":
            if values is None:
                raise ValueError("contribution oracle requires values")
            value_norm = np.sqrt(np.sum(
                values.astype(np.float32) ** 2, axis=-1
            ))
            selection_signal = np.mean(
                probability * value_norm[:, None, :], axis=0
            )
        elif mode == "influence":
            if values is None:
                raise ValueError("influence oracle requires values")
            values_float = values.astype(np.float32)
            dense_value = np.einsum(
                "hqk,hkd->hqd", probability, values_float, optimize=True
            )
            deviation = np.sqrt(np.sum(
                (values_float[:, None, :, :] - dense_value[:, :, None, :]) ** 2,
                axis=-1,
            ))
            selection_signal = np.mean(probability * deviation, axis=0)
        else:
            raise ValueError("oracle mode must be mass, contribution, or influence")
        for offset, position in enumerate(positions):
            distant = np.arange(
                sink_tokens, max(int(position) - window, sink_tokens),
                dtype=np.int32,
            )
            keep = min(topk, len(distant))
            selected = (
                distant[np.argpartition(
                    -selection_signal[offset, distant], keep - 1
                )[:keep]]
                if keep else np.empty((0,), dtype=np.int32)
            )
            local_start = max(sink_tokens, int(position) - window + 1)
            local = range(local_start, int(position) + 1)
            sinks = range(0, min(sink_tokens, int(position) + 1))
            rows.append(sorted(set(sinks) | set(local) | set(map(int, selected))))
    width = max(map(len, rows))
    result = np.full((length, width), -1, dtype=np.int32)
    for position, row in enumerate(rows):
        result[position, :len(row)] = row
    return result


def sparse_attention_output(attention, queries, keys, values, candidates):
    selected = mx.array(candidates[None], dtype=mx.int32)
    valid = selected >= 0
    safe = mx.maximum(selected, 0)
    length = queries.shape[2]
    query_by_position = queries.transpose(0, 2, 1, 3)
    key_by_position = keys.transpose(0, 2, 1, 3)
    value_by_position = values.transpose(0, 2, 1, 3)
    gathered_keys = key_by_position.reshape(-1, keys.shape[1], keys.shape[-1])[safe]
    gathered_values = value_by_position.reshape(
        -1, values.shape[1], values.shape[-1]
    )[safe]
    scores = mx.sum(
        query_by_position[:, :, None] * gathered_keys, axis=-1
    ) * attention.scale
    scores = mx.where(valid[..., None], scores, mx.array(-1e9, scores.dtype))
    probability = mx.softmax(scores.astype(mx.float32), axis=2).astype(values.dtype)
    probability = mx.where(valid[..., None], probability, mx.zeros_like(probability))
    output = mx.sum(probability[..., None] * gathered_values, axis=2)
    return attention.out_proj(output.reshape(1, length, -1))


def tail_loss(capture, attention_output, tokens, layer_index):
    body = capture["body"]
    layer = capture["layer"]
    residual = capture["h"] + attention_output
    h = residual + layer.feed_forward(layer.ffn_norm(residual))
    for later in body.layers[layer_index + 1:]:
        mask = capture["attention_mask"] if hasattr(later, "self_attn") else capture["state_mask"]
        h = later(h, mask)
    normalized = body.embedding_norm(h)
    logits = body.embed_tokens.as_linear(normalized)
    loss = nn.losses.cross_entropy(
        logits[:, :-1], tokens[:, 1:], reduction="mean"
    ).astype(mx.float32)
    mx.eval(loss)
    return float(loss)


def timed_positioned_queries(call, query, positions, warmups, repeats):
    for index in range(warmups):
        selected = index % len(positions)
        output = call(query[selected:selected + 1], positions[selected:selected + 1])
        mx.eval(output)
        mx.synchronize()
    samples = []
    for _ in range(repeats):
        for index in range(len(positions)):
            started = time.perf_counter()
            output = call(query[index:index + 1], positions[index:index + 1])
            mx.eval(output)
            mx.synchronize()
            samples.append((time.perf_counter() - started) * 1_000_000)
    return {
        "median_us_per_query": float(np.median(samples)),
        "p10_us_per_query": float(np.quantile(samples, 0.10)),
        "p90_us_per_query": float(np.quantile(samples, 0.90)),
        "timed_queries": len(samples),
    }


def latency_metrics(
    hidden, query_projection, key_projection, tables, bits, primary_probes,
    secondary_bits, secondary_probes, capacity, k, window, sink_tokens,
    warmups, repeats, reranker="full-hamming",
    rerank_query_projection=None, rerank_key_projection=None,
    retention_policy="reservoir", retention_projection=None,
    fingerprint_bytes=8,
    rerank_confidence_power=1.0, rerank_confidence_mix=1.0,
    pq_codebook=None,
    rerank_global_weights=None,
    storage_capacity=None,
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
    joint_vq_assignment_weight=None,
    joint_vq_assignment_bias=None,
    joint_vq_attention_decoder=None,
    query_lookup_weight=None,
    query_lookup_bias=None,
    address_query_assignment_weight=None,
    address_query_assignment_bias=None,
    address_key_assignment_weight=None,
    address_key_assignment_bias=None,
    secondary_query_assignment_weight=None,
    secondary_query_assignment_bias=None,
    secondary_key_assignment_weight=None,
    secondary_key_assignment_bias=None,
    secondary_query_primary_bias=None,
    secondary_key_primary_bias=None,
):
    query_capacity = capacity
    if probe_capacities is not None:
        storage_capacity = max(probe_capacities)
    storage_capacity = storage_capacity or capacity
    candidate_budget = (
        tables * primary_probes * secondary_probes * query_capacity
        if storage_capacity > query_capacity and probe_capacities is None else None
    )
    length = len(hidden)
    query_start = max(0, length - window)
    key_hidden = hidden[:query_start]
    query_hidden = hidden[query_start:]
    if address_key_assignment_weight is None:
        _, key_codes, address_key_bytes = router_codes(
            key_hidden, key_projection, tables, bits
        )
    else:
        _, key_codes, address_key_bytes = categorical_address_codes(
            key_hidden, address_key_assignment_weight,
            address_key_assignment_bias,
        )
    key_bytes = address_key_bytes
    secondary_key_codes = None
    if secondary_key_assignment_weight is not None:
        if secondary_key_primary_bias is None:
            _, secondary_key_codes, _ = categorical_address_codes(
                key_hidden, secondary_key_assignment_weight,
                secondary_key_assignment_bias,
            )
        else:
            _, secondary_key_codes, _ = (
                primary_conditioned_categorical_address_codes(
                    key_hidden, secondary_key_assignment_weight,
                    secondary_key_assignment_bias, key_projection,
                    secondary_key_primary_bias,
                )
            )
    if joint_vq_assignment_weight is not None:
        vq_logits = np.einsum(
            "nd,dtc->ntc", key_hidden.astype(np.float64),
            joint_vq_assignment_weight.astype(np.float64), optimize=False,
        ) + joint_vq_assignment_bias
        key_bytes = np.argmax(vq_logits, axis=-1).astype(np.uint8)
    elif rerank_key_projection is not None:
        key_bytes = (
            pq_key_codes(key_hidden, rerank_key_projection, pq_codebook)
            if reranker == "product-quantized"
            else binary_fingerprint_bytes(key_hidden, rerank_key_projection)
        )
    starts, counts, postings, occupancy = build_sparse_hierarchical_index(
        key_codes, storage_capacity, address_bits=bits, secondary_bits=secondary_bits,
        retention_policy=(
            "score" if retention_policy in ("norm", "learned")
            else retention_policy
        ),
        fingerprints=key_bytes if retention_policy == "diversity" else None,
        retention_scores=(
            np.linalg.norm(key_hidden.astype(np.float32), axis=-1)
            if retention_policy == "norm"
            else (
                retention_score_values(key_hidden, retention_projection)
                if retention_policy == "learned" else None
            )
        ),
        secondary_codes_override=secondary_key_codes,
    )
    query = mx.array(query_hidden.astype(np.float16))
    query_projection_mx = mx.array(query_projection.astype(np.float16))
    rerank_query_projection_mx = (
        None if rerank_query_projection is None
        else mx.array(rerank_query_projection.astype(np.float16))
    )
    starts_mx = mx.array(starts)
    counts_mx = mx.array(counts)
    postings_mx = mx.array(postings)
    key_bytes_mx = mx.array(key_bytes)
    popcount = mx.array(
        np.array([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    )
    pq_codebook_mx = (
        None if pq_codebook is None
        else mx.array(pq_codebook.astype(np.float16))
    )
    rerank_global_weights_mx = (
        None if rerank_global_weights is None
        else mx.array(rerank_global_weights.astype(np.float16))
    )
    rerank_bilinear_mx = (
        None if rerank_bilinear is None
        else mx.array(rerank_bilinear.astype(np.float16))
    )
    rerank_lookup_mx = (
        None if rerank_lookup is None
        else mx.array(rerank_lookup.astype(np.float16))
    )
    rerank_decoder_query_mx = (
        None if rerank_decoder_query is None
        else mx.array(rerank_decoder_query.astype(np.float16))
    )
    rerank_decoder_keys_mx = (
        None if rerank_decoder_keys is None
        else mx.array(rerank_decoder_keys.astype(np.float16))
    )
    rerank_distance_bias_mx = (
        None if rerank_distance_bias is None
        else mx.array(rerank_distance_bias.astype(np.float16))
    )
    attention_query_weight_mx = (
        None if attention_query_weight is None
        else mx.array(attention_query_weight.astype(np.float16))
    )
    attention_query_norm_mx = (
        None if attention_query_norm is None
        else mx.array(attention_query_norm.astype(np.float16))
    )
    attention_key_decoder_mx = (
        None if attention_key_decoder is None
        else mx.array(attention_key_decoder.astype(np.float16))
    )
    joint_binary_attention_decoder_mx = (
        None if joint_binary_attention_decoder is None
        else mx.array(joint_binary_attention_decoder.astype(np.float16))
    )
    joint_hidden_weight_mx = (
        None if joint_binary_attention_decoder_hidden_weight is None
        else mx.array(
            joint_binary_attention_decoder_hidden_weight.astype(np.float16)
        )
    )
    joint_hidden_bias_mx = (
        None if joint_binary_attention_decoder_hidden_bias is None
        else mx.array(
            joint_binary_attention_decoder_hidden_bias.astype(np.float16)
        )
    )
    joint_output_weight_mx = (
        None if joint_binary_attention_decoder_output_weight is None
        else mx.array(
            joint_binary_attention_decoder_output_weight.astype(np.float16)
        )
    )
    joint_output_bias_mx = (
        None if joint_binary_attention_decoder_output_bias is None
        else mx.array(
            joint_binary_attention_decoder_output_bias.astype(np.float16)
        )
    )
    joint_head_bias_weight_mx = (
        None if joint_binary_attention_head_bias_weight is None
        else mx.array(joint_binary_attention_head_bias_weight.astype(np.float16))
    )
    joint_head_bias_mx = (
        None if joint_binary_attention_head_bias is None
        else mx.array(joint_binary_attention_head_bias.astype(np.float16))
    )
    joint_vq_decoder_mx = (
        None if joint_vq_attention_decoder is None
        else mx.array(joint_vq_attention_decoder.astype(np.float16))
    )
    query_lookup_weight_mx = (
        None if query_lookup_weight is None
        else mx.array(query_lookup_weight.astype(np.float16))
    )
    query_lookup_bias_mx = (
        None if query_lookup_bias is None
        else mx.array(query_lookup_bias.astype(np.float16))
    )
    address_query_assignment_weight_mx = (
        None if address_query_assignment_weight is None
        else mx.array(address_query_assignment_weight.astype(np.float16))
    )
    address_query_assignment_bias_mx = (
        None if address_query_assignment_bias is None
        else mx.array(address_query_assignment_bias.astype(np.float16))
    )
    secondary_query_assignment_weight_mx = (
        None if secondary_query_assignment_weight is None
        else mx.array(secondary_query_assignment_weight.astype(np.float16))
    )
    secondary_query_assignment_bias_mx = (
        None if secondary_query_assignment_bias is None
        else mx.array(secondary_query_assignment_bias.astype(np.float16))
    )
    secondary_query_primary_bias_mx = (
        None if secondary_query_primary_bias is None
        else mx.array(secondary_query_primary_bias.astype(np.float16))
    )
    positions = mx.arange(query_start, length, dtype=mx.int32)
    mx.eval(
        query, query_projection_mx, starts_mx, counts_mx, postings_mx,
        key_bytes_mx, popcount, positions,
    )
    if pq_codebook_mx is not None:
        mx.eval(pq_codebook_mx)
    if rerank_global_weights_mx is not None:
        mx.eval(rerank_global_weights_mx)
    if rerank_bilinear_mx is not None:
        mx.eval(rerank_bilinear_mx)
    if rerank_lookup_mx is not None:
        mx.eval(rerank_lookup_mx)
    if rerank_decoder_query_mx is not None:
        mx.eval(rerank_decoder_query_mx)
    if rerank_decoder_keys_mx is not None:
        mx.eval(rerank_decoder_keys_mx)
    if rerank_distance_bias_mx is not None:
        mx.eval(rerank_distance_bias_mx)
    if attention_query_weight_mx is not None:
        mx.eval(attention_query_weight_mx)
    if attention_query_norm_mx is not None:
        mx.eval(attention_query_norm_mx)
    if attention_key_decoder_mx is not None:
        mx.eval(attention_key_decoder_mx)
    if joint_binary_attention_decoder_mx is not None:
        mx.eval(joint_binary_attention_decoder_mx)
    for value in (
        joint_hidden_weight_mx, joint_hidden_bias_mx,
        joint_output_weight_mx, joint_output_bias_mx,
        joint_head_bias_weight_mx, joint_head_bias_mx,
        joint_vq_decoder_mx,
    ):
        if value is not None:
            mx.eval(value)
    if query_lookup_weight_mx is not None:
        mx.eval(query_lookup_weight_mx)
    if query_lookup_bias_mx is not None:
        mx.eval(query_lookup_bias_mx)
    if address_query_assignment_weight_mx is not None:
        mx.eval(
            address_query_assignment_weight_mx,
            address_query_assignment_bias_mx,
        )
    if secondary_query_assignment_weight_mx is not None:
        mx.eval(
            secondary_query_assignment_weight_mx,
            secondary_query_assignment_bias_mx,
        )
    if secondary_query_primary_bias_mx is not None:
        mx.eval(secondary_query_primary_bias_mx)

    def call(current, current_positions):
        return sparse_hierarchical_bucket_lookup(
            current, query_projection_mx, starts_mx, counts_mx, postings_mx,
            key_bytes_mx, popcount, tables, bits, primary_probes,
            secondary_bits, secondary_probes, storage_capacity, k,
            query_positions=current_positions, min_distance=window,
            sink_tokens=sink_tokens,
            reranker=reranker,
            rerank_projection=rerank_query_projection_mx,
            rerank_confidence_power=rerank_confidence_power,
            rerank_confidence_mix=rerank_confidence_mix,
            pq_codebook=pq_codebook_mx,
            rerank_global_weights=rerank_global_weights_mx,
            candidate_budget=candidate_budget,
            probe_capacities=probe_capacities,
            rerank_bilinear=rerank_bilinear_mx,
            rerank_lookup=rerank_lookup_mx,
            rerank_decoder_query=rerank_decoder_query_mx,
            rerank_decoder_keys=rerank_decoder_keys_mx,
            rerank_distance_bias=rerank_distance_bias_mx,
            attention_query_weight=attention_query_weight_mx,
            attention_query_norm=attention_query_norm_mx,
            attention_key_decoder=attention_key_decoder_mx,
            joint_binary_attention_decoder=joint_binary_attention_decoder_mx,
            joint_binary_attention_decoder_hidden_weight=joint_hidden_weight_mx,
            joint_binary_attention_decoder_hidden_bias=joint_hidden_bias_mx,
            joint_binary_attention_decoder_output_weight=joint_output_weight_mx,
            joint_binary_attention_decoder_output_bias=joint_output_bias_mx,
            joint_binary_attention_head_bias_weight=joint_head_bias_weight_mx,
            joint_binary_attention_head_bias=joint_head_bias_mx,
            joint_vq_attention_decoder=joint_vq_decoder_mx,
            query_lookup_weight=query_lookup_weight_mx,
            query_lookup_bias=query_lookup_bias_mx,
            address_query_assignment_weight=(
                address_query_assignment_weight_mx
            ),
            address_query_assignment_bias=address_query_assignment_bias_mx,
            secondary_query_assignment_weight=(
                secondary_query_assignment_weight_mx
            ),
            secondary_query_assignment_bias=(
                secondary_query_assignment_bias_mx
            ),
            secondary_query_primary_bias=secondary_query_primary_bias_mx,
        )

    timing = timed_positioned_queries(call, query, positions, warmups, repeats)
    bytes_per_query = logical_history_bytes(
        query_start, hidden.shape[-1], tables, query_capacity,
        probes=primary_probes, secondary_probes=secondary_probes,
        directory_entry_bytes=5,
        fingerprint_bytes=fingerprint_bytes,
    )["bucket_lookup"]
    return timing | {
        "logical_history_bytes_per_query": bytes_per_query,
        "index_resident_bytes": int(starts.nbytes + counts.nbytes + postings.nbytes),
        "occupancy": occupancy,
    }


def aggregate_metric(rows, path):
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        values.append(value)
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "count": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--routers", default=(
        "runs/lfm2.5-layer14-hard-hamming-top32-pos10-seed0.safetensors"
    ))
    parser.add_argument("--lengths", default="256,512,1024")
    parser.add_argument("--corpora", default="wikitext2")
    parser.add_argument("--segments-per-corpus", type=int, default=1)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--primary-probes", type=int, default=1)
    parser.add_argument("--secondary-bits", type=int, default=8)
    parser.add_argument("--secondary-probes", type=int, default=4)
    parser.add_argument("--leaf-capacity", type=int, default=7)
    parser.add_argument(
        "--storage-capacity", type=int, default=0,
        help="optional larger reservoir storage with the same bounded query budget",
    )
    parser.add_argument(
        "--probe-capacities", default="",
        help="optional comma-separated fixed capacities per secondary probe",
    )
    parser.add_argument(
        "--retention-policy",
        choices=("reservoir", "tail", "diversity", "norm", "learned", "oracle"),
        default="reservoir",
        help="oracle is a noncausal future-attention retention ceiling",
    )
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument(
        "--fingerprint-bytes", type=int, choices=(4, 5, 6, 7, 8), default=8
    )
    parser.add_argument(
        "--exact-shortlist", type=int, default=0,
        help="exact-Q/K rerank this many approximate candidates (0 disables)",
    )
    parser.add_argument(
        "--reranker", choices=(
            "full-hamming", "path-hamming", "confidence-hamming",
            "product-quantized", "bilinear-code",
            "lookup-code",
            "decoder-code",
            "attention-key-decoder",
            "joint-binary-attention",
            "joint-binary-attention-normalized",
            "joint-vq-attention",
            "query-lookup",
        ),
        default="full-hamming",
    )
    parser.add_argument("--confidence-power", type=float, default=1.0)
    parser.add_argument("--confidence-mix", type=float, default=1.0)
    parser.add_argument("--distance-bias", action="store_true")
    parser.add_argument("--pq-codebook", type=pathlib.Path)
    parser.add_argument("--query-chunk", type=int, default=8)
    parser.add_argument("--key-chunk", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--memory-limit-mb", type=int, default=1792)
    parser.add_argument("--cache-limit-mb", type=int, default=64)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    probe_capacities = (
        parse_ints(args.probe_capacities) if args.probe_capacities else None
    )
    lengths = parse_ints(args.lengths)
    checkpoints = parse_paths(args.routers)
    corpora = [value.strip() for value in args.corpora.split(",") if value.strip()]
    if not corpora or any(value not in ("wikitext2", "pg19") for value in corpora):
        parser.error("--corpora must contain wikitext2 and/or pg19")
    if args.segments_per_corpus < 1:
        parser.error("--segments-per-corpus must be positive")
    if args.confidence_power <= 0:
        parser.error("--confidence-power must be positive")
    if args.exact_shortlist and args.exact_shortlist < args.k:
        parser.error("--exact-shortlist must be zero or at least K")
    if not 0.0 <= args.confidence_mix <= 1.0:
        parser.error("--confidence-mix must be in [0, 1]")
    if probe_capacities is not None:
        if len(probe_capacities) != args.secondary_probes:
            parser.error(
                "--probe-capacities must match --secondary-probes"
            )
        if sum(probe_capacities) != args.secondary_probes * args.leaf_capacity:
            parser.error(
                "--probe-capacities must preserve the configured query budget"
            )
        if args.storage_capacity and max(probe_capacities) > args.storage_capacity:
            parser.error("probe capacity exceeds --storage-capacity")
    if args.reranker == "product-quantized" and not (
        args.pq_codebook and args.pq_codebook.is_file()
    ):
        parser.error("product-quantized reranking requires --pq-codebook")
    if args.tables * args.bits != 64 or args.bits != 8:
        parser.error("current real-model evaluator requires eight 8-bit tables")
    if args.secondary_bits != args.bits:
        parser.error("current hierarchy uses the complete next 8-bit learned table")
    mx.set_memory_limit(args.memory_limit_mb * 2**20)
    mx.set_cache_limit(args.cache_limit_mb * 2**20)
    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    body = language_body(model)
    pq_codebook = None
    if args.pq_codebook:
        pq_weights = mx.load(str(args.pq_codebook))
        pq_codebook = np.array(
            pq_weights.get("pq_centroids", pq_weights.get("centroids")).astype(
                mx.float32
            )
        ).astype(np.float32)
        if pq_codebook.shape[0] != args.fingerprint_bytes:
            parser.error("PQ subquantizers must equal --fingerprint-bytes")
    if args.layer < 0 or args.layer >= len(body.layers):
        parser.error("--layer is outside the model")
    width = config.get("text_config", config)["hidden_size"]
    routers = []
    for checkpoint in checkpoints:
        checkpoint_weights = mx.load(str(checkpoint))
        has_separate_reranker = "rerank_query_projection" in checkpoint_weights
        router_class = (
            HierarchicalAttentionRouter if has_separate_reranker else DonorHashRouter
        )
        router = router_class(width, args.tables, args.bits)
        if has_separate_reranker:
            router.query_projection = checkpoint_weights["query_projection"]
            router.key_projection = checkpoint_weights["key_projection"]
            router.rerank_query_projection = checkpoint_weights[
                "rerank_query_projection"
            ]
            router.rerank_key_projection = checkpoint_weights[
                "rerank_key_projection"
            ]
            if "retention_projection" in checkpoint_weights:
                router.retention_projection = checkpoint_weights[
                    "retention_projection"
                ]
            if "rerank_bit_weights" in checkpoint_weights:
                router.rerank_bit_weights = checkpoint_weights[
                    "rerank_bit_weights"
                ][:args.fingerprint_bytes * 8]
            if "rerank_bilinear" in checkpoint_weights:
                width_bits = args.fingerprint_bytes * 8
                router.rerank_bilinear = checkpoint_weights[
                    "rerank_bilinear"
                ][:width_bits, :width_bits]
            if "rerank_lookup" in checkpoint_weights:
                router.rerank_lookup = checkpoint_weights[
                    "rerank_lookup"
                ][:args.fingerprint_bytes]
            if "rerank_decoder_query" in checkpoint_weights:
                router.rerank_decoder_query = checkpoint_weights[
                    "rerank_decoder_query"
                ]
            if "rerank_decoder_keys" in checkpoint_weights:
                router.rerank_decoder_keys = checkpoint_weights[
                    "rerank_decoder_keys"
                ][:args.fingerprint_bytes]
            if "rerank_distance_bias" in checkpoint_weights:
                router.rerank_distance_bias = checkpoint_weights[
                    "rerank_distance_bias"
                ]
        else:
            router.load_weights(str(checkpoint))
        mx.eval(router.parameters())
        routers.append({
            "checkpoint": str(checkpoint),
            "sha256": sha256(checkpoint),
            "query_projection": np.array(
                router.query_projection.astype(mx.float16).astype(mx.float32)
            ).copy(),
            "key_projection": np.array(
                router.key_projection.astype(mx.float16).astype(mx.float32)
            ).copy(),
            "address_query_bias": (
                np.array(
                    checkpoint_weights["address_query_bias"].astype(mx.float32)
                ).copy()
                if "address_query_bias" in checkpoint_weights else None
            ),
            "address_key_bias": (
                np.array(
                    checkpoint_weights["address_key_bias"].astype(mx.float32)
                ).copy()
                if "address_key_bias" in checkpoint_weights else None
            ),
            "address_query_assignment_weight": (
                np.array(checkpoint_weights[
                    "address_query_assignment_weight"
                ].astype(mx.float32)).copy()
                if "address_query_assignment_weight" in checkpoint_weights
                else None
            ),
            "address_query_assignment_bias": (
                np.array(checkpoint_weights[
                    "address_query_assignment_bias"
                ].astype(mx.float32)).copy()
                if "address_query_assignment_bias" in checkpoint_weights
                else None
            ),
            "address_key_assignment_weight": (
                np.array(checkpoint_weights[
                    "address_key_assignment_weight"
                ].astype(mx.float32)).copy()
                if "address_key_assignment_weight" in checkpoint_weights
                else None
            ),
            "address_key_assignment_bias": (
                np.array(checkpoint_weights[
                    "address_key_assignment_bias"
                ].astype(mx.float32)).copy()
                if "address_key_assignment_bias" in checkpoint_weights
                else None
            ),
            "secondary_query_assignment_weight": (
                np.array(checkpoint_weights[
                    "secondary_query_assignment_weight"
                ].astype(mx.float32)).copy()
                if "secondary_query_assignment_weight" in checkpoint_weights
                else None
            ),
            "secondary_query_assignment_bias": (
                np.array(checkpoint_weights[
                    "secondary_query_assignment_bias"
                ].astype(mx.float32)).copy()
                if "secondary_query_assignment_bias" in checkpoint_weights
                else None
            ),
            "secondary_key_assignment_weight": (
                np.array(checkpoint_weights[
                    "secondary_key_assignment_weight"
                ].astype(mx.float32)).copy()
                if "secondary_key_assignment_weight" in checkpoint_weights
                else None
            ),
            "secondary_key_assignment_bias": (
                np.array(checkpoint_weights[
                    "secondary_key_assignment_bias"
                ].astype(mx.float32)).copy()
                if "secondary_key_assignment_bias" in checkpoint_weights
                else None
            ),
            "secondary_query_primary_bias": (
                np.array(checkpoint_weights[
                    "secondary_query_primary_bias"
                ].astype(mx.float32)).copy()
                if "secondary_query_primary_bias" in checkpoint_weights
                else None
            ),
            "secondary_key_primary_bias": (
                np.array(checkpoint_weights[
                    "secondary_key_primary_bias"
                ].astype(mx.float32)).copy()
                if "secondary_key_primary_bias" in checkpoint_weights
                else None
            ),
            "rerank_query_projection": (
                np.array(
                    router.rerank_query_projection.astype(mx.float16).astype(mx.float32)
                ).copy()[:, :args.fingerprint_bytes * 8]
                if has_separate_reranker else None
            ),
            "rerank_key_projection": (
                np.array(
                    router.rerank_key_projection.astype(mx.float16).astype(mx.float32)
                ).copy()[:, :args.fingerprint_bytes * 8]
                if has_separate_reranker else None
            ),
            "retention_projection": (
                np.array(
                    router.retention_projection.astype(mx.float32)
                ).copy()
                if "retention_projection" in checkpoint_weights else None
            ),
            "rerank_global_weights": (
                np.logaddexp(
                    0.0,
                    np.array(
                        router.rerank_bit_weights.astype(mx.float32)
                    ).copy()[:args.fingerprint_bytes * 8],
                )
                if has_separate_reranker else None
            ),
            "rerank_bilinear": (
                np.array(router.rerank_bilinear.astype(mx.float32)).copy()[
                    :args.fingerprint_bytes * 8, :args.fingerprint_bytes * 8
                ]
                if has_separate_reranker else None
            ),
            "rerank_lookup": (
                np.array(router.rerank_lookup.astype(mx.float32)).copy()[
                    :args.fingerprint_bytes
                ]
                if has_separate_reranker else None
            ),
            "rerank_decoder_query": (
                np.array(router.rerank_decoder_query.astype(mx.float32)).copy()
                if has_separate_reranker else None
            ),
            "rerank_decoder_keys": (
                np.array(router.rerank_decoder_keys.astype(mx.float32)).copy()[
                    :args.fingerprint_bytes
                ]
                if has_separate_reranker else None
            ),
            "rerank_distance_bias": (
                np.array(router.rerank_distance_bias.astype(mx.float32)).copy()
                if has_separate_reranker and args.distance_bias else None
            ),
            "attention_key_decoder": (
                np.array(
                    checkpoint_weights["attention_key_decoder"].astype(mx.float32)
                ).copy()
                if "attention_key_decoder" in checkpoint_weights else None
            ),
            "joint_binary_attention_decoder": (
                np.array(
                    checkpoint_weights[
                        "joint_binary_attention_decoder"
                    ].astype(mx.float32)
                ).copy()
                if "joint_binary_attention_decoder" in checkpoint_weights
                else None
            ),
            "joint_binary_attention_decoder_hidden_weight": (
                np.array(checkpoint_weights[
                    "joint_binary_attention_decoder_hidden_weight"
                ].astype(mx.float32)).copy()
                if "joint_binary_attention_decoder_hidden_weight"
                in checkpoint_weights else None
            ),
            "joint_binary_attention_decoder_hidden_bias": (
                np.array(checkpoint_weights[
                    "joint_binary_attention_decoder_hidden_bias"
                ].astype(mx.float32)).copy()
                if "joint_binary_attention_decoder_hidden_bias"
                in checkpoint_weights else None
            ),
            "joint_binary_attention_decoder_output_weight": (
                np.array(checkpoint_weights[
                    "joint_binary_attention_decoder_output_weight"
                ].astype(mx.float32)).copy()
                if "joint_binary_attention_decoder_output_weight"
                in checkpoint_weights else None
            ),
            "joint_binary_attention_decoder_output_bias": (
                np.array(checkpoint_weights[
                    "joint_binary_attention_decoder_output_bias"
                ].astype(mx.float32)).copy()
                if "joint_binary_attention_decoder_output_bias"
                in checkpoint_weights else None
            ),
            "joint_binary_attention_head_bias_weight": (
                np.array(checkpoint_weights[
                    "joint_binary_attention_head_bias_weight"
                ].astype(mx.float32)).copy()
                if "joint_binary_attention_head_bias_weight"
                in checkpoint_weights else None
            ),
            "joint_binary_attention_head_bias": (
                np.array(checkpoint_weights[
                    "joint_binary_attention_head_bias"
                ].astype(mx.float32)).copy()
                if "joint_binary_attention_head_bias"
                in checkpoint_weights else None
            ),
            "joint_vq_assignment_weight": (
                np.array(checkpoint_weights[
                    "joint_vq_assignment_weight"
                ].astype(mx.float32)).copy()
                if "joint_vq_assignment_weight" in checkpoint_weights else None
            ),
            "joint_vq_assignment_bias": (
                np.array(checkpoint_weights[
                    "joint_vq_assignment_bias"
                ].astype(mx.float32)).copy()
                if "joint_vq_assignment_bias" in checkpoint_weights else None
            ),
            "joint_vq_attention_decoder": (
                np.array(checkpoint_weights[
                    "joint_vq_attention_decoder"
                ].astype(mx.float32)).copy()
                if "joint_vq_attention_decoder" in checkpoint_weights else None
            ),
            "attention_query_weight": (
                np.array(
                    checkpoint_weights["attention_query_weight"].astype(mx.float32)
                ).copy()
                if "attention_query_weight" in checkpoint_weights else None
            ),
            "attention_query_norm": (
                np.array(
                    checkpoint_weights["attention_query_norm"].astype(mx.float32)
                ).copy()
                if "attention_query_norm" in checkpoint_weights else None
            ),
            "attention_query_trained": (
                "attention_query_trained" in checkpoint_weights
            ),
            "query_lookup_weight": (
                np.array(
                    checkpoint_weights["rerank_query_lookup_weight"].astype(
                        mx.float32
                    )
                ).copy()
                if "rerank_query_lookup_weight" in checkpoint_weights else None
            ),
            "query_lookup_bias": (
                np.array(
                    checkpoint_weights["rerank_query_lookup_bias"].astype(
                        mx.float32
                    )
                ).copy()
                if "rerank_query_lookup_bias" in checkpoint_weights else None
            ),
        })
    rows = []
    for length in lengths:
        token_sets = []
        if "wikitext2" in corpora:
            token_sets.extend(("wikitext2-test", tokens) for tokens in wikitext_tokens(
                tokenizer, "test", length, args.segments_per_corpus
            ))
        if "pg19" in corpora:
            token_sets.extend(("pg19-validation", tokens) for tokens in pg19_tokens(
                tokenizer, "validation", length, args.segments_per_corpus
            ))
        for corpus, tokens in token_sets:
            capture = capture_layer(model, tokens, args.layer)
            hidden = np.array(capture["x"][0].astype(mx.float32)).copy()
            queries = np.array(capture["queries"][0].astype(mx.float32)).copy()
            keys = np.array(capture["keys"][0].astype(mx.float32)).copy()
            oracle_retention_scores = (
                oracle_future_distant_salience(
                    queries, keys, capture["layer"].self_attn.scale,
                    args.window, args.sink_tokens, args.query_chunk,
                )
                if args.retention_policy == "oracle" else None
            )
            dense_loss = tail_loss(
                capture, capture["dense_attention"], tokens, args.layer
            )
            oracle_candidates = dense_oracle_candidates(
                queries, keys, capture["layer"].self_attn.scale, args.k,
                args.window, args.sink_tokens, args.query_chunk,
            )
            oracle_attention = sparse_attention_output(
                capture["layer"].self_attn,
                capture["queries"], capture["keys"], capture["values"],
                oracle_candidates,
            )
            mx.eval(oracle_attention)
            oracle_sparse_loss = tail_loss(
                capture, oracle_attention, tokens, args.layer
            )
            contribution_oracle_candidates = dense_oracle_candidates(
                queries, keys, capture["layer"].self_attn.scale, args.k,
                args.window, args.sink_tokens, args.query_chunk,
                values=np.array(capture["values"][0].astype(mx.float32)),
                mode="contribution",
            )
            contribution_oracle_attention = sparse_attention_output(
                capture["layer"].self_attn,
                capture["queries"], capture["keys"], capture["values"],
                contribution_oracle_candidates,
            )
            mx.eval(contribution_oracle_attention)
            contribution_oracle_loss = tail_loss(
                capture, contribution_oracle_attention, tokens, args.layer
            )
            influence_oracle_candidates = dense_oracle_candidates(
                queries, keys, capture["layer"].self_attn.scale, args.k,
                args.window, args.sink_tokens, args.query_chunk,
                values=np.array(capture["values"][0].astype(mx.float32)),
                mode="influence",
            )
            influence_oracle_attention = sparse_attention_output(
                capture["layer"].self_attn,
                capture["queries"], capture["keys"], capture["values"],
                influence_oracle_candidates,
            )
            mx.eval(influence_oracle_attention)
            influence_oracle_loss = tail_loss(
                capture, influence_oracle_attention, tokens, args.layer
            )
            for router_info in routers:
                if router_info["address_query_assignment_weight"] is not None:
                    query_logits, query_codes, address_query_bytes = (
                        categorical_address_codes(
                            hidden,
                            router_info["address_query_assignment_weight"],
                            router_info["address_query_assignment_bias"],
                        )
                    )
                    _, key_codes, address_key_bytes = categorical_address_codes(
                        hidden,
                        router_info["address_key_assignment_weight"],
                        router_info["address_key_assignment_bias"],
                    )
                else:
                    query_logits, query_codes, address_query_bytes = router_codes(
                        hidden, router_info["query_projection"], args.tables,
                        args.bits, bias=router_info["address_query_bias"],
                    )
                    _, key_codes, address_key_bytes = router_codes(
                        hidden, router_info["key_projection"], args.tables,
                        args.bits, bias=router_info["address_key_bias"],
                    )
                _, frozen_primary_query_codes, _ = router_codes(
                    hidden, router_info["query_projection"], args.tables,
                    args.bits, bias=router_info["address_query_bias"],
                )
                _, frozen_primary_key_codes, _ = router_codes(
                    hidden, router_info["key_projection"], args.tables,
                    args.bits, bias=router_info["address_key_bias"],
                )
                frozen_primary_candidates = primary_address_candidates(
                    frozen_primary_query_codes, frozen_primary_key_codes,
                    args.window, args.sink_tokens,
                )
                secondary_query_logits = None
                secondary_key_codes = None
                if router_info["secondary_query_assignment_weight"] is not None:
                    if router_info["secondary_query_primary_bias"] is None:
                        secondary_query_logits, _, _ = categorical_address_codes(
                            hidden,
                            router_info["secondary_query_assignment_weight"],
                            router_info["secondary_query_assignment_bias"],
                        )
                        _, secondary_key_codes, _ = categorical_address_codes(
                            hidden,
                            router_info["secondary_key_assignment_weight"],
                            router_info["secondary_key_assignment_bias"],
                        )
                    else:
                        secondary_query_logits, _, _ = (
                            primary_conditioned_categorical_address_codes(
                                hidden,
                                router_info[
                                    "secondary_query_assignment_weight"
                                ],
                                router_info["secondary_query_assignment_bias"],
                                router_info["query_projection"],
                                router_info["secondary_query_primary_bias"],
                            )
                        )
                        _, secondary_key_codes, _ = (
                            primary_conditioned_categorical_address_codes(
                                hidden,
                                router_info[
                                    "secondary_key_assignment_weight"
                                ],
                                router_info["secondary_key_assignment_bias"],
                                router_info["key_projection"],
                                router_info["secondary_key_primary_bias"],
                            )
                        )
                query_bytes, key_bytes = address_query_bytes, address_key_bytes
                if router_info["rerank_query_projection"] is not None:
                    if args.reranker == "product-quantized":
                        query_bytes = np.zeros(
                            (len(hidden), args.fingerprint_bytes), dtype=np.uint8
                        )
                        key_bytes = pq_key_codes(
                            hidden, router_info["rerank_key_projection"],
                            pq_codebook,
                        )
                    else:
                        query_bytes = binary_fingerprint_bytes(
                            hidden, router_info["rerank_query_projection"]
                        )
                        key_bytes = binary_fingerprint_bytes(
                            hidden, router_info["rerank_key_projection"]
                        )
                if args.reranker == "joint-vq-attention":
                    if router_info["joint_vq_assignment_weight"] is None:
                        raise ValueError("joint-vq-attention checkpoint lacks VQ weights")
                    vq_logits = np.einsum(
                        "nd,dtc->ntc", hidden.astype(np.float64),
                        router_info["joint_vq_assignment_weight"].astype(
                            np.float64
                        ), optimize=False,
                    ) + router_info["joint_vq_assignment_bias"]
                    key_bytes = np.argmax(vq_logits, axis=-1).astype(np.uint8)
                    query_bytes = np.zeros_like(key_bytes)
                query_bit_weights = (
                    np.power(
                        np.abs(np.einsum(
                            "nd,df->nf",
                            hidden.astype(np.float64),
                            router_info["rerank_query_projection"].astype(np.float64),
                            optimize=False,
                        )),
                        args.confidence_power,
                    ).astype(np.float32)
                    if args.reranker in (
                        "confidence-hamming", "bilinear-code"
                    ) else None
                )
                if query_bit_weights is not None:
                    query_bit_weights *= router_info[
                        "rerank_global_weights"
                    ][None, :]
                    query_bit_weights /= np.maximum(
                        np.mean(query_bit_weights, axis=-1, keepdims=True), 1e-6
                    )
                    query_bit_weights = (
                        (1.0 - args.confidence_mix)
                        + args.confidence_mix * query_bit_weights
                    )
                query_pq = (
                    pq_query_vectors(
                        hidden, router_info["rerank_query_projection"],
                        pq_codebook,
                    )
                    if args.reranker == "product-quantized" else None
                )
                query_decoder = (
                    np.einsum(
                        "nd,df->nf", hidden.astype(np.float32),
                        router_info["rerank_decoder_query"].astype(np.float32),
                        optimize=False,
                    )
                    if args.reranker == "decoder-code" else None
                )
                if args.reranker in (
                    "attention-key-decoder", "joint-binary-attention",
                    "joint-binary-attention-normalized", "joint-vq-attention",
                ):
                    if args.reranker in (
                        "attention-key-decoder", "joint-vq-attention"
                    ):
                        decoder = (
                            router_info["joint_vq_attention_decoder"]
                            if args.reranker == "joint-vq-attention"
                            else router_info["attention_key_decoder"]
                        )
                        if decoder is None:
                            raise ValueError(
                                "attention-key-decoder checkpoint lacks decoder weights"
                            )
                        decoded_pre_rope = np.sum([
                            decoder[table, key_bytes[:, table]]
                            for table in range(args.fingerprint_bytes)
                        ], axis=0)
                        if decoded_pre_rope.ndim == 2:
                            decoded_pre_rope = decoded_pre_rope.reshape(
                                len(hidden),
                                capture["layer"].self_attn.n_kv_heads,
                                capture["layer"].self_attn.head_dim,
                            )
                    else:
                        if router_info["joint_binary_attention_decoder"] is None:
                            raise ValueError(
                                "joint-binary-attention checkpoint lacks decoder weights"
                            )
                        key_bits = (
                            np.unpackbits(
                                key_bytes, axis=-1, bitorder="little"
                            ).astype(np.float32) * 2.0 - 1.0
                        )
                        shape = (
                            len(hidden), capture["layer"].self_attn.n_kv_heads,
                            capture["layer"].self_attn.head_dim,
                        )
                        decoded_flat = np.einsum(
                            "nb,bf->nf", key_bits.astype(np.float64),
                            router_info[
                                "joint_binary_attention_decoder"
                            ].astype(np.float64),
                            optimize=False,
                        ).astype(np.float32)
                        if router_info[
                            "joint_binary_attention_decoder_hidden_weight"
                        ] is not None:
                            nonlinear_hidden = np.tanh(
                                np.einsum(
                                    "nb,bh->nh", key_bits.astype(np.float64),
                                    router_info[
                                        "joint_binary_attention_decoder_hidden_weight"
                                    ].astype(np.float64),
                                    optimize=False,
                                )
                                + router_info[
                                    "joint_binary_attention_decoder_hidden_bias"
                                ]
                            )
                            decoded_flat += np.einsum(
                                "nh,hf->nf", nonlinear_hidden.astype(np.float64),
                                router_info[
                                    "joint_binary_attention_decoder_output_weight"
                                ].astype(np.float64),
                                optimize=False,
                            ).astype(np.float32)
                            decoded_flat += router_info[
                                "joint_binary_attention_decoder_output_bias"
                            ]
                        decoded_pre_rope = decoded_flat.reshape(shape)
                    decoded_attention_keys = numpy_rope(
                        decoded_pre_rope, np.arange(len(hidden))
                    )
                    decoded_attention_keys = np.repeat(
                        decoded_attention_keys, queries.shape[0]
                        // decoded_attention_keys.shape[1], axis=1,
                    )
                    if router_info["attention_query_trained"]:
                        query_pre_rope = np.einsum(
                            "nd,od->no", hidden.astype(np.float64),
                            router_info["attention_query_weight"].astype(
                                np.float64
                            ), optimize=False,
                        ).astype(np.float32).reshape(
                            len(hidden),
                            capture["layer"].self_attn.n_heads,
                            capture["layer"].self_attn.head_dim,
                        )
                        query_pre_rope *= (
                            1.0 / np.sqrt(
                                np.mean(np.square(query_pre_rope), axis=-1,
                                        keepdims=True) + 1e-5
                            )
                        )
                        query_pre_rope *= router_info[
                            "attention_query_norm"
                        ][None, None, :]
                        attention_queries = numpy_rope(
                            query_pre_rope, np.arange(len(hidden))
                        )
                    else:
                        attention_queries = queries.transpose(1, 0, 2)
                    if router_info[
                        "joint_binary_attention_head_bias_weight"
                    ] is not None:
                        head_bias = (
                            np.einsum(
                                "nd,dh->nh", hidden.astype(np.float64),
                                router_info[
                                    "joint_binary_attention_head_bias_weight"
                                ].astype(np.float64), optimize=False,
                            ).astype(np.float32)
                            + router_info["joint_binary_attention_head_bias"]
                        )
                        attention_queries = np.concatenate([
                            attention_queries,
                            (head_bias / capture["layer"].self_attn.scale)[
                                :, :, None
                            ],
                        ], axis=-1)
                        decoded_attention_keys = np.concatenate([
                            decoded_attention_keys,
                            np.ones((*decoded_attention_keys.shape[:-1], 1),
                                    dtype=decoded_attention_keys.dtype),
                        ], axis=-1)
                else:
                    decoded_attention_keys = None
                    attention_queries = None
                query_lookup_scores = (
                    (
                        np.einsum(
                            "nd,df->nf", hidden.astype(np.float64),
                            router_info["query_lookup_weight"].astype(np.float64),
                            optimize=False,
                        ).astype(np.float32)
                        + router_info["query_lookup_bias"]
                    ).reshape(len(hidden), args.fingerprint_bytes, 256)
                    if args.reranker == "query-lookup" else None
                )
                (
                    distant, full, address_candidates, retained_candidates,
                    causal_index,
                ) = causal_hierarchical_candidates(
                    query_logits, query_codes, query_bytes, key_codes, key_bytes,
                    args.window, args.sink_tokens, args.secondary_probes,
                    args.leaf_capacity, args.k, args.reranker,
                    args.retention_policy,
                    np.linalg.norm(hidden, axis=-1)
                    if args.retention_policy == "norm"
                    else (
                        retention_score_values(
                            hidden, router_info["retention_projection"]
                        )
                        if args.retention_policy == "learned"
                        else oracle_retention_scores
                        if args.retention_policy == "oracle" else None
                    ),
                    query_bit_weights,
                    query_pq,
                    pq_codebook,
                    args.storage_capacity or None,
                    args.tables * args.primary_probes * args.secondary_probes
                    * args.leaf_capacity,
                    probe_capacities,
                    router_info["rerank_bilinear"],
                    router_info["rerank_lookup"],
                    query_decoder,
                    router_info["rerank_decoder_keys"],
                    router_info["rerank_distance_bias"],
                    attention_queries,
                    decoded_attention_keys,
                    capture["layer"].self_attn.scale,
                    query_lookup_scores,
                    secondary_query_logits,
                    secondary_key_codes,
                )
                shortlist_metrics = None
                if args.exact_shortlist:
                    if attention_queries is None or decoded_attention_keys is None:
                        raise ValueError(
                            "--exact-shortlist requires an attention decoder reranker"
                        )
                    exact_queries = queries.transpose(1, 0, 2)
                    exact_attention_keys = keys.transpose(1, 0, 2)
                    exact_attention_keys = np.repeat(
                        exact_attention_keys,
                        exact_queries.shape[1] // exact_attention_keys.shape[1],
                        axis=1,
                    )
                    distant, full, shortlist_metrics = exact_shortlist_rerank(
                        retained_candidates, attention_queries,
                        decoded_attention_keys, exact_queries,
                        exact_attention_keys, capture["layer"].self_attn.scale,
                        args.exact_shortlist, args.k, args.window,
                        args.sink_tokens,
                    )
                teacher = streaming_teacher_metrics(
                    queries, keys, full, distant,
                    capture["layer"].self_attn.scale, args.k,
                    args.window, args.sink_tokens, args.query_chunk, args.key_chunk,
                    query_bytes, key_bytes,
                    args.tables * args.primary_probes * args.secondary_probes
                    * args.leaf_capacity,
                    args.reranker,
                    address_candidates,
                    retained_candidates,
                    query_bit_weights,
                    query_pq,
                    pq_codebook,
                    router_info["rerank_bilinear"],
                    router_info["rerank_lookup"],
                    query_decoder,
                    router_info["rerank_decoder_keys"],
                    router_info["rerank_distance_bias"],
                    attention_queries,
                    decoded_attention_keys,
                    capture["layer"].self_attn.scale,
                    query_lookup_scores,
                    frozen_primary_candidates,
                )
                sparse_attention = sparse_attention_output(
                    capture["layer"].self_attn,
                    capture["queries"], capture["keys"], capture["values"], full,
                )
                mx.eval(sparse_attention)
                sparse_loss = tail_loss(capture, sparse_attention, tokens, args.layer)
                dense_np = np.array(capture["dense_attention"].astype(mx.float32))
                sparse_np = np.array(sparse_attention.astype(mx.float32))
                nrmse = float(
                    np.sqrt(np.mean((sparse_np - dense_np) ** 2))
                    / max(np.sqrt(np.mean(dense_np ** 2)), 1e-12)
                )
                latency = latency_metrics(
                    hidden, router_info["query_projection"],
                    router_info["key_projection"], args.tables, args.bits,
                    args.primary_probes, args.secondary_bits,
                    args.secondary_probes, args.leaf_capacity, args.k,
                    args.window, args.sink_tokens, args.warmups, args.repeats,
                    args.reranker,
                    router_info["rerank_query_projection"],
                    router_info["rerank_key_projection"],
                    (
                        "reservoir" if args.retention_policy == "oracle"
                        else args.retention_policy
                    ),
                    router_info["retention_projection"],
                    args.fingerprint_bytes,
                    args.confidence_power,
                    args.confidence_mix,
                    pq_codebook,
                    router_info["rerank_global_weights"],
                    args.storage_capacity or None,
                    probe_capacities,
                    router_info["rerank_bilinear"],
                    router_info["rerank_lookup"],
                    router_info["rerank_decoder_query"],
                    router_info["rerank_decoder_keys"],
                    router_info["rerank_distance_bias"],
                    router_info["attention_query_weight"],
                    router_info["attention_query_norm"],
                    router_info["attention_key_decoder"],
                    router_info["joint_binary_attention_decoder"],
                    router_info[
                        "joint_binary_attention_decoder_hidden_weight"
                    ],
                    router_info[
                        "joint_binary_attention_decoder_hidden_bias"
                    ],
                    router_info[
                        "joint_binary_attention_decoder_output_weight"
                    ],
                    router_info[
                        "joint_binary_attention_decoder_output_bias"
                    ],
                    router_info[
                        "joint_binary_attention_head_bias_weight"
                    ],
                    router_info["joint_binary_attention_head_bias"],
                    router_info["joint_vq_assignment_weight"],
                    router_info["joint_vq_assignment_bias"],
                    router_info["joint_vq_attention_decoder"],
                    router_info["query_lookup_weight"],
                    router_info["query_lookup_bias"],
                    router_info["address_query_assignment_weight"],
                    router_info["address_query_assignment_bias"],
                    router_info["address_key_assignment_weight"],
                    router_info["address_key_assignment_bias"],
                    router_info["secondary_query_assignment_weight"],
                    router_info["secondary_query_assignment_bias"],
                    router_info["secondary_key_assignment_weight"],
                    router_info["secondary_key_assignment_bias"],
                    router_info["secondary_query_primary_bias"],
                    router_info["secondary_key_primary_bias"],
                )
                if args.retention_policy == "oracle":
                    latency["retention_timing_scope"] = (
                        "reservoir-equivalent lookup proxy; noncausal future-attention "
                        "score computation excluded"
                    )
                row = {
                    "length": length,
                    "corpus": corpus,
                    "router": router_info["checkpoint"],
                    "router_sha256": router_info["sha256"],
                    "teacher": teacher,
                    "causal_index": causal_index,
                    "dense_loss": dense_loss,
                    "oracle_sparse_loss": oracle_sparse_loss,
                    "oracle_perplexity_ratio": math.exp(
                        oracle_sparse_loss - dense_loss
                    ),
                    "contribution_oracle_sparse_loss": contribution_oracle_loss,
                    "contribution_oracle_perplexity_ratio": math.exp(
                        contribution_oracle_loss - dense_loss
                    ),
                    "influence_oracle_sparse_loss": influence_oracle_loss,
                    "influence_oracle_perplexity_ratio": math.exp(
                        influence_oracle_loss - dense_loss
                    ),
                    "sparse_loss": sparse_loss,
                    "loss_delta": sparse_loss - dense_loss,
                    "perplexity_ratio": math.exp(sparse_loss - dense_loss),
                    "attention_output_nrmse": nrmse,
                    "routing": latency,
                    "retention_scope": (
                        "noncausal_future_distant_attention_oracle"
                        if args.retention_policy == "oracle" else "causal"
                    ),
                    "exact_shortlist": (
                        shortlist_metrics | {
                            "exact_key_bytes_per_query": (
                                args.exact_shortlist
                                * capture["layer"].self_attn.n_kv_heads
                                * capture["layer"].self_attn.head_dim * 2
                            ),
                            "index_plus_exact_key_bytes_per_query": (
                                latency["logical_history_bytes_per_query"]
                                + args.exact_shortlist
                                * capture["layer"].self_attn.n_kv_heads
                                * capture["layer"].self_attn.head_dim * 2
                            ),
                            "timing_scope": (
                                "NumPy approximate scoring plus exact shortlist "
                                "rerank only; excludes index lookup and KV gather"
                            ),
                        }
                        if shortlist_metrics is not None else None
                    ),
                    "peak_memory_mb_so_far": mx.get_peak_memory() / 2**20,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                del sparse_attention
                mx.clear_cache()
            del capture
            mx.clear_cache()
    summary = {}
    for checkpoint in checkpoints:
        checkpoint_rows = [row for row in rows if row["router"] == str(checkpoint)]
        summary[str(checkpoint)] = {
            "attention_mass_recall": aggregate_metric(
                checkpoint_rows, ("teacher", "attention_mass_recall", "mean")
            ),
            "distant_attention_mass_recall": aggregate_metric(
                checkpoint_rows,
                ("teacher", "distant_attention_mass_recall", "mean"),
            ),
            "topk_recall": aggregate_metric(
                checkpoint_rows, ("teacher", "topk_recall", "mean")
            ),
            "perplexity_ratio": aggregate_metric(
                checkpoint_rows, ("perplexity_ratio",)
            ),
            "routing_us": aggregate_metric(
                checkpoint_rows, ("routing", "median_us_per_query")
            ),
            "logical_bytes_per_query": aggregate_metric(
                checkpoint_rows,
                ("routing", "logical_history_bytes_per_query"),
            ),
        }
    report = {
        "format_version": 1,
        "scope": (
            "held-out real LFM2.5 layer states, streaming dense attention teacher, "
            "causal hierarchical candidates, and paired layer-14 replacement loss"
        ),
        "config": vars(args) | {
            "output": str(args.output),
            "routers": [str(path) for path in checkpoints],
            "lengths": lengths,
            "corpora": corpora,
        },
        "platform": platform.platform(),
        "mlx_version": mx.__version__,
        "rows": rows,
        "summary": summary,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "peak_memory_mb": report["peak_memory_mb"],
    }), flush=True)


if __name__ == "__main__":
    main()
