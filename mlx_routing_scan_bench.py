"""Compare linear routing scans with persistent fixed-budget bucket lookup."""

import argparse
import json
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


def build_bucket_tails(table_codes, members, address_bits=16):
    tails, _ = build_bucket_index(
        table_codes, members, address_bits=address_bits, retention_policy="tail"
    )
    return tails


def logical_history_bytes(
    length, dim, tables, bucket_capacity, probes=1, fp_bytes=2
):
    """Logical historical index payload read per query, excluding query encoding."""
    return {
        "fp_scan": length * dim * fp_bytes,
        "binary64_scan": length * 8,
        "bucket_lookup": tables * probes * bucket_capacity * (4 + 8),
    }


def candidate_recall(selected, reference, k):
    recalls = []
    for selected_row, reference_row in zip(selected, reference):
        selected_set = {int(value) for value in selected_row if value >= 0}
        reference_set = {int(value) for value in reference_row if value >= 0}
        recalls.append(len(selected_set & reference_set) / k)
    return float(np.mean(recalls))


def query_byte_codes(query, projection):
    logits = query @ projection
    bits = (logits >= 0).reshape(query.shape[0], 8, 8).astype(mx.int32)
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
    if args.tables * args.probes * args.bucket_capacity < args.k:
        parser.error("addressed candidate pool must contain at least K slots")
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
                "persistent direct-address bounded postings followed by Hamming rerank"
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
        query_probe_codes_np = host_probe_codes(
            queries_np.astype(np.float16), routing_projection_np,
            args.tables, args.bits, args.probes,
        )
        target_probe_address_match = np.any(
            query_probe_codes_np == table_codes_np[targets, :, None], axis=(1, 2)
        )
        build_started = time.perf_counter()
        tails_np, occupancy = build_bucket_index(
            table_codes_np[:length],
            args.bucket_capacity,
            address_bits=args.bits,
            retention_policy=args.retention_policy,
        )
        build_ms = (time.perf_counter() - build_started) * 1000.0

        keys = mx.array(keys_np[:length])
        query = mx.array(queries_np.astype(np.float16))
        key_bytes = mx.array(key_bytes_np[:length])
        tails = mx.array(tails_np)
        mx.eval(keys, query, key_bytes, tails)

        methods = (
            ("fp_scan", lambda current: fp_scan(current, keys, args.k)),
            ("binary64_scan", lambda current: binary64_scan(
                current, projection, key_bytes, popcount_lut, args.k
            )),
            ("bucket_lookup", lambda current: bucket_lookup(
                current, projection, tails, key_bytes, popcount_lut,
                args.tables, args.bits, args.probes, args.k,
            )),
        )
        outputs = {}
        timing = {}
        for name, call in methods:
            outputs[name] = batched_output(call, query, args.recall_batch)
            timing[name] = timed_queries(call, query, args.warmups, args.repeats)
        reference = outputs["fp_scan"]
        byte_counts = logical_history_bytes(
            length, args.dim, args.tables, args.bucket_capacity,
            probes=args.probes,
        )
        row = {
            "length": length,
            "queries": args.queries,
            "k": args.k,
            "bucket_index_build_ms": build_ms,
            "bucket_index_resident_bytes": int(tails_np.nbytes),
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
        del keys, query, key_bytes, tails
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
