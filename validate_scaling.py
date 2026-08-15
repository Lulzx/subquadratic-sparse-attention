#!/usr/bin/env python3
"""Run a bounded empirical validation of SubQ scaling and hash capacity.

The report deliberately separates finite-range measurements from asymptotic claims.
It benchmarks the portable MLX selector, gathered sparse attention, a safely capped
dense MLX reference, and the real selector under controlled collision pressure.
"""

import argparse
import datetime as dt
import json
import math
import pathlib
import platform
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np

from capacity_scaling import binomial_tail_survival, retrieval_probability
from mlx_attention_bench import timed as timed_sparse_attention
from mlx_selector import make_needles, timed_selector
from ssa.mlx_attention import apply_rope, random_weights
from ssa.mlx_selector import hash_codes, select_indices


DEFAULT_LENGTHS = "1024,2048,4096,8192,16384"
DEFAULT_DENSE_LENGTHS = "512,1024,2048,4096,8192"
DEFAULT_CAPACITY_CONFIGS = (
    "8:256,8:1024,10:1024,10:4096,12:4096,12:16384,14:16384"
)


def parse_ints(value, minimum=1):
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item < minimum for item in values):
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers >= {minimum}"
        )
    return values


def parse_capacity_configs(value):
    configs = []
    for item in value.split(","):
        try:
            bits, length = map(int, item.split(":"))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "capacity configs must have the form bits:length,bits:length"
            ) from error
        if bits < 1 or bits > 30 or length < 2:
            raise argparse.ArgumentTypeError(
                "capacity bits must be in [1, 30] and lengths must be >= 2"
            )
        configs.append((bits, length))
    if not configs:
        raise argparse.ArgumentTypeError("at least one capacity config is required")
    return configs


def fit_models(rows, field="milliseconds"):
    """Fit y = intercept + coefficient*f(n) and a log-log exponent."""
    points = [(float(row["length"]), float(row[field])) for row in rows]
    if len(points) < 3:
        return {"error": "at least three measured lengths are required"}
    n = np.array([point[0] for point in points], dtype=np.float64)
    y = np.array([point[1] for point in points], dtype=np.float64)
    functions = {
        "n": n,
        "n_log_n": n * np.log2(n),
        "n_squared": n * n,
    }
    models = {}
    total = float(np.sum((y - y.mean()) ** 2))
    for name, basis in functions.items():
        design = np.column_stack([np.ones_like(basis), basis])
        intercept, coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
        prediction = design @ np.array([intercept, coefficient])
        residual = y - prediction
        rmse = float(np.sqrt(np.mean(residual * residual)))
        models[name] = {
            "intercept": float(intercept),
            "coefficient": float(coefficient),
            "rmse_ms": rmse,
            "normalized_rmse": rmse / max(float(y.mean()), 1e-12),
            "r_squared": 1.0 - float(np.sum(residual * residual)) / total
            if total > 0.0
            else None,
        }
    log_n, log_y = np.log(n), np.log(np.maximum(y, 1e-12))
    exponent = float(np.polyfit(log_n, log_y, 1)[0])
    doubling_ratios = [
        float(y[index] / y[index - 1])
        for index in range(1, len(y))
        if math.isclose(n[index], 2.0 * n[index - 1])
    ]
    return {
        "models": models,
        "best_normalized_rmse": min(
            models, key=lambda name: models[name]["normalized_rmse"]
        ),
        "empirical_log_log_exponent": exponent,
        "doubling_ratios": doubling_ratios,
    }


def scaling_ratios(length, milliseconds, peak_memory_mb):
    n = float(length)
    return {
        "milliseconds_per_n": milliseconds / n,
        "milliseconds_per_n_log_n": milliseconds / (n * math.log2(n)),
        "milliseconds_per_n_squared": milliseconds / (n * n),
        "peak_memory_mb_per_n": peak_memory_mb / n,
    }


def benchmark_selector(args):
    rng = np.random.default_rng(args.seed)
    projection = mx.array(
        rng.standard_normal(
            (args.dim, args.tables * args.bits), dtype=np.float32
        )
        / np.sqrt(args.dim)
    )
    rows = []
    for length in args.lengths:
        if length > args.max_length:
            rows.append({"length": length, "skipped": "above --max-length"})
            continue
        mx.reset_peak_memory()
        x, stored, queries = make_needles(
            length, args.dim, min(args.needles, length // 4), args.seed + length
        )
        milliseconds, selected = timed_selector(
            x,
            projection,
            args.repeats,
            tables=args.tables,
            bits=args.bits,
            members=args.members,
            probes=args.probes,
        )
        selected_np = np.array(selected)[0, queries]
        hits = sum(
            bool(np.any(row == target) or np.any(row == target + 1))
            for row, target in zip(selected_np, stored)
        )
        peak = mx.get_peak_memory() / 2**20
        row = {
            "length": length,
            "milliseconds": float(milliseconds),
            "recall": hits / len(stored),
            "needles": len(stored),
            "selected_per_query": int(selected.shape[-1]),
            "active_memory_mb": mx.get_active_memory() / 2**20,
            "peak_memory_mb": peak,
        }
        row.update(scaling_ratios(length, milliseconds, peak))
        rows.append(row)
        del x, selected
        mx.clear_cache()
    measured = [row for row in rows if "milliseconds" in row]
    return {"rows": rows, "latency_fit": fit_models(measured)}


def benchmark_sparse_attention(args):
    weights = random_weights(
        args.width, tables=args.tables, bits=args.bits, seed=args.seed
    )
    rows = []
    for length in args.lengths:
        if length > args.max_length:
            rows.append({"length": length, "skipped": "above --max-length"})
            continue
        mx.reset_peak_memory()
        mx.random.seed(args.seed + length)
        x = mx.random.normal((1, length, args.width)).astype(mx.float16)
        milliseconds, output, selected = timed_sparse_attention(
            x, weights, args.heads, args.repeats, args.chunk_q
        )
        peak = mx.get_peak_memory() / 2**20
        row = {
            "length": length,
            "milliseconds": float(milliseconds),
            "output_shape": list(output.shape),
            "selected_per_query": int(selected.shape[-1]),
            "active_memory_mb": mx.get_active_memory() / 2**20,
            "peak_memory_mb": peak,
        }
        row.update(scaling_ratios(length, milliseconds, peak))
        rows.append(row)
        del x, output, selected
        mx.clear_cache()
    measured = [row for row in rows if "milliseconds" in row]
    return {"rows": rows, "latency_fit": fit_models(measured)}


def dense_attention_chunked(x, weights, heads, chunk_q):
    """Causal dense attention with bounded score memory but quadratic work."""
    wq, wk, wv, wo, _ = weights
    batch, length, width = x.shape
    head_dim = width // heads
    q = (x @ wq).reshape(batch, length, heads, head_dim)
    k = (x @ wk).reshape(batch, length, heads, head_dim)
    v = (x @ wv).reshape(batch, length, heads, head_dim)
    positions = mx.arange(length).reshape(1, length)
    q = apply_rope(q, positions).transpose(0, 2, 1, 3)
    k = apply_rope(k, positions).transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    chunks = []
    for start in range(0, length, chunk_q):
        end = min(start + chunk_q, length)
        query_positions = mx.arange(start, end).reshape(-1, 1)
        key_positions = mx.arange(end).reshape(1, -1)
        mask = query_positions >= key_positions
        output = mx.fast.scaled_dot_product_attention(
            q[:, :, start:end],
            k[:, :, :end],
            v[:, :, :end],
            scale=1.0 / math.sqrt(head_dim),
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            batch, end - start, width
        ) @ wo
        mx.eval(output)
        chunks.append(output)
    return mx.concatenate(chunks, axis=1)


def timed_dense_attention(x, weights, heads, repeats, chunk_q):
    warm = dense_attention_chunked(x, weights, heads, chunk_q)
    mx.eval(warm)
    mx.synchronize()
    samples, output = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        output = dense_attention_chunked(x, weights, heads, chunk_q)
        mx.eval(output)
        mx.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return sorted(samples)[len(samples) // 2], output


def benchmark_dense_attention(args):
    weights = random_weights(
        args.width, tables=args.tables, bits=args.bits, seed=args.seed
    )
    rows = []
    for length in args.dense_lengths:
        if length > args.max_dense_length:
            rows.append(
                {"length": length, "skipped": "above --max-dense-length"}
            )
            continue
        mx.reset_peak_memory()
        mx.random.seed(args.seed + length)
        x = mx.random.normal((1, length, args.width)).astype(mx.float16)
        milliseconds, output = timed_dense_attention(
            x, weights, args.heads, args.repeats, args.dense_chunk_q
        )
        peak = mx.get_peak_memory() / 2**20
        row = {
            "length": length,
            "milliseconds": float(milliseconds),
            "output_shape": list(output.shape),
            "active_memory_mb": mx.get_active_memory() / 2**20,
            "peak_memory_mb": peak,
        }
        row.update(scaling_ratios(length, milliseconds, peak))
        rows.append(row)
        del x, output
        mx.clear_cache()
    measured = [row for row in rows if "milliseconds" in row]
    return {"rows": rows, "latency_fit": fit_models(measured)}


def wilson_interval(successes, count, z=1.959963984540054):
    if count < 1:
        return [None, None]
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def mean_pairwise_correlation(values):
    """Mean off-diagonal correlation, omitting constant table columns."""
    values = np.asarray(values, dtype=np.float64)
    variable = values[:, np.std(values, axis=0) > 0.0]
    if variable.shape[1] < 2:
        return None
    correlation = np.corrcoef(variable, rowvar=False)
    upper = correlation[np.triu_indices(correlation.shape[0], 1)]
    finite = upper[np.isfinite(upper)]
    return float(finite.mean()) if finite.size else None


def occupancy_statistics(codes, bits):
    table_stats, collision_probabilities = [], []
    bucket_count = 1 << bits
    for table in range(codes.shape[-1]):
        counts = np.bincount(codes[..., table].reshape(-1), minlength=bucket_count)
        occupied = counts[counts > 0]
        total = int(counts.sum())
        pair_denominator = total * (total - 1)
        collision_probability = (
            float(np.sum(counts * (counts - 1)) / pair_denominator)
            if pair_denominator
            else 0.0
        )
        probabilities = occupied / max(total, 1)
        entropy = -float(np.sum(probabilities * np.log2(probabilities)))
        table_stats.append(
            {
                "empty_bucket_fraction": float(np.mean(counts == 0)),
                "maximum_load": int(occupied.max()) if occupied.size else 0,
                "p95_occupied_load": float(np.percentile(occupied, 95))
                if occupied.size
                else 0.0,
                "p99_occupied_load": float(np.percentile(occupied, 99))
                if occupied.size
                else 0.0,
                "entropy_bits": entropy,
                "normalized_entropy": entropy / bits,
                "pair_collision_probability": collision_probability,
            }
        )
        collision_probabilities.append(collision_probability)
    return table_stats, collision_probabilities


def capacity_trial(length, bits, args, seed):
    rng = np.random.default_rng(seed)
    projection = mx.array(
        rng.standard_normal(
            (args.dim, args.tables * bits), dtype=np.float32
        )
        / np.sqrt(args.dim)
    )
    all_success, all_agreement, all_collision_survival = [], [], []
    unique_candidates, later_key_counts = [], []
    occupancy_rows, collision_probability_rows = [], []
    remaining = args.capacity_needles
    while remaining:
        batch = min(args.capacity_batch, remaining)
        x_np = rng.standard_normal((batch, length, args.dim), dtype=np.float32)
        # Put every target in the oldest eighth and copy it to the final query.
        # Each batch row is an independent selector sample, avoiding interactions
        # between multiple copied needles in one sequence.
        stored = rng.integers(0, max(1, length // 8), size=batch, dtype=np.int32)
        x_np[np.arange(batch), length - 1] = x_np[np.arange(batch), stored]
        x = mx.array(x_np)
        selected = select_indices(
            x,
            projection,
            tables=args.tables,
            bits=bits,
            members=args.members,
            probes=1,
            block=False,
        )
        codes = hash_codes(x, projection, args.tables, bits)
        mx.eval(selected, codes)
        selected_np = np.array(selected)[:, -1].reshape(
            batch, args.tables, args.members
        )
        codes_np = np.array(codes)
        success = np.any(selected_np == stored[:, None, None], axis=-1)
        agreement = codes_np[np.arange(batch), -1] == codes_np[
            np.arange(batch), stored
        ]
        collision_survival = np.zeros((batch, args.tables), dtype=bool)
        for sample, target in enumerate(stored):
            target_codes = codes_np[sample, target]
            later_codes = codes_np[sample, target + 1 : -1]
            collision_survival[sample] = np.sum(
                later_codes == target_codes[None, :], axis=0
            ) < args.members
        all_success.append(success)
        all_agreement.append(agreement)
        all_collision_survival.append(collision_survival)
        unique_candidates.extend(
            len(np.unique(row[row >= 0]))
            for row in selected_np.reshape(batch, -1)
        )
        later_key_counts.extend((length - stored - 2).tolist())
        for sample in range(batch):
            sample_stats, sample_collision_probabilities = occupancy_statistics(
                codes_np[sample : sample + 1, :-1], bits
            )
            occupancy_rows.extend(sample_stats)
            collision_probability_rows.append(sample_collision_probabilities)
        remaining -= batch
        del x, selected, codes
        mx.clear_cache()

    table_success = np.concatenate(all_success, axis=0)
    agreement = np.concatenate(all_agreement, axis=0)
    collision_survival = np.concatenate(all_collision_survival, axis=0)
    hits = np.any(table_success, axis=-1)
    ideal_predictions, pair_collision_predictions, loads = [], [], []
    for later_keys, agreements, collision_probabilities in zip(
        later_key_counts, agreement, collision_probability_rows
    ):
        loads.append(later_keys * (2.0**-bits))
        ideal_predictions.append(
            retrieval_probability(
                later_keys, bits, args.tables, args.members, agreement=1.0
            )
        )
        failures = 1.0
        for table, agrees in enumerate(agreements):
            survival = binomial_tail_survival(
                later_keys, collision_probabilities[table], args.members
            )
            failures *= 1.0 - float(agrees) * survival
        pair_collision_predictions.append(1.0 - failures)
    measured_table_success = np.mean(agreement & collision_survival, axis=0)
    measured_occupancy_prediction = 1.0 - float(
        np.prod(1.0 - measured_table_success)
    )
    result = {
        "hits": int(hits.sum()),
        "trials": int(args.capacity_needles),
        "mean_load": float(np.mean(loads)),
        "ideal_prediction": float(np.mean(ideal_predictions)),
        "occupancy_adjusted_prediction": measured_occupancy_prediction,
        "pair_collision_binomial_prediction": float(
            np.mean(pair_collision_predictions)
        ),
        "query_key_agreement": float(np.mean(agreement)),
        "query_key_agreement_by_table": np.mean(agreement, axis=0).tolist(),
        "measured_collision_survival_by_table": np.mean(
            collision_survival, axis=0
        ).tolist(),
        "selector_collision_survival_mismatches": int(
            np.sum(table_success != (agreement & collision_survival))
        ),
        "mean_unique_candidates": float(np.mean(unique_candidates)),
        "mean_pairwise_table_success_correlation": mean_pairwise_correlation(
            table_success
        ),
        "occupancy": occupancy_rows,
    }
    return result


def aggregate_capacity(config, trials):
    bits, length = config
    hits = sum(trial["hits"] for trial in trials)
    count = sum(trial["trials"] for trial in trials)
    occupancy_rows = [row for trial in trials for row in trial["occupancy"]]

    def mean_field(field, rows=trials):
        values = [row[field] for row in rows if row[field] is not None]
        return float(np.mean(values)) if values else None

    agreement_by_table = np.mean(
        [trial["query_key_agreement_by_table"] for trial in trials], axis=0
    )
    survival_by_table = np.mean(
        [trial["measured_collision_survival_by_table"] for trial in trials], axis=0
    )
    measured_table_success = agreement_by_table * survival_by_table
    occupancy_adjusted_prediction = 1.0 - float(
        np.prod(1.0 - measured_table_success)
    )
    return {
        "bits": bits,
        "length": length,
        "seeds": len(trials),
        "hits": hits,
        "trials": count,
        "observed_recall": hits / count,
        "recall_wilson_95": wilson_interval(hits, count),
        "mean_load": mean_field("mean_load"),
        "ideal_prediction": mean_field("ideal_prediction"),
        "occupancy_adjusted_prediction": occupancy_adjusted_prediction,
        "pair_collision_binomial_prediction": mean_field(
            "pair_collision_binomial_prediction"
        ),
        "query_key_agreement": mean_field("query_key_agreement"),
        "query_key_agreement_by_table": agreement_by_table.tolist(),
        "measured_collision_survival_by_table": survival_by_table.tolist(),
        "selector_collision_survival_mismatches": sum(
            trial["selector_collision_survival_mismatches"] for trial in trials
        ),
        "mean_unique_candidates": mean_field("mean_unique_candidates"),
        "mean_pairwise_table_success_correlation": mean_field(
            "mean_pairwise_table_success_correlation"
        ),
        "occupancy": {
            field: mean_field(field, occupancy_rows)
            for field in (
                "empty_bucket_fraction",
                "maximum_load",
                "p95_occupied_load",
                "p99_occupied_load",
                "entropy_bits",
                "normalized_entropy",
                "pair_collision_probability",
            )
        },
        "seed_results": trials,
    }


def benchmark_capacity(args):
    rows = []
    for bits, length in args.capacity_configs:
        if length > args.max_length:
            rows.append(
                {
                    "bits": bits,
                    "length": length,
                    "skipped": "above --max-length",
                }
            )
            continue
        trials = [
            capacity_trial(length, bits, args, seed)
            for seed in args.capacity_seeds
        ]
        rows.append(aggregate_capacity((bits, length), trials))
    measured = [row for row in rows if "observed_recall" in row]
    if measured:
        ideal_mae = float(
            np.mean(
                [
                    abs(row["observed_recall"] - row["ideal_prediction"])
                    for row in measured
                ]
            )
        )
        occupancy_mae = float(
            np.mean(
                [
                    abs(
                        row["observed_recall"]
                        - row["occupancy_adjusted_prediction"]
                    )
                    for row in measured
                ]
            )
        )
    else:
        ideal_mae = occupancy_mae = None
    load_groups = {}
    for row in measured:
        group = f"log2_load_{round(math.log2(max(row['mean_load'], 1e-12)))}"
        load_groups.setdefault(group, []).append(row["observed_recall"])
    matched_load_spreads = {
        group: max(values) - min(values)
        for group, values in load_groups.items()
        if len(values) > 1
    }
    return {
        "rows": rows,
        "ideal_mean_absolute_error": ideal_mae,
        "occupancy_adjusted_mean_absolute_error": occupancy_mae,
        "matched_load_recall_spreads": matched_load_spreads,
        "selector_collision_survival_mismatches": sum(
            row.get("selector_collision_survival_mismatches", 0) for row in measured
        ),
        "note": (
            "Exact copied query/key vectors force agreement near one. Differences "
            "from the ideal curve therefore isolate hash imbalance and table "
            "dependence. The occupancy-adjusted value combines measured per-table "
            "collision survival under an independence assumption; its remaining gap "
            "to actual recall reflects table dependence and sampling noise. This does "
            "not measure a learned router."
        ),
    }


def run_causality_tests(repo):
    completed = subprocess.run(
        [sys.executable, "mlx_tests.py"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    pass_lines = [line for line in output.splitlines() if line.startswith("PASS ")]
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "pass_count": len(pass_lines),
        "pass_lines": pass_lines,
        "output": output,
    }


def finite_claims(report):
    selector_rows = [
        row for row in report["selector"]["rows"] if "milliseconds" in row
    ]
    sparse_rows = [
        row for row in report["sparse_attention"]["rows"] if "milliseconds" in row
    ]
    candidate_counts = {
        row["selected_per_query"] for row in selector_rows + sparse_rows
    }
    capacity_rows = [
        row
        for row in report["capacity"]["rows"]
        if "observed_recall" in row
    ]
    collision_tracking = all(
        row["ideal_prediction"] >= row["recall_wilson_95"][0] - 0.10
        and row["ideal_prediction"] <= row["recall_wilson_95"][1] + 0.10
        for row in capacity_rows
    )
    occupancy_tracking = all(
        abs(row["observed_recall"] - row["occupancy_adjusted_prediction"]) <= 0.05
        for row in capacity_rows
    )
    matched_load_tracking = all(
        spread <= 0.05
        for spread in report["capacity"]["matched_load_recall_spreads"].values()
    )
    return {
        "candidate_count_constant_over_measured_lengths": len(candidate_counts) == 1,
        "candidate_count": next(iter(candidate_counts)) if len(candidate_counts) == 1 else None,
        "selector_exact_needle_recall_all_one": all(
            row["recall"] == 1.0 for row in selector_rows
        ),
        "causality_suite_passed": report["causality"]["passed"],
        "capacity_curve_tracks_ideal_within_ci_plus_0_10": collision_tracking,
        "capacity_curve_tracks_measured_occupancy_within_0_05": occupancy_tracking,
        "matched_load_recall_spread_at_most_0_05": matched_load_tracking,
        "selector_matches_direct_collision_survival": (
            report["capacity"]["selector_collision_survival_mismatches"] == 0
        ),
        "scope": (
            "These are finite-range diagnostics. Model-fit ranking and doubling ratios "
            "can distinguish behavior over the measured range but cannot prove Big-O."
        ),
    }


def format_number(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def scaling_table(section):
    lines = [
        "| n | Median ms | Peak MB | Candidates/query | ms/n | ms/(n log2 n) | ms/n^2 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in section["rows"]:
        if "milliseconds" not in row:
            lines.append(f"| {row['length']:,} | skipped | | | | | |")
            continue
        candidates = row.get("selected_per_query")
        lines.append(
            f"| {row['length']:,} | {row['milliseconds']:.3f} | "
            f"{row['peak_memory_mb']:.2f} | "
            f"{candidates if candidates is not None else 'all history'} | "
            f"{row['milliseconds_per_n']:.3e} | "
            f"{row['milliseconds_per_n_log_n']:.3e} | "
            f"{row['milliseconds_per_n_squared']:.3e} |"
        )
    return lines


def fit_summary(section):
    fit = section["latency_fit"]
    if "error" in fit:
        return fit["error"]
    model_bits = ", ".join(
        f"{name} NRMSE={values['normalized_rmse']:.4f}"
        for name, values in fit["models"].items()
    )
    ratios = ", ".join(f"{value:.2f}x" for value in fit["doubling_ratios"])
    return (
        f"Best finite-range fit: `{fit['best_normalized_rmse']}`; log-log exponent "
        f"{fit['empirical_log_log_exponent']:.3f}. Fits: {model_bits}. "
        f"Doubling ratios: {ratios or 'n/a'}."
    )


def render_markdown(report):
    metadata = report["metadata"]
    claims = report["finite_range_findings"]
    lines = [
        "# Empirical scaling and capacity validation",
        "",
        f"Generated `{metadata['generated_at_utc']}` on "
        f"`{metadata['platform']}` with MLX `{metadata['mlx_version']}`.",
        "",
        "This report tests finite lengths and controlled collision loads. It does not "
        "prove an asymptotic bound or establish learned-router quality.",
        "",
        "## Selector scaling",
        "",
        *scaling_table(report["selector"]),
        "",
        fit_summary(report["selector"]),
        "",
        "## Gathered sparse-attention scaling",
        "",
        *scaling_table(report["sparse_attention"]),
        "",
        fit_summary(report["sparse_attention"]),
        "",
        "## Safely capped dense-attention reference",
        "",
        *scaling_table(report["dense_attention"]),
        "",
        fit_summary(report["dense_attention"]),
        "",
        "The dense reference chunks queries to bound temporary score memory; chunking "
        "does not change its all-history, quadratic arithmetic count.",
        "",
        "## Collision-capacity sweep through the real selector",
        "",
        "| Bits | n | Mean load | Observed recall | Wilson 95% | Ideal | Occupancy-adjusted | Agreement | Unique candidates |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["capacity"]["rows"]:
        if "observed_recall" not in row:
            lines.append(f"| {row['bits']} | {row['length']:,} | skipped | | | | | | |")
            continue
        low, high = row["recall_wilson_95"]
        lines.append(
            f"| {row['bits']} | {row['length']:,} | {row['mean_load']:.3f} | "
            f"{row['observed_recall']:.4f} ({row['hits']}/{row['trials']}) | "
            f"[{low:.4f}, {high:.4f}] | {row['ideal_prediction']:.4f} | "
            f"{row['occupancy_adjusted_prediction']:.4f} | "
            f"{row['query_key_agreement']:.4f} | "
            f"{row['mean_unique_candidates']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Ideal-curve MAE: {format_number(report['capacity']['ideal_mean_absolute_error'], 4)}. "
            f"Occupancy-adjusted MAE: "
            f"{format_number(report['capacity']['occupancy_adjusted_mean_absolute_error'], 4)}.",
            "",
            report["capacity"]["note"],
            "",
            "## Causality and finite-range findings",
            "",
            f"The MLX suite returned `{report['causality']['returncode']}` with "
            f"{report['causality']['pass_count']} passing checks.",
            "",
            f"- Fixed candidate count: `{claims['candidate_count_constant_over_measured_lengths']}` "
            f"({claims['candidate_count']} per query).",
            f"- Exact-needle recall at every measured selector length: "
            f"`{claims['selector_exact_needle_recall_all_one']}`.",
            f"- Future-mutation and other causality suite checks: "
            f"`{claims['causality_suite_passed']}`.",
            f"- Capacity curve within Wilson interval plus 0.10 tolerance: "
            f"`{claims['capacity_curve_tracks_ideal_within_ci_plus_0_10']}`.",
            f"- Capacity curve within 0.05 of measured-occupancy prediction: "
            f"`{claims['capacity_curve_tracks_measured_occupancy_within_0_05']}`.",
            f"- Matched-load recall spread at most 0.05: "
            f"`{claims['matched_load_recall_spread_at_most_0_05']}`.",
            f"- Selector/table outcomes match direct collision-tail survival: "
            f"`{claims['selector_matches_direct_collision_survival']}`.",
            "",
            claims["scope"],
            "",
            "## Reproduction",
            "",
            "```bash",
            metadata["command"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def self_test():
    rows = [
        {"length": n, "milliseconds": 2.0 + 0.01 * n * math.log2(n)}
        for n in (128, 256, 512, 1024, 2048)
    ]
    fit = fit_models(rows)
    assert fit["best_normalized_rmse"] == "n_log_n"
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high
    assert parse_capacity_configs("8:256,10:1024") == [(8, 256), (10, 1024)]
    assert math.isclose(retrieval_probability(1, 1, 2, 1), 0.75)
    mx.random.seed(41)
    x = mx.random.normal((1, 17, 16)).astype(mx.float16)
    weights = random_weights(16, tables=2, bits=4, seed=42)
    one_chunk = dense_attention_chunked(x, weights, heads=2, chunk_q=17)
    many_chunks = dense_attention_chunked(x, weights, heads=2, chunk_q=4)
    changed = mx.concatenate(
        [x[:, :11], mx.random.normal((1, 6, 16)).astype(mx.float16)], axis=1
    )
    changed_output = dense_attention_chunked(changed, weights, heads=2, chunk_q=4)
    mx.eval(one_chunk, many_chunks, changed_output)
    np.testing.assert_allclose(
        np.array(one_chunk), np.array(many_chunks), atol=2e-3, rtol=2e-3
    )
    np.testing.assert_allclose(
        np.array(many_chunks)[:, :11],
        np.array(changed_output)[:, :11],
        atol=2e-3,
        rtol=2e-3,
    )
    print("PASS validation harness self-test")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=parse_ints, default=parse_ints(DEFAULT_LENGTHS))
    parser.add_argument(
        "--dense-lengths",
        type=parse_ints,
        default=parse_ints(DEFAULT_DENSE_LENGTHS),
    )
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--max-dense-length", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--tables", type=int, default=4)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--needles", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--chunk-q", type=int, default=1024)
    parser.add_argument("--dense-chunk-q", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--capacity-configs",
        type=parse_capacity_configs,
        default=parse_capacity_configs(DEFAULT_CAPACITY_CONFIGS),
    )
    parser.add_argument(
        "--capacity-seeds",
        type=lambda value: parse_ints(value, minimum=0),
        default=parse_ints("0,1,2", minimum=0),
    )
    parser.add_argument("--capacity-needles", type=int, default=100)
    parser.add_argument(
        "--capacity-batch",
        type=int,
        default=2,
        help="independent sequences evaluated together; keep small for memory safety",
    )
    parser.add_argument(
        "--json-output", default="runs/scaling-validation.json"
    )
    parser.add_argument(
        "--markdown-output", default="runs/scaling-validation.md"
    )
    parser.add_argument("--skip-causality", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.width % args.heads:
        parser.error("--width must be divisible by --heads")
    if args.repeats < 1 or args.capacity_needles < 1 or args.capacity_batch < 1:
        parser.error("--repeats, --capacity-needles, and --capacity-batch must be positive")
    if args.max_dense_length > 8192:
        parser.error("--max-dense-length is capped at 8192 by repository memory policy")

    repo = pathlib.Path(__file__).resolve().parent
    command = " ".join(["python3", pathlib.Path(__file__).name, *sys.argv[1:]])
    report = {
        "metadata": {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "command": command,
            "parameters": vars(args),
        },
        "selector": benchmark_selector(args),
        "sparse_attention": benchmark_sparse_attention(args),
        "dense_attention": benchmark_dense_attention(args),
        "capacity": benchmark_capacity(args),
        "causality": (
            {
                "passed": None,
                "returncode": None,
                "pass_count": 0,
                "pass_lines": [],
                "output": "skipped by --skip-causality",
            }
            if args.skip_causality
            else run_causality_tests(repo)
        ),
    }
    report["finite_range_findings"] = finite_claims(report)

    json_path = repo / args.json_output
    markdown_path = repo / args.markdown_output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(render_markdown(report))
    print(json.dumps(report["finite_range_findings"], indent=2))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
