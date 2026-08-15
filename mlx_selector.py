import argparse
import json
import time

import mlx.core as mx
import numpy as np

from ssa.mlx_selector import select_indices


def timed_selector(x, projection, repeats, **kwargs):
    mx.eval(select_indices(x, projection, **kwargs))
    mx.synchronize()
    samples = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = select_indices(x, projection, **kwargs)
        mx.eval(output)
        mx.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return sorted(samples)[len(samples) // 2], output


def make_needles(length, dim, count, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, length, dim), dtype=np.float32)
    stored = np.linspace(32, length // 2, count, endpoint=False, dtype=np.int32)
    queries = np.arange(length - count, length, dtype=np.int32)
    x[0, queries] = x[0, stored]
    return mx.array(x), stored, queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="1024,2048,4096,8192,16384")
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--tables", type=int, default=4)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--needles", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    projection = mx.array(
        rng.standard_normal((args.dim, args.tables * args.bits), dtype=np.float32)
        / np.sqrt(args.dim)
    )
    for length in map(int, args.lengths.split(",")):
        if length > args.max_length:
            print(json.dumps({"length": length, "skipped": "above --max-length"}), flush=True)
            continue
        mx.reset_peak_memory()
        x, stored, queries = make_needles(length, args.dim, args.needles, args.seed + length)
        ms, selected = timed_selector(
            x, projection, args.repeats,
            tables=args.tables, bits=args.bits, members=args.members,
            probes=args.probes,
        )
        selected_np = np.array(selected)[0, queries]
        hits = sum(
            bool(np.any(row == target) or np.any(row == target + 1))
            for row, target in zip(selected_np, stored)
        )
        print(json.dumps({
            "length": length,
            "milliseconds": round(ms, 3),
            "recall": hits / len(stored),
            "selected_per_query": int(selected.shape[-1]),
            "active_memory_mb": round(mx.get_active_memory() / 2**20, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 2),
        }), flush=True)
        del x, selected
        mx.clear_cache()


if __name__ == "__main__":
    main()
