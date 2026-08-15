import argparse
import json
import time

import mlx.core as mx

from ssa.mlx_attention import random_weights, sparse_attention_chunked


def timed(x, weights, heads, repeats, chunk_q):
    warm_output, warm_selected = sparse_attention_chunked(
        x, *weights, heads=heads, chunk_q=chunk_q
    )
    mx.eval(warm_output, warm_selected)
    mx.synchronize()
    del warm_output, warm_selected
    samples, output, selected = [], None, None
    for _ in range(repeats):
        started = time.perf_counter()
        output, selected = sparse_attention_chunked(
            x, *weights, heads=heads, chunk_q=chunk_q
        )
        mx.eval(output, selected)
        mx.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return sorted(samples)[len(samples) // 2], output, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="1024,2048,4096,8192,16384")
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chunk-q", type=int, default=1024)
    args = parser.parse_args()
    weights = random_weights(args.width)
    for length in map(int, args.lengths.split(",")):
        if length > args.max_length:
            print(json.dumps({"length": length, "skipped": "above --max-length"}), flush=True)
            continue
        mx.reset_peak_memory()
        x = mx.random.normal((1, length, args.width)).astype(mx.float16)
        milliseconds, output, selected = timed(
            x, weights, args.heads, args.repeats, args.chunk_q
        )
        print(json.dumps({
            "length": length,
            "milliseconds": round(milliseconds, 3),
            "output_shape": list(output.shape),
            "selected_per_query": int(selected.shape[-1]),
            "active_memory_mb": round(mx.get_active_memory() / 2**20, 2),
            "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 2),
        }), flush=True)
        del x, output, selected
        mx.clear_cache()


if __name__ == "__main__":
    main()
