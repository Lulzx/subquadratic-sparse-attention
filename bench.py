import argparse
import json
import math
import time

import torch

from ssa.experiment import device_name
from ssa.model import DenseAttention, SSAAttention


def analytic(n, d, heads, window, selected):
    projections = 8 * n * d * d
    dense_attention = 4 * n * n * d
    sparse_attention = 4 * n * (window + selected) * d
    return {
        "length": n,
        "dense_flops": projections + dense_attention,
        "ssa_flops": projections + sparse_attention,
        "reduction": (projections + dense_attention) / (projections + sparse_attention),
    }


@torch.no_grad()
def median_ms(module, x, warmup, repeats, device):
    for _ in range(warmup):
        output = module(x)
        del output
    if device == "mps":
        torch.mps.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = module(x)
        if device == "mps":
            torch.mps.synchronize()
        values.append((time.perf_counter() - start) * 1000)
        del output
    return sorted(values)[len(values) // 2]


def slope(points):
    xs = [math.log(n) for n, _ in points]
    ys = [math.log(t) for _, t in points]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", default="256,512,1024,2048,4096,8192")
    p.add_argument("--d", type=int, default=96)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--selected", type=int, default=32)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max-dense-length", type=int, default=8192)
    p.add_argument("--max-ssa-length", type=int, default=16384)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = device_name(args.device)
    dense = DenseAttention(args.d, args.heads).to(device).eval()
    ssa = SSAAttention(
        args.d, args.heads, window=args.window, n_select_buckets=4,
        members_per_bucket=args.selected // 8, capacity=512, chunk_q=512,
    ).to(device).eval()
    dense_points, ssa_points = [], []
    for n in map(int, args.lengths.split(",")):
        if n > args.max_ssa_length:
            print(json.dumps({"length": n, "skipped": "above --max-ssa-length"}), flush=True)
            continue
        x = torch.randn(1, n, args.d, device=device)
        row = analytic(n, args.d, args.heads, args.window, args.selected)
        if n <= args.max_dense_length:
            try:
                row["dense_ms"] = median_ms(dense, x, 1, args.repeats, device)
                dense_points.append((n, row["dense_ms"]))
            except RuntimeError as error:
                row["dense_error"] = str(error).splitlines()[0]
                if device == "mps":
                    torch.mps.empty_cache()
        else:
            row["dense_skipped"] = f"length exceeds safe limit {args.max_dense_length}"
        row["ssa_ms"] = median_ms(ssa, x, 1, args.repeats, device)
        ssa_points.append((n, row["ssa_ms"]))
        print(json.dumps(row), flush=True)
        del x
        if device == "mps":
            torch.mps.empty_cache()
    print(json.dumps({
        "dense_empirical_exponent": slope(dense_points) if len(dense_points) > 1 else None,
        "ssa_empirical_exponent": slope(ssa_points),
    }, indent=2))


if __name__ == "__main__":
    main()
