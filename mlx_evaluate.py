import argparse
import json
import pathlib

import mlx.core as mx
import numpy as np

from mlx_train import answer_arrays
from ssa.mlx_model import MLXTinyLM
from ssa.tasks import VOCAB, mqar_numpy_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--lengths", default="128,256,512,1024")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--inference-chunk", type=int, default=1024)
    args = parser.parse_args()
    config = json.loads(pathlib.Path(args.checkpoint).with_suffix(".json").read_text())
    model = MLXTinyLM(
        vocab=VOCAB,
        width=config["width"], layers=config["layers"], heads=config["heads"],
        window=config["window"],
    )
    model.load_weights(args.checkpoint)
    for length in map(int, args.lengths.split(",")):
        if length > args.max_length:
            print(json.dumps({"length": length, "skipped": "above --max-length"}), flush=True)
            continue
        mx.reset_peak_memory()
        rng = np.random.default_rng(args.seed + length)
        correct, count = 0, 0
        for _ in range(args.batches):
            tokens, answers = mqar_numpy_batch(args.batch, length, rng)
            batches, positions, targets = answer_arrays(answers)
            logits = model(mx.array(tokens), inference_chunk=args.inference_chunk)
            prediction = logits[batches, positions].argmax(-1)
            mx.eval(prediction)
            correct += int(mx.sum(prediction == targets))
            count += targets.size
        print(json.dumps({
            "length": length,
            "accuracy": correct / count,
            "examples": count,
            "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 2),
        }), flush=True)
        mx.clear_cache()


if __name__ == "__main__":
    main()
