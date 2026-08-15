import argparse
import json
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from ssa.mlx_model import MLXTinyLM
from ssa.tasks import VOCAB, mqar_numpy_batch


def answer_arrays(answers):
    batches, positions, targets = [], [], []
    for batch, pairs in enumerate(answers):
        for position, target in pairs:
            batches.append(batch)
            positions.append(position - 1)
            targets.append(target)
    return mx.array(batches), mx.array(positions), mx.array(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--output", default="runs/mlx-model.safetensors")
    args = parser.parse_args()
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = MLXTinyLM(
        vocab=VOCAB, width=args.width, layers=args.layers,
        heads=args.heads, window=args.window,
    )
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(model, tokens, batches, positions, targets):
        logits = model(tokens)
        selected = logits[batches, positions]
        task = nn.losses.cross_entropy(selected, targets, reduction="mean")
        return task + args.aux_weight * model.router_loss(tokens), task

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        tokens, answers = mqar_numpy_batch(args.batch, args.seq_len, rng)
        batches, positions, targets = answer_arrays(answers)
        tokens_mx = mx.array(tokens)
        (loss, task), gradients = loss_and_grad(model, tokens_mx, batches, positions, targets)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss, task)
        if step == 1 or step % args.log_every == 0:
            logits = model(tokens_mx)
            prediction = logits[batches, positions].argmax(-1)
            accuracy = mx.mean(prediction == targets)
            mx.eval(accuracy)
            print(json.dumps({
                "step": step,
                "loss": round(float(loss), 4),
                "task_loss": round(float(task), 4),
                "accuracy": round(float(accuracy), 4),
                "steps_per_second": round(step / (time.perf_counter() - started), 3),
                "active_memory_mb": round(mx.get_active_memory() / 2**20, 2),
                "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 2),
            }), flush=True)
    eval_rng = np.random.default_rng(args.seed + 1)
    correct, count = 0, 0
    for _ in range(args.eval_batches):
        tokens, answers = mqar_numpy_batch(args.batch, args.seq_len, eval_rng)
        batches, positions, targets = answer_arrays(answers)
        logits = model(mx.array(tokens))
        prediction = logits[batches, positions].argmax(-1)
        mx.eval(prediction)
        correct += int(mx.sum(prediction == targets))
        count += targets.size
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output))
    output.with_suffix(".json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(json.dumps({
        "validation_accuracy": correct / count,
        "validation_examples": count,
        "checkpoint": str(output),
    }), flush=True)


if __name__ == "__main__":
    main()
