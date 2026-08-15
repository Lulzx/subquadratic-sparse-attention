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


def parse_train_lengths(spec, fallback):
    lengths = [int(value) for value in spec.split(",") if value.strip()] if spec else [fallback]
    if not lengths or any(length < 8 for length in lengths):
        raise ValueError("training lengths must be at least 8")
    return sorted(set(lengths))


def curriculum_length(step, steps, lengths):
    """Advance through equal-duration, shortest-to-longest curriculum stages."""
    stage = min(len(lengths) - 1, (step - 1) * len(lengths) // max(1, steps))
    return lengths[stage]


def scaled_batch_size(base_batch, shortest_length, length):
    """Keep the approximate number of tokens per optimizer step bounded."""
    return max(1, base_batch * shortest_length // length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--train-lengths", default="",
        help="comma-separated shortest-to-longest curriculum; overrides --seq-len",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--tables", type=int, default=4)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument(
        "--probes", type=int, default=1,
        help="codes per table: exact code plus lowest-margin one-bit neighbors",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument(
        "--resume", default="",
        help="initialize from an existing safetensors checkpoint; optimizer state starts fresh",
    )
    parser.add_argument("--output", default="runs/mlx-model.safetensors")
    args = parser.parse_args()
    train_lengths = parse_train_lengths(args.train_lengths, args.seq_len)
    if args.steps < len(train_lengths):
        parser.error("--steps must be at least the number of curriculum lengths")
    if args.probes < 1 or args.probes > args.bits + 1:
        parser.error("--probes must be between 1 and --bits + 1")
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = MLXTinyLM(
        vocab=VOCAB, width=args.width, layers=args.layers,
        heads=args.heads, window=args.window,
        tables=args.tables, bits=args.bits, members=args.members, probes=args.probes,
    )
    if args.resume:
        resume_path = pathlib.Path(args.resume)
        if not resume_path.is_file():
            parser.error(f"resume checkpoint does not exist: {resume_path}")
        model.load_weights(str(resume_path))
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(model, tokens, batches, positions, targets):
        logits = model(tokens)
        selected = logits[batches, positions]
        task = nn.losses.cross_entropy(selected, targets, reduction="mean")
        return task + args.aux_weight * model.router_loss(tokens), task

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        train_length = curriculum_length(step, args.steps, train_lengths)
        step_batch = scaled_batch_size(args.batch, train_lengths[0], train_length)
        tokens, answers = mqar_numpy_batch(step_batch, train_length, rng)
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
                "train_length": train_length,
                "batch": step_batch,
                "loss": round(float(loss), 4),
                "task_loss": round(float(task), 4),
                "accuracy": round(float(accuracy), 4),
                "steps_per_second": round(step / (time.perf_counter() - started), 3),
                "active_memory_mb": round(mx.get_active_memory() / 2**20, 2),
                "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 2),
            }), flush=True)
    validation = {}
    for eval_length in train_lengths:
        eval_rng = np.random.default_rng(args.seed + 1 + eval_length)
        eval_batch = scaled_batch_size(args.batch, train_lengths[0], eval_length)
        correct, count = 0, 0
        for _ in range(args.eval_batches):
            tokens, answers = mqar_numpy_batch(eval_batch, eval_length, eval_rng)
            batches, positions, targets = answer_arrays(answers)
            logits = model(mx.array(tokens))
            prediction = logits[batches, positions].argmax(-1)
            mx.eval(prediction)
            correct += int(mx.sum(prediction == targets))
            count += targets.size
        validation[str(eval_length)] = {"accuracy": correct / count, "examples": count}
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output))
    metadata = vars(args) | {"train_lengths": train_lengths}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "validation": validation,
        "checkpoint": str(output),
    }), flush=True)


if __name__ == "__main__":
    main()
