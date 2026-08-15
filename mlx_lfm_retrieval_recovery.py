"""Recover sparse LFM attention on retrieval answer positions without caching states."""

import argparse
import json
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load

from mlx_donor_router import language_body
from mlx_lfm_behavior_eval import parse_ints, set_sparse
from mlx_lfm_multilayer_eval import parse_layers
from mlx_lfm_quality_eval import install_replacements
from mlx_lfm_retrieval_router import TRAIN_VALUES, training_targets


VARIABLE_WORDS = (
    "CEDAR", "RIVER", "STONE", "CLOUD", "FIELD", "DAWN", "PINE", "FLAME",
    "MAPLE", "OCEAN", "RIDGE", "COMET", "BIRCH", "GLASS", "NIGHT", "CORAL",
    "NORTH", "SOUTH", "EAST", "WEST", "EMBER", "QUARTZ", "IRON", "MOON",
)


def diverse_train_values(count):
    if count <= 0:
        return TRAIN_VALUES
    values = []
    for index in range(count):
        first = VARIABLE_WORDS[index % len(VARIABLE_WORDS)]
        second = VARIABLE_WORDS[(index * 5 + 3) % len(VARIABLE_WORDS)]
        last = VARIABLE_WORDS[(index * 7 + 9) % len(VARIABLE_WORDS)]
        digits = 1000 + (index * 7919) % 9000
        values.append(f"{first}-{second}-{digits:04d}-{last}")
    return dict(TRAIN_VALUES) | {"variable": tuple(values)}


def teacher_target(model, tokens, topk):
    logits = model(tokens)[:, -1].astype(mx.float32)
    indices = mx.argpartition(logits, kth=-topk, axis=-1)[..., -topk:]
    selected_logits = mx.take_along_axis(logits, indices, axis=-1)
    log_normalizer = mx.logsumexp(logits, axis=-1, keepdims=True)
    probability = mx.exp(selected_logits - log_normalizer)
    other_probability = mx.maximum(
        1.0 - mx.sum(probability, axis=-1), mx.array(1e-8)
    )
    values = tuple(mx.stop_gradient(value) for value in (
        indices, probability, other_probability
    ))
    mx.eval(*values)
    del logits, selected_logits, log_normalizer
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layers", default="12,14")
    parser.add_argument("--checkpoint-template", default=(
        "runs/lfm2.5-layer{layer}-retrieval-router-12-14-seed{seed}.safetensors"
    ))
    parser.add_argument("--output-template", default=(
        "runs/lfm2.5-layer{layer}-retrieval-recovered-12-14-seed{seed}.safetensors"
    ))
    parser.add_argument("--lengths", default="256")
    parser.add_argument("--positions", default="0.1,0.5")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--span-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--teacher-topk", type=int, default=8)
    parser.add_argument("--objective", choices=["kl", "lm"], default="kl")
    parser.add_argument("--variable-values", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--memory-limit-mb", type=int, default=1400)
    parser.add_argument("--cache-limit-mb", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    layers = parse_layers(args.layers)
    lengths = parse_ints(args.lengths)
    positions = [float(value) for value in args.positions.split(",")]
    mx.set_memory_limit(args.memory_limit_mb * 2**20)
    mx.set_cache_limit(args.cache_limit_mb * 2**20)
    mx.random.seed(args.seed)

    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    replacements = install_replacements(language_body(model), config, args, layers)
    targets, base_cases, skipped_local = training_targets(
        tokenizer,
        lengths,
        positions,
        args.window,
        diverse_train_values(args.variable_values),
    )
    if args.variable_values:
        permutation = np.random.default_rng(args.seed).permutation(len(targets))
        targets = [targets[index] for index in permutation]
    model.freeze()
    for replacement in replacements.values():
        replacement.sparse_attention.unfreeze()
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(current_model, tokens, target):
        if args.objective == "lm":
            logits = current_model(tokens)[:, -1]
            return nn.losses.cross_entropy(logits, target, reduction="mean")
        indices, teacher_probability, teacher_other_probability = target
        logits = current_model(tokens)[:, -1].astype(mx.float32)
        log_normalizer = mx.logsumexp(logits, axis=-1, keepdims=True)
        selected_log_probability = (
            mx.take_along_axis(logits, indices, axis=-1) - log_normalizer
        )
        selected_probability = mx.exp(selected_log_probability)
        other_log_probability = mx.log(mx.maximum(
            1.0 - mx.sum(selected_probability, axis=-1), mx.array(1e-8)
        ))
        return -mx.mean(
            mx.sum(teacher_probability * selected_log_probability, axis=-1)
            + teacher_other_probability * other_log_probability
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        prefix_ids, _, _, expected_token_id = targets[(step - 1) % len(targets)]
        tokens = mx.array([prefix_ids], dtype=mx.int32)
        if args.objective == "kl":
            set_sparse(replacements, False)
            target = teacher_target(model, tokens, args.teacher_topk)
        else:
            target = mx.array([expected_token_id], dtype=mx.int32)
        set_sparse(replacements, True)
        loss, gradients = loss_and_grad(model, tokens, target)
        optimizer.update(model, gradients)
        mx.eval(model.trainable_parameters(), optimizer.state, loss)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({
                "step": step,
                "loss": round(float(loss), 6),
                "steps_per_second": round(
                    step / (time.perf_counter() - started), 3
                ),
                "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 2),
            }), flush=True)
        del tokens, target, loss, gradients
        mx.clear_cache()

    checkpoints = {}
    for layer, replacement in replacements.items():
        checkpoint = pathlib.Path(args.output_template.format(
            layer=layer, seed=args.seed
        ))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        replacement.save_weights(str(checkpoint))
        checkpoints[str(layer)] = str(checkpoint)
    result = vars(args) | {
        "layers_resolved": layers,
        "base_cases": base_cases,
        "skipped_local_targets": skipped_local,
        "training_targets": len(targets),
        "checkpoints": checkpoints,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = pathlib.Path(
        args.output or f"runs/lfm2.5-retrieval-recovery-12-14-seed{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
