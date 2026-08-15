"""Fine-tune LFM hash routers on supervised distant retrieval positions."""

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
from mlx_lfm_behavior_eval import (
    find_text_span,
    parse_ints,
    retrieval_case,
    set_sparse,
)
from mlx_lfm_multilayer_eval import parse_layers
from mlx_lfm_quality_eval import install_replacements


TRAIN_VALUES = {
    "exact": ("KILO-1042", "RAVEN-6835"),
    "lexical": ("FROST-2941", "AMBER-5076"),
    "variable": ("SOUTH-CEDAR-2468-MOON", "WEST-IRON-1357-CLOUD"),
}


def captured_input(model, tokens, replacement):
    captured = []
    original = replacement.candidate_indices

    def capture(x):
        captured.append(x)
        return original(x)

    replacement.candidate_indices = capture
    try:
        logits = model(tokens)
        mx.eval(logits, *captured)
    finally:
        replacement.candidate_indices = original
    if len(captured) != 1:
        raise RuntimeError("target replacement did not run exactly once")
    return np.array(captured[0].astype(mx.float16))


def training_targets(tokenizer, lengths, positions, window):
    targets = []
    base_cases = 0
    skipped_local = 0
    for task, values in TRAIN_VALUES.items():
        for value in values:
            for length in lengths:
                for position in positions:
                    case = retrieval_case(tokenizer, task, length, position, value=value)
                    prompt_ids = tokenizer.encode(
                        case["prompt"], add_special_tokens=False
                    )
                    answer_ids = tokenizer.encode(value, add_special_tokens=False)
                    source_span = find_text_span(tokenizer, prompt_ids, value)
                    if source_span is None:
                        raise RuntimeError(f"could not locate source value {value}")
                    source_start = source_span[1] - len(answer_ids)
                    base_cases += 1
                    for offset in range(len(answer_ids)):
                        prefix_ids = prompt_ids + answer_ids[:offset]
                        source_position = source_start + offset
                        if source_position >= len(prefix_ids) - window:
                            skipped_local += 1
                            continue
                        targets.append((
                            prefix_ids,
                            source_position,
                            len(prefix_ids) - 1,
                        ))
    return targets, base_cases, skipped_local


def hard_source_recall(
    model, replacement, targets, members, probes, window, sample_limit=48
):
    from ssa.mlx_selector import select_indices_qk

    recalled = []
    candidate_counts = []
    stride = max(1, len(targets) // sample_limit)
    sampled = targets[::stride][:sample_limit]
    for prefix_ids, source_position, query_position in sampled:
        tokens = mx.array([prefix_ids], dtype=mx.int32)
        x_np = captured_input(model, tokens, replacement)
        x = mx.array(x_np)
        selected = select_indices_qk(
            x,
            x,
            replacement.router.query_projection,
            replacement.router.key_projection,
            tables=replacement.router.tables,
            bits=replacement.router.bits,
            members=members,
            probes=probes,
            block=False,
            min_distance=window,
        )
        mx.eval(selected)
        indices = np.array(selected[0, query_position])
        recalled.append(source_position in indices)
        candidate_counts.append(len(np.unique(indices[indices >= 0])))
        del tokens, x, x_np, selected
        mx.clear_cache()
    return {
        "source_token_recall": float(np.mean(recalled)),
        "examples": len(recalled),
        "mean_unique_candidates": float(np.mean(candidate_counts)),
    }


def train_router(model, replacement, targets, args, layer):
    router = replacement.router
    router.unfreeze()
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(current_router, x, teacher, query_start):
        loss, parts = current_router.loss(
            x,
            teacher,
            query_start,
            args.window,
            args.sink_tokens,
            args.alignment_weight,
            args.balance_weight,
        )
        return loss, parts

    loss_and_grad = nn.value_and_grad(router, loss_fn)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        prefix_ids, source_position, query_position = targets[(step - 1) % len(targets)]
        tokens = mx.array([prefix_ids], dtype=mx.int32)
        x_np = captured_input(model, tokens, replacement)
        x = mx.array(x_np)
        teacher = mx.zeros((1, 1, x.shape[1]), dtype=mx.float32)
        teacher = teacher.at[0, 0, source_position].add(1.0)
        (loss, parts), gradients = loss_and_grad(router, x, teacher, query_position)
        optimizer.update(router, gradients)
        mx.eval(router.parameters(), optimizer.state, loss, parts)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({
                "layer": layer,
                "step": step,
                "loss": round(float(loss), 5),
                "cross_entropy": round(float(parts[0]), 5),
                "alignment": round(float(parts[1]), 5),
                "steps_per_second": round(
                    step / (time.perf_counter() - started), 3
                ),
                "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 2),
            }), flush=True)
        del tokens, x, x_np, teacher, loss, parts, gradients
        mx.clear_cache()
    router.freeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layers", default="12,14")
    parser.add_argument("--checkpoint-template", default=(
        "runs/lfm2.5-layer{layer}-joint-kl-12-14-seed{seed}.safetensors"
    ))
    parser.add_argument("--output-template", default=(
        "runs/lfm2.5-layer{layer}-retrieval-router-12-14-seed{seed}.safetensors"
    ))
    parser.add_argument("--lengths", default="256,512")
    parser.add_argument("--positions", default="0.1,0.5")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--balance-weight", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--memory-limit-mb", type=int, default=1792)
    parser.add_argument("--cache-limit-mb", type=int, default=256)
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
    set_sparse(replacements, True)
    targets, base_cases, skipped_local = training_targets(
        tokenizer, lengths, positions, args.window
    )

    results = {}
    for layer, replacement in replacements.items():
        before = hard_source_recall(
            model,
            replacement,
            targets,
            args.members,
            args.probes,
            args.window,
        )
        train_router(model, replacement, targets, args, layer)
        after = hard_source_recall(
            model,
            replacement,
            targets,
            args.members,
            args.probes,
            args.window,
        )
        checkpoint = pathlib.Path(args.output_template.format(
            layer=layer, seed=args.seed
        ))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        replacement.save_weights(str(checkpoint))
        results[str(layer)] = {
            "before": before,
            "after": after,
            "checkpoint": str(checkpoint),
        }

    result = vars(args) | {
        "layers_resolved": layers,
        "base_cases": base_cases,
        "skipped_local_targets": skipped_local,
        "training_examples_per_layer": len(targets),
        "results": results,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
    }
    output = pathlib.Path(
        args.output or f"runs/lfm2.5-retrieval-router-12-14-seed{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
