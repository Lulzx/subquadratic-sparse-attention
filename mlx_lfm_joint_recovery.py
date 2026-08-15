"""Jointly recover multiple independently converted LFM2.5 sparse layers."""

import argparse
import json
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load

from mlx_donor_router import DonorHashRouter, language_body
from mlx_lfm_multilayer_eval import parse_layers
from mlx_lfm_replacement import GatedLFMReplacement, perplexity, wikitext_tokens


def hidden_targets(body, token_batches):
    targets = []
    for tokens in token_batches:
        hidden = mx.stop_gradient(body(tokens))
        mx.eval(hidden)
        targets.append(hidden)
    return targets


def hidden_metrics(body, token_batches, targets):
    nrmse = []
    cosine = []
    for tokens, target in zip(token_batches, targets):
        predicted = body(tokens)
        mx.eval(predicted)
        predicted_np = np.array(predicted.astype(mx.float32))
        target_np = np.array(target.astype(mx.float32))
        nrmse.append(
            np.sqrt(np.mean((predicted_np - target_np) ** 2))
            / max(np.sqrt(np.mean(target_np ** 2)), 1e-12)
        )
        numerator = np.sum(predicted_np * target_np, axis=-1)
        denominator = np.linalg.norm(predicted_np, axis=-1) * np.linalg.norm(target_np, axis=-1)
        cosine.append(np.mean(numerator / np.maximum(denominator, 1e-12)))
    return {"final_hidden_nrmse": float(np.mean(nrmse)), "final_hidden_cosine": float(np.mean(cosine))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layers", default="12,14")
    parser.add_argument("--checkpoint-template", default=(
        "runs/lfm2.5-layer{layer}-replacement-wikitext-seed{seed}.safetensors"
    ))
    parser.add_argument("--output-template", default=(
        "runs/lfm2.5-layer{layer}-joint-12-14-seed{seed}.safetensors"
    ))
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--train-segments", type=int, default=32)
    parser.add_argument("--eval-segments", type=int, default=8)
    parser.add_argument("--quality-segments", type=int, default=16)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    layers = parse_layers(args.layers)
    mx.random.seed(args.seed)

    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    body = language_body(model)
    train_tokens = wikitext_tokens(tokenizer, "train", args.seq_len, args.train_segments)
    eval_tokens = wikitext_tokens(tokenizer, "validation", args.seq_len, args.eval_segments)
    quality_tokens = wikitext_tokens(tokenizer, "test", args.seq_len, args.quality_segments)
    train_targets = hidden_targets(body, train_tokens)
    eval_targets = hidden_targets(body, eval_tokens)
    dense_quality = perplexity(model, quality_tokens)

    replacements = {}
    for layer_index in layers:
        checkpoint = pathlib.Path(args.checkpoint_template.format(layer=layer_index, seed=args.seed))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"replacement checkpoint not found: {checkpoint}")
        router = DonorHashRouter(config["hidden_size"], args.tables, args.bits)
        replacement = GatedLFMReplacement(
            body.layers[layer_index].self_attn, router,
            args.window, args.sink_tokens, args.members, args.probes,
            replacement_alpha=1.0,
        )
        replacement.load_weights(str(checkpoint))
        replacement.replacement_alpha = 1.0
        body.layers[layer_index].self_attn = replacement
        replacements[layer_index] = replacement

    before = hidden_metrics(body, eval_tokens, eval_targets)
    sparse_quality_before = perplexity(model, quality_tokens)
    model.freeze()
    for replacement in replacements.values():
        replacement.sparse_attention.unfreeze()
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(current_model, tokens, target):
        predicted = language_body(current_model)(tokens).astype(mx.float32)
        target = target.astype(mx.float32)
        mse = mx.mean(mx.square(predicted - target))
        scale = mx.maximum(mx.mean(mx.square(target)), 1e-8)
        return mse / scale

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        index = (step - 1) % len(train_tokens)
        loss, gradients = loss_and_grad(model, train_tokens[index], train_targets[index])
        optimizer.update(model, gradients)
        mx.eval(model.trainable_parameters(), optimizer.state, loss)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({
                "step": step,
                "normalized_final_hidden_mse": round(float(loss), 6),
                "steps_per_second": round(step / (time.perf_counter() - started), 3),
                "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 2),
            }), flush=True)

    after = hidden_metrics(body, eval_tokens, eval_targets)
    sparse_quality_after = perplexity(model, quality_tokens)
    for replacement in replacements.values():
        replacement.replacement_alpha = 0.0
    gated_dense = perplexity(model, quality_tokens)
    gate_zero_loss_delta = gated_dense["loss"] - dense_quality["loss"]
    if abs(gate_zero_loss_delta) > 1e-7:
        raise RuntimeError("joint recovery changed the frozen dense path")

    checkpoints = {}
    for layer_index, replacement in replacements.items():
        replacement.replacement_alpha = 1.0
        path = pathlib.Path(args.output_template.format(layer=layer_index, seed=args.seed))
        path.parent.mkdir(parents=True, exist_ok=True)
        replacement.save_weights(str(path))
        checkpoints[str(layer_index)] = str(path)
    result = vars(args) | {
        "layers_resolved": layers,
        "dense": dense_quality,
        "before": before,
        "after": after,
        "sparse_quality_before": sparse_quality_before,
        "sparse_quality_after": sparse_quality_after,
        "perplexity_ratio_before": sparse_quality_before["perplexity"] / dense_quality["perplexity"],
        "perplexity_ratio_after": sparse_quality_after["perplexity"] / dense_quality["perplexity"],
        "gate_zero_loss_delta": gate_zero_loss_delta,
        "checkpoints": checkpoints,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = pathlib.Path(
        args.output or f"runs/lfm2.5-joint-12-14-seed{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
