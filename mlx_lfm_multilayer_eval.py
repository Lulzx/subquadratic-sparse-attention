"""Evaluate independently aligned LFM2.5 sparse replacements in combination."""

import argparse
import json
import pathlib

import mlx.core as mx
from mlx_lm import load

from mlx_donor_router import DonorHashRouter, language_body
from mlx_lfm_replacement import GatedLFMReplacement, perplexity, wikitext_tokens


def parse_layers(spec):
    layers = [int(value.strip()) for value in spec.split(",") if value.strip()]
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("--layers must contain unique comma-separated indices")
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layers", default="12,14")
    parser.add_argument("--checkpoint-template", default=(
        "runs/lfm2.5-layer{layer}-replacement-wikitext-seed{seed}.safetensors"
    ))
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--quality-segments", type=int, default=16)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    layers = parse_layers(args.layers)

    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    body = language_body(model)
    for layer_index in layers:
        if layer_index < 0 or layer_index >= len(body.layers) or not hasattr(
            body.layers[layer_index], "self_attn"
        ):
            parser.error(f"layer {layer_index} is not a full-attention layer")
    quality_tokens = wikitext_tokens(
        tokenizer, "test", args.seq_len, args.quality_segments
    )
    original_dense = perplexity(model, quality_tokens)

    replacements = {}
    checkpoint_metadata = {}
    for layer_index in layers:
        checkpoint = pathlib.Path(args.checkpoint_template.format(
            layer=layer_index, seed=args.seed
        ))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"replacement checkpoint not found: {checkpoint}")
        metadata_path = checkpoint.with_suffix(".json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("layer") != layer_index or metadata.get("seed") != args.seed:
                raise ValueError(f"checkpoint metadata mismatch: {metadata_path}")
            checkpoint_metadata[str(layer_index)] = {
                key: metadata.get(key) for key in (
                    "layer", "seed", "window", "tables", "bits", "members", "probes",
                    "perplexity_ratio",
                )
            }
        router = DonorHashRouter(config["hidden_size"], args.tables, args.bits)
        replacement = GatedLFMReplacement(
            body.layers[layer_index].self_attn, router,
            args.window, args.sink_tokens, args.members, args.probes,
            replacement_alpha=0.0,
        )
        replacement.load_weights(str(checkpoint))
        replacement.replacement_alpha = 0.0
        body.layers[layer_index].self_attn = replacement
        replacements[layer_index] = replacement

    gated_dense = perplexity(model, quality_tokens)
    gate_zero_loss_delta = gated_dense["loss"] - original_dense["loss"]
    if abs(gate_zero_loss_delta) > 1e-7:
        raise RuntimeError("all-zero replacement gates do not reproduce the donor")

    individual = {}
    for layer_index, replacement in replacements.items():
        replacement.replacement_alpha = 1.0
        quality = perplexity(model, quality_tokens)
        individual[str(layer_index)] = quality | {
            "perplexity_ratio": quality["perplexity"] / original_dense["perplexity"]
        }
        replacement.replacement_alpha = 0.0

    for replacement in replacements.values():
        replacement.replacement_alpha = 1.0
    combined = perplexity(model, quality_tokens)
    combined_ratio = combined["perplexity"] / original_dense["perplexity"]

    result = vars(args) | {
        "layers_resolved": layers,
        "quality_dataset": "Salesforce/wikitext wikitext-2-raw-v1 test",
        "dense": original_dense,
        "gate_zero_loss_delta": gate_zero_loss_delta,
        "individual": individual,
        "combined": combined,
        "combined_perplexity_ratio": combined_ratio,
        "independent_penalty_product": float(
            mx.prod(mx.array([
                value["perplexity_ratio"] for value in individual.values()
            ]))
        ),
        "checkpoint_metadata": checkpoint_metadata,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
    }
    output = pathlib.Path(
        args.output or f"runs/lfm2.5-layers-{'-'.join(map(str, layers))}-seed{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
