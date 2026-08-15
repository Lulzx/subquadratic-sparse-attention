"""Paired large-sample quality evaluation for converted LFM2.5 layers."""

import argparse
import json
import math
import pathlib

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import load_dataset
from mlx_lm import load

from mlx_donor_router import DonorHashRouter, language_body
from mlx_lfm_multilayer_eval import parse_layers
from mlx_lfm_replacement import GatedLFMReplacement, wikitext_tokens


def pg19_tokens(tokenizer, split, seq_len, segments, segments_per_book=8):
    dataset = load_dataset("emozilla/pg19", split=split, streaming=True)
    batches = []
    for row in dataset:
        token_ids = tokenizer.encode(row["text"])
        book_segments = 0
        for start in range(0, len(token_ids) - seq_len + 1, seq_len):
            batches.append(mx.array([token_ids[start:start + seq_len]], dtype=mx.int32))
            book_segments += 1
            if len(batches) == segments:
                return batches
            if book_segments == segments_per_book:
                break
    raise ValueError(f"PG-19 yielded only {len(batches)} complete segments")


def segment_losses(model, token_batches, batch_size):
    losses = []
    for start in range(0, len(token_batches), batch_size):
        tokens = mx.concatenate(token_batches[start:start + batch_size], axis=0)
        logits = model(tokens[:, :-1])
        token_loss = nn.losses.cross_entropy(logits, tokens[:, 1:], reduction="none")
        batch_loss = mx.mean(token_loss.astype(mx.float32), axis=-1)
        mx.eval(batch_loss)
        losses.extend(np.array(batch_loss).tolist())
    return np.array(losses, dtype=np.float64)


def paired_summary(dense_losses, sparse_losses, bootstrap_samples, seed):
    delta = sparse_losses - dense_losses
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
    boot_mean = delta[draw].mean(axis=1)
    lower, upper = np.quantile(np.exp(boot_mean), [0.025, 0.975])
    mean_dense = float(dense_losses.mean())
    mean_sparse = float(sparse_losses.mean())
    return {
        "segments": len(delta),
        "dense_loss": mean_dense,
        "sparse_loss": mean_sparse,
        "mean_paired_loss_delta": float(delta.mean()),
        "perplexity_ratio": float(math.exp(delta.mean())),
        "perplexity_ratio_ci95": [float(lower), float(upper)],
        "segments_sparse_better_fraction": float(np.mean(delta < 0)),
        "dense_perplexity": float(math.exp(min(mean_dense, 50.0))),
        "sparse_perplexity": float(math.exp(min(mean_sparse, 50.0))),
    }


def install_replacements(body, config, args, layers):
    replacements = {}
    for layer_index in layers:
        checkpoint = pathlib.Path(args.checkpoint_template.format(
            layer=layer_index, seed=args.seed
        ))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"replacement checkpoint not found: {checkpoint}")
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
    return replacements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layers", default="12,14")
    parser.add_argument("--checkpoint-template", default=(
        "runs/lfm2.5-layer{layer}-joint-12-14-seed{seed}.safetensors"
    ))
    parser.add_argument("--corpora", default="wikitext2,pg19")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--tokens-per-corpus", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
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
    corpora = [value.strip() for value in args.corpora.split(",") if value.strip()]
    segment_count = math.ceil(args.tokens_per_corpus / args.seq_len)

    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    body = language_body(model)
    replacements = install_replacements(body, config, args, layers)
    results = {}
    for corpus_index, corpus in enumerate(corpora):
        if corpus == "wikitext2":
            token_batches = wikitext_tokens(tokenizer, "test", args.seq_len, segment_count)
            provenance = "Salesforce/wikitext wikitext-2-raw-v1 test"
        elif corpus == "pg19":
            token_batches = pg19_tokens(tokenizer, "validation", args.seq_len, segment_count)
            provenance = "emozilla/pg19 validation, max 8 segments per book"
        else:
            parser.error(f"unsupported corpus: {corpus}")

        for replacement in replacements.values():
            replacement.replacement_alpha = 0.0
        dense_losses = segment_losses(model, token_batches, args.batch_size)
        for replacement in replacements.values():
            replacement.replacement_alpha = 1.0
        sparse_losses = segment_losses(model, token_batches, args.batch_size)
        results[corpus] = paired_summary(
            dense_losses, sparse_losses, args.bootstrap_samples,
            args.seed * 100 + corpus_index,
        ) | {"provenance": provenance, "evaluated_tokens": len(token_batches) * args.seq_len}

    result = vars(args) | {
        "layers_resolved": layers,
        "segments_per_corpus": segment_count,
        "results": results,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
    }
    output = pathlib.Path(
        args.output or f"runs/lfm2.5-quality-layers-{'-'.join(map(str, layers))}-seed{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
