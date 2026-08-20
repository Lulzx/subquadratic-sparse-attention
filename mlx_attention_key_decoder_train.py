"""Fit four-byte router codes to normalized LFM attention-key heads."""

import argparse
import hashlib
import json
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load

from mlx_donor_router import language_body
from mlx_hierarchical_router_train import corpus_segments, parse_corpora
from mlx_lfm_hierarchical_eval import binary_fingerprint_bytes, capture_layer


class AttentionKeyDecoder(nn.Module):
    def __init__(self, tables, heads, head_dim):
        super().__init__()
        self.embeddings = mx.random.normal(
            (tables, 256, heads, head_dim)
        ) * 0.01

    def __call__(self, codes):
        return mx.sum(mx.stack([
            self.embeddings[table, codes[:, table]]
            for table in range(codes.shape[-1])
        ], axis=1), axis=1)


def apply_rope(values, base=1_000_000.0):
    dimensions = values.shape[-1]
    frequencies = mx.exp(
        -mx.arange(0, dimensions, 2, dtype=mx.float32)
        * (np.log(base) / dimensions)
    )
    positions = mx.arange(values.shape[0], dtype=mx.float32)
    angles = positions[:, None] * frequencies[None, :]
    cosine = mx.cos(angles)[:, None, :]
    sine = mx.sin(angles)[:, None, :]
    first, second = mx.split(values, 2, axis=-1)
    return mx.concatenate([
        first * cosine - second * sine,
        first * sine + second * cosine,
    ], axis=-1)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--router", required=True, type=pathlib.Path)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--fingerprint-bytes", type=int, default=4)
    parser.add_argument("--corpora", default="wikitext2,pg19")
    parser.add_argument("--train-segments-per-corpus", type=int, default=64)
    parser.add_argument("--eval-segments-per-corpus", type=int, default=8)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument(
        "--objective", choices=("reconstruction", "attention"),
        default="reconstruction",
    )
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--memory-limit-mb", type=int, default=1792)
    parser.add_argument("--cache-limit-mb", type=int, default=64)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if not args.router.is_file():
        parser.error("--router does not exist")
    corpora = parse_corpora(args.corpora)
    mx.set_memory_limit(args.memory_limit_mb * 2**20)
    mx.set_cache_limit(args.cache_limit_mb * 2**20)
    mx.random.seed(args.seed)

    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    layer = language_body(model).layers[args.layer]
    attention = layer.self_attn
    router_weights = mx.load(str(args.router))
    key_projection = np.array(
        router_weights["rerank_key_projection"].astype(mx.float16).astype(mx.float32)
    )[:, :args.fingerprint_bytes * 8]
    train_rows = corpus_segments(
        tokenizer, corpora, args.seq_len,
        args.train_segments_per_corpus, skip=0,
    )
    eval_rows = corpus_segments(
        tokenizer, corpora, args.seq_len,
        args.eval_segments_per_corpus, skip=args.train_segments_per_corpus,
    )
    examples = []
    for partition, rows in (("train", train_rows), ("eval", eval_rows)):
        for corpus, tokens in rows:
            capture = capture_layer(model, tokens, args.layer)
            hidden = np.array(capture["x"][0].astype(mx.float32)).copy()
            codes = binary_fingerprint_bytes(hidden, key_projection).astype(np.int32)
            if args.objective == "reconstruction":
                target = attention.k_proj(capture["x"]).reshape(
                    1, args.seq_len, attention.n_kv_heads, attention.head_dim
                )
                target = attention.k_layernorm(target)[0].astype(mx.float32)
                query_heads = mx.zeros((1, 1, 1), dtype=mx.float32)
            else:
                query_start = 32 + 4 + 1
                query_heads = capture["queries"][0, :, query_start:].transpose(
                    1, 0, 2
                ).astype(mx.float32)
                exact_keys = capture["keys"][0].astype(mx.float32)
                exact_logits = mx.einsum(
                    "qhd,hkd->qhk", query_heads, exact_keys
                ) * attention.scale
                query_positions = mx.arange(
                    query_start, args.seq_len
                ).reshape(-1, 1)
                key_positions = mx.arange(args.seq_len).reshape(1, -1)
                eligible = (key_positions < query_positions - 32) & (
                    key_positions >= 4
                )
                exact_logits = mx.where(
                    eligible[:, None, :], exact_logits,
                    mx.array(-1e9, exact_logits.dtype),
                )
                target = mx.mean(
                    mx.softmax(exact_logits, axis=-1), axis=1
                ).astype(mx.float32)
            target = mx.stop_gradient(target)
            query_heads = mx.stop_gradient(query_heads)
            mx.eval(target, query_heads)
            examples.append((
                partition, corpus, mx.array(codes), target, query_heads
            ))
            del capture
            mx.clear_cache()

    train_examples = [row[1:] for row in examples if row[0] == "train"]
    eval_examples = [row[1:] for row in examples if row[0] == "eval"]
    decoder = AttentionKeyDecoder(
        args.fingerprint_bytes, attention.n_kv_heads, attention.head_dim
    )
    if "attention_key_decoder" in router_weights:
        decoder.embeddings = router_weights["attention_key_decoder"]
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)

    def loss_fn(module, codes, target, query_heads):
        prediction = module(codes)
        if args.objective == "reconstruction":
            return mx.mean(mx.square(prediction - target))
        decoded_keys = apply_rope(prediction)
        decoded_keys = mx.repeat(
            decoded_keys, query_heads.shape[1] // decoded_keys.shape[1], axis=1
        )
        logits = mx.einsum(
            "qhd,khd->qkh", query_heads, decoded_keys
        ) * attention.scale
        scores = mx.logsumexp(logits, axis=-1)
        query_start = args.seq_len - query_heads.shape[0]
        query_positions = mx.arange(query_start, args.seq_len).reshape(-1, 1)
        key_positions = mx.arange(args.seq_len).reshape(1, -1)
        eligible = (key_positions < query_positions - 32) & (key_positions >= 4)
        scores = mx.where(eligible, scores, mx.array(-1e9, scores.dtype))
        log_probability = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
        return -mx.mean(mx.sum(target * log_probability, axis=-1))

    loss_and_grad = nn.value_and_grad(decoder, loss_fn)

    def mean_loss(rows):
        values = []
        for _, codes, target, query_heads in rows:
            value = loss_fn(decoder, codes, target, query_heads)
            mx.eval(value)
            values.append(float(value))
        return float(np.mean(values))

    before = mean_loss(eval_examples)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_examples))
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        if (step - 1) % len(order) == 0 and step > 1:
            order = rng.permutation(len(train_examples))
        _, codes, target, query_heads = train_examples[
            order[(step - 1) % len(order)]
        ]
        loss, gradients = loss_and_grad(
            decoder, codes, target, query_heads
        )
        optimizer.update(decoder, gradients)
        mx.eval(decoder.parameters(), optimizer.state, loss)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({
                "step": step,
                "key_mse": float(loss),
                "steps_per_second": step / (time.perf_counter() - started),
                "peak_memory_mb": mx.get_peak_memory() / 2**20,
            }), flush=True)

    after = mean_loss(eval_examples)
    output_weights = dict(router_weights)
    output_weights["attention_key_decoder"] = decoder.embeddings
    output_weights["attention_query_weight"] = attention.q_proj.weight
    output_weights["attention_query_norm"] = attention.q_layernorm.weight
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), output_weights)
    metadata = vars(args) | {
        "router": str(args.router),
        "router_sha256": sha256(args.router),
        "before_key_mse": before,
        "after_key_mse": after,
        "kv_heads": attention.n_kv_heads,
        "query_heads": attention.n_heads,
        "head_dim": attention.head_dim,
        "rope_base": attention.rope.base,
        "rope_traditional": attention.rope.traditional,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
