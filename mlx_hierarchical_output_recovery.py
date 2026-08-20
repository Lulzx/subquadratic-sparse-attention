"""Align a trainable sparse attention layer to hierarchical K=32 candidates.

The router and its discrete candidate sets stay frozen.  Only a copy of the
donor attention projections is trained against the donor's dense attention
output, preserving the routing budget while testing whether the fixed-K
quality ceiling can be recovered downstream.
"""

import argparse
import copy
import hashlib
import json
import math
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load

from mlx_donor_router import HierarchicalAttentionRouter, language_body
from mlx_hierarchical_router_train import corpus_segments, parse_corpora
from mlx_lfm_hierarchical_eval import (
    attention_qkv,
    binary_fingerprint_bytes,
    capture_layer,
    causal_hierarchical_candidates,
    router_codes,
    sparse_attention_output,
    tail_loss,
    parse_ints,
)
from mlx_lfm_quality_eval import pg19_tokens
from mlx_lfm_replacement import wikitext_tokens


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_router(path, width, tables, bits, fingerprint_bytes):
    weights = mx.load(str(path))
    router = HierarchicalAttentionRouter(
        width, tables, bits, rerank_bytes=fingerprint_bytes
    )
    router.query_projection = weights["query_projection"]
    router.key_projection = weights["key_projection"]
    rerank_width = fingerprint_bytes * 8
    router.rerank_query_projection = weights.get(
        "rerank_query_projection", weights["query_projection"]
    )[:, :rerank_width]
    router.rerank_key_projection = weights.get(
        "rerank_key_projection", weights["key_projection"]
    )[:, :rerank_width]
    if "rerank_bit_weights" in weights:
        router.rerank_bit_weights = weights["rerank_bit_weights"][:rerank_width]
    if "rerank_decoder_query" in weights:
        router.rerank_decoder_query = weights["rerank_decoder_query"]
    if "rerank_decoder_keys" in weights:
        router.rerank_decoder_keys = weights["rerank_decoder_keys"][
            :fingerprint_bytes
        ]
    router.freeze()
    mx.eval(router.parameters())
    return router


def candidates_for_hidden(hidden, router, args):
    hidden_np = np.array(hidden[0].astype(mx.float32)).copy()
    query_projection = np.array(
        router.query_projection.astype(mx.float16).astype(mx.float32)
    )
    key_projection = np.array(
        router.key_projection.astype(mx.float16).astype(mx.float32)
    )
    rerank_query_projection = np.array(
        router.rerank_query_projection.astype(mx.float16).astype(mx.float32)
    )[:, :args.fingerprint_bytes * 8]
    rerank_key_projection = np.array(
        router.rerank_key_projection.astype(mx.float16).astype(mx.float32)
    )[:, :args.fingerprint_bytes * 8]
    query_logits, query_codes, _ = router_codes(
        hidden_np, query_projection, args.tables, args.bits
    )
    _, key_codes, _ = router_codes(
        hidden_np, key_projection, args.tables, args.bits
    )
    query_bytes = binary_fingerprint_bytes(hidden_np, rerank_query_projection)
    key_bytes = binary_fingerprint_bytes(hidden_np, rerank_key_projection)
    query_bit_weights = None
    query_decoder = None
    if args.reranker == "confidence-hamming":
        query_bit_weights = np.power(
            np.abs(np.einsum(
                "nd,df->nf", hidden_np.astype(np.float64),
                rerank_query_projection.astype(np.float64), optimize=False,
            )),
            args.confidence_power,
        ).astype(np.float32)
        global_weights = np.logaddexp(
            0.0,
            np.array(router.rerank_bit_weights.astype(mx.float32))[
                :args.fingerprint_bytes * 8
            ],
        )
        query_bit_weights *= global_weights[None, :]
        query_bit_weights /= np.maximum(
            np.mean(query_bit_weights, axis=-1, keepdims=True), 1e-6
        )
        query_bit_weights = (
            (1.0 - args.confidence_mix)
            + args.confidence_mix * query_bit_weights
        )
    if args.reranker == "decoder-code":
        decoder_projection = np.array(
            router.rerank_decoder_query.astype(mx.float32)
        )
        query_decoder = np.einsum(
            "nd,df->nf", hidden_np, decoder_projection, optimize=False
        )
        decoder_keys = np.array(
            router.rerank_decoder_keys.astype(mx.float32)
        )[:args.fingerprint_bytes]
    else:
        decoder_keys = None
    _, full, _, _, index = causal_hierarchical_candidates(
        query_logits, query_codes, query_bytes, key_codes, key_bytes,
        args.window, args.sink_tokens, args.secondary_probes,
        args.leaf_capacity, args.k, args.reranker, args.retention_policy,
        storage_capacity=args.storage_capacity or None,
        candidate_budget=(
            args.tables * args.secondary_probes * args.leaf_capacity
        ),
        probe_capacities=args.probe_capacities_resolved,
        query_bit_weights=query_bit_weights,
        query_decoder_vectors=query_decoder,
        decoder_key_embeddings=decoder_keys,
    )
    return full, index


def capture_examples(model, rows, router, layer_index, args):
    examples = []
    for corpus, tokens in rows:
        capture = capture_layer(model, tokens, layer_index)
        candidates, index = candidates_for_hidden(capture["x"], router, args)
        x = mx.stop_gradient(capture["x"])
        target = mx.stop_gradient(capture["dense_attention"])
        mx.eval(x, target)
        examples.append((corpus, x, target, candidates, index))
        del capture
        mx.clear_cache()
    return examples


def aligned_sparse_output(attention, x, candidates):
    queries, keys, values = attention_qkv(attention, x)
    return sparse_attention_output(attention, queries, keys, values, candidates)


def alignment_metrics(attention, examples):
    rows = []
    for corpus, x, target, candidates, _ in examples:
        predicted = aligned_sparse_output(attention, x, candidates)
        mx.eval(predicted)
        target_np = np.array(target.astype(mx.float32))
        predicted_np = np.array(predicted.astype(mx.float32))
        error = predicted_np - target_np
        numerator = np.sum(predicted_np * target_np, axis=-1)
        denominator = np.linalg.norm(predicted_np, axis=-1) * np.linalg.norm(
            target_np, axis=-1
        )
        rows.append({
            "corpus": corpus,
            "normalized_mse": float(
                np.mean(error ** 2) / max(np.mean(target_np ** 2), 1e-12)
            ),
            "nrmse": float(
                np.sqrt(np.mean(error ** 2))
                / max(np.sqrt(np.mean(target_np ** 2)), 1e-12)
            ),
            "cosine": float(np.mean(
                numerator / np.maximum(denominator, 1e-12)
            )),
        })
    return rows


def heldout_metrics(model, attention, router, token_rows, layer_index, args):
    rows = []
    for corpus, tokens in token_rows:
        capture = capture_layer(model, tokens, layer_index)
        candidates, index = candidates_for_hidden(capture["x"], router, args)
        sparse = aligned_sparse_output(attention, capture["x"], candidates)
        mx.eval(sparse)
        dense_loss = tail_loss(capture, capture["dense_attention"], tokens, layer_index)
        sparse_loss = tail_loss(capture, sparse, tokens, layer_index)
        dense_np = np.array(capture["dense_attention"].astype(mx.float32))
        sparse_np = np.array(sparse.astype(mx.float32))
        rows.append({
            "corpus": corpus,
            "dense_loss": dense_loss,
            "sparse_loss": sparse_loss,
            "loss_delta": sparse_loss - dense_loss,
            "perplexity_ratio": math.exp(sparse_loss - dense_loss),
            "attention_output_nrmse": float(
                np.sqrt(np.mean((sparse_np - dense_np) ** 2))
                / max(np.sqrt(np.mean(dense_np ** 2)), 1e-12)
            ),
            "causal_index": index,
        })
        del capture, sparse
        mx.clear_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--router", required=True, type=pathlib.Path)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--secondary-probes", type=int, default=3)
    parser.add_argument("--leaf-capacity", type=int, default=14)
    parser.add_argument("--storage-capacity", type=int, default=0)
    parser.add_argument("--probe-capacities", default="")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--fingerprint-bytes", type=int, choices=(4, 6, 8), default=4)
    parser.add_argument(
        "--reranker", choices=(
            "full-hamming", "path-hamming", "confidence-hamming",
            "decoder-code",
        ), default="full-hamming"
    )
    parser.add_argument("--confidence-power", type=float, default=1.0)
    parser.add_argument("--confidence-mix", type=float, default=0.75)
    parser.add_argument("--retention-policy", choices=("reservoir", "tail"), default="reservoir")
    parser.add_argument("--corpora", default="wikitext2,pg19")
    parser.add_argument("--train-segments-per-corpus", type=int, default=8)
    parser.add_argument("--eval-segments-per-corpus", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--memory-limit-mb", type=int, default=1792)
    parser.add_argument("--cache-limit-mb", type=int, default=64)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.probe_capacities_resolved = (
        parse_ints(args.probe_capacities) if args.probe_capacities else None
    )
    if not args.router.is_file():
        parser.error(f"--router does not exist: {args.router}")
    if args.tables != 8 or args.bits != 8:
        parser.error("the deployed hierarchy requires eight 8-bit tables")
    corpora = parse_corpora(args.corpora)
    mx.set_memory_limit(args.memory_limit_mb * 2**20)
    mx.set_cache_limit(args.cache_limit_mb * 2**20)
    mx.random.seed(args.seed)

    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    body = language_body(model)
    layer = body.layers[args.layer]
    width = config.get("text_config", config)["hidden_size"]
    router = load_router(
        args.router, width, args.tables, args.bits, args.fingerprint_bytes
    )
    train_tokens = corpus_segments(
        tokenizer, corpora, args.seq_len,
        args.train_segments_per_corpus, skip=0,
    )
    internal_eval_tokens = corpus_segments(
        tokenizer, corpora, args.seq_len,
        args.eval_segments_per_corpus, skip=args.train_segments_per_corpus,
    )
    train_examples = capture_examples(
        model, train_tokens, router, args.layer, args
    )
    eval_examples = capture_examples(
        model, internal_eval_tokens, router, args.layer, args
    )
    attention = copy.deepcopy(layer.self_attn)
    attention.unfreeze()
    before = alignment_metrics(attention, eval_examples)
    optimizer = optim.AdamW(
        learning_rate=args.lr, weight_decay=args.weight_decay
    )

    def loss_fn(module, x, target, candidates):
        predicted = aligned_sparse_output(module, x, candidates)
        difference = predicted.astype(mx.float32) - target.astype(mx.float32)
        scale = mx.maximum(
            mx.mean(mx.square(target.astype(mx.float32))), mx.array(1e-8)
        )
        return mx.mean(mx.square(difference)) / scale

    loss_and_grad = nn.value_and_grad(attention, loss_fn)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        _, x, target, candidates, _ = train_examples[
            (step - 1) % len(train_examples)
        ]
        loss, gradients = loss_and_grad(attention, x, target, candidates)
        optimizer.update(attention, gradients)
        mx.eval(attention.trainable_parameters(), optimizer.state, loss)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({
                "step": step,
                "normalized_mse": round(float(loss), 6),
                "steps_per_second": round(step / (time.perf_counter() - started), 3),
                "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 2),
            }), flush=True)

    after = alignment_metrics(attention, eval_examples)
    heldout_tokens = []
    if "wikitext2" in corpora:
        heldout_tokens.extend(
            ("wikitext2-test", tokens) for tokens in wikitext_tokens(
                tokenizer, "test", args.seq_len, 1
            )
        )
    if "pg19" in corpora:
        heldout_tokens.extend(
            ("pg19-validation", tokens) for tokens in pg19_tokens(
                tokenizer, "validation", args.seq_len, 1
            )
        )
    heldout = heldout_metrics(
        model, attention, router, heldout_tokens, args.layer, args
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    attention.save_weights(str(args.output))
    result = vars(args) | {
        "router": str(args.router),
        "router_sha256": sha256(args.router),
        "output": str(args.output),
        "before": before,
        "after": after,
        "heldout": heldout,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
        "scope": "frozen hierarchical K=32 routing with trainable sparse attention projections",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
