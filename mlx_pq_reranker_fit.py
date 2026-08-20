"""Fit a four-byte product-quantized reranker on LFM2.5 train states."""

import argparse
import hashlib
import json
import pathlib

import mlx.core as mx
import numpy as np
from mlx_lm import load
from sklearn.cluster import MiniBatchKMeans

from mlx_donor_router import language_body
from mlx_hierarchical_router_train import corpus_segments, parse_corpora
from mlx_lfm_hierarchical_eval import capture_layer


def token_sha256(tokens):
    values = np.asarray(tokens, dtype=np.int32)
    return hashlib.sha256(values.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--router", required=True, type=pathlib.Path)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--corpora", default="wikitext2,pg19")
    parser.add_argument("--segments-per-corpus", type=int, default=64)
    parser.add_argument("--subquantizers", type=int, default=4)
    parser.add_argument("--centroids", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-limit-mb", type=int, default=1792)
    parser.add_argument("--cache-limit-mb", type=int, default=64)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if not args.router.is_file():
        parser.error(f"--router does not exist: {args.router}")
    if args.centroids != 256:
        parser.error("one-byte subquantizers require exactly 256 centroids")
    corpora = parse_corpora(args.corpora)
    mx.set_memory_limit(args.memory_limit_mb * 2**20)
    mx.set_cache_limit(args.cache_limit_mb * 2**20)
    model, tokenizer, _ = load(args.model, lazy=True, return_config=True)
    rows = corpus_segments(
        tokenizer, corpora, args.seq_len, args.segments_per_corpus, skip=0
    )
    weights = mx.load(str(args.router))
    projection = np.array(
        weights["rerank_key_projection"].astype(mx.float32)
    ).astype(np.float32)
    if projection.shape[1] % args.subquantizers:
        parser.error("rerank width must divide evenly across subquantizers")
    width = projection.shape[1] // args.subquantizers
    projected = []
    manifest = []
    for corpus, tokens in rows:
        capture = capture_layer(model, tokens, args.layer)
        hidden = np.array(capture["x"][0].astype(mx.float32)).copy()
        logits = np.einsum(
            "nd,df->nf", hidden.astype(np.float64),
            projection.astype(np.float64), optimize=False,
        ).astype(np.float32).reshape(-1, args.subquantizers, width)
        logits /= np.maximum(
            np.linalg.norm(logits, axis=-1, keepdims=True), 1e-6
        )
        projected.append(logits)
        manifest.append({"corpus": corpus, "sha256": token_sha256(tokens)})
        del capture
        mx.clear_cache()
    values = np.concatenate(projected, axis=0)
    codebooks = []
    inertia = []
    for subspace in range(args.subquantizers):
        estimator = MiniBatchKMeans(
            n_clusters=args.centroids,
            batch_size=4096,
            n_init=3,
            max_iter=100,
            random_state=args.seed + subspace,
            reassignment_ratio=0.01,
        ).fit(values[:, subspace])
        centers = estimator.cluster_centers_.astype(np.float32)
        centers /= np.maximum(
            np.linalg.norm(centers, axis=-1, keepdims=True), 1e-6
        )
        codebooks.append(centers)
        inertia.append(float(estimator.inertia_))
    codebook = np.stack(codebooks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.output), {"centroids": mx.array(codebook.astype(np.float16))}
    )
    metadata = vars(args) | {
        "router": str(args.router),
        "output": str(args.output),
        "shape": list(codebook.shape),
        "inertia": inertia,
        "token_manifest": manifest,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
        "scope": "four one-byte subquantizers fit only on train-split states",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
