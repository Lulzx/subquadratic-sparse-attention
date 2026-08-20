"""Select whole-table address hybrids on reserved training-split segments."""

import argparse
import hashlib
import json
import pathlib

import mlx.core as mx
import numpy as np
from mlx_lm import load

from mlx_donor_router import donor_example, language_body
from mlx_hierarchical_router_train import corpus_segments, parse_corpora
from mlx_lfm_hierarchical_eval import (
    binary_fingerprint_bytes,
    causal_hierarchical_candidates,
    router_codes,
)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def declared_masks(tables):
    masks = {
        "original": 0,
        "groupdro": (1 << tables) - 1,
        "alternating_even": sum(1 << table for table in range(0, tables, 2)),
        "alternating_odd": sum(1 << table for table in range(1, tables, 2)),
    }
    for count in range(1, tables):
        masks[f"prefix_{count}"] = (1 << count) - 1
        masks[f"suffix_{count}"] = ((1 << count) - 1) << (tables - count)
    for start in range(tables):
        mask = sum(1 << ((start + offset) % tables) for offset in range(tables // 2))
        masks[f"cyclic_half_{start}"] = mask
    unique = {}
    for name, mask in masks.items():
        unique.setdefault(mask, name)
    return {name: mask for mask, name in sorted(unique.items())}


def mixed_projection(original, groupdro, mask, tables, bits):
    width = original.shape[0]
    original = original.reshape(width, tables, bits)
    groupdro = groupdro.reshape(width, tables, bits)
    choose = np.asarray(
        [(mask >> table) & 1 for table in range(tables)], dtype=bool
    ).reshape(1, tables, 1)
    return np.where(choose, groupdro, original).reshape(width, tables * bits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--original", required=True, type=pathlib.Path)
    parser.add_argument("--groupdro", required=True, type=pathlib.Path)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--secondary-probes", type=int, default=6)
    parser.add_argument("--leaf-capacity", type=int, default=6)
    parser.add_argument("--storage-capacity", type=int, default=32)
    parser.add_argument("--fingerprint-bytes", type=int, default=5)
    parser.add_argument("--corpora", default="wikitext2,pg19")
    parser.add_argument("--segments-per-corpus", type=int, default=8)
    parser.add_argument("--segment-skip", type=int, default=128)
    parser.add_argument("--memory-limit-mb", type=int, default=1792)
    parser.add_argument("--cache-limit-mb", type=int, default=64)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint-output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if not args.original.is_file() or not args.groupdro.is_file():
        parser.error("both checkpoints must exist")
    if args.tables != 8 or args.bits != 8:
        parser.error("the deployed hierarchy requires eight 8-bit tables")
    corpora = parse_corpora(args.corpora)
    mx.set_memory_limit(args.memory_limit_mb * 2**20)
    mx.set_cache_limit(args.cache_limit_mb * 2**20)

    original_weights = mx.load(str(args.original))
    groupdro_weights = mx.load(str(args.groupdro))
    original_query = np.array(original_weights["query_projection"].astype(mx.float32))
    original_key = np.array(original_weights["key_projection"].astype(mx.float32))
    groupdro_query = np.array(groupdro_weights["query_projection"].astype(mx.float32))
    groupdro_key = np.array(groupdro_weights["key_projection"].astype(mx.float32))
    rerank_width = args.fingerprint_bytes * 8
    rerank_query = np.array(
        original_weights["rerank_query_projection"][:, :rerank_width].astype(mx.float32)
    )
    rerank_key = np.array(
        original_weights["rerank_key_projection"][:, :rerank_width].astype(mx.float32)
    )
    masks = declared_masks(args.tables)
    projections = {
        name: (
            mixed_projection(
                original_query, groupdro_query, mask, args.tables, args.bits
            ),
            mixed_projection(
                original_key, groupdro_key, mask, args.tables, args.bits
            ),
        )
        for name, mask in masks.items()
    }
    accumulators = {
        name: {
            corpus: {"candidate": [], "topk": [], "oracle": []}
            for corpus in corpora
        }
        for name in masks
    }

    model, tokenizer, _ = load(args.model, lazy=True, return_config=True)
    rows = corpus_segments(
        tokenizer, corpora, args.seq_len, args.segments_per_corpus,
        skip=args.segment_skip,
    )
    for corpus, tokens in rows:
        hidden, teacher, query_start = donor_example(
            model, tokens, args.layer, args.window, args.sink_tokens,
            teacher_target="attention",
        )
        hidden_np = np.array(hidden[0].astype(mx.float32)).copy()
        teacher_np = np.array(teacher[0].astype(mx.float32)).copy()
        query_bytes = binary_fingerprint_bytes(hidden_np, rerank_query)
        key_bytes = binary_fingerprint_bytes(hidden_np, rerank_key)
        for name, (query_projection, key_projection) in projections.items():
            query_logits, query_codes, _ = router_codes(
                hidden_np, query_projection, args.tables, args.bits
            )
            _, key_codes, _ = router_codes(
                hidden_np, key_projection, args.tables, args.bits
            )
            _, _, _, retained, _ = causal_hierarchical_candidates(
                query_logits, query_codes, query_bytes, key_codes, key_bytes,
                args.window, args.sink_tokens, args.secondary_probes,
                args.leaf_capacity, 32, "full-hamming", "reservoir",
                storage_capacity=args.storage_capacity,
                candidate_budget=(
                    args.tables * args.secondary_probes * args.leaf_capacity
                ),
            )
            metrics = accumulators[name][corpus]
            for offset, position in enumerate(range(query_start, args.seq_len)):
                candidates = retained[position]
                candidates = candidates[candidates >= 0]
                probability = teacher_np[offset]
                candidate_mass = float(np.sum(probability[candidates]))
                keep = min(32, len(candidates))
                top = (
                    candidates[np.argpartition(-probability[candidates], keep - 1)[:keep]]
                    if keep else candidates
                )
                oracle_keep = min(32, position - args.window - args.sink_tokens)
                eligible = np.arange(args.sink_tokens, position - args.window)
                oracle = (
                    eligible[np.argpartition(-probability[eligible], oracle_keep - 1)[:oracle_keep]]
                    if oracle_keep else eligible[:0]
                )
                metrics["candidate"].append(candidate_mass)
                metrics["topk"].append(float(np.sum(probability[top])))
                metrics["oracle"].append(float(np.sum(probability[oracle])))
        del hidden, teacher
        mx.clear_cache()

    results = []
    for name, mask in masks.items():
        corpus_rows = {}
        for corpus in corpora:
            values = accumulators[name][corpus]
            oracle = float(np.mean(values["oracle"]))
            topk = float(np.mean(values["topk"]))
            corpus_rows[corpus] = {
                "candidate_mass": float(np.mean(values["candidate"])),
                "retained_top32_mass": topk,
                "oracle_top32_mass": oracle,
                "oracle_relative_retained_top32": topk / max(oracle, 1e-12),
                "queries": len(values["topk"]),
            }
        worst = min(
            row["oracle_relative_retained_top32"] for row in corpus_rows.values()
        )
        results.append({
            "name": name,
            "mask": mask,
            "groupdro_tables": [
                table for table in range(args.tables) if (mask >> table) & 1
            ],
            "by_corpus": corpus_rows,
            "worst_corpus_oracle_relative_retained_top32": worst,
        })
    results.sort(
        key=lambda row: (
            -row["worst_corpus_oracle_relative_retained_top32"], row["mask"]
        )
    )
    best = results[0]
    output_weights = dict(original_weights)
    best_query, best_key = projections[best["name"]]
    output_weights["query_projection"] = mx.array(best_query)
    output_weights["key_projection"] = mx.array(best_key)
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.checkpoint_output), output_weights)
    report = {
        "format_version": 1,
        "scope": "reserved_training_split_whole_table_address_selection",
        "selection_metric": "maximize worst-corpus retained-top32/oracle-top32",
        "config": vars(args) | {
            "original": str(args.original),
            "groupdro": str(args.groupdro),
            "output": str(args.output),
            "checkpoint_output": str(args.checkpoint_output),
        },
        "original_sha256": sha256(args.original),
        "groupdro_sha256": sha256(args.groupdro),
        "best": best,
        "results": results,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({
        "best": best,
        "checkpoint": str(args.checkpoint_output),
        "output": str(args.output),
        "peak_memory_mb": report["peak_memory_mb"],
    }), flush=True)


if __name__ == "__main__":
    main()
