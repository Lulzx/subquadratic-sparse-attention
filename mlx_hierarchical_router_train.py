"""Train adjacent-table hierarchical addresses against distant attention mass."""

import argparse
import hashlib
import json
import math
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from datasets import load_dataset
from mlx_lm import load

from mlx_donor_router import (
    HierarchicalAttentionRouter,
    ProductQuantizedAttentionRouter,
    donor_example,
    language_body,
)


def parse_corpora(spec):
    corpora = [value.strip() for value in spec.split(",") if value.strip()]
    if not corpora or any(value not in ("wikitext2", "pg19") for value in corpora):
        raise ValueError("corpora must contain wikitext2 and/or pg19")
    return corpora


def wikitext_train_segments(tokenizer, seq_len, count, skip, split="train"):
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split
    )
    token_ids = tokenizer.encode("\n".join(
        row["text"] for row in dataset if row["text"]
    ))
    result = []
    for segment in range(skip, skip + count):
        start = segment * seq_len
        values = token_ids[start:start + seq_len]
        if len(values) != seq_len:
            break
        result.append(mx.array([values], dtype=mx.int32))
    if len(result) != count:
        raise ValueError(f"WikiText train yielded only {len(result)} segments")
    return result


def pg19_train_segments(
    tokenizer, seq_len, count, skip, segments_per_book=8, split="train"
):
    dataset = load_dataset("emozilla/pg19", split=split, streaming=True)
    result = []
    seen = 0
    for row in dataset:
        token_ids = tokenizer.encode(row["text"])
        within_book = 0
        for start in range(0, len(token_ids) - seq_len + 1, seq_len):
            if seen >= skip and len(result) < count:
                result.append(mx.array(
                    [token_ids[start:start + seq_len]], dtype=mx.int32
                ))
            seen += 1
            within_book += 1
            if len(result) == count:
                return result
            if within_book == segments_per_book:
                break
    raise ValueError(f"PG-19 train yielded only {len(result)} segments")


def corpus_segments(tokenizer, corpora, seq_len, count, skip):
    result = []
    for corpus in corpora:
        if corpus == "wikitext2":
            segments = wikitext_train_segments(tokenizer, seq_len, count, skip)
        else:
            segments = pg19_train_segments(tokenizer, seq_len, count, skip)
        result.extend((corpus, tokens) for tokens in segments)
    return result


def evaluation_domain_segments(tokenizer, corpora, seq_len, count, skip):
    result = []
    for corpus in corpora:
        if corpus == "wikitext2":
            segments = wikitext_train_segments(
                tokenizer, seq_len, count, skip, split="test"
            )
        else:
            segments = pg19_train_segments(
                tokenizer, seq_len, count, skip, split="validation"
            )
        result.extend((corpus, tokens) for tokens in segments)
    return result


def token_sha256(tokens):
    values = np.asarray(tokens, dtype=np.int32)
    return hashlib.sha256(values.tobytes()).hexdigest()


def training_loss(router, x, teacher, query_start, args, candidate_mask=None):
    if args.train_component == "pq":
        return router.pq_rerank_loss(
            x, teacher, query_start, args.window, args.sink_tokens,
            candidate_mask=candidate_mask,
            retrieval_topk=args.retrieval_topk,
            hard_negatives=args.hard_negatives,
            score_temperature=args.pq_score_temperature,
            assignment_temperature=args.pq_assignment_temperature,
            pairwise_weight=args.pairwise_weight,
            pairwise_margin=args.pq_pairwise_margin,
            balance_weight=args.pq_balance_weight,
            quantization_weight=args.pq_quantization_weight,
        )
    if args.train_component == "retention":
        return router.attention_retention_loss(
            x, teacher,
            retrieval_topk=args.retrieval_topk,
            pairwise_weight=args.pairwise_weight,
            pairwise_margin=args.pairwise_margin,
            leaf_pairwise_weight=args.leaf_retention_weight,
            leaf_storage_capacity=(args.storage_capacity or args.leaf_capacity),
        )
    if args.train_component == "rerank":
        return router.attention_rerank_loss(
            x, teacher, query_start, args.window, args.sink_tokens,
            temperature=args.rerank_temperature,
            mass_gamma=args.mass_gamma,
            balance_weight=args.balance_weight,
            decorrelation_weight=args.decorrelation_weight,
            pairwise_weight=args.pairwise_weight,
            retrieval_topk=args.retrieval_topk,
            hard_negatives=args.hard_negatives,
            pairwise_margin=args.pairwise_margin,
            candidate_mask=candidate_mask,
            confidence_weighted=args.confidence_weighted_loss,
            confidence_power=args.confidence_power,
            confidence_mix=args.confidence_mix,
            bilinear=args.bilinear_reranker_loss,
            lookup=args.lookup_reranker_loss,
            decoder=args.decoder_reranker_loss,
            distance_bias=args.distance_bias_loss,
            query_lookup=args.query_lookup_loss,
        )
    return router.hierarchical_loss(
        x, teacher, query_start, args.window, args.sink_tokens,
        mass_cover=args.mass_cover,
        mass_gamma=args.mass_gamma,
        max_positives=args.max_positives,
        positive_weight=args.positive_weight,
        hard_negative_weight=args.hard_negative_weight,
        hard_negative_temperature=args.hard_negative_temperature,
        rerank_weight=args.rerank_weight,
        rerank_temperature=args.rerank_temperature,
        balance_weight=args.balance_weight,
        decorrelation_weight=args.decorrelation_weight,
        address_entropy_weight=args.address_entropy_weight,
        leaf_overflow_weight=args.leaf_overflow_weight,
        leaf_storage_capacity=(args.storage_capacity or args.leaf_capacity),
        candidate_set_weight=args.candidate_set_weight,
        candidate_set_temperature=args.candidate_set_temperature,
        candidate_set_query_stride=args.candidate_set_query_stride,
        secondary_probes=args.secondary_probes,
        retrieval_topk=args.retrieval_topk,
        deployed_candidate_mask=candidate_mask,
        exact_boundary_weight=args.exact_boundary_weight,
        exact_boundary_negative_weight=args.exact_boundary_negative_weight,
        exact_boundary_query_stride=args.exact_boundary_query_stride,
        reranker=args.reranker,
    )


def materialize_candidate_mask(candidate_mask):
    if isinstance(candidate_mask, tuple):
        packed, width = candidate_mask
        return mx.array(np.unpackbits(
            packed, axis=-1, count=width, bitorder="little"
        ).astype(np.bool_))
    return candidate_mask


def materialize_example_tensor(value):
    return mx.array(value) if isinstance(value, np.ndarray) else value


def mean_training_loss(router, examples, args, return_by_corpus=False):
    rows = []
    rows_by_corpus = {}
    for example in examples:
        corpus, x, teacher, query_start, *optional = example
        x = materialize_example_tensor(x)
        teacher = materialize_example_tensor(teacher)
        candidate_mask = materialize_candidate_mask(
            optional[0] if optional else None
        )
        loss, parts = training_loss(
            router, x, teacher, query_start, args, candidate_mask
        )
        mx.eval(loss, parts)
        row = [float(loss), *map(float, parts)]
        rows.append(row)
        rows_by_corpus.setdefault(corpus, []).append(row)
    names = (
        ["loss", "retention_cross_entropy", "pairwise", "leaf_pairwise"]
        if args.train_component == "retention"
        else ["loss", "pq_cross_entropy", "pairwise", "balance", "quantization"]
        if args.train_component == "pq"
        else
        [
            "loss", "rerank_cross_entropy", "pairwise", "balance",
            "confidence", "decorrelation",
        ]
        if args.train_component == "rerank"
        else [
            "loss", "mass_positive", "hard_negative", "rerank_cross_entropy",
            "balance", "confidence", "selected_positive_count",
            "selected_positive_mass", "decorrelation", "address_entropy",
            "leaf_overflow",
            "candidate_set_mass",
            "exact_boundary_positive", "exact_boundary_negative",
        ]
    )
    values = np.mean(np.asarray(rows, dtype=np.float64), axis=0)
    overall = dict(zip(names, map(float, values)))
    if not return_by_corpus:
        return overall
    by_corpus = {
        corpus: dict(zip(
            names,
            map(float, np.mean(np.asarray(corpus_rows, dtype=np.float64), axis=0)),
        ))
        for corpus, corpus_rows in rows_by_corpus.items()
    }
    return overall, by_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument(
        "--fingerprint-bytes", type=int, choices=(4, 5, 6, 7, 8), default=8
    )
    parser.add_argument("--corpora", default="wikitext2,pg19")
    parser.add_argument("--train-segments-per-corpus", type=int, default=8)
    parser.add_argument("--eval-segments-per-corpus", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--train-component", choices=("address", "rerank", "retention", "pq"),
        default="address"
    )
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--mass-cover", type=float, default=0.95)
    parser.add_argument("--mass-gamma", type=float, default=0.5)
    parser.add_argument("--max-positives", type=int, default=56)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-temperature", type=float, default=1.0)
    parser.add_argument("--rerank-weight", type=float, default=1.0)
    parser.add_argument("--rerank-temperature", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-margin", type=float, default=2.0)
    parser.add_argument("--retrieval-topk", type=int, default=32)
    parser.add_argument("--hard-negatives", type=int, default=32)
    parser.add_argument(
        "--deployed-pool-loss", action="store_true",
        help="train reranking only within the frozen deployed hierarchy pool",
    )
    parser.add_argument("--confidence-weighted-loss", action="store_true")
    parser.add_argument("--bit-weights-only", action="store_true")
    parser.add_argument("--bilinear-reranker-loss", action="store_true")
    parser.add_argument("--bilinear-only", action="store_true")
    parser.add_argument("--lookup-reranker-loss", action="store_true")
    parser.add_argument("--lookup-only", action="store_true")
    parser.add_argument("--decoder-reranker-loss", action="store_true")
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--distance-bias-loss", action="store_true")
    parser.add_argument("--distance-bias-only", action="store_true")
    parser.add_argument("--query-lookup-loss", action="store_true")
    parser.add_argument("--query-lookup-only", action="store_true")
    parser.add_argument("--confidence-power", type=float, default=1.0)
    parser.add_argument("--confidence-mix", type=float, default=0.75)
    parser.add_argument("--pq-codebook", type=pathlib.Path)
    parser.add_argument("--pq-score-temperature", type=float, default=0.25)
    parser.add_argument("--pq-assignment-temperature", type=float, default=0.1)
    parser.add_argument("--pq-pairwise-margin", type=float, default=0.2)
    parser.add_argument("--pq-balance-weight", type=float, default=0.1)
    parser.add_argument("--pq-quantization-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--secondary-probes", type=int, default=3)
    parser.add_argument("--leaf-capacity", type=int, default=14)
    parser.add_argument("--storage-capacity", type=int, default=0)
    parser.add_argument(
        "--retention-policy", choices=("reservoir", "tail"),
        default="reservoir",
    )
    parser.add_argument(
        "--reranker", choices=("path-hamming", "full-hamming"),
        default="path-hamming",
    )
    parser.add_argument("--balance-weight", type=float, default=10.0)
    parser.add_argument("--decorrelation-weight", type=float, default=0.0)
    parser.add_argument("--address-entropy-weight", type=float, default=0.0)
    parser.add_argument("--leaf-overflow-weight", type=float, default=0.0)
    parser.add_argument("--candidate-set-weight", type=float, default=0.0)
    parser.add_argument("--candidate-set-temperature", type=float, default=16.0)
    parser.add_argument("--candidate-set-query-stride", type=int, default=1)
    parser.add_argument("--exact-boundary-weight", type=float, default=0.0)
    parser.add_argument("--exact-boundary-negative-weight", type=float, default=0.0)
    parser.add_argument("--exact-boundary-query-stride", type=int, default=1)
    parser.add_argument("--leaf-retention-weight", type=float, default=0.0)
    parser.add_argument(
        "--group-dro-beta", type=float, default=0.0,
        help="softmax strength for worst-corpus loss upweighting (0 disables)",
    )
    parser.add_argument(
        "--group-dro-ema", type=float, default=0.99,
        help="EMA decay for per-corpus training losses",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--memory-limit-mb", type=int, default=1792)
    parser.add_argument("--cache-limit-mb", type=int, default=64)
    parser.add_argument("--evaluation-domain-training", action="store_true")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.seq_len <= args.window + args.sink_tokens + 1:
        parser.error("sequence leaves no distant training queries")
    if args.tables != 8 or args.bits != 8:
        parser.error("deployed hierarchical fingerprint requires eight 8-bit tables")
    if not 0.0 < args.mass_cover <= 1.0:
        parser.error("--mass-cover must be in (0, 1]")
    if not 0.0 < args.mass_gamma <= 1.0:
        parser.error("--mass-gamma must be in (0, 1]")
    if args.max_positives < 1:
        parser.error("--max-positives must be positive")
    if args.hard_negative_temperature <= 0 or args.rerank_temperature <= 0:
        parser.error("temperatures must be positive")
    if args.candidate_set_temperature <= 0:
        parser.error("--candidate-set-temperature must be positive")
    if args.candidate_set_query_stride < 1:
        parser.error("--candidate-set-query-stride must be positive")
    if args.secondary_probes > args.bits + 1:
        parser.error("--secondary-probes cannot exceed bits + 1")
    if args.confidence_power <= 0:
        parser.error("--confidence-power must be positive")
    if args.group_dro_beta < 0:
        parser.error("--group-dro-beta must be nonnegative")
    if args.candidate_set_weight < 0:
        parser.error("--candidate-set-weight must be nonnegative")
    if args.exact_boundary_weight < 0 or args.exact_boundary_negative_weight < 0:
        parser.error("exact boundary weights must be nonnegative")
    if args.exact_boundary_query_stride < 1:
        parser.error("--exact-boundary-query-stride must be positive")
    if not 0.0 <= args.group_dro_ema < 1.0:
        parser.error("--group-dro-ema must be in [0, 1)")
    if not 0.0 <= args.confidence_mix <= 1.0:
        parser.error("--confidence-mix must be in [0, 1]")
    if args.train_component == "pq" and not (
        args.pq_codebook and args.pq_codebook.is_file()
    ):
        parser.error("--train-component pq requires --pq-codebook")
    try:
        corpora = parse_corpora(args.corpora)
    except ValueError as error:
        parser.error(str(error))

    mx.set_memory_limit(args.memory_limit_mb * 2**20)
    mx.set_cache_limit(args.cache_limit_mb * 2**20)
    mx.random.seed(args.seed)
    donor, tokenizer, config = load(args.model, lazy=True, return_config=True)
    body = language_body(donor)
    if args.layer < 0 or args.layer >= len(body.layers):
        parser.error("--layer is outside the donor model")
    segment_loader = (
        evaluation_domain_segments
        if args.evaluation_domain_training else corpus_segments
    )
    initial_skip = 1 if args.evaluation_domain_training else 0
    train_tokens = segment_loader(
        tokenizer, corpora, args.seq_len,
        args.train_segments_per_corpus, skip=initial_skip,
    )
    eval_tokens = segment_loader(
        tokenizer, corpora, args.seq_len,
        args.eval_segments_per_corpus,
        skip=initial_skip + args.train_segments_per_corpus,
    )

    def capture(rows):
        examples = []
        for corpus, tokens in rows:
            x, teacher, query_start = donor_example(
                donor, tokens, args.layer, args.window, args.sink_tokens,
                teacher_target="attention",
            )
            examples.append((corpus, x, teacher, query_start))
        return examples

    train_examples = capture(train_tokens)
    eval_examples = capture(eval_tokens)
    width = train_examples[0][1].shape[-1]
    del donor
    mx.clear_cache()

    router_class = (
        ProductQuantizedAttentionRouter
        if args.train_component == "pq" else HierarchicalAttentionRouter
    )
    router = router_class(
        width, args.tables, args.bits, rerank_bytes=args.fingerprint_bytes
    )
    initial_weights = None
    if args.init_checkpoint:
        checkpoint = pathlib.Path(args.init_checkpoint)
        if not checkpoint.is_file():
            parser.error(f"--init-checkpoint does not exist: {checkpoint}")
        weights = mx.load(str(checkpoint))
        initial_weights = dict(weights)
        router.query_projection = weights["query_projection"]
        router.key_projection = weights["key_projection"]
        rerank_width = args.fingerprint_bytes * 8
        source_query = weights.get(
            "rerank_query_projection", weights["query_projection"]
        )
        source_key = weights.get(
            "rerank_key_projection", weights["key_projection"]
        )
        if source_query.shape[1] < rerank_width:
            source_query = mx.concatenate([
                source_query,
                weights["query_projection"][:, source_query.shape[1]:rerank_width],
            ], axis=1)
            source_key = mx.concatenate([
                source_key,
                weights["key_projection"][:, source_key.shape[1]:rerank_width],
            ], axis=1)
        router.rerank_query_projection = source_query[:, :rerank_width]
        router.rerank_key_projection = source_key[:, :rerank_width]
        if "rerank_bit_weights" in weights:
            source_weights = weights["rerank_bit_weights"]
            if source_weights.shape[0] < rerank_width:
                source_weights = mx.concatenate([
                    source_weights,
                    mx.zeros((rerank_width - source_weights.shape[0],)),
                ])
            router.rerank_bit_weights = source_weights[:rerank_width]
        if "rerank_bilinear" in weights:
            router.rerank_bilinear = weights["rerank_bilinear"][
                :rerank_width, :rerank_width
            ]
        if "rerank_lookup" in weights:
            router.rerank_lookup = weights["rerank_lookup"][
                :args.fingerprint_bytes
            ]
        if "rerank_decoder_query" in weights:
            router.rerank_decoder_query = weights["rerank_decoder_query"]
        if "rerank_decoder_keys" in weights:
            router.rerank_decoder_keys = weights["rerank_decoder_keys"][
                :args.fingerprint_bytes
            ]
        if "rerank_distance_bias" in weights:
            router.rerank_distance_bias = weights["rerank_distance_bias"]
        if "rerank_query_lookup_weight" in weights:
            router.rerank_query_lookup_weight = weights[
                "rerank_query_lookup_weight"
            ][:, :args.fingerprint_bytes * 256]
            router.rerank_query_lookup_bias = weights[
                "rerank_query_lookup_bias"
            ][:args.fingerprint_bytes * 256]
        if "retention_projection" in weights:
            router.retention_projection = weights["retention_projection"]
    if args.train_component == "pq":
        pq_weights = mx.load(str(args.pq_codebook))
        router.pq_centroids = pq_weights.get(
            "pq_centroids", pq_weights.get("centroids")
        )
    mx.eval(router.parameters())
    needs_deployed_mask = args.deployed_pool_loss or (
        args.exact_boundary_weight != 0.0
        or args.exact_boundary_negative_weight != 0.0
    )
    if needs_deployed_mask:
        if args.train_component not in ("rerank", "pq"):
            if args.train_component != "address" or args.deployed_pool_loss:
                parser.error(
                    "deployed masks require rerank/pq training or exact-boundary "
                    "address training"
                )
        from mlx_lfm_hierarchical_eval import (
            binary_fingerprint_bytes,
            causal_hierarchical_candidates,
            router_codes,
        )

        def add_candidate_mask(example):
            corpus, x, teacher, query_start = example
            hidden = np.array(x[0].astype(mx.float32)).copy()
            query_projection = np.array(
                router.query_projection.astype(mx.float16).astype(mx.float32)
            )
            key_projection = np.array(
                router.key_projection.astype(mx.float16).astype(mx.float32)
            )
            rerank_query_projection = np.array(
                router.rerank_query_projection.astype(mx.float16).astype(mx.float32)
            )
            rerank_key_projection = np.array(
                router.rerank_key_projection.astype(mx.float16).astype(mx.float32)
            )
            query_logits, query_codes, _ = router_codes(
                hidden, query_projection, args.tables, args.bits
            )
            _, key_codes, _ = router_codes(
                hidden, key_projection, args.tables, args.bits
            )
            query_bytes = binary_fingerprint_bytes(
                hidden, rerank_query_projection
            )
            key_bytes = binary_fingerprint_bytes(hidden, rerank_key_projection)
            _, _, _, retained, _ = causal_hierarchical_candidates(
                query_logits, query_codes, query_bytes, key_codes, key_bytes,
                args.window, args.sink_tokens, args.secondary_probes,
                args.leaf_capacity, args.retrieval_topk, "full-hamming",
                args.retention_policy,
                storage_capacity=args.storage_capacity or None,
                candidate_budget=(
                    args.tables * args.secondary_probes * args.leaf_capacity
                ),
            )
            mask = np.zeros(
                (1, len(hidden) - query_start, len(hidden)), dtype=np.bool_
            )
            for offset, positions in enumerate(retained[query_start:]):
                valid = positions[positions >= 0]
                mask[0, offset, valid] = True
            # Dense bool masks across 256 long-context examples consume tens of
            # megabytes. Keep them bit-packed on the host and materialize only
            # the current example in the training/evaluation call.
            packed = np.packbits(mask, axis=-1, bitorder="little")
            host_x = np.array(x.astype(mx.float16)).copy()
            host_teacher = np.array(teacher.astype(mx.float32)).copy()
            return (
                corpus, host_x, host_teacher, query_start,
                (packed, mask.shape[-1]),
            )

        def attach_candidate_masks(examples):
            packed_examples = []
            for index in range(len(examples)):
                packed_examples.append(add_candidate_mask(examples[index]))
                examples[index] = None
                mx.clear_cache()
            return packed_examples

        train_examples = attach_candidate_masks(train_examples)
        eval_examples = attach_candidate_masks(eval_examples)
    if args.train_component in ("rerank", "pq"):
        router.freeze(keys=["query_projection", "key_projection"], strict=True)
        router.freeze(keys=["retention_projection"], strict=True)
        if args.bit_weights_only:
            if args.train_component != "rerank":
                parser.error("--bit-weights-only requires rerank training")
            router.freeze(
                keys=["rerank_query_projection", "rerank_key_projection"],
                strict=True,
            )
        if args.bilinear_only:
            if args.train_component != "rerank":
                parser.error("--bilinear-only requires rerank training")
            router.freeze(
                keys=[
                    "rerank_query_projection", "rerank_key_projection",
                    "rerank_bit_weights",
                ],
                strict=True,
            )
        if args.lookup_only:
            if args.train_component != "rerank":
                parser.error("--lookup-only requires rerank training")
            router.freeze(
                keys=[
                    "rerank_query_projection", "rerank_key_projection",
                    "rerank_bit_weights", "rerank_bilinear",
                    "rerank_decoder_query", "rerank_decoder_keys",
                    "rerank_distance_bias", "rerank_query_lookup_weight",
                    "rerank_query_lookup_bias",
                ],
                strict=True,
            )
        if args.decoder_only:
            if args.train_component != "rerank":
                parser.error("--decoder-only requires rerank training")
            router.freeze(
                keys=[
                    "rerank_query_projection", "rerank_key_projection",
                    "rerank_bit_weights", "rerank_bilinear", "rerank_lookup",
                    "rerank_query_lookup_weight", "rerank_query_lookup_bias",
                ],
                strict=True,
            )
        if args.distance_bias_only:
            if args.train_component != "rerank":
                parser.error("--distance-bias-only requires rerank training")
            router.freeze(
                keys=[
                    "rerank_query_projection", "rerank_key_projection",
                    "rerank_bit_weights", "rerank_bilinear", "rerank_lookup",
                    "rerank_decoder_query", "rerank_decoder_keys",
                    "rerank_query_lookup_weight", "rerank_query_lookup_bias",
                ],
                strict=True,
            )
        if args.query_lookup_only:
            if args.train_component != "rerank":
                parser.error("--query-lookup-only requires rerank training")
            router.freeze(
                keys=[
                    "rerank_query_projection", "rerank_key_projection",
                    "rerank_bit_weights", "rerank_bilinear", "rerank_lookup",
                    "rerank_decoder_query", "rerank_decoder_keys",
                    "rerank_distance_bias",
                ],
                strict=True,
            )
    elif args.train_component == "retention":
        router.freeze(
            keys=[
                "query_projection", "key_projection",
                "rerank_query_projection", "rerank_key_projection",
            ],
            strict=True,
        )
    else:
        router.freeze(
            keys=["rerank_query_projection", "rerank_key_projection"], strict=True
        )
    before, before_by_corpus = mean_training_loss(
        router, eval_examples, args, return_by_corpus=True
    )
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(
        model, x, teacher, query_start, candidate_mask=None,
        example_weight=1.0,
    ):
        loss, parts = training_loss(
            model, x, teacher, query_start, args, candidate_mask
        )
        return loss * example_weight, parts

    loss_and_grad = nn.value_and_grad(router, loss_fn)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_examples))
    group_loss_ema = {
        corpus: metrics["loss"] for corpus, metrics in before_by_corpus.items()
    }

    def group_weights():
        if args.group_dro_beta == 0.0:
            return {corpus: 1.0 for corpus in group_loss_ema}
        corpus_order = sorted(group_loss_ema)
        logits = args.group_dro_beta * np.asarray(
            [group_loss_ema[corpus] for corpus in corpus_order], dtype=np.float64
        )
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= np.sum(probabilities)
        return {
            corpus: float(probability * len(corpus_order))
            for corpus, probability in zip(corpus_order, probabilities)
        }

    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        if (step - 1) % len(order) == 0 and step > 1:
            order = rng.permutation(len(train_examples))
        example = train_examples[order[(step - 1) % len(order)]]
        corpus, x, teacher, query_start, *optional = example
        x = materialize_example_tensor(x)
        teacher = materialize_example_tensor(teacher)
        candidate_mask = materialize_candidate_mask(
            optional[0] if optional else None
        )
        example_weight = group_weights()[corpus]
        (loss, parts), gradients = loss_and_grad(
            router, x, teacher, query_start, candidate_mask, example_weight
        )
        gradient_norm = None
        if args.train_component == "pq":
            gradients, gradient_norm = optim.clip_grad_norm(
                gradients, args.max_grad_norm
            )
            mx.eval(loss, parts, gradient_norm)
            if not math.isfinite(float(loss)) or not math.isfinite(
                float(gradient_norm)
            ):
                raise FloatingPointError(
                    f"non-finite PQ optimization at step {step}: "
                    f"loss={float(loss)}, gradient_norm={float(gradient_norm)}"
                )
        optimizer.update(router, gradients)
        mx.eval(router.parameters(), optimizer.state, loss, parts)
        raw_loss = float(loss) / max(example_weight, 1e-12)
        group_loss_ema[corpus] = (
            args.group_dro_ema * group_loss_ema[corpus]
            + (1.0 - args.group_dro_ema) * raw_loss
        )
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            payload = {
                "step": step,
                "loss": float(loss),
                "steps_per_second": step / (time.perf_counter() - started),
                "peak_memory_mb": mx.get_peak_memory() / 2**20,
                "corpus": corpus,
                "group_weight": example_weight,
                "group_loss_ema": dict(group_loss_ema),
            }
            if args.train_component == "retention":
                payload.update({
                    "retention_cross_entropy": float(parts[0]),
                    "pairwise": float(parts[1]),
                    "leaf_pairwise": float(parts[2]),
                })
            elif args.train_component == "rerank":
                payload.update({
                    "rerank_cross_entropy": float(parts[0]),
                    "pairwise": float(parts[1]),
                    "balance": float(parts[2]),
                    "decorrelation": float(parts[4]),
                })
            elif args.train_component == "pq":
                payload.update({
                    "pq_cross_entropy": float(parts[0]),
                    "pairwise": float(parts[1]),
                    "balance": float(parts[2]),
                    "quantization": float(parts[3]),
                    "gradient_norm": float(gradient_norm),
                })
            else:
                payload.update({
                    "mass_positive": float(parts[0]),
                    "hard_negative": float(parts[1]),
                    "rerank_cross_entropy": float(parts[2]),
                    "selected_positive_count": float(parts[5]),
                    "selected_positive_mass": float(parts[6]),
                    "address_entropy": float(parts[8]),
                    "leaf_overflow": float(parts[9]),
                    "candidate_set_mass": float(parts[10]),
                    "exact_boundary_positive": float(parts[11]),
                    "exact_boundary_negative": float(parts[12]),
                })
            print(json.dumps(payload), flush=True)

    after, after_by_corpus = mean_training_loss(
        router, eval_examples, args, return_by_corpus=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if initial_weights is not None and args.train_component in (
        "address", "retention"
    ):
        # Component-only retraining must not silently discard the fixed
        # deployed scorer carried by the initialization checkpoint.
        if args.train_component == "address":
            initial_weights["query_projection"] = router.query_projection
            initial_weights["key_projection"] = router.key_projection
        else:
            initial_weights["retention_projection"] = router.retention_projection
        mx.save_safetensors(str(args.output), initial_weights)
    else:
        router.save_weights(str(args.output))
    token_manifest = {
        "train": [
            {"corpus": corpus, "sha256": token_sha256(tokens)}
            for corpus, tokens in train_tokens
        ],
        "eval": [
            {"corpus": corpus, "sha256": token_sha256(tokens)}
            for corpus, tokens in eval_tokens
        ],
    }
    metadata = {
        "format_version": 1,
        "objective": (
            "future_attention_mass_retention"
            if args.train_component == "retention"
            else (
                "separate_full_hamming_attention_mass_reranker"
                if args.train_component == "rerank"
                else (
                    "exact_causal_retained_boundary_attention_mass"
                    if args.exact_boundary_weight != 0.0
                    or args.exact_boundary_negative_weight != 0.0
                    else "capacity_aware_candidate_set_attention_mass"
                    if args.candidate_set_weight != 0.0
                    else "joint_adjacent_path_attention_mass"
                )
            )
        ),
        "teacher_target": "mean_dense_distant_attention_probability",
        "reranker": args.reranker,
        "config": vars(args) | {"output": str(args.output), "corpora": corpora},
        "donor_config": config.get("text_config", config),
        "token_manifest": token_manifest,
        "before": before,
        "before_by_corpus": before_by_corpus,
        "after": after,
        "after_by_corpus": after_by_corpus,
        "final_group_loss_ema": group_loss_ema,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    print(json.dumps({
        "checkpoint": str(args.output),
        "before": before,
        "before_by_corpus": before_by_corpus,
        "after": after,
        "after_by_corpus": after_by_corpus,
        "final_group_loss_ema": group_loss_ema,
        "peak_memory_mb": metadata["peak_memory_mb"],
    }), flush=True)


if __name__ == "__main__":
    main()
