import argparse
import json
import math
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from ssa.mlx_selector import select_indices_qk


DEFAULT_TRAIN_FILES = [
    "README.md",
    "docs/architecture.md",
    "docs/design-history.md",
    "docs/experiments.md",
    "docs/reproduction.md",
]
DEFAULT_EVAL_FILES = ["docs/limitations.md", "docs/model-card-audit.md"]


def language_body(model):
    """Return the text decoder body for text-only and multimodal MLX-LM models."""
    if hasattr(model, "language_model"):
        return model.language_model.model
    return model.model


class DonorHashRouter(nn.Module):
    def __init__(self, width, tables, bits):
        super().__init__()
        scale = 1.0 / math.sqrt(width)
        projection = mx.random.normal((width, tables * bits)) * scale
        self.query_projection = projection
        self.key_projection = projection + mx.zeros_like(projection)
        self.tables = tables
        self.bits = bits

    def logits(self, x):
        shape = (*x.shape[:-1], self.tables, self.bits)
        return (x @ self.query_projection).reshape(shape), (x @ self.key_projection).reshape(shape)

    @staticmethod
    def straight_through_sign(logits):
        continuous = mx.tanh(logits)
        binary = mx.where(logits >= 0, mx.ones_like(logits), -mx.ones_like(logits))
        return mx.stop_gradient(binary - continuous) + continuous

    def loss(self, x, teacher_probability, query_start, window, sink_tokens,
             alignment_weight, balance_weight):
        query_logits, key_logits = self.logits(x)
        query_logits = query_logits[:, query_start:]
        # Forward values are the exact +/-1 bits used by Hamming lookup while
        # backward gradients flow through tanh.
        query_code = self.straight_through_sign(query_logits)
        key_code = self.straight_through_sign(key_logits)
        student_by_table = mx.einsum("bqtd,bktd->bqkt", query_code, key_code)
        student_scores = mx.max(student_by_table, axis=-1) / math.sqrt(self.bits)
        query_positions = mx.arange(query_start, x.shape[1]).reshape(-1, 1)
        key_positions = mx.arange(x.shape[1]).reshape(1, -1)
        eligible = (key_positions < query_positions - window) & (key_positions >= sink_tokens)
        student_scores = mx.where(eligible, student_scores, mx.array(-1e9, student_scores.dtype))
        student_log_probability = student_scores - mx.logsumexp(
            student_scores, axis=-1, keepdims=True
        )
        cross_entropy = -mx.mean(
            mx.sum(teacher_probability * student_log_probability, axis=-1)
        )

        top_key = mx.argmax(teacher_probability, axis=-1)
        batch_index = mx.arange(x.shape[0]).reshape(-1, 1)
        query_probability = mx.sigmoid(query_logits)
        key_probability = mx.sigmoid(key_logits)[batch_index, top_key]
        same_bit = (
            query_probability * key_probability
            + (1.0 - query_probability) * (1.0 - key_probability)
        )
        log_table_match = mx.sum(
            mx.log(mx.maximum(same_bit, mx.array(1e-6, same_bit.dtype))), axis=-1
        )
        alignment = -mx.mean(mx.max(log_table_match, axis=-1))

        probabilities = [mx.sigmoid(query_logits), mx.sigmoid(key_logits)]
        balance = mx.mean(mx.stack([
            mx.mean(mx.square(mx.mean(probability, axis=(0, 1)) - 0.5))
            for probability in probabilities
        ]))
        confidence = mx.mean(mx.stack([
            mx.mean(probability * (1.0 - probability)) for probability in probabilities
        ]))
        total = (
            cross_entropy + alignment_weight * alignment
            + balance_weight * balance + 0.01 * confidence
        )
        return total, (cross_entropy, alignment, balance, confidence)


def parse_paths(spec, defaults):
    values = [value.strip() for value in spec.split(",") if value.strip()] if spec else defaults
    paths = [pathlib.Path(value) for value in values]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing corpus files: {', '.join(missing)}")
    return paths


def token_segments(tokenizer, paths, seq_len, stride, limit):
    segments = []
    for path in paths:
        token_ids = tokenizer.encode(path.read_text())
        for start in range(0, max(0, len(token_ids) - seq_len + 1), stride):
            segments.append(mx.array([token_ids[start:start + seq_len]], dtype=mx.int32))
            if len(segments) == limit:
                return segments
    if not segments:
        raise ValueError(f"corpus contains fewer than {seq_len} tokens")
    return segments


def donor_example(model, tokens, layer_index, window, sink_tokens):
    body = language_body(model)
    h = body.embed_tokens(tokens)
    attention_mask = create_attention_mask(h)
    state_mask = create_ssm_mask(h)
    for layer in body.layers[:layer_index]:
        layer_mask = attention_mask if hasattr(layer, "self_attn") else state_mask
        h = layer(h, layer_mask)
    layer = body.layers[layer_index]
    if getattr(layer, "is_linear", False) or not hasattr(layer, "self_attn"):
        raise ValueError(
            f"layer {layer_index} is not a full-attention layer; choose one with self_attn"
        )
    layer_norm = getattr(layer, "input_layernorm", None)
    if layer_norm is None:
        layer_norm = getattr(layer, "operator_norm", None)
    if layer_norm is None:
        raise ValueError(f"cannot find the input normalization for layer {layer_index}")
    x = layer_norm(h)
    attention = layer.self_attn
    batch, length, _ = x.shape
    heads = getattr(attention, "n_heads", getattr(attention, "num_attention_heads", None))
    kv_heads = getattr(
        attention, "n_kv_heads", getattr(attention, "num_key_value_heads", None)
    )
    head_dim = getattr(attention, "head_dim", x.shape[-1] // heads)
    query_projection = attention.q_proj(x)
    if query_projection.shape[-1] == heads * head_dim * 2:
        queries, _ = mx.split(
            query_projection.reshape(batch, length, heads, -1), 2, axis=-1
        )
    else:
        queries = query_projection.reshape(batch, length, heads, head_dim)
    keys = attention.k_proj(x).reshape(batch, length, kv_heads, head_dim)
    query_norm = getattr(attention, "q_norm", None)
    if query_norm is None:
        query_norm = getattr(attention, "q_layernorm", None)
    key_norm = getattr(attention, "k_norm", None)
    if key_norm is None:
        key_norm = getattr(attention, "k_layernorm", None)
    if query_norm is not None:
        queries = query_norm(queries)
    if key_norm is not None:
        keys = key_norm(keys)
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    queries = attention.rope(queries)
    keys = attention.rope(keys)
    if heads != kv_heads:
        keys = mx.repeat(keys, heads // kv_heads, axis=1)
    query_start = window + sink_tokens + 1
    queries = queries[:, :, query_start:]
    scores = mx.einsum("bhqd,bhkd->bhqk", queries, keys) * attention.scale
    query_positions = mx.arange(query_start, length).reshape(-1, 1)
    key_positions = mx.arange(length).reshape(1, -1)
    eligible = (key_positions < query_positions - window) & (key_positions >= sink_tokens)
    scores = mx.where(eligible, scores, mx.array(-1e9, scores.dtype))
    teacher_probability = mx.mean(mx.softmax(scores.astype(mx.float32), axis=-1), axis=1)
    x = mx.stop_gradient(x)
    teacher_probability = mx.stop_gradient(teacher_probability)
    mx.eval(x, teacher_probability)
    return x, teacher_probability, query_start


def hard_metrics(router, examples, members, probes, window, sink_tokens):
    retained_mass = []
    top_one = []
    candidates = []
    soft_retained_mass = []
    soft_top_one = []
    teacher_top_positions = []
    for x, teacher_probability, query_start in examples:
        query_logits, key_logits = router.logits(x)
        query_code = mx.tanh(query_logits[:, query_start:])
        key_code = mx.tanh(key_logits)
        soft_scores = mx.max(
            mx.einsum("bqtd,bktd->bqkt", query_code, key_code), axis=-1
        )
        selected = select_indices_qk(
            x, x, router.query_projection, router.key_projection,
            tables=router.tables, bits=router.bits, members=members, probes=probes, block=False,
            min_distance=window,
        )
        mx.eval(selected, teacher_probability, soft_scores)
        selected_np = np.array(selected)
        teacher_np = np.array(teacher_probability)
        soft_np = np.array(soft_scores)
        soft_budget = router.tables * members * probes
        for batch in range(selected_np.shape[0]):
            for offset, position in enumerate(range(query_start, selected_np.shape[1])):
                valid = np.unique(selected_np[batch, position])
                valid = valid[(valid >= sink_tokens) & (valid < position - window)]
                candidates.append(len(valid))
                teacher_top = int(np.argmax(teacher_np[batch, offset]))
                teacher_top_positions.append(teacher_top)
                eligible_positions = np.arange(sink_tokens, position - window)
                count = min(soft_budget, len(eligible_positions))
                soft_order = np.argsort(soft_np[batch, offset, eligible_positions])[-count:]
                soft_selected = eligible_positions[soft_order]
                soft_retained_mass.append(float(teacher_np[batch, offset, soft_selected].sum()))
                soft_top_one.append(teacher_top in soft_selected)
                if len(valid):
                    retained_mass.append(float(teacher_np[batch, offset, valid].sum()))
                    top_one.append(teacher_top in valid)
                else:
                    retained_mass.append(0.0)
                    top_one.append(False)
    return {
        "retained_teacher_mass": float(np.mean(retained_mass)),
        "teacher_top1_recall": float(np.mean(top_one)),
        "unique_candidates": float(np.mean(candidates)),
        "soft_topk_retained_teacher_mass": float(np.mean(soft_retained_mass)),
        "soft_topk_teacher_top1_recall": float(np.mean(soft_top_one)),
        "teacher_top1_unique_positions": len(set(teacher_top_positions)),
        "teacher_top1_mode_fraction": float(
            np.max(np.bincount(teacher_top_positions)) / len(teacher_top_positions)
        ),
        "queries": len(retained_mass),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--balance-weight", type=float, default=10.0)
    parser.add_argument("--train-segments", type=int, default=8)
    parser.add_argument("--eval-segments", type=int, default=2)
    parser.add_argument("--train-files", default="")
    parser.add_argument("--eval-files", default="")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="runs/lfm2.5-router.safetensors")
    args = parser.parse_args()
    if args.seq_len <= args.window + 1:
        parser.error("--seq-len must exceed --window + 1")
    if args.sink_tokens < 0 or args.sink_tokens >= args.seq_len - args.window - 1:
        parser.error("--sink-tokens leaves no eligible distant keys")
    if args.bits < 1 or args.bits > 30:
        parser.error("--bits must be between 1 and 30")
    if args.probes < 1 or args.probes > args.bits + 1:
        parser.error("--probes must be between 1 and --bits + 1")
    mx.random.seed(args.seed)

    train_paths = parse_paths(args.train_files, DEFAULT_TRAIN_FILES)
    eval_paths = parse_paths(args.eval_files, DEFAULT_EVAL_FILES)
    donor, tokenizer, config = load(args.model, lazy=True, return_config=True)
    body = language_body(donor)
    if args.layer < 0 or args.layer >= len(body.layers):
        parser.error(f"--layer must be between 0 and {len(body.layers) - 1}")
    if not hasattr(body.layers[args.layer], "self_attn"):
        full_layers = [
            index for index, layer in enumerate(body.layers)
            if hasattr(layer, "self_attn")
        ]
        parser.error(
            f"--layer {args.layer} is not full attention; full-attention layers: {full_layers}"
        )
    train_tokens = token_segments(
        tokenizer, train_paths, args.seq_len, args.stride, args.train_segments
    )
    eval_tokens = token_segments(
        tokenizer, eval_paths, args.seq_len, args.stride, args.eval_segments
    )
    train_examples = [
        donor_example(donor, tokens, args.layer, args.window, args.sink_tokens)
        for tokens in train_tokens
    ]
    eval_examples = [
        donor_example(donor, tokens, args.layer, args.window, args.sink_tokens)
        for tokens in eval_tokens
    ]
    width = train_examples[0][0].shape[-1]
    del donor
    mx.clear_cache()

    router = DonorHashRouter(width, args.tables, args.bits)
    before = hard_metrics(
        router, eval_examples, args.members, args.probes, args.window, args.sink_tokens
    )
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(model, x, teacher, query_start):
        return model.loss(
            x, teacher, query_start, args.window, args.sink_tokens,
            args.alignment_weight, args.balance_weight,
        )

    loss_and_grad = nn.value_and_grad(router, loss_fn)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, teacher, query_start = train_examples[(step - 1) % len(train_examples)]
        (loss, parts), gradients = loss_and_grad(router, x, teacher, query_start)
        optimizer.update(router, gradients)
        mx.eval(router.parameters(), optimizer.state, loss, parts)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            cross_entropy, alignment, balance, confidence = map(float, parts)
            print(json.dumps({
                "step": step,
                "loss": round(float(loss), 5),
                "cross_entropy": round(cross_entropy, 5),
                "alignment": round(alignment, 5),
                "balance": round(balance, 5),
                "confidence": round(confidence, 5),
                "steps_per_second": round(step / (time.perf_counter() - started), 3),
                "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 2),
            }), flush=True)

    after = hard_metrics(
        router, eval_examples, args.members, args.probes, args.window, args.sink_tokens
    )
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    router.save_weights(str(output))
    donor_config = config.get("text_config", config)
    metadata = vars(args) | {
        "donor_config": {
            key: donor_config.get(key) for key in [
                "model_type", "hidden_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
            ]
        },
        "train_files_resolved": [str(path) for path in train_paths],
        "eval_files_resolved": [str(path) for path in eval_paths],
        "before": before,
        "after": after,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"before": before, "after": after, "checkpoint": str(output)}), flush=True)


if __name__ == "__main__":
    main()
