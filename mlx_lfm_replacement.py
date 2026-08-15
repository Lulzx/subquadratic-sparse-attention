"""Replace one LFM2.5 dense-attention layer with routed sparse attention.

The dense branch is frozen and retained only as an evaluation/gating reference. At
replacement_alpha=0 the model is exactly the donor; at 1 it executes only the sparse
branch. Training aligns the sparse attention output to cached dense teacher outputs.
"""

import argparse
import copy
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

from mlx_donor_router import (
    DEFAULT_EVAL_FILES,
    DEFAULT_TRAIN_FILES,
    DonorHashRouter,
    language_body,
    parse_paths,
    token_segments,
)
from ssa.mlx_selector import select_indices_qk


class GatedLFMReplacement(nn.Module):
    """Dense-compatible LFM attention wrapper with a sparse replacement branch."""

    def __init__(self, dense_attention, router, window, sink_tokens, members, probes,
                 replacement_alpha=1.0, block_expansion=False):
        super().__init__()
        self.dense_attention = dense_attention
        self.sparse_attention = copy.deepcopy(dense_attention)
        self.router = router
        self.window = window
        self.sink_tokens = sink_tokens
        self.members = members
        self.probes = probes
        self.block_expansion = block_expansion
        self.span_size = 2 if block_expansion else 1
        self.replacement_alpha = replacement_alpha
        self.dense_attention.freeze()
        self.router.freeze()
        self.sparse_attention.unfreeze()

    def candidate_indices(self, x):
        batch, length, _ = x.shape
        distant = select_indices_qk(
            mx.stop_gradient(x), mx.stop_gradient(x),
            mx.stop_gradient(self.router.query_projection),
            mx.stop_gradient(self.router.key_projection),
            tables=self.router.tables, bits=self.router.bits,
            members=self.members, probes=self.probes, block=False,
            min_distance=self.window,
        )
        if self.span_size > 1:
            offsets = mx.arange(self.span_size).reshape(1, 1, 1, self.span_size)
            distant = distant[..., None] + offsets
            query_positions = mx.arange(length).reshape(1, length, 1, 1)
            valid_distant = (distant >= self.sink_tokens) & (
                distant < query_positions - self.window
            )
            distant = mx.where(valid_distant, distant, length).reshape(
                batch, length, -1
            )
            # Expanded spans overlap frequently. Sort and retain each position once
            # so duplicate anchors do not receive extra softmax weight.
            distant = mx.sort(distant, axis=-1)
            first = distant[..., :1] < length
            unique = distant[..., 1:] != distant[..., :-1]
            unique = mx.concatenate([first, unique], axis=-1)
            distant = mx.where(unique & (distant < length), distant, -1)
        distant = mx.where(distant >= self.sink_tokens, distant, -1)

        positions = mx.arange(length).reshape(1, length, 1)
        local = positions - mx.arange(self.window).reshape(1, 1, self.window)
        local = mx.broadcast_to(local, (batch, length, self.window))
        # Sink tokens are added explicitly below; exclude them from the local
        # window so they are not double-counted in the attention softmax.
        local = mx.where(local >= self.sink_tokens, local, -1)
        if self.sink_tokens:
            sinks = mx.arange(self.sink_tokens).reshape(1, 1, self.sink_tokens)
            sinks = mx.broadcast_to(sinks, (batch, length, self.sink_tokens))
            sinks = mx.where(sinks <= positions, sinks, -1)
            return mx.concatenate([sinks, local, distant], axis=-1)
        return mx.concatenate([local, distant], axis=-1)

    def sparse_forward(self, x):
        attention = self.sparse_attention
        batch, length, _ = x.shape
        queries = attention.q_proj(x).reshape(batch, length, attention.n_heads, -1)
        keys = attention.k_proj(x).reshape(batch, length, attention.n_kv_heads, -1)
        values = attention.v_proj(x).reshape(batch, length, attention.n_kv_heads, -1)
        queries = attention.q_layernorm(queries).transpose(0, 2, 1, 3)
        keys = attention.k_layernorm(keys).transpose(0, 2, 1, 3)
        queries = attention.rope(queries).transpose(0, 2, 1, 3)
        keys = attention.rope(keys).transpose(0, 2, 1, 3)
        if attention.n_heads != attention.n_kv_heads:
            repeats = attention.n_heads // attention.n_kv_heads
            keys = mx.repeat(keys, repeats, axis=2)
            values = mx.repeat(values, repeats, axis=2)

        selected = self.candidate_indices(x)
        valid = selected >= 0
        selected_safe = mx.maximum(selected, 0)
        batch_offsets = mx.arange(batch).reshape(batch, 1, 1) * length
        global_indices = selected_safe + batch_offsets
        gathered_keys = keys.reshape(batch * length, attention.n_heads, -1)[global_indices]
        gathered_values = values.reshape(batch * length, attention.n_heads, -1)[global_indices]
        scores = mx.sum(queries[:, :, None] * gathered_keys, axis=-1) * attention.scale
        scores = mx.where(valid[..., None], scores, mx.array(-1e9, scores.dtype))
        probability = mx.softmax(scores.astype(mx.float32), axis=2).astype(values.dtype)
        probability = mx.where(valid[..., None], probability, mx.zeros_like(probability))
        probability = probability / mx.maximum(
            mx.sum(probability, axis=2, keepdims=True),
            mx.array(1e-9, probability.dtype),
        )
        output = mx.sum(probability[..., None] * gathered_values, axis=2)
        return attention.out_proj(output.reshape(batch, length, -1))

    def __call__(self, x, mask=None, cache=None):
        if cache is not None:
            raise NotImplementedError("sparse replacement decoding cache is not implemented")
        if self.replacement_alpha <= 0.0:
            return self.dense_attention(x, mask=mask, cache=cache)
        sparse = self.sparse_forward(x)
        if self.replacement_alpha >= 1.0:
            return sparse
        dense = self.dense_attention(x, mask=mask, cache=cache)
        return dense + self.replacement_alpha * (sparse - dense)


def layer_inputs_and_targets(model, token_batches, layer_index):
    body = language_body(model)
    layer = body.layers[layer_index]
    examples = []
    for tokens in token_batches:
        h = body.embed_tokens(tokens)
        attention_mask = create_attention_mask(h)
        state_mask = create_ssm_mask(h)
        for earlier in body.layers[:layer_index]:
            earlier_mask = attention_mask if hasattr(earlier, "self_attn") else state_mask
            h = earlier(h, earlier_mask)
        x = layer.operator_norm(h)
        target_attention = layer.self_attn(x, mask=attention_mask)
        residual = h + target_attention
        target_layer = residual + layer.feed_forward(layer.ffn_norm(residual))
        values = tuple(mx.stop_gradient(value) for value in (x, target_attention, h, target_layer))
        mx.eval(*values)
        examples.append(values)
    return examples


def alignment_metrics(replacement, examples, layer):
    attention_nrmse = []
    layer_nrmse = []
    cosine = []
    for x, target_attention, h, target_layer in examples:
        predicted_attention = replacement.sparse_forward(x)
        residual = h + predicted_attention
        predicted_layer = residual + layer.feed_forward(layer.ffn_norm(residual))
        mx.eval(predicted_attention, predicted_layer)
        target_attention_np = np.array(target_attention.astype(mx.float32))
        predicted_attention_np = np.array(predicted_attention.astype(mx.float32))
        target_layer_np = np.array(target_layer.astype(mx.float32))
        predicted_layer_np = np.array(predicted_layer.astype(mx.float32))
        attention_nrmse.append(
            np.sqrt(np.mean((predicted_attention_np - target_attention_np) ** 2))
            / max(np.sqrt(np.mean(target_attention_np ** 2)), 1e-12)
        )
        layer_nrmse.append(
            np.sqrt(np.mean((predicted_layer_np - target_layer_np) ** 2))
            / max(np.sqrt(np.mean(target_layer_np ** 2)), 1e-12)
        )
        numerator = np.sum(predicted_attention_np * target_attention_np, axis=-1)
        denominator = (
            np.linalg.norm(predicted_attention_np, axis=-1)
            * np.linalg.norm(target_attention_np, axis=-1)
        )
        cosine.append(np.mean(numerator / np.maximum(denominator, 1e-12)))
    return {
        "attention_nrmse": float(np.mean(attention_nrmse)),
        "attention_cosine": float(np.mean(cosine)),
        "layer_nrmse": float(np.mean(layer_nrmse)),
    }


def perplexity(model, token_batches):
    losses = []
    for tokens in token_batches:
        logits = model(tokens[:, :-1])
        loss = nn.losses.cross_entropy(logits, tokens[:, 1:], reduction="mean")
        mx.eval(loss)
        losses.append(float(loss))
    mean_loss = float(np.mean(losses))
    return {"loss": mean_loss, "perplexity": float(math.exp(min(mean_loss, 50.0)))}


def wikitext_tokens(tokenizer, split, seq_len, segments):
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split
    )
    token_ids = tokenizer.encode("\n".join(row["text"] for row in dataset if row["text"]))
    batches = []
    for start in range(0, len(token_ids) - seq_len + 1, seq_len):
        batches.append(mx.array([token_ids[start:start + seq_len]], dtype=mx.int32))
        if len(batches) == segments:
            break
    if len(batches) < segments:
        raise ValueError(f"quality dataset yielded only {len(batches)} complete segments")
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--router", default="")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alignment-dataset", choices=["repo", "wikitext2"], default="wikitext2")
    parser.add_argument("--train-segments", type=int, default=32)
    parser.add_argument("--eval-segments", type=int, default=8)
    parser.add_argument("--quality-dataset", choices=["repo", "wikitext2"], default="wikitext2")
    parser.add_argument("--quality-segments", type=int, default=16)
    parser.add_argument("--train-files", default="")
    parser.add_argument("--eval-files", default="")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.window < 1 or args.window >= args.seq_len:
        parser.error("--window must be positive and smaller than --seq-len")
    mx.random.seed(args.seed)

    train_paths = parse_paths(args.train_files, DEFAULT_TRAIN_FILES)
    eval_paths = parse_paths(args.eval_files, DEFAULT_EVAL_FILES)
    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    body = language_body(model)
    if args.layer < 0 or args.layer >= len(body.layers) or not hasattr(
        body.layers[args.layer], "self_attn"
    ):
        parser.error("--layer must select a full-attention layer")
    if args.alignment_dataset == "wikitext2":
        train_tokens = wikitext_tokens(
            tokenizer, "train", args.seq_len, args.train_segments
        )
        eval_tokens = wikitext_tokens(
            tokenizer, "validation", args.seq_len, args.eval_segments
        )
    else:
        train_tokens = token_segments(
            tokenizer, train_paths, args.seq_len, args.stride, args.train_segments
        )
        eval_tokens = token_segments(
            tokenizer, eval_paths, args.seq_len, args.stride, args.eval_segments
        )
    if args.quality_dataset == "wikitext2":
        quality_tokens = wikitext_tokens(
            tokenizer, "test", args.seq_len, args.quality_segments
        )
    else:
        quality_tokens = token_segments(
            tokenizer, eval_paths, args.seq_len, args.stride, args.quality_segments
        )
    train_examples = layer_inputs_and_targets(model, train_tokens, args.layer)
    eval_examples = layer_inputs_and_targets(model, eval_tokens, args.layer)
    layer = body.layers[args.layer]
    router = DonorHashRouter(config["hidden_size"], args.tables, args.bits)
    router_path = args.router or f"runs/lfm2.5-layer{args.layer}-router-seed{args.seed}.safetensors"
    if not pathlib.Path(router_path).is_file():
        raise FileNotFoundError(
            f"router checkpoint not found: {router_path}; run mlx_donor_router.py first"
        )
    router.load_weights(router_path)
    replacement = GatedLFMReplacement(
        layer.self_attn, router, args.window, args.sink_tokens,
        args.members, args.probes, replacement_alpha=1.0,
    )
    before = alignment_metrics(replacement, eval_examples, layer)
    baseline_quality = perplexity(model, quality_tokens)
    layer.self_attn = replacement
    replacement.replacement_alpha = 0.0
    dense_quality = perplexity(model, quality_tokens)
    gate_zero_loss_delta = dense_quality["loss"] - baseline_quality["loss"]
    if abs(gate_zero_loss_delta) > 1e-7:
        raise RuntimeError("replacement_alpha=0 does not exactly reproduce the donor")
    replacement.replacement_alpha = 1.0
    before_sparse_quality = perplexity(model, quality_tokens)
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(module, x, target):
        predicted = module.sparse_forward(x)
        difference = predicted.astype(mx.float32) - target.astype(mx.float32)
        mse = mx.mean(mx.square(difference))
        target_scale = mx.maximum(mx.mean(mx.square(target.astype(mx.float32))), 1e-8)
        return mse / target_scale

    loss_and_grad = nn.value_and_grad(replacement, loss_fn)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, target, _, _ = train_examples[(step - 1) % len(train_examples)]
        loss, gradients = loss_and_grad(replacement, x, target)
        optimizer.update(replacement, gradients)
        mx.eval(replacement.trainable_parameters(), optimizer.state, loss)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({
                "step": step,
                "normalized_mse": round(float(loss), 6),
                "steps_per_second": round(step / (time.perf_counter() - started), 3),
                "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 2),
            }), flush=True)

    after = alignment_metrics(replacement, eval_examples, layer)
    replacement.replacement_alpha = 1.0
    sparse_quality = perplexity(model, quality_tokens)

    output = pathlib.Path(
        args.output
        or f"runs/lfm2.5-layer{args.layer}-replacement-wikitext-seed{args.seed}.safetensors"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    replacement.save_weights(str(output))
    result = vars(args) | {
        "output_resolved": str(output),
        "router_resolved": router_path,
        "train_files_resolved": [str(path) for path in train_paths],
        "eval_files_resolved": [str(path) for path in eval_paths],
        "before": before,
        "after": after,
        "dense_eval": dense_quality,
        "gate_zero_loss_delta": gate_zero_loss_delta,
        "sparse_eval_before_alignment": before_sparse_quality,
        "sparse_eval": sparse_quality,
        "perplexity_ratio": sparse_quality["perplexity"] / dense_quality["perplexity"],
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": f"one-layer output alignment on {args.alignment_dataset} and perplexity on {args.quality_dataset}",
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
