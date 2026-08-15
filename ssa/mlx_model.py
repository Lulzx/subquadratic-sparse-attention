import math

import mlx.core as mx
import mlx.nn as nn

from .mlx_selector import select_indices, select_indices_qk
from .mlx_attention import attend_selected


def window_attention(q, k, v, window):
    batch, length, heads, head_dim = q.shape
    chunks = []
    for start in range(0, length, window):
        end = min(start + window, length)
        key_start = max(0, start - window)
        query = q[:, start:end].transpose(0, 2, 1, 3)
        key = k[:, key_start:end].transpose(0, 2, 1, 3)
        value = v[:, key_start:end].transpose(0, 2, 1, 3)
        query_pos = mx.arange(start, end).reshape(-1, 1)
        key_pos = mx.arange(key_start, end).reshape(1, -1)
        mask = (query_pos >= key_pos) & (query_pos - key_pos < window)
        output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=1.0 / math.sqrt(head_dim), mask=mask
        )
        chunks.append(output.transpose(0, 2, 1, 3))
    return mx.concatenate(chunks, axis=1)


def causal_slot_attention(q, k, v, slots):
    """Attend to fixed-count causal summaries of interleaved history slots."""
    batch, length, heads, head_dim = q.shape
    positions = mx.arange(length)
    assignment = ((positions[:, None] % slots) == mx.arange(slots)[None, :]).astype(q.dtype)
    assignment = assignment.reshape(1, length, slots, 1, 1)
    key_sum = mx.cumsum(k[:, :, None, :, :] * assignment, axis=1)
    value_sum = mx.cumsum(v[:, :, None, :, :] * assignment, axis=1)
    count = mx.cumsum(assignment, axis=1)
    zero_state = mx.zeros((batch, 1, slots, heads, head_dim), dtype=q.dtype)
    zero_count = mx.zeros((1, 1, slots, 1, 1), dtype=q.dtype)
    key_sum = mx.concatenate([zero_state, key_sum[:, :-1]], axis=1)
    value_sum = mx.concatenate([zero_state, value_sum[:, :-1]], axis=1)
    count = mx.concatenate([zero_count, count[:, :-1]], axis=1)
    summary_k = key_sum / mx.maximum(count, mx.array(1.0, dtype=q.dtype))
    summary_v = value_sum / mx.maximum(count, mx.array(1.0, dtype=q.dtype))
    scores = mx.sum(q[:, :, None, :, :] * summary_k, axis=-1) / math.sqrt(head_dim)
    valid = count[..., 0] > 0
    scores = mx.where(valid, scores, mx.array(-1e9, dtype=scores.dtype))
    probability = mx.softmax(scores.astype(mx.float32), axis=2).astype(q.dtype)
    probability = mx.where(valid, probability, mx.zeros_like(probability))
    probability = probability / mx.maximum(
        mx.sum(probability, axis=2, keepdims=True), mx.array(1e-9, dtype=probability.dtype)
    )
    return mx.sum(probability[..., None] * summary_v, axis=2)


class MLXSSAAttention(nn.Module):
    def __init__(self, width, heads, window=32, tables=4, bits=16, members=4, probes=1,
                 semantic_router=False, global_slots=0, router_teacher_tokens=256,
                 semantic_loss_weight=1.0):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if tables < 1 or members < 1:
            raise ValueError("tables and members must be positive")
        if bits < 1 or bits > 30:
            raise ValueError("bits must be between 1 and 30")
        if probes < 1 or probes > bits + 1:
            raise ValueError("probes must be between 1 and bits + 1")
        if semantic_router and tables < 2:
            raise ValueError("semantic_router requires at least two tables for hybrid fallback")
        if global_slots < 0:
            raise ValueError("global_slots must be non-negative")
        if router_teacher_tokens < 2:
            raise ValueError("router_teacher_tokens must be at least 2")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.window = window
        self.tables = tables
        self.bits = bits
        self.members = members
        self.probes = probes
        self.semantic_router = semantic_router
        self.global_slots = global_slots
        self.router_teacher_tokens = router_teacher_tokens
        self.semantic_loss_weight = semantic_loss_weight
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        if semantic_router:
            self.shared_tables = tables // 2
            self.semantic_tables = tables - self.shared_tables
            self.hash_projection = (
                mx.random.normal((width, self.shared_tables * bits)) / math.sqrt(width)
            )
            semantic_projection = (
                mx.random.normal((width, self.semantic_tables * bits)) / math.sqrt(width)
            )
            self.query_hash_projection = semantic_projection
            self.key_hash_projection = semantic_projection + mx.zeros_like(semantic_projection)
        else:
            self.hash_projection = mx.random.normal((width, tables * bits)) / math.sqrt(width)
        self.sparse_gate = mx.full((heads,), 2.0)
        self.window_gate = mx.full((heads,), 2.0)
        if global_slots:
            self.global_gate = mx.full((heads,), -2.0)

    def __call__(self, x, inference_chunk=None):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.heads, self.head_dim)
        router_x = mx.stop_gradient(x)
        if self.semantic_router:
            shared_selected = select_indices(
                router_x, mx.stop_gradient(self.hash_projection),
                tables=self.shared_tables, bits=self.bits,
                members=self.members, probes=self.probes,
            )
            semantic_selected = select_indices_qk(
                router_x, router_x,
                mx.stop_gradient(self.query_hash_projection),
                mx.stop_gradient(self.key_hash_projection),
                tables=self.semantic_tables, bits=self.bits,
                members=self.members, probes=self.probes,
            )
            selected = mx.concatenate([shared_selected, semantic_selected], axis=-1)
        else:
            selected = select_indices(
                router_x, mx.stop_gradient(self.hash_projection),
                tables=self.tables, bits=self.bits, members=self.members, probes=self.probes,
            )
        if inference_chunk is None:
            sparse = attend_selected(q, k, v, selected)
        else:
            pieces = []
            mx.eval(q, k, v, selected)
            for start in range(0, length, inference_chunk):
                end = min(start + inference_chunk, length)
                piece = attend_selected(
                    q[:, start:end], k, v, selected[:, start:end], query_start=start
                )
                mx.eval(piece)
                pieces.append(piece)
            sparse = mx.concatenate(pieces, axis=1)
        local = window_attention(q, k, v, self.window)
        sparse_gate = mx.sigmoid(self.sparse_gate).reshape(1, 1, self.heads, 1)
        window_gate = mx.sigmoid(self.window_gate).reshape(1, 1, self.heads, 1)
        output = sparse_gate * sparse + window_gate * local
        if self.global_slots:
            global_summary = causal_slot_attention(q, k, v, self.global_slots)
            global_gate = mx.sigmoid(self.global_gate).reshape(1, 1, self.heads, 1)
            output = output + global_gate * global_summary
        return self.o_proj(output.reshape(batch, length, self.width))

    def router_loss(self, x):
        if not self.semantic_router:
            logits = (x @ self.hash_projection).reshape(*x.shape[:-1], self.tables, self.bits)
            probability = mx.sigmoid(logits)
            balance = mx.mean(mx.square(mx.mean(probability, axis=(0, 1)) - 0.5))
            confidence = mx.mean(probability * (1.0 - probability))
            return balance + 0.01 * confidence

        shared_logits = (x @ self.hash_projection).reshape(
            *x.shape[:-1], self.shared_tables, self.bits
        )
        query_logits = (x @ self.query_hash_projection).reshape(
            *x.shape[:-1], self.semantic_tables, self.bits
        )
        key_logits = (x @ self.key_hash_projection).reshape(
            *x.shape[:-1], self.semantic_tables, self.bits
        )
        probabilities = [
            mx.sigmoid(shared_logits), mx.sigmoid(query_logits), mx.sigmoid(key_logits)
        ]
        balance = mx.mean(mx.stack([
            mx.mean(mx.square(mx.mean(probability, axis=(0, 1)) - 0.5))
            for probability in probabilities
        ]))
        confidence = mx.mean(mx.stack([
            mx.mean(probability * (1.0 - probability)) for probability in probabilities
        ]))

        limit = min(x.shape[1], self.router_teacher_tokens)
        teacher_q = self.q_proj(x[:, 1:limit]).reshape(
            x.shape[0], limit - 1, self.heads, self.head_dim
        )
        teacher_k = self.k_proj(x[:, :limit]).reshape(
            x.shape[0], limit, self.heads, self.head_dim
        )
        teacher_scores = mx.einsum("bqhd,bkhd->bqk", teacher_q, teacher_k)
        teacher_scores = teacher_scores / math.sqrt(self.head_dim * self.heads)
        query_positions = mx.arange(1, limit).reshape(-1, 1)
        key_positions = mx.arange(limit).reshape(1, -1)
        causal = key_positions < query_positions
        teacher_scores = mx.where(causal, teacher_scores, mx.array(-1e9, dtype=teacher_scores.dtype))
        teacher_probability = mx.stop_gradient(mx.softmax(teacher_scores, axis=-1))

        query_code = mx.tanh(query_logits[:, 1:limit])
        key_code = mx.tanh(key_logits[:, :limit])
        student_by_table = mx.einsum("bqtd,bktd->bqkt", query_code, key_code) / math.sqrt(self.bits)
        student_scores = mx.max(student_by_table, axis=-1)
        student_scores = mx.where(causal, student_scores, mx.array(-1e9, dtype=student_scores.dtype))
        student_log_probability = student_scores - mx.logsumexp(
            student_scores, axis=-1, keepdims=True
        )
        semantic = -mx.mean(mx.sum(teacher_probability * student_log_probability, axis=-1))
        return balance + 0.01 * confidence + self.semantic_loss_weight * semantic


class MLXBlock(nn.Module):
    def __init__(self, width, heads, window, tables, bits, members, probes,
                 semantic_router, global_slots, router_teacher_tokens, semantic_loss_weight):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = MLXSSAAttention(
            width, heads, window, tables, bits, members, probes,
            semantic_router=semantic_router, global_slots=global_slots,
            router_teacher_tokens=router_teacher_tokens,
            semantic_loss_weight=semantic_loss_weight,
        )
        self.norm2 = nn.LayerNorm(width)
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(4 * width, width)

    def __call__(self, x, inference_chunk=None):
        x = x + self.attention(self.norm1(x), inference_chunk=inference_chunk)
        return x + self.down(nn.gelu(self.up(self.norm2(x))))


class MLXTinyLM(nn.Module):
    def __init__(self, vocab=2048, width=64, layers=2, heads=4, window=32,
                 tables=4, bits=16, members=4, probes=1, semantic_router=False,
                 global_slots=0, router_teacher_tokens=256, semantic_loss_weight=1.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.blocks = [
            MLXBlock(
                width, heads, window, tables, bits, members, probes,
                semantic_router, global_slots, router_teacher_tokens, semantic_loss_weight,
            )
            for _ in range(layers)
        ]
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab, bias=False)

    def __call__(self, tokens, inference_chunk=None):
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x, inference_chunk=inference_chunk)
        return self.output(self.norm(x))

    def router_loss(self, tokens):
        x = self.embedding(tokens)
        losses = []
        for block in self.blocks:
            normalized = block.norm1(x)
            losses.append(block.attention.router_loss(normalized))
            x = block(x)
        return mx.mean(mx.stack(losses))
