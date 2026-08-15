import math

import mlx.core as mx
import mlx.nn as nn

from .mlx_selector import select_indices
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


class MLXSSAAttention(nn.Module):
    def __init__(self, width, heads, window=32, tables=4, bits=16, members=4):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.window = window
        self.tables = tables
        self.bits = bits
        self.members = members
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.hash_projection = mx.random.normal((width, tables * bits)) / math.sqrt(width)
        self.sparse_gate = mx.full((heads,), 2.0)
        self.window_gate = mx.full((heads,), 2.0)

    def __call__(self, x, inference_chunk=None):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.heads, self.head_dim)
        selected = select_indices(
            mx.stop_gradient(x), mx.stop_gradient(self.hash_projection),
            tables=self.tables, bits=self.bits, members=self.members,
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
        return self.o_proj(output.reshape(batch, length, self.width))

    def router_loss(self, x):
        logits = (x @ self.hash_projection).reshape(*x.shape[:-1], self.tables, self.bits)
        probability = mx.sigmoid(logits)
        balance = mx.mean(mx.square(mx.mean(probability, axis=(0, 1)) - 0.5))
        confidence = mx.mean(probability * (1.0 - probability))
        return balance + 0.01 * confidence


class MLXBlock(nn.Module):
    def __init__(self, width, heads, window, tables, bits, members):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = MLXSSAAttention(width, heads, window, tables, bits, members)
        self.norm2 = nn.LayerNorm(width)
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(4 * width, width)

    def __call__(self, x, inference_chunk=None):
        x = x + self.attention(self.norm1(x), inference_chunk=inference_chunk)
        return x + self.down(nn.gelu(self.up(self.norm2(x))))


class MLXTinyLM(nn.Module):
    def __init__(self, vocab=2048, width=64, layers=2, heads=4, window=32,
                 tables=4, bits=16, members=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.blocks = [MLXBlock(width, heads, window, tables, bits, members) for _ in range(layers)]
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
