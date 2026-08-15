import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self, dim, base=50000.0):
        super().__init__()
        self.dim = dim
        self.base = base

    def freqs(self, n, device, theta_scale=1.0):
        base = self.base * theta_scale
        inv = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        t = torch.arange(n, device=device).float()
        return torch.polar(torch.ones_like(t.unsqueeze(-1)), torch.outer(t, inv))

    @staticmethod
    def rotate(x, freqs):
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        xo = torch.view_as_real(xc * freqs).flatten(3)
        return xo.type_as(x)


class LSHRouter(nn.Module):
    def __init__(self, d, tables=4, bits=16):
        super().__init__()
        if bits > 62:
            raise ValueError("hash bits must fit in int64")
        self.tables = tables
        self.bits = bits
        self.proj = nn.Linear(d, tables * bits, bias=False)
        self.register_buffer("powers", 2 ** torch.arange(bits, dtype=torch.long), persistent=False)

    def logits(self, x):
        return self.proj(x).view(*x.shape[:-1], self.tables, self.bits)

    @torch.no_grad()
    def assign(self, x):
        return ((self.logits(x) >= 0).long() * self.powers).sum(-1)


class SSAAttention(nn.Module):
    def __init__(self, d, heads, window=256, n_select_buckets=4, members_per_bucket=4,
                 block=True, tables=4, hash_bits=16, chunk_q=1024, **kw):
        super().__init__()
        assert d % heads == 0
        self.d, self.h, self.dh = d, heads, d // heads
        self.window = window
        self.b = min(n_select_buckets, tables)
        self.m = members_per_bucket
        self.block = block
        self.chunk_q = chunk_q
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.router = LSHRouter(d, tables=tables, bits=hash_bits)
        self.gate_s = nn.Parameter(torch.full((heads,), 2.0))
        self.gate_w = nn.Parameter(torch.full((heads,), 2.0))
        self.rope = RoPE(self.dh)

    def _qkv(self, x):
        B, n, _ = x.shape
        q = self.wq(x).view(B, n, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, n, self.h, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, n, self.h, self.dh).transpose(1, 2)
        return q, k, v

    def _select_indices(self, x):
        B, n, _ = x.shape
        device = x.device
        T = self.router.tables
        space = 1 << self.router.bits
        codes = self.router.assign(x)
        table = torch.arange(T, device=device).view(1, 1, T)
        sample = torch.arange(B, device=device).view(B, 1, 1)
        global_codes = (sample * T + table) * space + codes
        flat_codes = global_codes.reshape(-1)
        flat_pos = torch.arange(B * n * T, device=device) // T % n
        order = torch.argsort(flat_codes, stable=True)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=device)
        current = torch.arange(B * n * T, device=device).reshape(B, n, T)
        rank = inverse[current][:, :, : self.b]
        offs = torch.arange(self.m, device=device).view(1, 1, 1, -1)
        take = rank.unsqueeze(-1) - 1 - offs
        valid = take >= 0
        take = take.clamp(min=0)
        candidate_entry = order[take]
        anchor = flat_pos[candidate_entry]
        candidate_code = flat_codes[candidate_entry]
        own_code = global_codes[:, :, : self.b].unsqueeze(-1)
        valid = valid & (candidate_code == own_code)
        anchor = torch.where(valid, anchor, torch.full_like(anchor, -1))
        if self.block:
            nxt = torch.where(anchor >= 0, (anchor + 1).clamp(max=n - 1), torch.full_like(anchor, -1))
            sel = torch.stack([anchor, nxt], dim=-1).reshape(B, n, -1)
        else:
            sel = anchor.reshape(B, n, -1)
        return sel

    def _sparse_attn(self, q, k, v, sel, theta_scale):
        B, H, n, dh = q.shape
        K = sel.shape[-1]
        out = torch.zeros_like(q)
        for s0 in range(0, n, self.chunk_q):
            s1 = min(s0 + self.chunk_q, n)
            selc = sel[:, s0:s1]
            valid = selc >= 0
            selc_ = selc.clamp(min=0)
            cq = s1 - s0
            idx_k = selc_.unsqueeze(-1).unsqueeze(1).expand(-1, H, -1, -1, dh)
            kc = k.gather(2, idx_k.reshape(B, H, -1, dh))
            vc = v.gather(2, idx_k.reshape(B, H, -1, dh))
            qc = q[:, :, s0:s1]
            qpos = (
                torch.arange(s0, s1, device=q.device)
                .view(1, 1, -1, 1)
                .expand(B, H, -1, dh // 2)
            )
            kpos = selc_.unsqueeze(1).expand(-1, H, -1, -1).reshape(B, H, -1, 1).expand(
                B, H, -1, dh // 2
            )
            qc = self._rope_pos(qc, qpos, theta_scale)
            kc = self._rope_pos(kc, kpos, theta_scale)
            kc = kc.view(B, H, cq, K, dh)
            vc = vc.view(B, H, cq, K, dh)
            att = torch.einsum("bhqd,bhqkd->bhqk", qc, kc) / math.sqrt(dh)
            att = att.masked_fill(~valid.unsqueeze(1), float("-inf")).softmax(-1)
            p = torch.nan_to_num(att, nan=0.0)
            out[:, :, s0:s1] = torch.einsum("bhqk,bhqkd->bhqd", p, vc)
        return out

    def _rope_pos(self, x, pos, theta_scale):
        base = self.rope.base * theta_scale
        inv = 1.0 / (
            base ** (torch.arange(0, self.rope.dim, 2, device=x.device).float() / self.rope.dim)
        )
        ang = pos.float() * inv
        f = torch.polar(torch.ones_like(ang), ang)
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(xc * f).flatten(3).type_as(x)

    def _window_attn(self, q, k, v, theta_scale):
        B, H, n, dh = q.shape
        W = self.window
        out = torch.zeros_like(q)
        for s0 in range(0, n, W):
            s1 = min(s0 + W, n)
            k0 = max(0, s0 - W)
            qc = q[:, :, s0:s1]
            kc = k[:, :, k0:s1]
            vc = v[:, :, k0:s1]
            f = self.rope.freqs(s1, q.device, theta_scale)
            qc = RoPE.rotate(qc, f[s0:s1])
            kc = RoPE.rotate(kc, f[k0:s1])
            att = torch.einsum("bhqd,bhkd->bhqk", qc, kc) / math.sqrt(dh)
            qpos = torch.arange(s0, s1, device=q.device).view(-1, 1)
            kpos = torch.arange(k0, s1, device=q.device).view(1, -1)
            keep = (qpos >= kpos) & (qpos - kpos < W)
            att = att.masked_fill(~keep, float("-inf"))
            out[:, :, s0:s1] = torch.einsum("bhqk,bhkd->bhqd", att.softmax(-1), vc)
        return out

    def forward(self, x, theta_scale=1.0):
        B, n, d = x.shape
        q, k, v = self._qkv(x)
        if self.b > 0:
            sel = self._select_indices(x)
            o_s = self._sparse_attn(q, k, v, sel, theta_scale)
        else:
            o_s = torch.zeros_like(q)
        o_w = self._window_attn(q, k, v, theta_scale)
        gs = torch.sigmoid(self.gate_s).view(1, -1, 1, 1)
        gw = torch.sigmoid(self.gate_w).view(1, -1, 1, 1)
        o = gs * o_s + gw * o_w
        return self.wo(o.transpose(1, 2).reshape(B, n, d)), None

    def aux_loss(self, x, theta_scale=1.0):
        if self.b == 0:
            return torch.zeros((), device=x.device)
        p = self.router.logits(x).sigmoid()
        balance = (p.mean(dim=(0, 1)) - 0.5).square().mean()
        confidence = (p * (1.0 - p)).mean()
        return balance + 0.01 * confidence


class DenseAttention(nn.Module):
    def __init__(self, d, heads, **kw):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.rope = RoPE(self.dh)

    def forward(self, x, theta_scale=1.0):
        B, n, d = x.shape
        q = self.wq(x).view(B, n, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, n, self.h, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, n, self.h, self.dh).transpose(1, 2)
        f = self.rope.freqs(n, x.device, theta_scale)
        q = RoPE.rotate(q, f)
        k = RoPE.rotate(k, f)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, n, d)), None

    def aux_loss(self, x, theta_scale=1.0):
        return torch.zeros((), device=x.device)


class Block(nn.Module):
    def __init__(self, d, attn_cls, heads, **kw):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = attn_cls(d, heads=heads, **kw)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, theta_scale=1.0):
        a, _ = self.attn(self.ln1(x), theta_scale)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


ATTN = {"ssa": SSAAttention, "dense": DenseAttention}


class TinyLM(nn.Module):
    def __init__(self, vocab=2048, d=192, layers=3, heads=6, attn="ssa", **kw):
        super().__init__()
        if attn == "window":
            cls = SSAAttention
            kw = dict(kw)
            kw["n_select_buckets"] = 0
        else:
            cls = ATTN[attn]
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(
            [Block(d, cls, heads=heads, **kw) for _ in range(layers)]
        )
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        self.attn_kind = attn
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, theta_scale=1.0):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, theta_scale)
        return self.head(self.ln(x))

    def aux_loss(self, idx, theta_scale=1.0):
        if self.attn_kind != "ssa":
            return torch.zeros((), device=idx.device)
        x = self.emb(idx)
        total = torch.zeros((), device=idx.device)
        for b in self.blocks:
            total = total + b.attn.aux_loss(b.ln1(x), theta_scale)
            x = b(x, theta_scale)
        return total / len(self.blocks)
