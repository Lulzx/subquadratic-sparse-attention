import math
import pathlib
import tempfile

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_train import curriculum_length, scaled_batch_size
from mlx_donor_router import DonorHashRouter
from mlx_lfm_replacement import GatedLFMReplacement
from ssa.mlx_attention import random_weights, sparse_attention, sparse_attention_chunked
from ssa.mlx_model import MLXSSAAttention, causal_slot_attention
from ssa.mlx_selector import probe_codes, select_indices, select_indices_qk


def numpy_rope(x, positions, base=50000.0):
    dim = x.shape[-1]
    inv = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    angles = positions.astype(np.float32)[..., None] * inv
    while angles.ndim < x.ndim:
        angles = np.expand_dims(angles, -2)
    even, odd = x[..., 0::2], x[..., 1::2]
    return np.stack([
        even * np.cos(angles) - odd * np.sin(angles),
        even * np.sin(angles) + odd * np.cos(angles),
    ], axis=-1).reshape(x.shape)


def numpy_reference(x, weights, selected, heads):
    wq, wk, wv, wo, _ = weights
    batch, length, width = x.shape
    dim = width // heads
    with np.errstate(all="ignore"):
        q = (x @ wq).reshape(batch, length, heads, dim)
        k = (x @ wk).reshape(batch, length, heads, dim)
        v = (x @ wv).reshape(batch, length, heads, dim)
    safe = np.maximum(selected, 0)
    gathered_k = np.stack([k[b][safe[b]] for b in range(batch)])
    gathered_v = np.stack([v[b][safe[b]] for b in range(batch)])
    q = numpy_rope(q, np.arange(length)[None, :])
    gathered_k = numpy_rope(gathered_k, safe)
    scores = np.sum(q[:, :, None] * gathered_k, axis=-1) / math.sqrt(dim)
    valid = selected >= 0
    scores = np.where(valid[..., None], scores, -1e9)
    scores = scores - np.max(scores, axis=2, keepdims=True)
    probability = np.exp(scores) * valid[..., None]
    probability /= np.maximum(probability.sum(axis=2, keepdims=True), 1e-9)
    output = np.sum(probability[..., None] * gathered_v, axis=2).reshape(batch, length, width)
    with np.errstate(all="ignore"):
        return output @ wo


def test_reference():
    mx.random.seed(0)
    base = mx.random.normal((2, 32, 32)).astype(mx.float32)
    x = mx.concatenate([base, base], axis=1)
    weights = tuple(w.astype(mx.float32) for w in random_weights(32, seed=1))
    output, selected = sparse_attention(x, *weights, heads=4)
    mx.eval(x, output, selected, *weights)
    reference = numpy_reference(
        np.array(x.tolist(), dtype=np.float64),
        tuple(np.array(w.tolist(), dtype=np.float64) for w in weights),
        np.array(selected),
        heads=4,
    )
    error = float(np.max(np.abs(np.array(output) - reference)))
    assert np.abs(reference).max() > 0.01
    assert error < 2e-5, error
    assert np.isfinite(np.array(output)).all()
    print("PASS MLX sparse attention vs NumPy, max error", error)

    chunked, chunked_selected = sparse_attention_chunked(x, *weights, heads=4, chunk_q=17)
    mx.eval(chunked, chunked_selected)
    chunk_error = float(np.max(np.abs(np.array(chunked) - np.array(output))))
    assert np.array_equal(np.array(chunked_selected), np.array(selected))
    assert chunk_error < 2e-5, chunk_error
    print("PASS chunked MLX attention, max error", chunk_error)


def test_multiprobe_neighbor():
    # Positions 0 and 2 differ only in the low-margin first hash bit.
    x = mx.array([[[0.1, 1.0], [-2.0, -2.0], [-0.1, 1.0]]], dtype=mx.float32)
    projection = mx.eye(2, dtype=mx.float32)
    codes = probe_codes(x, projection, tables=1, bits=2, probes=2)
    exact = select_indices(
        x, projection, tables=1, bits=2, members=1, probes=1, block=False
    )
    multiprobe = select_indices(
        x, projection, tables=1, bits=2, members=1, probes=2, block=False
    )
    mx.eval(codes, exact, multiprobe)
    assert int(codes[0, 0, 0, 1]) == int(codes[0, 2, 0, 0])
    assert not bool(mx.any(exact[0, 2] == 0))
    assert bool(mx.any(multiprobe[0, 2] == 0))
    assert not bool(mx.any(multiprobe[0, 0] > 0)), "future position leaked"
    print("PASS multiprobe retrieves a Hamming-1 causal neighbor")


def test_length_curriculum():
    lengths = [128, 256, 512, 1024]
    schedule = [curriculum_length(step, 8, lengths) for step in range(1, 9)]
    assert schedule == [128, 128, 256, 256, 512, 512, 1024, 1024]
    assert [scaled_batch_size(16, 128, length) for length in lengths] == [16, 8, 4, 2]
    print("PASS staged length curriculum and constant-token batches")


def test_multiprobe_causality():
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((1, 32, 8), dtype=np.float32)
    projection_np = rng.standard_normal((8, 12), dtype=np.float32)
    # Construct owned MLX buffers. mx.array(np_array) may share NumPy storage,
    # which makes this a host-buffer lifetime test instead of a causality test.
    projection = mx.array(projection_np.tolist(), dtype=mx.float32)
    before_input_np = x_np.copy()
    before_input = mx.array(before_input_np.tolist(), dtype=mx.float32)
    before = select_indices(before_input, projection, tables=3, bits=4, probes=3)
    mx.eval(before)
    before_np = np.array(before).copy()
    x_np[:, 20:] = rng.standard_normal((1, 12, 8), dtype=np.float32)
    after_input_np = x_np.copy()
    after_input = mx.array(after_input_np.tolist(), dtype=mx.float32)
    after = select_indices(after_input, projection, tables=3, bits=4, probes=3)
    mx.eval(before, after)
    np.testing.assert_array_equal(before_np[:, :20], np.array(after)[:, :20])
    print("PASS multiprobe selector causality under future mutation")


def test_semantic_selector_parity_and_causality():
    rng = np.random.default_rng(11)
    x_np = rng.standard_normal((2, 32, 8), dtype=np.float32)
    projection = mx.array(rng.standard_normal((8, 12), dtype=np.float32))
    shared = select_indices(
        mx.array(x_np), projection, tables=3, bits=4, members=2, probes=1
    )
    separate = select_indices_qk(
        mx.array(x_np), mx.array(x_np), projection, projection,
        tables=3, bits=4, members=2, probes=1,
    )
    changed = x_np.copy()
    changed[:, 20:] = rng.standard_normal((2, 12, 8), dtype=np.float32)
    after = select_indices_qk(
        mx.array(changed), mx.array(changed), projection, projection,
        tables=3, bits=4, members=2, probes=1,
    )
    mx.eval(shared, separate, after)
    np.testing.assert_array_equal(np.array(shared), np.array(separate))
    np.testing.assert_array_equal(np.array(separate)[:, :20], np.array(after)[:, :20])
    distant = select_indices_qk(
        mx.array(x_np), mx.array(x_np), projection, projection,
        tables=3, bits=4, members=2, probes=1, block=False, min_distance=4,
    )
    mx.eval(distant)
    positions = np.arange(32).reshape(1, 32, 1)
    valid = np.array(distant) >= 0
    assert np.all(np.array(distant)[valid] < np.broadcast_to(positions - 4, distant.shape)[valid])
    print("PASS separate Q/K selector parity and causality")


def test_causal_global_slots():
    rng = np.random.default_rng(13)
    q = rng.standard_normal((1, 24, 2, 4), dtype=np.float32)
    k = rng.standard_normal((1, 24, 2, 4), dtype=np.float32)
    v = rng.standard_normal((1, 24, 2, 4), dtype=np.float32)
    before = causal_slot_attention(mx.array(q), mx.array(k), mx.array(v), slots=4)
    q[:, 16:] = rng.standard_normal((1, 8, 2, 4), dtype=np.float32)
    k[:, 16:] = rng.standard_normal((1, 8, 2, 4), dtype=np.float32)
    v[:, 16:] = rng.standard_normal((1, 8, 2, 4), dtype=np.float32)
    after = causal_slot_attention(mx.array(q), mx.array(k), mx.array(v), slots=4)
    mx.eval(before, after)
    np.testing.assert_allclose(np.array(before)[:, :16], np.array(after)[:, :16], atol=0, rtol=0)
    assert np.isfinite(np.array(before)).all()
    print("PASS compressed global slots are finite and causal")


def test_semantic_router_gradient():
    mx.random.seed(17)
    attention = MLXSSAAttention(
        width=16, heads=2, tables=2, bits=4, members=1,
        semantic_router=True, global_slots=2, router_teacher_tokens=16,
    )
    x = mx.random.normal((2, 16, 16))
    before = np.array(attention.query_hash_projection)
    loss_and_grad = nn.value_and_grad(attention, lambda model, inputs: model.router_loss(inputs))
    loss, gradients = loss_and_grad(attention, x)
    optimizer = optim.SGD(learning_rate=1e-2)
    optimizer.update(attention, gradients)
    mx.eval(loss, attention.parameters(), optimizer.state)
    assert math.isfinite(float(loss))
    assert not np.array_equal(before, np.array(attention.query_hash_projection))
    print("PASS semantic router loss is finite and updates hash projections")


def test_weight_resume():
    from ssa.mlx_model import MLXTinyLM

    original = MLXTinyLM(width=16, layers=1, heads=2, tables=2, bits=4, members=1)
    tokens = mx.array([[1, 2, 3, 4]])
    expected = original(tokens)
    mx.eval(expected)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = pathlib.Path(directory) / "resume.safetensors"
        original.save_weights(str(checkpoint))
        restored = MLXTinyLM(width=16, layers=1, heads=2, tables=2, bits=4, members=1)
        restored.load_weights(str(checkpoint))
        actual = restored(tokens)
        mx.eval(actual)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))
    print("PASS weights-only checkpoint resume")


def test_lfm_replacement_gate_and_causality():
    class FakeLFMAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads = 2
            self.n_kv_heads = 1
            self.scale = 0.5
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.k_proj = nn.Linear(8, 4, bias=False)
            self.v_proj = nn.Linear(8, 4, bias=False)
            self.out_proj = nn.Linear(8, 8, bias=False)
            self.q_layernorm = nn.RMSNorm(4)
            self.k_layernorm = nn.RMSNorm(4)
            self.rope = nn.RoPE(4, base=10000.0, traditional=False)

        def __call__(self, x, mask=None, cache=None):
            del mask, cache
            return self.q_proj(x)

    mx.random.seed(19)
    dense = FakeLFMAttention()
    router = DonorHashRouter(8, tables=2, bits=4)
    replacement = GatedLFMReplacement(
        dense, router, window=4, sink_tokens=1, members=2, probes=1,
        replacement_alpha=0.0,
    )
    x = mx.random.normal((1, 12, 8))
    expected = dense(x)
    actual = replacement(x)
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.array(expected), np.array(actual))

    before = replacement.candidate_indices(x)
    replacement.block_expansion = True
    expanded = replacement.candidate_indices(x)
    replacement.block_expansion = False
    changed = mx.concatenate([x[:, :8], mx.random.normal((1, 4, 8))], axis=1)
    after = replacement.candidate_indices(changed)
    replacement.replacement_alpha = 1.0
    sparse = replacement(x)
    mx.eval(before, expanded, after, sparse)
    assert expanded.shape[-1] == before.shape[-1] + 2 * 2 * 1
    np.testing.assert_array_equal(np.array(before)[:, :8], np.array(after)[:, :8])
    assert bool(mx.all(mx.isfinite(sparse)))
    assert sparse.shape == x.shape
    print("PASS LFM replacement exact gate, sparse path, and routing causality")


if __name__ == "__main__":
    test_reference()
    test_multiprobe_neighbor()
    test_length_curriculum()
    test_multiprobe_causality()
    test_semantic_selector_parity_and_causality()
    test_causal_global_slots()
    test_semantic_router_gradient()
    test_weight_resume()
    test_lfm_replacement_gate_and_causality()
