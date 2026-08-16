import math
import json
import pathlib
import re
import tempfile

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_train import curriculum_length, scaled_batch_size
from mlx_donor_router import DonorHashRouter, hard_metrics
from mlx_lfm_replacement import GatedLFMReplacement
from mlx_lfm_retrieval_generalization import (
    TASK_VALUES,
    build_manifest,
    distant_candidate_budget,
    load_manifest,
)
from mlx_lfm_retrieval_generalization_report import aggregate, wilson_interval
from mlx_lfm_retrieval_router import TRAIN_VALUES
from mlx_routing_scan_bench import (
    build_bucket_index,
    build_bucket_tails,
    candidate_recall,
    logical_history_bytes,
    packed_codes,
    per_length_query_rng,
)
from ssa.mlx_attention import random_weights, sparse_attention, sparse_attention_chunked
from ssa.mlx_model import MLXSSAAttention, causal_slot_attention
from ssa.mlx_selector import (
    hash_codes,
    probe_codes,
    select_block_indices_qk,
    select_indices,
    select_indices_qk,
)


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


def test_hybrid_bucket_history():
    x = mx.ones((1, 12, 1))
    projection = mx.ones((1, 1))
    recent = np.array(select_indices_qk(
        x, x, projection, projection,
        tables=1, bits=1, members=4, probes=1, block=False,
        member_policy="recent",
    ))
    hybrid = np.array(select_indices_qk(
        x, x, projection, projection,
        tables=1, bits=1, members=4, probes=1, block=False,
        member_policy="hybrid",
    ))
    assert recent[0, 11].tolist() == [10, 9, 8, 7]
    assert hybrid[0, 11].tolist() == [10, 9, 8, 5]
    changed = mx.concatenate([x[:, :8], -mx.ones((1, 4, 1))], axis=1)
    changed_hybrid = np.array(select_indices_qk(
        changed, changed, projection, projection,
        tables=1, bits=1, members=4, probes=1, block=False,
        member_policy="hybrid",
    ))
    np.testing.assert_array_equal(hybrid[:, :8], changed_hybrid[:, :8])
    print("PASS hybrid bucket retention spans history and remains causal")


def test_hybrid_selector_reference():
    rng = np.random.default_rng(41)
    batch, length, width, tables, bits = 2, 17, 4, 2, 3
    query = mx.array(rng.standard_normal((batch, length, width), dtype=np.float32))
    key = mx.array(rng.standard_normal((batch, length, width), dtype=np.float32))
    query_projection = mx.array(
        rng.standard_normal((width, tables * bits), dtype=np.float32)
    )
    key_projection = mx.array(
        rng.standard_normal((width, tables * bits), dtype=np.float32)
    )
    actual = select_indices_qk(
        query, key, query_projection, key_projection,
        tables=tables, bits=bits, members=4, probes=1, block=False,
        min_distance=2, member_policy="hybrid", history_fraction=0.5,
    )
    query_code = np.array(probe_codes(
        query, query_projection, tables, bits, probes=1
    ))[..., 0]
    key_code = np.array(hash_codes(key, key_projection, tables, bits))
    expected = np.full((batch, length, tables, 4), -1, dtype=np.int32)
    for sample in range(batch):
        for position in range(length):
            for table in range(tables):
                matches = [
                    key_position
                    for key_position in range(max(position - 2, 0))
                    if key_code[sample, key_position, table]
                    == query_code[sample, position, table]
                ]
                maximum_offset = len(matches) - 1
                offsets = [0, 1, 2, max(3, math.floor(maximum_offset * 0.5))]
                for member, offset in enumerate(offsets):
                    if offset < len(matches):
                        expected[sample, position, table, member] = matches[-1 - offset]
    np.testing.assert_array_equal(
        np.array(actual).reshape(batch, length, tables, 4), expected
    )
    print("PASS hybrid selector matches causal NumPy reference")


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


def test_selector_causality_matrix():
    """Cover batches, table counts, probes, distance boundaries, and successors."""
    rng = np.random.default_rng(29)
    length, width = 33, 8
    original = rng.standard_normal((2, length, width), dtype=np.float32)
    projection_np = rng.standard_normal((width, 12), dtype=np.float32)
    projection = mx.array(projection_np.tolist(), dtype=mx.float32)
    for batch in (1, 2):
        for tables, bits, probes in ((1, 4, 1), (3, 4, 3)):
            table_projection = projection[:, : tables * bits]
            for block in (False, True):
                for min_distance in (0, 4):
                    for boundary in (1, 4, 8, 16, 32):
                        before_np = original[:batch].copy()
                        after_np = before_np.copy()
                        after_np[:, boundary:] = rng.standard_normal(
                            (batch, length - boundary, width), dtype=np.float32
                        )
                        before = select_indices_qk(
                            mx.array(before_np.tolist(), dtype=mx.float32),
                            mx.array(before_np.tolist(), dtype=mx.float32),
                            table_projection,
                            table_projection,
                            tables=tables,
                            bits=bits,
                            members=2,
                            probes=probes,
                            block=block,
                            min_distance=min_distance,
                        )
                        after = select_indices_qk(
                            mx.array(after_np.tolist(), dtype=mx.float32),
                            mx.array(after_np.tolist(), dtype=mx.float32),
                            table_projection,
                            table_projection,
                            tables=tables,
                            bits=bits,
                            members=2,
                            probes=probes,
                            block=block,
                            min_distance=min_distance,
                        )
                        mx.eval(before, after)
                        before_selected = np.array(before)
                        np.testing.assert_array_equal(
                            before_selected[:, :boundary], np.array(after)[:, :boundary]
                        )
                        for query_position in range(boundary):
                            valid = before_selected[:, query_position]
                            valid = valid[valid >= 0]
                            if block:
                                assert np.all(valid <= query_position - min_distance)
                            else:
                                assert np.all(valid < query_position - min_distance)
    print("PASS selector causality matrix across boundaries, batches, tables, and probes")


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


def test_block_selector_causality():
    mx.random.seed(23)
    x = mx.random.normal((1, 12, 8))
    projection = mx.random.normal((8, 8))

    def select(values):
        blocks = mx.mean(values.reshape(1, 3, 4, 8), axis=2)
        return select_block_indices_qk(
            values,
            blocks,
            projection,
            projection,
            block_size=4,
            context_length=12,
            tables=2,
            bits=4,
            members=1,
            probes=1,
        )

    before = select(x)
    changed = mx.concatenate([x[:, :8], mx.random.normal((1, 4, 8))], axis=1)
    after = select(changed)
    mx.eval(before, after)
    np.testing.assert_array_equal(np.array(before)[:, :8], np.array(after)[:, :8])
    selected = np.array(before)[0]
    for query_position, anchors in enumerate(selected):
        valid = anchors[anchors >= 0]
        assert np.all(valid % 4 == 0)
        assert np.all(valid + 3 < query_position)
    print("PASS completed-block selector causality")


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


def test_learned_router_diagnostics():
    mx.random.seed(31)
    length, query_start = 24, 6
    x = mx.random.normal((1, length, 8))
    teacher = np.zeros((1, length - query_start, length), dtype=np.float32)
    for offset, position in enumerate(range(query_start, length)):
        teacher[0, offset, max(1, position - 5)] = 1.0
    router = DonorHashRouter(width=8, tables=2, bits=4)
    loss, parts = router.loss(
        x,
        mx.array(teacher),
        query_start,
        window=2,
        sink_tokens=1,
        alignment_weight=0.1,
        balance_weight=10.0,
        decorrelation_weight=1.0,
        retrieval_weight=1.0,
        retrieval_topk=4,
        retrieval_positive_weight=20.0,
    )
    mx.eval(loss, parts)
    assert math.isfinite(float(loss))
    assert len(parts) == 6 and all(math.isfinite(float(part)) for part in parts)
    metrics = hard_metrics(
        router,
        [(x, mx.array(teacher), query_start)],
        members=2,
        probes=1,
        window=2,
        sink_tokens=1,
    )
    assert metrics["queries"] == length - query_start
    assert len(metrics["query_key_agreement"]["exact_by_table"]) == 2
    assert len(metrics["bucket_occupancy"]["by_table"]) == 2
    assert len(metrics["table_retrieval"]["success_by_table"]) == 2
    attribution = metrics["failure_attribution"]
    assert (
        attribution["no_probed_address_agreement"]
        + attribution["agreement_without_selection"]
        + attribution["selected"]
        == metrics["queries"]
    )
    assert metrics["distance"]["1-64"]["queries"] == metrics["queries"]
    print("PASS learned-router agreement, occupancy, correlation, and distance diagnostics")


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
    replacement.span_size = 2
    expanded = replacement.candidate_indices(x)
    replacement.span_size = 1
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


class FakeTokenizer:
    def __init__(self):
        self.vocabulary = {}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return f"USER: {messages[0]['content']}\nASSISTANT:"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        pieces = re.findall(r"\w+|[^\w\s]", text)
        return [
            self.vocabulary.setdefault(piece, len(self.vocabulary) + 1)
            for piece in pieces
        ]


def test_retrieval_generalization_manifest():
    manifest = build_manifest(
        FakeTokenizer(), "test/model", lengths=(256,), positions=(0.1, 0.9)
    )
    assert len(manifest["cases"]) == 48
    assert len({case["case_id"] for case in manifest["cases"]}) == 48
    assert all(case["actual_length"] >= 256 for case in manifest["cases"])
    assert all(case["source_start"] < case["query_position"] for case in manifest["cases"])
    assert {case["task"] for case in manifest["cases"]} == set(TASK_VALUES)
    training_values = {
        value for values in TRAIN_VALUES.values() for value in values
    }
    evaluation_values = {
        value for values in TASK_VALUES.values() for value in values
    }
    assert training_values.isdisjoint(evaluation_values)
    with tempfile.TemporaryDirectory() as temporary:
        path = pathlib.Path(temporary) / "manifest.json"
        path.write_text(json.dumps(manifest))
        loaded = load_manifest(path)
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]
    assert loaded["cases"] == manifest["cases"]
    assert distant_candidate_budget(8, 1, 4, span_size=2) == 64
    assert distant_candidate_budget(8, 1, 2, block_size=4) == 64
    try:
        distant_candidate_budget(8, 1, 4, span_size=2, block_size=4)
    except ValueError:
        pass
    else:
        raise AssertionError("mixed span/block expansion must be rejected")
    print("PASS deterministic unseen retrieval-generalization manifest")


def test_retrieval_generalization_report():
    manifest = build_manifest(
        FakeTokenizer(), "test/model", lengths=(256,), positions=(0.1,)
    )
    cases = manifest["cases"][:2]

    def result(case, correct):
        return {
            key: value for key, value in case.items()
            if key not in ("prompt", "needle", "expected")
        } | {"contains_expected": correct}

    common = {
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_template": None,
        "distant_candidate_budget": None,
        "member_policy": None,
        "history_fraction": None,
        "block_size": None,
        "peak_memory_mb": 1.0,
        "elapsed_seconds": 1.0,
    }
    dense = common | {
        "variant": "dense",
        "mode": "dense",
        "seed": None,
        "results": [result(cases[0], True), result(cases[1], False)],
    }
    sparse = common | {
        "variant": "hybrid_k32",
        "mode": "sparse",
        "seed": 0,
        "checkpoint_template": "checkpoint-{seed}",
        "distant_candidate_budget": 32,
        "member_policy": "hybrid",
        "history_fraction": 0.5,
        "block_size": 0,
        "results": [result(cases[0], True), result(cases[1], True)],
    }
    report = aggregate(manifest, [dense, sparse])
    overall = report["slices"]["overall"]["dense_pass_preservation"]
    assert overall == [{
        "variant": "hybrid_k32",
        "successes": 1,
        "trials": 1,
        "accuracy": 1.0,
        "wilson_ci95": wilson_interval(1, 1),
        "mean_retrieval_distance": cases[0]["retrieval_distance"],
    }]
    by_value = report["slices"]["by_value"]["dense_pass_preservation"]
    assert {row["value"] for row in by_value} == {
        f"{case['task']}:{case['expected']}" for case in cases
    }
    interval = wilson_interval(50, 100)
    assert interval[0] < 0.5 < interval[1]
    print("PASS matched retrieval report and Wilson uncertainty")


def test_routing_scan_benchmark():
    full_sweep_targets = {
        length: per_length_query_rng(7, length, 64).integers(0, length, size=64)
        for length in (262_144, 1_048_576, 2_097_152)
    }
    subset_targets = per_length_query_rng(7, 2_097_152, 64).integers(
        0, 2_097_152, size=64
    )
    np.testing.assert_array_equal(
        full_sweep_targets[2_097_152], subset_targets,
    )
    codes = np.array([
        [1, 2],
        [1, 3],
        [1, 2],
        [4, 2],
        [1, 2],
    ], dtype=np.uint16)
    tails = build_bucket_tails(codes, members=2, address_bits=4)
    np.testing.assert_array_equal(tails[0, 1], np.array([4, 2]))
    np.testing.assert_array_equal(tails[1, 2], np.array([4, 3]))
    crowded = np.ones((10, 2), dtype=np.uint16)
    reservoir_a, occupancy = build_bucket_index(
        crowded, capacity=3, address_bits=4, retention_policy="reservoir"
    )
    reservoir_b, _ = build_bucket_index(
        crowded, capacity=3, address_bits=4, retention_policy="reservoir"
    )
    np.testing.assert_array_equal(reservoir_a, reservoir_b)
    assert len(set(reservoir_a[0, 1])) == 3
    assert occupancy["max_occupancy"] == 10
    assert occupancy["eviction_count"] == 14
    assert occupancy["evicted_fraction"] == 0.7
    fingerprint, fingerprint_stats = build_bucket_index(
        crowded, capacity=4, address_bits=4, retention_policy="fingerprint"
    )
    assert np.sum(fingerprint >= 0) == 2
    assert fingerprint_stats["eviction_count"] == 18
    assert candidate_recall(
        np.array([[1, 2], [3, -1]]),
        np.array([[1, 2], [3, 4]]),
        k=2,
    ) == 0.75
    short = logical_history_bytes(16, 64, 4, 8, probes=2)
    long = logical_history_bytes(256, 64, 4, 8, probes=2)
    assert long["fp_scan"] == 16 * short["fp_scan"]
    assert long["binary64_scan"] == 16 * short["binary64_scan"]
    assert long["bucket_lookup"] == short["bucket_lookup"] == 768
    vectors = np.eye(4, dtype=np.float32)
    projection = np.arange(4 * 64, dtype=np.float32).reshape(4, 64) - 100.0
    byte_codes, table_codes = packed_codes(vectors, projection, chunk_size=2)
    assert byte_codes.shape == (4, 8)
    reconstructed = (
        byte_codes[:, 0::2].astype(np.uint16)
        | (byte_codes[:, 1::2].astype(np.uint16) << np.uint16(8))
    )
    np.testing.assert_array_equal(table_codes, reconstructed)
    print("PASS routing-scan index, byte accounting, and recall metrics")


if __name__ == "__main__":
    test_reference()
    test_multiprobe_neighbor()
    test_hybrid_bucket_history()
    test_hybrid_selector_reference()
    test_length_curriculum()
    test_multiprobe_causality()
    test_selector_causality_matrix()
    test_semantic_selector_parity_and_causality()
    test_block_selector_causality()
    test_causal_global_slots()
    test_semantic_router_gradient()
    test_learned_router_diagnostics()
    test_weight_resume()
    test_lfm_replacement_gate_and_causality()
    test_retrieval_generalization_manifest()
    test_retrieval_generalization_report()
    test_routing_scan_benchmark()
