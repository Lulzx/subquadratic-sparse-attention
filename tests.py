import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from ssa.model import SSAAttention, TinyLM
from ssa.tasks import mqar_batch


def test_shapes():
    attn = SSAAttention(192, heads=6)
    x = torch.randn(2, 512, 192)
    o, _ = attn(x)
    assert o.shape == x.shape
    print("PASS shapes")


def test_causality():
    torch.manual_seed(0)
    attn = SSAAttention(192, heads=6)
    attn.eval()
    x = torch.randn(1, 256, 192)
    with torch.no_grad():
        o1, _ = attn(x)
        x2 = x.clone()
        x2[0, 200:] = torch.randn_like(x2[0, 200:])
        o2, _ = attn(x2)
    assert torch.allclose(o1[0, :200], o2[0, :200], atol=1e-5), "future leaked into past"
    print("PASS causality")


def test_content_colocation():
    attn = SSAAttention(192, heads=6)
    x = torch.randn(1, 512, 192)
    x[0, 400] = x[0, 100]
    x[0, 450] = x[0, 50]
    bid = attn.router.assign(x)
    assert torch.equal(bid[0, 400], bid[0, 100])
    assert torch.equal(bid[0, 450], bid[0, 50])
    sel = attn._select_indices(x)
    print("PASS content colocation | sel K:", sel.shape[-1])


def test_selection_contains_match():
    attn = SSAAttention(192, heads=6)
    x = torch.randn(1, 512, 192)
    x[0, 400] = x[0, 100]
    sel = attn._select_indices(x)
    found = (sel[0, 400] == 100).any() or (sel[0, 400] == 101).any()
    print("query@400 retrieves stored anchor@100 (or +1):", bool(found))


def test_long_selector_recall():
    torch.manual_seed(1)
    n = 16384
    attn = SSAAttention(64, heads=4, window=64, chunk_q=512)
    x = torch.randn(1, n, 64)
    stored = torch.arange(100, 10100, 100)
    query = torch.arange(n - len(stored), n)
    x[0, query] = x[0, stored]
    with torch.no_grad():
        selected = attn._select_indices(x)
    hits = [
        bool((selected[0, q] == s).any() or (selected[0, q] == s + 1).any())
        for s, q in zip(stored, query)
    ]
    assert all(hits)
    print("PASS 16K selector recall: 100/100")


def test_train_step():
    torch.manual_seed(0)
    model = TinyLM(attn="ssa")
    rng = np.random.default_rng(0)
    x, ans = mqar_batch(2, 512, rng)
    logits = model(x)
    assert logits.shape == (2, 512, 2048)
    aux = model.aux_loss(x)
    assert torch.isfinite(aux), f"aux not finite: {aux.item()}"
    print("PASS train step, aux =", round(aux.item(), 4))


def test_window_mode():
    model = TinyLM(attn="window")
    x = torch.randint(0, 2048, (2, 256))
    out = model(x)
    assert torch.isfinite(out).all()
    print("PASS window mode finite")


if __name__ == "__main__":
    test_shapes()
    test_causality()
    test_content_colocation()
    test_selection_contains_match()
    test_long_selector_recall()
    test_train_step()
    test_window_mode()
