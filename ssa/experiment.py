import math

import numpy as np
import torch
import torch.nn.functional as F

from .tasks import VOCAB, mqar_batch


def device_name(requested="auto"):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def answer_tensors(answers, device):
    batches, positions, targets = [], [], []
    for batch, pairs in enumerate(answers):
        for position, target in pairs:
            batches.append(batch)
            positions.append(position - 1)
            targets.append(target)
    return (
        torch.tensor(batches, device=device),
        torch.tensor(positions, device=device),
        torch.tensor(targets, device=device),
    )


def answer_metrics(logits, answers):
    b, p, y = answer_tensors(answers, logits.device)
    selected = logits[b, p]
    loss = F.cross_entropy(selected, y)
    accuracy = (selected.argmax(-1) == y).float().mean()
    return loss, accuracy


def theta_scale(seq_len, train_len, head_dim):
    ratio = max(1.0, seq_len / train_len)
    return ratio ** (head_dim / max(1, head_dim - 2))


@torch.no_grad()
def evaluate(model, seq_len, train_len, batches, batch_size, seed, device):
    model.eval()
    rng = np.random.default_rng(seed)
    losses, correct, count = [], 0, 0
    scale = theta_scale(seq_len, train_len, model.blocks[0].attn.dh)
    for _ in range(batches):
        x, answers = mqar_batch(batch_size, seq_len, rng)
        logits = model(x.to(device), theta_scale=scale)
        loss, accuracy = answer_metrics(logits, answers)
        n = sum(len(row) for row in answers)
        losses.append(float(loss.cpu()))
        correct += int(round(float(accuracy.cpu()) * n))
        count += n
    return {"length": seq_len, "loss": sum(losses) / len(losses), "accuracy": correct / count, "n": count}


def model_config(args):
    return {
        "vocab": VOCAB,
        "d": args.d,
        "layers": args.layers,
        "heads": args.heads,
        "attn": args.attn,
        "window": args.window,
        "n_select_buckets": args.select_buckets,
        "members_per_bucket": args.members,
        "capacity": args.capacity,
        "ca": args.codes,
        "cb": args.codes,
        "aux_window": args.aux_window,
        "chunk_q": args.chunk_q,
    }
