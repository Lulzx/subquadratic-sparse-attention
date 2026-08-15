"""Distill LFM2.5 semantic embeddings into a multi-table binary router.

This is a block-retrieval probe, not a language-model quality experiment. Markdown
headings act as queries and their section bodies act as positive document blocks.
"""

import argparse
import json
import pathlib
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv


MODEL = "LiquidAI/LFM2.5-Embedding-350M"
REVISION = "f35ae2c91d687658dbf1f2b449382f0b019b9808"
TRAIN_FILES = [
    "README.md",
    "docs/architecture.md",
    "docs/design-history.md",
    "docs/experiments.md",
    "docs/reproduction.md",
]
EVAL_FILES = [
    "docs/limitations.md",
    "docs/model-card-audit.md",
    "docs/replication-roadmap.md",
]


def sections(paths, min_words):
    examples = []
    heading_pattern = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)
    for path in paths:
        text = pathlib.Path(path).read_text()
        matches = list(heading_pattern.finditer(text))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[match.end():end].strip()
            body = re.sub(r"```.*?```", " ", body, flags=re.DOTALL)
            body = re.sub(r"\s+", " ", body)
            if len(body.split()) >= min_words:
                examples.append((match.group(2).strip(), body, str(path)))
    if len(examples) < 2:
        raise ValueError("need at least two Markdown sections after filtering")
    return examples


def install_transformers_compatibility_shim():
    """Accept Transformers 5's seq_idx argument in LiquidAI's pinned wrapper."""
    original = Lfm2ShortConv.slow_forward
    if getattr(original, "_subq_compatible", False):
        return

    def compatible(self, hidden_states, past_key_values=None, cache_position=None,
                   attention_mask=None, seq_idx=None, **kwargs):
        del seq_idx, kwargs
        return original(
            self, hidden_states, past_key_values=past_key_values,
            cache_position=cache_position, attention_mask=attention_mask,
        )

    compatible._subq_compatible = True
    Lfm2ShortConv.slow_forward = compatible


def embed(model, examples, batch_size):
    queries = [f"query: {heading}" for heading, _, _ in examples]
    documents = [f"document: {body}" for _, body, _ in examples]
    query_vectors = model.encode(
        queries, batch_size=batch_size, normalize_embeddings=True,
        convert_to_tensor=True, show_progress_bar=False,
    )
    document_vectors = model.encode(
        documents, batch_size=batch_size, normalize_embeddings=True,
        convert_to_tensor=True, show_progress_bar=False,
    )
    return query_vectors.float(), document_vectors.float()


def hash_metrics(query_vectors, document_vectors, query_projection, key_projection,
                 tables, bits, radius):
    query_codes = (query_vectors @ query_projection >= 0).reshape(-1, tables, bits)
    key_codes = (document_vectors @ key_projection >= 0).reshape(-1, tables, bits)
    distances = (query_codes[:, None] != key_codes[None]).sum(dim=-1)
    selected = (distances <= radius).any(dim=-1)
    positive = selected.diagonal()
    counts = selected.sum(dim=-1)
    return {
        "positive_recall": float(positive.float().mean().cpu()),
        "mean_candidates": float(counts.float().mean().cpu()),
        "candidate_fraction": float(counts.float().mean().cpu() / len(document_vectors)),
        "zero_candidate_fraction": float((counts == 0).float().mean().cpu()),
    }


def train_router(query_vectors, document_vectors, tables, bits, steps, lr):
    width = query_vectors.shape[-1]
    scale = width ** -0.5
    projection = torch.nn.Parameter(
        torch.randn(width, tables * bits, device=query_vectors.device) * scale
    )
    optimizer = torch.optim.AdamW([projection], lr=lr)
    for _ in range(steps):
        query_codes = torch.tanh(query_vectors @ projection).reshape(-1, tables, bits)
        key_codes = torch.tanh(document_vectors @ projection).reshape(-1, tables, bits)
        scores = torch.einsum("itb,jtb->ijt", query_codes, key_codes).amax(dim=-1)
        scores = scores / bits ** 0.5
        labels = torch.arange(len(query_vectors), device=query_vectors.device)
        retrieval_loss = (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2
        bit_means = torch.cat([query_codes, key_codes]).mean(dim=0)
        balance_loss = bit_means.square().mean()
        loss = retrieval_loss + 0.1 * balance_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return projection.detach(), float(loss.detach().cpu())


def tradeoff_grid(query_vectors, document_vectors, projection, tables, bits, max_radius):
    grid = []
    table_counts = sorted(set([1, 2, 4, tables]))
    for table_count in table_counts:
        if table_count > tables:
            continue
        truncated = projection[:, :table_count * bits]
        for radius in range(max_radius + 1):
            grid.append({
                "tables": table_count,
                "radius": radius,
                **hash_metrics(
                    query_vectors, document_vectors, truncated, truncated,
                    table_count, bits, radius,
                ),
            })
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-words", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="runs/lfm2.5-embedding-router.json")
    args = parser.parse_args()
    if args.radius < 0 or args.radius > args.bits:
        parser.error("--radius must be between zero and --bits")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_examples = sections(TRAIN_FILES, args.min_words)
    eval_examples = sections(EVAL_FILES, args.min_words)

    started = time.perf_counter()
    model = SentenceTransformer(
        args.model, revision=args.revision, trust_remote_code=True, device=device
    )
    install_transformers_compatibility_shim()
    train_queries, train_documents = embed(model, train_examples, args.batch_size)
    eval_queries, eval_documents = embed(model, eval_examples, args.batch_size)
    del model

    width = train_queries.shape[-1]
    initial = torch.randn(width, args.tables * args.bits, device=device) / width ** 0.5
    random_metrics = tradeoff_grid(
        eval_queries, eval_documents, initial, args.tables, args.bits, args.radius
    )
    projection, final_loss = train_router(
        train_queries, train_documents, args.tables, args.bits, args.steps, args.lr
    )
    learned_metrics = tradeoff_grid(
        eval_queries, eval_documents, projection, args.tables, args.bits, args.radius
    )
    continuous_scores = eval_queries @ eval_documents.T
    continuous_top1 = float(
        (continuous_scores.argmax(dim=-1) == torch.arange(len(eval_queries), device=device))
        .float().mean().cpu()
    )
    result = {
        "model": args.model,
        "revision": args.revision,
        "device": device,
        "task": "Markdown heading-to-own-section retrieval",
        "train_sections": len(train_examples),
        "eval_sections": len(eval_examples),
        "embedding_width": width,
        "tables": args.tables,
        "bits": args.bits,
        "multiprobe_hamming_radius": args.radius,
        "continuous_top1_recall": continuous_top1,
        "random_hash_tradeoff": random_metrics,
        "learned_shared_hash_tradeoff": learned_metrics,
        "final_train_loss": final_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "This probes block retrieval, not causal language-model perplexity.",
            "The small documentation corpus is a plumbing benchmark, not evidence of generalization.",
        ],
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
