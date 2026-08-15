#!/usr/bin/env python3
"""Audit trained donor hash routers on held-out dense-attention targets."""

import argparse
import hashlib
import json
import pathlib
import platform
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

from mlx_donor_router import (
    DEFAULT_EVAL_FILES,
    DonorHashRouter,
    donor_example,
    hard_metrics,
    language_body,
    parse_paths,
    token_segments,
)


CONFIG_FIELDS = (
    "model",
    "layer",
    "seq_len",
    "stride",
    "window",
    "sink_tokens",
    "tables",
    "bits",
    "members",
    "probes",
    "eval_segments",
    "eval_files",
    "teacher_target",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(checkpoint):
    metadata_path = checkpoint.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing checkpoint metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    metadata["probes"] = metadata.get("probes") or 1
    metadata.setdefault(
        "teacher_target",
        "normalized_mean_attention_probability_times_value_l2_norm",
    )
    return metadata_path, metadata


def validate_compatible(configs):
    reference = configs[0]
    for config in configs[1:]:
        mismatches = [
            field for field in CONFIG_FIELDS
            if config.get(field) != reference.get(field)
            and field not in ("eval_segments",)
        ]
        if mismatches:
            raise ValueError(
                "checkpoints require different evaluation configurations: "
                + ", ".join(mismatches)
            )


def scalar_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def aggregate(seed_results):
    metrics = [row["metrics"] for row in seed_results]
    fields = {
        "retained_teacher_mass": [row["retained_teacher_mass"] for row in metrics],
        "teacher_top1_recall": [row["teacher_top1_recall"] for row in metrics],
        "unique_candidates": [row["unique_candidates"] for row in metrics],
        "soft_topk_retained_teacher_mass": [
            row["soft_topk_retained_teacher_mass"] for row in metrics
        ],
        "soft_topk_teacher_top1_recall": [
            row["soft_topk_teacher_top1_recall"] for row in metrics
        ],
        "exact_agreement_any_table": [
            row["query_key_agreement"]["exact_any_table"] for row in metrics
        ],
        "probed_agreement_any_table": [
            row["query_key_agreement"]["probed_any_table"] for row in metrics
        ],
        "empty_bucket_fraction": [
            row["bucket_occupancy"]["mean"]["empty_bucket_fraction"]
            for row in metrics
        ],
        "normalized_bucket_entropy": [
            row["bucket_occupancy"]["mean"]["normalized_entropy"]
            for row in metrics
        ],
        "collision_inflation_vs_balanced": [
            row["bucket_occupancy"]["mean"]["collision_inflation_vs_balanced"]
            for row in metrics
        ],
        "mean_pairwise_table_success_correlation": [
            row["table_retrieval"]["mean_pairwise_success_correlation"] or 0.0
            for row in metrics
        ],
        "independence_prediction_gap": [
            row["table_retrieval"]["prediction_gap"] for row in metrics
        ],
        "no_agreement_fraction": [
            row["failure_attribution"]["fractions"][
                "no_probed_address_agreement"
            ]
            for row in metrics
        ],
        "agreement_without_selection_fraction": [
            row["failure_attribution"]["fractions"][
                "agreement_without_selection"
            ]
            for row in metrics
        ],
    }
    distance_order = {"1-64": 0, "65-128": 1, "129-256": 2, "257-512": 3, "513+": 4}
    distance_labels = sorted(
        {label for row in metrics for label in row["distance"]},
        key=lambda label: distance_order[label],
    )
    return {
        "metrics": {name: scalar_summary(values) for name, values in fields.items()},
        "distance": {
            label: {
                field: scalar_summary([
                    row["distance"][label][field]
                    for row in metrics if label in row["distance"]
                ])
                for field in (
                    "teacher_top1_recall",
                    "retained_teacher_mass",
                    "mean_unique_candidates",
                )
            }
            for label in distance_labels
        },
    }


def render_markdown(report):
    aggregate_metrics = report["aggregate"]["metrics"]

    def mean(name):
        return aggregate_metrics[name]["mean"]

    lines = [
        "# Learned-router audit",
        "",
        f"Model: `{report['configuration']['model']}`, layer "
        f"`{report['configuration']['layer']}`, "
        f"{len(report['seeds'])} checkpoint seeds, "
        f"{report['configuration']['eval_segments']} held-out segments per seed.",
        "Teacher importance is normalized mean attention probability multiplied "
        "by the value-vector L2 norm.",
        "",
        "## Aggregate",
        "",
        "| Measurement | Mean |",
        "|---|---:|",
        f"| Hard retained teacher contribution | {mean('retained_teacher_mass'):.4f} |",
        f"| Hard teacher top-1 recall | {mean('teacher_top1_recall'):.4f} |",
        f"| Soft top-k retained contribution | {mean('soft_topk_retained_teacher_mass'):.4f} |",
        f"| Soft top-k teacher top-1 recall | {mean('soft_topk_teacher_top1_recall'):.4f} |",
        f"| Mean unique hard candidates | {mean('unique_candidates'):.2f} |",
        f"| Exact address agreement, any table | {mean('exact_agreement_any_table'):.4f} |",
        f"| Empty bucket fraction | {mean('empty_bucket_fraction'):.4f} |",
        f"| Normalized bucket entropy | {mean('normalized_bucket_entropy'):.4f} |",
        f"| Collision inflation vs balanced | {mean('collision_inflation_vs_balanced'):.3f}x |",
        f"| Mean table-success correlation | {mean('mean_pairwise_table_success_correlation'):.4f} |",
        f"| Actual minus independent-table prediction | {mean('independence_prediction_gap'):.4f} |",
        f"| Failure: no address agreement | {mean('no_agreement_fraction'):.4f} |",
        f"| Failure: agreement but not selected | {mean('agreement_without_selection_fraction'):.4f} |",
        "",
        "## Retrieval distance",
        "",
        "| Distance | Teacher top-1 recall | Retained contribution | Unique candidates |",
        "|---|---:|---:|---:|",
    ]
    for label, row in report["aggregate"]["distance"].items():
        lines.append(
            f"| {label} | {row['teacher_top1_recall']['mean']:.4f} | "
            f"{row['retained_teacher_mass']['mean']:.4f} | "
            f"{row['mean_unique_candidates']['mean']:.2f} |"
        )
    lines.extend([
        "",
        "Exact/probed agreement is measured against the dense teacher's top distant "
        "key. Occupancy is measured per held-out segment and table. The independent-"
        "table prediction combines measured per-table success marginals; its gap to "
        "actual recall measures dependence, not model quality.",
        "",
        "This audit attributes hard-routing loss but does not measure perplexity, "
        "generation behavior, or end-to-end speed.",
        "",
        "## Reproduction",
        "",
        "```bash",
        report["command"],
        "```",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--eval-segments", type=int, default=0)
    parser.add_argument("--memory-limit-mb", type=int, default=1400)
    parser.add_argument("--cache-limit-mb", type=int, default=128)
    parser.add_argument("--output", default="runs/learned-router-audit.json")
    parser.add_argument("--markdown-output", default="runs/learned-router-audit.md")
    args = parser.parse_args()
    checkpoints = [pathlib.Path(value) for value in args.checkpoints]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        parser.error("missing checkpoints: " + ", ".join(missing))
    metadata_rows = [load_metadata(path) for path in checkpoints]
    configs = [row[1] for row in metadata_rows]
    validate_compatible(configs)
    config = configs[0].copy()
    config["teacher_target"] = (
        "normalized_mean_attention_probability_times_value_l2_norm"
    )
    if args.eval_segments:
        config["eval_segments"] = args.eval_segments
    if config["eval_segments"] < 1:
        parser.error("evaluation requires at least one segment")

    mx.set_memory_limit(args.memory_limit_mb * 2**20)
    mx.set_cache_limit(args.cache_limit_mb * 2**20)
    model, tokenizer = load(config["model"], lazy=True)
    body = language_body(model)
    if config["layer"] >= len(body.layers) or not hasattr(
        body.layers[config["layer"]], "self_attn"
    ):
        parser.error("configured layer is not an available dense-attention layer")
    eval_paths = parse_paths(config.get("eval_files", ""), DEFAULT_EVAL_FILES)
    tokens = token_segments(
        tokenizer,
        eval_paths,
        config["seq_len"],
        config["stride"],
        config["eval_segments"],
    )
    examples = [
        donor_example(
            model,
            token_batch,
            config["layer"],
            config["window"],
            config["sink_tokens"],
        )
        for token_batch in tokens
    ]
    width = examples[0][0].shape[-1]
    del model
    mx.clear_cache()

    seed_results = []
    for checkpoint, (metadata_path, metadata) in zip(checkpoints, metadata_rows):
        router = DonorHashRouter(width, config["tables"], config["bits"])
        router.load_weights(str(checkpoint))
        router.freeze()
        mx.eval(router.parameters())
        metrics = hard_metrics(
            router,
            examples,
            config["members"],
            config["probes"],
            config["window"],
            config["sink_tokens"],
        )
        seed_results.append({
            "seed": metadata.get("seed"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "metadata": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "metrics": metrics,
        })
        del router
        mx.clear_cache()

    command = " ".join(["python3", pathlib.Path(__file__).name, *sys.argv[1:]])
    report = {
        "configuration": {field: config.get(field) for field in CONFIG_FIELDS},
        "evaluation_corpus": [
            {"path": str(path), "sha256": sha256(path)} for path in eval_paths
        ],
        "platform": platform.platform(),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "command": command,
        "seeds": seed_results,
        "aggregate": aggregate(seed_results),
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
    }
    output = pathlib.Path(args.output)
    markdown_output = pathlib.Path(args.markdown_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    markdown_output.write_text(render_markdown(report))
    print(json.dumps({
        "aggregate": report["aggregate"]["metrics"],
        "peak_memory_mb": report["peak_memory_mb"],
        "output": str(output),
        "markdown_output": str(markdown_output),
    }, indent=2))


if __name__ == "__main__":
    main()
