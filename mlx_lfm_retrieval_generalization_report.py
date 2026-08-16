"""Aggregate matched LFM2.5 retrieval-generalization reports."""

import argparse
import json
import math
import pathlib
import statistics

from mlx_lfm_retrieval_generalization import load_manifest


def wilson_interval(successes, trials, z=1.959963984540054):
    if trials == 0:
        return None
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(
        probability * (1.0 - probability) / trials
        + z * z / (4.0 * trials * trials)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def distance_band(position):
    if position <= 0.2:
        return "far"
    if position >= 0.8:
        return "near"
    return "middle"


def metric(rows, eligible_field=None):
    if eligible_field:
        rows = [row for row in rows if row.get(eligible_field) is not None]
        values = [row[eligible_field] for row in rows]
    else:
        values = [row["correct"] for row in rows]
    successes = sum(values)
    trials = len(values)
    return {
        "successes": successes,
        "trials": trials,
        "accuracy": successes / trials if trials else None,
        "wilson_ci95": wilson_interval(successes, trials),
        "mean_retrieval_distance": (
            sum(row["retrieval_distance"] for row in rows) / len(rows)
            if rows else None
        ),
    }


def group_rows(rows, dimensions, eligible_field=None):
    groups = {}
    for row in rows:
        key = tuple(row[dimension] for dimension in dimensions)
        groups.setdefault(key, []).append(row)
    result = []
    for key in sorted(groups, key=lambda value: tuple(str(item) for item in value)):
        result.append({
            **dict(zip(dimensions, key)),
            **metric(groups[key], eligible_field=eligible_field),
        })
    return result


def seed_uncertainty(rows):
    result = []
    variants = sorted({row["variant"] for row in rows if row["seed"] is not None})
    for variant in variants:
        seed_rows = group_rows(
            [row for row in rows if row["variant"] == variant],
            ("seed",),
            eligible_field="dense_pass_preserved",
        )
        accuracies = [row["accuracy"] for row in seed_rows if row["accuracy"] is not None]
        result.append({
            "variant": variant,
            "seeds": len(accuracies),
            "mean_seed_accuracy": statistics.mean(accuracies) if accuracies else None,
            "seed_standard_deviation": (
                statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0
            ),
            "minimum_seed_accuracy": min(accuracies) if accuracies else None,
            "maximum_seed_accuracy": max(accuracies) if accuracies else None,
        })
    return result


def aggregate(manifest, reports, skipped=None):
    if not reports:
        raise ValueError("at least one input report is required")
    manifest_hash = manifest["manifest_sha256"]
    if any(report.get("manifest_sha256") != manifest_hash for report in reports):
        raise ValueError("input reports do not share the requested manifest")
    manifest_cases = {case["case_id"]: case for case in manifest["cases"]}
    identities = [(report["variant"], report.get("seed")) for report in reports]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate variant/seed input report")
    dense_reports = [report for report in reports if report["mode"] == "dense"]
    if len(dense_reports) != 1:
        raise ValueError("exactly one dense report is required")

    dense_results = {
        row["case_id"]: row["contains_expected"]
        for row in dense_reports[0]["results"]
    }
    rows = []
    variants = []
    coverage = []
    for report in reports:
        variants.append({
            key: report.get(key)
            for key in (
                "variant",
                "mode",
                "seed",
                "checkpoint_template",
                "distant_candidate_budget",
                "member_policy",
                "history_fraction",
                "span_size",
                "block_size",
            )
        })
        coverage.append({
            "variant": report["variant"],
            "seed": report.get("seed"),
            "cases": len(report["results"]),
            "target_lengths": sorted({
                row["target_length"] for row in report["results"]
            }),
            "peak_memory_mb": report.get("peak_memory_mb"),
            "elapsed_seconds": report.get("elapsed_seconds"),
        })
        for result in report["results"]:
            manifest_case = manifest_cases.get(result["case_id"])
            if manifest_case is None:
                raise ValueError(f"unknown case id: {result['case_id']}")
            if (
                result.get("expected") is not None
                and result["expected"] != manifest_case["expected"]
            ):
                raise ValueError(f"expected value mismatch: {result['case_id']}")
            dense_correct = dense_results.get(result["case_id"])
            rows.append({
                "variant": report["variant"],
                "seed": report.get("seed"),
                "task": result["task"],
                "value": f"{result['task']}:{manifest_case['expected']}",
                "template": f"{result['task']}:{result['template']}",
                "target_length": result["target_length"],
                "distance_band": distance_band(result["position"]),
                "retrieval_distance": result["retrieval_distance"],
                "correct": result["contains_expected"],
                "dense_correct": dense_correct,
                "dense_pass_preserved": (
                    result["contains_expected"] if dense_correct else None
                ),
            })

    slices = {}
    dimensions = {
        "overall": ("variant",),
        "by_seed": ("variant", "seed"),
        "by_task": ("variant", "task"),
        "by_value": ("variant", "value"),
        "by_template": ("variant", "template"),
        "by_length": ("variant", "target_length"),
        "by_distance": ("variant", "distance_band"),
        "by_seed_task": ("variant", "seed", "task"),
    }
    for name, fields in dimensions.items():
        slices[name] = {
            "absolute": group_rows(rows, fields),
            "dense_pass_preservation": group_rows(
                [row for row in rows if row["variant"] != "dense"],
                fields,
                eligible_field="dense_pass_preserved",
            ),
        }
    return {
        "manifest_sha256": manifest_hash,
        "manifest_cases": len(manifest["cases"]),
        "manifest_lengths": manifest["lengths"],
        "variants": variants,
        "coverage": coverage,
        "skipped": skipped or [],
        "seed_uncertainty": seed_uncertainty(rows),
        "slices": slices,
    }


def percent(value):
    return "—" if value is None else f"{100.0 * value:.2f}%"


def render_metric_table(title, rows, dimensions):
    lines = [f"## {title}", ""]
    headings = [dimension.replace("_", " ").title() for dimension in dimensions]
    lines.extend([
        "| " + " | ".join(headings + ["Correct", "Accuracy", "Wilson 95% CI"]) + " |",
        "|" + "|".join(["---"] * len(headings) + ["---:", "---:", "---:"]) + "|",
    ])
    for row in rows:
        interval = row["wilson_ci95"]
        interval_text = (
            "—" if interval is None
            else f"{percent(interval[0])}–{percent(interval[1])}"
        )
        lines.append(
            "| " + " | ".join(
                [str(row[dimension]) for dimension in dimensions]
                + [
                    f"{row['successes']}/{row['trials']}",
                    percent(row["accuracy"]),
                    interval_text,
                ]
            ) + " |"
        )
    lines.append("")
    return lines


def render_markdown(report):
    lines = [
        "# LFM2.5 retrieval-generalization gate",
        "",
        f"Manifest: `{report['manifest_sha256']}` with "
        f"{report['manifest_cases']} matched cases.",
        "Wilson intervals describe case-level binomial uncertainty; the separate "
        "seed summary exposes between-seed variation.",
        "",
    ]
    dense_pass = report["slices"]
    for title, name, dimensions in (
        ("Dense-pass preservation", "overall", ("variant",)),
        ("Preservation by seed", "by_seed", ("variant", "seed")),
        ("Preservation by task", "by_task", ("variant", "task")),
        ("Preservation by context length", "by_length", ("variant", "target_length")),
        ("Preservation by retrieval distance", "by_distance", ("variant", "distance_band")),
        ("Preservation by value", "by_value", ("variant", "value")),
        ("Preservation by prompt template", "by_template", ("variant", "template")),
    ):
        lines.extend(render_metric_table(
            title,
            dense_pass[name]["dense_pass_preservation"],
            dimensions,
        ))
    lines.extend([
        "## Coverage and resource bounds",
        "",
        "| Variant | Seed | Cases | Lengths | Peak MLX memory |",
        "|---|---:|---:|---|---:|",
    ])
    for row in report["coverage"]:
        seed = "—" if row["seed"] is None else row["seed"]
        lengths = ", ".join(map(str, row["target_lengths"]))
        peak = row["peak_memory_mb"]
        lines.append(
            f"| {row['variant']} | {seed} | {row['cases']} | {lengths} | "
            f"{peak:.2f} MB |"
        )
    if report["skipped"]:
        lines.extend(["", "Skipped evaluations:", ""])
        for row in report["skipped"]:
            lines.append(
                f"- `{row['variant']}` at `{row['length']}`: {row['reason']}"
            )
    lines.append("")
    return "\n".join(lines)


def parse_skip(spec):
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--skip must use VARIANT:LENGTH:REASON")
    return {"variant": parts[0], "length": int(parts[1]), "reason": parts[2]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--markdown-output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    reports = [json.loads(path.read_text()) for path in args.inputs]
    report = aggregate(manifest, reports, [parse_skip(spec) for spec in args.skip])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(report))
    print(json.dumps({
        "output": str(args.output),
        "markdown_output": str(args.markdown_output),
        "manifest_sha256": report["manifest_sha256"],
        "coverage": report["coverage"],
    }, indent=2))


if __name__ == "__main__":
    main()
