"""Aggregate the fixed-budget bucket-retention ablation."""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlx_routing_scan_report import format_bytes, length_label


def variant_name(report):
    config = report["config"]
    return f"c{config['bucket_capacity']}_{config['retention_policy']}"


def load_reports(paths):
    reports = [json.loads(path.read_text()) for path in paths]
    if len(reports) != 5:
        raise ValueError("expected exactly five retention reports")
    reference = reports[0]["config"]
    matched = (
        "lengths", "dim", "queries", "k", "tables", "bits", "probes",
        "noise", "warmups", "repeats", "recall_batch", "seed",
    )
    for report in reports[1:]:
        if any(report["config"][key] != reference[key] for key in matched):
            raise ValueError("retention reports do not use a matched protocol")
    identities = {variant_name(report) for report in reports}
    expected = {
        "c16_tail", "c32_tail", "c64_tail", "c16_reservoir", "c32_reservoir",
    }
    if identities != expected:
        raise ValueError(f"expected variants {sorted(expected)}, got {sorted(identities)}")
    return {variant_name(report): report for report in reports}


def render_markdown(reports):
    lines = [
        "# Bucket-retention tradeoff",
        "",
        "All variants use four 16-bit tables, two probes, and rerank a bounded "
        "addressed pool to final K=32. Only bucket capacity and retention policy vary.",
        "",
        "| Context | Variant | Routing | Bytes/query | Needle recall | FP-top32 recall | Address agreement | Recall given address | Mean occupancy | P99 | Max | Evicted postings |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = (
        "c16_tail", "c16_reservoir", "c32_tail", "c32_reservoir", "c64_tail",
    )
    lengths = [row["length"] for row in reports["c16_tail"]["results"]]
    for length_index, length in enumerate(lengths):
        for identity in ordered:
            row = reports[identity]["results"][length_index]
            method = row["methods"]["bucket_lookup"]
            occupancy = row["bucket_occupancy"]
            address = row["query_target_probe_address_match"]
            conditional = method["needle_recall"] / address if address else 0.0
            lines.append(
                f"| {length_label(length)} | `{identity}` | "
                f"{method['median_routing_us_per_query']:.1f} us | "
                f"{format_bytes(method['logical_history_bytes_per_query'])} | "
                f"{100 * method['needle_recall']:.2f}% | "
                f"{100 * method['candidate_recall_at_k']:.2f}% | "
                f"{100 * address:.2f}% | {100 * conditional:.2f}% | "
                f"{occupancy['mean_occupancy']:.2f} | "
                f"{occupancy['p99_occupancy']:.0f} | "
                f"{occupancy['max_occupancy']} | "
                f"{occupancy['eviction_count']:,} "
                f"({100 * occupancy['evicted_fraction']:.2f}%) |"
            )
    c64_2m = reports["c64_tail"]["results"][-1]
    c64_method = c64_2m["methods"]["bucket_lookup"]
    c32_reservoir_2m = reports["c32_reservoir"]["results"][-1]
    c32_reservoir_method = c32_reservoir_2m["methods"]["bucket_lookup"]
    lines.extend([
        "",
        "At 2M, capacity 64 raises needle recall to "
        f"{100 * c64_method['needle_recall']:.2f}% at "
        f"{c64_method['median_routing_us_per_query']:.1f} us/query and "
        f"{format_bytes(c64_method['logical_history_bytes_per_query'])}/query. "
        "The best tested point at or below 3 KiB is capacity-32 reservoir at "
        f"{100 * c32_reservoir_method['needle_recall']:.2f}% and "
        f"{c32_reservoir_method['median_routing_us_per_query']:.1f} us/query. "
        "Neither reaches the active target of at least 95% recall within 3 KiB. "
        "The result supports a bounded storage/recall tradeoff, not a general "
        "attention-quality claim: FP-top32 overlap remains "
        f"{100 * c64_method['candidate_recall_at_k']:.2f}%.",
        "",
    ])
    return "\n".join(lines)


def render_plot(reports, output):
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    colors = ("#1b9e77", "#7570b3", "#d95f02")
    tail_reports = [reports[name] for name in ("c16_tail", "c32_tail", "c64_tail")]
    lengths = [row["length"] for row in tail_reports[0]["results"]]
    for length_index, (length, color) in enumerate(zip(lengths, colors)):
        byte_values = [
            report["results"][length_index]["methods"]["bucket_lookup"]
            ["logical_history_bytes_per_query"]
            for report in tail_reports
        ]
        recalls = [
            100 * report["results"][length_index]["methods"]["bucket_lookup"]
            ["needle_recall"]
            for report in tail_reports
        ]
        axis.plot(
            byte_values, recalls, marker="o", linewidth=2,
            label=f"Tail, {length_label(length)}", color=color,
        )
        for reservoir_name in ("c16_reservoir", "c32_reservoir"):
            reservoir = reports[reservoir_name]["results"][length_index]
            axis.scatter(
                reservoir["methods"]["bucket_lookup"]
                ["logical_history_bytes_per_query"],
                100 * reservoir["methods"]["bucket_lookup"]["needle_recall"],
                marker="x", s=90, linewidths=2.5, color=color,
            )
    axis.set_xscale("log", base=2)
    axis.set_xticks((1536, 3072, 6144), ("1.5 KiB", "3 KiB", "6 KiB"))
    axis.set_ylim(45, 102)
    axis.set_xlabel("Logical historical-index bytes per query")
    axis.set_ylabel("Needle recall (%)")
    axis.set_title("Bounded bucket storage trades bytes for million-token recall")
    axis.grid(alpha=0.25)
    axis.legend(
        frameon=False, ncol=3, fontsize=9, loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    axis.text(
        1580, 47, "x = deterministic reservoir at capacity 16 or 32",
        fontsize=9, color="#333333",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format=output.suffix.lstrip(".") or "svg")
    plt.close(figure)
    if output.suffix == ".svg":
        output.write_text(
            "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--markdown-output", required=True, type=pathlib.Path)
    parser.add_argument("--plot-output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    reports = load_reports(args.inputs)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(reports) + "\n")
    render_plot(reports, args.plot_output)
    print(json.dumps({
        "markdown_output": str(args.markdown_output),
        "plot_output": str(args.plot_output),
    }))


if __name__ == "__main__":
    main()
