"""Render a concise report and scaling plot from a routing-scan benchmark."""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "fp_scan": "FP16 full scan",
    "binary64_scan": "64-bit full scan",
    "bucket_lookup": "Addressed pool + rerank",
}
COLORS = {
    "fp_scan": "#d95f02",
    "binary64_scan": "#7570b3",
    "bucket_lookup": "#1b9e77",
}


def format_bytes(value):
    if value >= 2**20:
        return f"{value / 2**20:.2f} MiB"
    if value >= 2**10:
        return f"{value / 2**10:.2f} KiB"
    return f"{value} B"


def length_label(length):
    if length >= 2**20 and length % 2**20 == 0:
        return f"{length // 2**20}M"
    return f"{length // 2**10}K"


def endpoint_growth(report, method):
    first, last = report["results"][0], report["results"][-1]
    return (
        last["methods"][method]["median_routing_us_per_query"]
        / first["methods"][method]["median_routing_us_per_query"]
    )


def render_markdown(report):
    rows = report["results"]
    last = rows[-1]
    bucket_last = last["methods"]["bucket_lookup"]
    lines = [
        "# Routing-only scan benchmark",
        "",
        "Timed scope: one query encoding plus routing and K=32 candidate selection. "
        "Index construction, KV gather, attention, and model execution are excluded.",
        "",
        "| Context | FP16 scan | Binary64 scan | Addressed | FP bytes/query | Binary bytes/query | Addressed bytes/query | Addressed FP-top32 recall | Addressed needle recall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fp = row["methods"]["fp_scan"]
        binary = row["methods"]["binary64_scan"]
        bucket = row["methods"]["bucket_lookup"]
        lines.append(
            f"| {row['length']:,} | {fp['median_routing_us_per_query']:.1f} us | "
            f"{binary['median_routing_us_per_query']:.1f} us | "
            f"{bucket['median_routing_us_per_query']:.1f} us | "
            f"{format_bytes(fp['logical_history_bytes_per_query'])} | "
            f"{format_bytes(binary['logical_history_bytes_per_query'])} | "
            f"{format_bytes(bucket['logical_history_bytes_per_query'])} | "
            f"{100 * bucket['candidate_recall_at_k']:.2f}% | "
            f"{100 * bucket['needle_recall']:.2f}% |"
        )
    fp_last = last["methods"]["fp_scan"]["median_routing_us_per_query"]
    binary_last = last["methods"]["binary64_scan"]["median_routing_us_per_query"]
    addressed_last = bucket_last["median_routing_us_per_query"]
    needle_last = bucket_last["needle_recall"]
    addressed_recalls = [
        row["methods"]["bucket_lookup"]["candidate_recall_at_k"] for row in rows
    ]
    lines.extend([
        "",
        f"Endpoint scaling from {length_label(rows[0]['length'])} to "
        f"{length_label(last['length'])}:",
        "",
        f"- FP16 full scan: {endpoint_growth(report, 'fp_scan'):.2f}x latency.",
        f"- 64-bit full scan: {endpoint_growth(report, 'binary64_scan'):.2f}x latency.",
        f"- Addressed pool: {endpoint_growth(report, 'bucket_lookup'):.2f}x latency.",
        f"- At {length_label(last['length'])}, addressed lookup is "
        f"{fp_last / addressed_last:.2f}x faster "
        f"than FP16 scan and {binary_last / addressed_last:.2f}x faster than the "
        "64-bit scan in this routing-only prototype.",
        "",
        "The addressed curve supports the bounded-lookup mechanism, not a full-model "
        f"speed claim. At the longest context, needle recall is {100 * needle_last:.1f}%; "
        f"{100 * min(addressed_recalls):.1f}–"
        f"{100 * max(addressed_recalls):.1f}% FP-top32 overlap is "
        "too low for a general attention-quality claim; learned addressing or a "
        "stronger multiprobe policy remains necessary.",
        "",
    ])
    return "\n".join(lines)


def render_plot(report, output):
    rows = report["results"]
    lengths = [row["length"] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), constrained_layout=True)
    latency = axes[0]
    for method in LABELS:
        latency.plot(
            lengths,
            [row["methods"][method]["median_routing_us_per_query"] for row in rows],
            marker="o",
            linewidth=2,
            label=LABELS[method],
            color=COLORS[method],
        )
    latency.set_xscale("log", base=2)
    latency.set_xticks(lengths, [length_label(length) for length in lengths])
    latency.set_ylabel("Median routing latency (us/query)")
    latency.set_title("Routing scan grows; addressed lookup stays bounded")
    latency.grid(alpha=0.25)
    latency.legend(frameon=False, ncol=3, fontsize=9)

    recall = axes[1]
    recall.plot(
        lengths,
        [100 * row["methods"]["binary64_scan"]["candidate_recall_at_k"] for row in rows],
        marker="o",
        linewidth=2,
        label="Binary FP-top32 overlap",
        color=COLORS["binary64_scan"],
    )
    recall.plot(
        lengths,
        [100 * row["methods"]["bucket_lookup"]["candidate_recall_at_k"] for row in rows],
        marker="o",
        linewidth=2,
        label="Addressed FP-top32 overlap",
        color=COLORS["bucket_lookup"],
    )
    recall.plot(
        lengths,
        [100 * row["methods"]["bucket_lookup"]["needle_recall"] for row in rows],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Addressed needle recall",
        color=COLORS["fp_scan"],
    )
    recall.set_xscale("log", base=2)
    recall.set_xticks(lengths, [length_label(length) for length in lengths])
    recall.set_ylim(0, 105)
    recall.set_xlabel("Historical tokens")
    recall.set_ylabel("Recall (%)")
    recall.set_title("Fixed bucket tails lose needles as occupancy grows")
    recall.grid(alpha=0.25)
    recall.legend(frameon=False, ncol=1, fontsize=9, loc="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format=output.suffix.lstrip(".") or "svg")
    plt.close(figure)
    if output.suffix == ".svg":
        output.write_text(
            "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--markdown-output", type=pathlib.Path, required=True)
    parser.add_argument("--plot-output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    required = {"fp_scan", "binary64_scan", "bucket_lookup"}
    if not report.get("results") or any(
        set(row.get("methods", {})) != required for row in report["results"]
    ):
        raise ValueError("input does not contain the three matched routing methods")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(report) + "\n")
    render_plot(report, args.plot_output)
    print(json.dumps({
        "markdown_output": str(args.markdown_output),
        "plot_output": str(args.plot_output),
    }))


if __name__ == "__main__":
    main()
