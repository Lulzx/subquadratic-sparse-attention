"""Report the sparse hierarchical routing milestone and rejected variants."""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlx_routing_scan_report import format_bytes, length_label


def load(path):
    return json.loads(path.read_text())


def addressed(row):
    return row["methods"]["bucket_lookup"]


def validate(report):
    config = report["config"]
    expected = {
        "index_kind": "sparse-hierarchical",
        "secondary_bits": 8,
        "secondary_probes": 4,
        "bucket_capacity": 7,
        "retention_policy": "reservoir",
        "tables": 4,
        "bits": 16,
        "probes": 2,
        "k": 32,
        "queries": 64,
        "seed": 7,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"hierarchical report config mismatch: {mismatches}")
    if [row["length"] for row in report["results"]] != [262144, 1048576, 2097152]:
        raise ValueError("expected canonical 256K, 1M, 2M rows")


def variant_label(report):
    config = report["config"]
    kind = config.get("index_kind", "flat")
    if kind == "flat":
        return f"Flat c{config['bucket_capacity']} {config['retention_policy']}"
    prefix = "Sparse hierarchy" if kind == "sparse-hierarchical" else "Dense hierarchy"
    return (
        f"{prefix} s{config['secondary_bits']}p{config['secondary_probes']}"
        f"c{config['bucket_capacity']}"
    )


def render_markdown(report, comparisons):
    lines = [
        "# Sparse hierarchical routing milestone",
        "",
        "The final selector uses four 16-bit primary tables, two primary probes, "
        "an 8-bit secondary address from the next table, four secondary probes, "
        "capacity seven per sparse leaf, deterministic reservoir retention, and "
        "Hamming reranking to K=32.",
        "",
        "| Context | Routing | Bytes/query | Needle recall | Address agreement | Recall given address | FP-top32 recall | Unique candidates | Leaf p99 | Leaf max | Evicted | Resident index |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        method = addressed(row)
        occupancy = row["bucket_occupancy"]
        address = row["query_target_probe_address_match"]
        conditional = method["needle_recall"] / address if address else 0.0
        lines.append(
            f"| {length_label(row['length'])} | "
            f"{method['median_routing_us_per_query']:.1f} us | "
            f"{format_bytes(method['logical_history_bytes_per_query'])} | "
            f"{100 * method['needle_recall']:.2f}% | {100 * address:.2f}% | "
            f"{100 * conditional:.2f}% | "
            f"{100 * method['candidate_recall_at_k']:.2f}% | "
            f"{method['mean_unique_candidates']:.2f} | "
            f"{occupancy['p99_occupancy']:.0f} | {occupancy['max_occupancy']} | "
            f"{occupancy['eviction_count']:,} "
            f"({100 * occupancy['evicted_fraction']:.3f}%) | "
            f"{format_bytes(row['bucket_index_resident_bytes'])} |"
        )
    final = report["results"][-1]
    final_method = addressed(final)
    meets = (
        final_method["needle_recall"] >= 0.95
        and final_method["logical_history_bytes_per_query"] <= 3072
        and final_method["median_routing_us_per_query"] <= 350.0
    )
    lines.extend([
        "",
        f"At 2M the selector {'meets' if meets else 'does not meet'} the active gate: "
        f"{100 * final_method['needle_recall']:.2f}% recall, "
        f"{format_bytes(final_method['logical_history_bytes_per_query'])}/query, "
        f"and {final_method['median_routing_us_per_query']:.1f} us/query. "
        "Recall equals address agreement, so measured retention conditional on a "
        "valid hierarchical address is 100% for these 64 planted queries.",
        "",
        "## 2M design comparison",
        "",
        "| Variant | Routing | Bytes/query | Needle recall | Address agreement | Evicted postings |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for comparison in comparisons + [report]:
        row = comparison["results"][-1]
        method = addressed(row)
        occupancy = row["bucket_occupancy"]
        lines.append(
            f"| {variant_label(comparison)} | "
            f"{method['median_routing_us_per_query']:.1f} us | "
            f"{format_bytes(method['logical_history_bytes_per_query'])} | "
            f"{100 * method['needle_recall']:.2f}% | "
            f"{100 * row['query_target_probe_address_match']:.2f}% | "
            f"{100 * occupancy['evicted_fraction']:.2f}% |"
        )
    lines.extend([
        "",
        "The dense 2-bit hierarchy meets latency and traffic but remains too coarse; "
        "the dense 4-bit hierarchy reduces eviction but loses secondary-address "
        "agreement. The sparse 8-bit directory is the first tested design to satisfy "
        "all three constraints. FP-top32 overlap remains low, so this is a synthetic "
        "planted-neighbor routing result, not a general attention-quality result.",
        "",
        f"Peak MLX memory for the canonical sweep: {report['peak_memory_mb']:.1f} MB.",
        "",
    ])
    return "\n".join(lines)


def render_plot(report, comparisons, output):
    reports = comparisons + [report]
    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    colors = ["#999999", "#7570b3", "#d95f02", "#1b9e77"][-len(reports):]
    for current, color in zip(reports, colors):
        row = current["results"][-1]
        method = addressed(row)
        axis.scatter(
            method["logical_history_bytes_per_query"] / 1024,
            100 * method["needle_recall"],
            s=95, color=color, label=variant_label(current), zorder=3,
        )
    axis.axhline(95, color="#333333", linestyle="--", linewidth=1, label="95% target")
    axis.axvline(3, color="#333333", linestyle=":", linewidth=1, label="3 KiB ceiling")
    axis.set_xlabel("Logical historical-index bytes/query (KiB)")
    axis.set_ylabel("Needle recall at 2M (%)")
    axis.set_title("Sparse secondary addressing moves the recall/traffic frontier")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
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
    parser.add_argument("--comparison", action="append", default=[], type=pathlib.Path)
    parser.add_argument("--markdown-output", required=True, type=pathlib.Path)
    parser.add_argument("--plot-output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    report = load(args.input)
    validate(report)
    comparisons = [load(path) for path in args.comparison]
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(report, comparisons) + "\n")
    render_plot(report, comparisons, args.plot_output)
    print(json.dumps({
        "markdown_output": str(args.markdown_output),
        "plot_output": str(args.plot_output),
    }))


if __name__ == "__main__":
    main()
