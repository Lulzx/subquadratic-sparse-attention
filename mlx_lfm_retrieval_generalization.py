"""Deterministic, resumable LFM2.5 retrieval-generalization evaluation."""

import argparse
import hashlib
import json
import math
import pathlib
import re
import time

import mlx.core as mx
from mlx_lm import load

from mlx_donor_router import language_body
from mlx_lfm_behavior_eval import answer_metrics, contains_answer, greedy_generate
from mlx_lfm_multilayer_eval import parse_layers
from mlx_lfm_quality_eval import install_replacements


MANIFEST_VERSION = 1
DEFAULT_LENGTHS = (1024, 4096, 8192, 16384)
DEFAULT_POSITIONS = (0.1, 0.5, 0.9)

TASK_VALUES = {
    "exact": (
        "LYRA-6047",
        "FALCON-3185",
        "CEDAR-9724",
    ),
    "lexical": (
        "IVORY-5268",
        "MARBLE-4073",
        "TOPAZ-8614",
    ),
    "multi_token": (
        "EAST-WILLOW-6047-AURORA",
        "SILVER-HARBOR-3185-COMET",
        "QUIET-MAPLE-9724-ORBIT",
    ),
    "niah": (
        "739184",
        "260573",
        "814926",
    ),
}

TASK_TEMPLATES = {
    "exact": (
        {
            "id": "project_passkey",
            "prefix": "Reference records follow.",
            "needle": "The passkey for Project Juniper is {value}.",
            "question": (
                "What is the passkey for Project Juniper? Reply with only the passkey."
            ),
        },
        {
            "id": "shipment_access",
            "prefix": "Archived shipment notices follow.",
            "needle": "Shipment Kestrel uses access code {value}.",
            "question": (
                "Give the access code assigned to Shipment Kestrel. Reply with only "
                "the code."
            ),
        },
    ),
    "lexical": (
        {
            "id": "archive_alias",
            "prefix": (
                "Terminology note: staff call the cobalt archive the blue records "
                "collection. Reference records follow."
            ),
            "needle": "The opening sequence for the cobalt archive is {value}.",
            "question": (
                "What code unlocks the blue records collection? Reply with only the code."
            ),
        },
        {
            "id": "facility_role",
            "prefix": (
                "Terminology note: the night-observation facility is the observatory. "
                "Facility notices follow."
            ),
            "needle": "The observatory authorization code is {value}.",
            "question": (
                "What is the authorization code for the facility used to observe the "
                "night sky? Reply with only the code."
            ),
        },
    ),
    "multi_token": (
        {
            "id": "station_phrase",
            "prefix": "Emergency recovery records follow.",
            "needle": "The complete recovery phrase for Station Alder is {value}.",
            "question": (
                "What is the complete recovery phrase for Station Alder? Reply with "
                "only the phrase."
            ),
        },
        {
            "id": "relay_phrase",
            "prefix": "Relay restoration records follow.",
            "needle": "Relay Meridian has restoration phrase {value}.",
            "question": (
                "Give the entire restoration phrase for Relay Meridian. Reply with "
                "only the phrase."
            ),
        },
    ),
    "niah": (
        {
            "id": "magic_number",
            "prefix": "A long collection of unrelated operational notes follows.",
            "needle": "The special magic number is {value}.",
            "question": "What is the special magic number? Reply with digits only.",
        },
        {
            "id": "needle_serial",
            "prefix": "A long collection of unrelated maintenance notes follows.",
            "needle": "The hidden needle serial is {value}.",
            "question": "What is the hidden needle serial? Reply with digits only.",
        },
    ),
}


def parse_csv(spec, cast=str):
    values = [cast(value.strip()) for value in spec.split(",") if value.strip()]
    if not values:
        raise ValueError("comma-separated value list must not be empty")
    return values


def distant_candidate_budget(tables, probes, members, span_size=1, block_size=0):
    if min(tables, probes, members, span_size) < 1 or block_size < 0:
        raise ValueError("candidate dimensions must be positive")
    if block_size and span_size != 1:
        raise ValueError("span and completed-block expansion are mutually exclusive")
    return tables * probes * members * (block_size or span_size)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def chat_prompt(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def filler_lines(count):
    topics = ("storage", "weather", "inventory", "routing", "maintenance")
    return [
        f"Routine note {index:05d}: {topics[index % len(topics)]} status is nominal."
        for index in range(count)
    ]


def render_prompt(tokenizer, template, value, position, filler_count):
    lines = filler_lines(filler_count)
    insert_at = round(position * len(lines))
    needle = template["needle"].format(value=value)
    context = "\n".join(
        [template["prefix"]] + lines[:insert_at] + [needle] + lines[insert_at:]
    )
    prompt = chat_prompt(tokenizer, context + "\n\nQuestion: " + template["question"])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return prompt, prompt_ids, needle


def find_subsequence(sequence, subsequence):
    for start in range(len(sequence) - len(subsequence) + 1):
        if sequence[start:start + len(subsequence)] == subsequence:
            return start, start + len(subsequence)
    return None


def find_text_span(tokenizer, prompt_ids, text):
    for prefix in ("", " ", "\n"):
        span = find_subsequence(
            prompt_ids,
            tokenizer.encode(prefix + text, add_special_tokens=False),
        )
        if span is not None:
            return span
    return None


def build_case(tokenizer, task, value_id, value, template, target_length, position):
    low = 0
    high = max(1, target_length // 4)
    while True:
        _, prompt_ids, _ = render_prompt(
            tokenizer, template, value, position, high
        )
        if len(prompt_ids) >= target_length:
            break
        high *= 2
    while low < high:
        middle = (low + high) // 2
        _, prompt_ids, _ = render_prompt(
            tokenizer, template, value, position, middle
        )
        if len(prompt_ids) < target_length:
            low = middle + 1
        else:
            high = middle
    prompt, prompt_ids, needle = render_prompt(
        tokenizer, template, value, position, low
    )
    span = find_text_span(tokenizer, prompt_ids, value)
    if span is None:
        raise RuntimeError(f"could not locate value tokens for {task}/{value_id}")
    source_start, source_end = span
    query_position = len(prompt_ids) - 1
    case_id = (
        f"{task}:{value_id}:{template['id']}:"
        f"n{target_length}:p{position:.2f}"
    )
    return {
        "case_id": case_id,
        "task": task,
        "value_id": value_id,
        "template": template["id"],
        "target_length": target_length,
        "actual_length": len(prompt_ids),
        "position": position,
        "source_start": source_start,
        "source_end": source_end,
        "query_position": query_position,
        "retrieval_distance": query_position - source_start,
        "prompt": prompt,
        "needle": needle,
        "expected": value,
    }


def build_manifest(tokenizer, model_name, lengths, positions):
    cases = []
    for task, values in TASK_VALUES.items():
        for value_index, value in enumerate(values):
            for template in TASK_TEMPLATES[task]:
                for target_length in lengths:
                    for position in positions:
                        cases.append(build_case(
                            tokenizer,
                            task,
                            f"v{value_index}",
                            value,
                            template,
                            target_length,
                            position,
                        ))
    manifest = {
        "version": MANIFEST_VERSION,
        "model": model_name,
        "lengths": list(lengths),
        "positions": list(positions),
        "tasks": list(TASK_VALUES),
        "values_per_task": len(next(iter(TASK_VALUES.values()))),
        "templates_per_task": len(next(iter(TASK_TEMPLATES.values()))),
        "cases": cases,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def load_manifest(path):
    manifest = json.loads(path.read_text())
    expected_hash = manifest.pop("manifest_sha256", None)
    actual_hash = sha256_json(manifest)
    manifest["manifest_sha256"] = expected_hash
    if expected_hash != actual_hash:
        raise ValueError("manifest SHA-256 does not match its contents")
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("unsupported manifest version")
    case_ids = [case["case_id"] for case in manifest["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("manifest contains duplicate case IDs")
    return manifest


def selected_case_metadata(case):
    return {
        key: value
        for key, value in case.items()
        if key not in ("prompt", "needle")
    }


def case_result(model, tokenizer, case, max_new_tokens):
    mx.reset_peak_memory()
    started = time.perf_counter()
    output = greedy_generate(model, tokenizer, case["prompt"], max_new_tokens)
    metrics = answer_metrics(
        model, tokenizer, case["prompt"], case["expected"]
    )
    result = selected_case_metadata(case) | {
        "output": output,
        "contains_expected": contains_answer(output, case["expected"]),
        "metrics": metrics,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
        "elapsed_seconds": time.perf_counter() - started,
    }
    mx.clear_cache()
    return result


def output_skeleton(args, manifest, candidate_budget):
    return {
        "format_version": 1,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "variant": args.variant,
        "mode": args.mode,
        "model": args.model,
        "seed": None if args.mode == "dense" else args.seed,
        "layers": args.layers,
        "checkpoint_template": (
            None if args.mode == "dense" else args.checkpoint_template
        ),
        "tables": None if args.mode == "dense" else args.tables,
        "bits": None if args.mode == "dense" else args.bits,
        "window": None if args.mode == "dense" else args.window,
        "sink_tokens": None if args.mode == "dense" else args.sink_tokens,
        "members": None if args.mode == "dense" else args.members,
        "probes": None if args.mode == "dense" else args.probes,
        "member_policy": None if args.mode == "dense" else args.member_policy,
        "history_fraction": (
            None if args.mode == "dense" else args.history_fraction
        ),
        "span_size": None if args.mode == "dense" else args.span_size,
        "block_size": None if args.mode == "dense" else args.block_size,
        "distant_candidate_budget": candidate_budget,
        "memory_limit_mb": args.memory_limit_mb,
        "cache_limit_mb": args.cache_limit_mb,
        "max_new_tokens": args.max_new_tokens,
        "results": [],
    }


def validate_resume(report, expected):
    keys = (
        "manifest_sha256",
        "variant",
        "mode",
        "seed",
        "checkpoint_template",
        "window",
        "sink_tokens",
        "members",
        "probes",
        "member_policy",
        "history_fraction",
        "span_size",
        "block_size",
        "distant_candidate_budget",
        "max_new_tokens",
    )
    mismatches = [key for key in keys if report.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError("resume configuration mismatch: " + ", ".join(mismatches))


def write_report(path, report, started):
    report["completed_cases"] = len(report["results"])
    report["peak_memory_mb"] = max(
        (row["peak_memory_mb"] for row in report["results"]), default=0.0
    )
    report["elapsed_seconds"] = report.get("elapsed_seconds", 0.0) + (
        time.perf_counter() - started
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--make-manifest", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument(
        "--lengths", default=",".join(map(str, DEFAULT_LENGTHS))
    )
    parser.add_argument(
        "--positions", default=",".join(map(str, DEFAULT_POSITIONS))
    )
    parser.add_argument("--variant", default="")
    parser.add_argument("--mode", choices=("dense", "sparse"))
    parser.add_argument("--layers", default="12,14")
    parser.add_argument("--checkpoint-template", default="")
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument(
        "--member-policy", choices=("recent", "hybrid"), default="recent"
    )
    parser.add_argument("--history-fraction", type=float, default=0.5)
    parser.add_argument("--span-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--memory-limit-mb", type=int, default=1792)
    parser.add_argument("--cache-limit-mb", type=int, default=128)
    parser.add_argument("--only-lengths", default="")
    parser.add_argument("--only-tasks", default="")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.make_manifest:
        lengths = parse_csv(args.lengths, int)
        positions = parse_csv(args.positions, float)
        if any(length < 256 for length in lengths):
            parser.error("manifest lengths must be at least 256")
        if any(position <= 0.0 or position >= 1.0 for position in positions):
            parser.error("manifest positions must be strictly between zero and one")
        _, tokenizer = load(args.model, lazy=True)
        manifest = build_manifest(tokenizer, args.model, lengths, positions)
        args.make_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.make_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({
            "manifest": str(args.make_manifest),
            "manifest_sha256": manifest["manifest_sha256"],
            "cases": len(manifest["cases"]),
        }, indent=2))
        return

    if not args.manifest or not args.output or not args.variant or not args.mode:
        parser.error(
            "evaluation requires --manifest, --output, --variant, and --mode"
        )
    if args.mode == "sparse" and not args.checkpoint_template:
        parser.error("sparse evaluation requires --checkpoint-template")
    if args.mode == "dense" and args.variant != "dense":
        parser.error("dense evaluation requires --variant dense")
    if args.span_size < 1:
        parser.error("--span-size must be positive")
    if args.block_size and args.span_size != 1:
        parser.error("--span-size cannot be combined with --block-size")
    manifest = load_manifest(args.manifest)
    if manifest["model"] != args.model:
        parser.error("manifest model does not match --model")

    only_lengths = set(parse_csv(args.only_lengths, int)) if args.only_lengths else None
    only_tasks = set(parse_csv(args.only_tasks)) if args.only_tasks else None
    cases = [
        case for case in manifest["cases"]
        if (only_lengths is None or case["target_length"] in only_lengths)
        and (only_tasks is None or case["task"] in only_tasks)
    ]
    if args.case_limit:
        cases = cases[:args.case_limit]

    candidate_budget = None
    if args.mode == "sparse":
        candidate_budget = distant_candidate_budget(
            args.tables,
            args.probes,
            args.members,
            span_size=args.span_size,
            block_size=args.block_size,
        )
    expected = output_skeleton(args, manifest, candidate_budget)
    if args.resume and args.output.is_file():
        report = json.loads(args.output.read_text())
        validate_resume(report, expected)
    else:
        report = expected
    completed = {row["case_id"] for row in report["results"]}

    mx.set_memory_limit(args.memory_limit_mb * 2**20)
    mx.set_cache_limit(args.cache_limit_mb * 2**20)
    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    if args.mode == "sparse":
        layers = parse_layers(args.layers)
        replacements = install_replacements(
            language_body(model), config, args, layers
        )
        for replacement in replacements.values():
            replacement.replacement_alpha = 1.0
            replacement.span_size = args.span_size
            replacement.block_size = args.block_size

    started = time.perf_counter()
    for case in cases:
        if case["case_id"] in completed:
            continue
        result = case_result(
            model, tokenizer, case, args.max_new_tokens
        )
        report["results"].append(result)
        completed.add(case["case_id"])
        write_report(args.output, report, started)
        started = time.perf_counter()
        print(json.dumps({
            "variant": args.variant,
            "seed": report["seed"],
            "case_id": case["case_id"],
            "correct": result["contains_expected"],
            "peak_memory_mb": round(result["peak_memory_mb"], 2),
            "elapsed_seconds": round(result["elapsed_seconds"], 2),
        }), flush=True)

    write_report(args.output, report, started)


if __name__ == "__main__":
    main()
