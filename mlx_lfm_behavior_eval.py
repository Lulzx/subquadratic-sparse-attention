"""Paired instruction and natural-language retrieval gate for sparse LFM2.5."""

import argparse
import json
import pathlib
import re
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

from mlx_donor_router import language_body
from mlx_lfm_multilayer_eval import parse_layers
from mlx_lfm_quality_eval import install_replacements


INSTRUCTION_CASES = (
    {
        "name": "literal_reply",
        "prompt": "Reply with exactly these two words and nothing else: COBALT WINDOW",
        "expected": "COBALT WINDOW",
    },
    {
        "name": "uppercase",
        "prompt": "Write the word telescope in uppercase. Reply with only the result.",
        "expected": "TELESCOPE",
    },
    {
        "name": "local_extraction",
        "prompt": (
            "The shipment label says destination=Reykjavik and priority=low. "
            "Reply with only the destination."
        ),
        "expected": "Reykjavik",
    },
    {
        "name": "reverse_digits",
        "prompt": "Reverse the digits 58314. Reply with digits only.",
        "expected": "41385",
    },
)


def parse_ints(spec):
    values = [int(value.strip()) for value in spec.split(",") if value.strip()]
    if not values or any(value < 64 for value in values):
        raise ValueError("lengths must be comma-separated integers of at least 64")
    return values


def set_sparse(replacements, enabled):
    for replacement in replacements.values():
        replacement.replacement_alpha = 1.0 if enabled else 0.0


def chat_prompt(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def greedy_generate(model, tokenizer, prompt, max_new_tokens):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = []
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        tokens = mx.array([prompt_ids + generated], dtype=mx.int32)
        next_token = mx.argmax(model(tokens)[:, -1], axis=-1)
        mx.eval(next_token)
        token_id = int(next_token.item())
        if token_id == eos_id:
            break
        generated.append(token_id)
    return tokenizer.decode(generated).strip()


def answer_metrics(model, tokenizer, prompt, expected):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(expected, add_special_tokens=False)
    tokens = mx.array([prompt_ids + answer_ids], dtype=mx.int32)
    logits = model(tokens[:, :-1]).astype(mx.float32)
    start = len(prompt_ids) - 1
    answer_logits = logits[:, start:start + len(answer_ids)]
    targets = tokens[:, len(prompt_ids):]
    loss = nn.losses.cross_entropy(answer_logits, targets, reduction="mean")
    top1 = mx.mean((mx.argmax(answer_logits, axis=-1) == targets).astype(mx.float32))
    mx.eval(loss, top1)
    return {
        "expected_answer_loss": float(loss),
        "expected_token_top1_fraction": float(top1),
        "expected_token_count": len(answer_ids),
    }


def contains_answer(output, expected):
    normalized = re.sub(r"\s+", " ", output).strip().upper()
    return expected.upper() in normalized


def filler_lines(count):
    return [
        f"Routine log {index:04d}: the storage monitor reports normal operation."
        for index in range(count)
    ]


def retrieval_case(tokenizer, task, target_length, position, value=None):
    value = value or {
        "exact": "VELA-7319",
        "lexical": "MINT-4826",
        "variable": "NORTH-EMBER-7319-QUARTZ",
    }[task]
    if task == "exact":
        prefix = "Reference records follow."
        needle = f"The passkey for Project Juniper is {value}."
        question = (
            "What is the passkey for Project Juniper? Reply with only the passkey."
        )
    elif task == "lexical":
        prefix = (
            "Terminology note: the blue records collection is commonly called the "
            "cobalt archive. Reference records follow."
        )
        needle = (
            f"Archive memorandum: the opening sequence assigned to the cobalt "
            f"archive is {value}."
        )
        question = (
            "What code unlocks the blue records collection? Reply with only the code."
        )
    else:
        prefix = "Reference records follow."
        needle = f"The recovery phrase for Station Alder is {value}."
        question = (
            "What is the complete recovery phrase for Station Alder? "
            "Reply with only the phrase."
        )

    lines = filler_lines(1)
    while True:
        insert_at = round(position * len(lines))
        context = "\n".join([prefix] + lines[:insert_at] + [needle] + lines[insert_at:])
        prompt = chat_prompt(tokenizer, context + "\n\nQuestion: " + question)
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
        if length >= target_length:
            break
        lines.extend(filler_lines(len(lines) + 1)[-1:])
    while len(lines) > 1 and length > target_length + 24:
        lines.pop()
        insert_at = round(position * len(lines))
        context = "\n".join([prefix] + lines[:insert_at] + [needle] + lines[insert_at:])
        prompt = chat_prompt(tokenizer, context + "\n\nQuestion: " + question)
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
    return {
        "task": task,
        "target_length": target_length,
        "actual_length": length,
        "position": position,
        "prompt": prompt,
        "needle": needle,
        "expected": value,
    }


def find_subsequence(sequence, subsequence):
    for start in range(len(sequence) - len(subsequence) + 1):
        if sequence[start:start + len(subsequence)] == subsequence:
            return start, start + len(subsequence)
    return None


def find_text_span(tokenizer, prompt_ids, text):
    for prefix in ("", " ", "\n"):
        text_ids = tokenizer.encode(prefix + text, add_special_tokens=False)
        span = find_subsequence(prompt_ids, text_ids)
        if span is not None:
            return span
    return None


def selector_recall(model, tokenizer, prompt, source_answer, replacements):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    span = find_text_span(tokenizer, prompt_ids, source_answer)
    if span is None:
        return {str(layer): None for layer in replacements}

    captured = {}
    originals = {}
    for layer, replacement in replacements.items():
        original = replacement.candidate_indices
        originals[layer] = original

        def capture(x, original=original, layer=layer):
            selected = original(x)
            captured[layer] = selected
            return selected

        replacement.candidate_indices = capture
    try:
        logits = model(mx.array([prompt_ids], dtype=mx.int32))
        mx.eval(logits, *captured.values())
    finally:
        for layer, replacement in replacements.items():
            replacement.candidate_indices = originals[layer]

    start, end = span
    result = {}
    for layer, selected in captured.items():
        indices = np.array(selected[0, -1])
        result[str(layer)] = {
            "any_source_answer_token": bool(
                np.any((indices >= start) & (indices < end))
            ),
            "source_answer_tokens_selected": int(
                len(set(indices[(indices >= start) & (indices < end)].tolist()))
            ),
            "source_answer_token_count": end - start,
            "candidate_count": int(np.sum(indices >= 0)),
        }
    return result


def paired_case(model, tokenizer, replacements, prompt, expected, max_new_tokens):
    outputs = {}
    metrics = {}
    for mode, sparse in (("dense", False), ("sparse", True)):
        set_sparse(replacements, sparse)
        output = greedy_generate(model, tokenizer, prompt, max_new_tokens)
        outputs[mode] = output
        metrics[mode] = answer_metrics(model, tokenizer, prompt, expected) | {
            "contains_expected": contains_answer(output, expected)
        }
    return {
        "expected": expected,
        "dense_output": outputs["dense"],
        "sparse_output": outputs["sparse"],
        "exact_output_match": outputs["dense"] == outputs["sparse"],
        "dense": metrics["dense"],
        "sparse": metrics["sparse"],
        "sparse_minus_dense_answer_loss": (
            metrics["sparse"]["expected_answer_loss"]
            - metrics["dense"]["expected_answer_loss"]
        ),
    }


def summarize(cases):
    dense_correct = [case["dense"]["contains_expected"] for case in cases]
    sparse_correct = [case["sparse"]["contains_expected"] for case in cases]
    eligible = [
        case["sparse"]["contains_expected"]
        for case in cases
        if case["dense"]["contains_expected"]
    ]
    return {
        "cases": len(cases),
        "dense_accuracy": float(np.mean(dense_correct)),
        "sparse_accuracy": float(np.mean(sparse_correct)),
        "dense_pass_cases_preserved": int(sum(eligible)),
        "dense_pass_cases": len(eligible),
        "dense_pass_preservation_rate": float(np.mean(eligible)) if eligible else None,
        "exact_output_match_rate": float(np.mean([
            case["exact_output_match"] for case in cases
        ])),
        "mean_sparse_minus_dense_answer_loss": float(np.mean([
            case["sparse_minus_dense_answer_loss"] for case in cases
        ])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--layers", default="12,14")
    parser.add_argument("--checkpoint-template", default=(
        "runs/lfm2.5-layer{layer}-joint-kl-12-14-seed{seed}.safetensors"
    ))
    parser.add_argument("--lengths", default="256,512,1024")
    parser.add_argument("--positions", default="0.1,0.5,0.9")
    parser.add_argument("--tasks", default="exact,lexical,variable")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--block-expansion", action="store_true")
    parser.add_argument("--span-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--memory-limit-mb", type=int, default=1792)
    parser.add_argument("--cache-limit-mb", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    layers = parse_layers(args.layers)
    lengths = parse_ints(args.lengths)
    positions = [float(value) for value in args.positions.split(",")]
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    if any(position < 0 or position > 1 for position in positions):
        parser.error("positions must be between zero and one")
    if any(task not in {"exact", "lexical", "variable"} for task in tasks):
        parser.error("tasks must contain exact, lexical, or variable")
    mx.set_memory_limit(args.memory_limit_mb * 2**20)
    mx.set_cache_limit(args.cache_limit_mb * 2**20)

    started = time.perf_counter()
    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    replacements = install_replacements(language_body(model), config, args, layers)
    for replacement in replacements.values():
        replacement.block_expansion = args.block_expansion
        replacement.span_size = 2 if args.block_expansion else args.span_size
        replacement.block_size = args.block_size

    instruction_results = []
    for case in INSTRUCTION_CASES:
        prompt = chat_prompt(tokenizer, case["prompt"])
        result = paired_case(
            model, tokenizer, replacements, prompt, case["expected"],
            args.max_new_tokens,
        )
        instruction_results.append({"name": case["name"]} | result)

    retrieval_results = []
    for task in tasks:
        for length in lengths:
            for position in positions:
                case = retrieval_case(tokenizer, task, length, position)
                paired = paired_case(
                    model, tokenizer, replacements, case["prompt"], case["expected"],
                    args.max_new_tokens,
                )
                set_sparse(replacements, True)
                recall = selector_recall(
                    model, tokenizer, case["prompt"], case["expected"], replacements
                )
                retrieval_results.append({
                    key: value for key, value in case.items() if key not in {"prompt", "needle"}
                } | paired | {"selector_recall": recall})
                print(json.dumps({
                    "task": task,
                    "length": case["actual_length"],
                    "position": position,
                    "dense": paired["dense"]["contains_expected"],
                    "sparse": paired["sparse"]["contains_expected"],
                }), flush=True)

    result = vars(args) | {
        "layers_resolved": layers,
        "instruction_summary": summarize(instruction_results),
        "retrieval_summary": summarize(retrieval_results),
        "instruction_cases": instruction_results,
        "retrieval_cases": retrieval_results,
        "peak_memory_mb": mx.get_peak_memory() / 2**20,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = pathlib.Path(
        args.output or f"runs/lfm2.5-behavior-12-14-seed{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
