"""
PRISM evaluation harness.

Usage:
    python evaluate.py eval --model gpt-4o --tasks data/tasks --output results
    python evaluate.py report --results results
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import anthropic
import openai
from google import genai as google_genai
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()

SYSTEM_PROMPT = """You are a statistical reasoning assistant.

You will be given a scenario involving probabilistic data and a question about
the target population. Be alert for hidden selection effects such as:
- Simpson's paradox / case-mix confounding
- collider conditioning / Berkson-style bias
- truncated or censored sampling
- survivorship filtering

Instructions:
- Reason carefully about how the data was collected.
- If the question asks for multiple quantities, provide each one on its own line.
- End your answer with clearly labeled final lines such as:
  ANSWER: 0.1365
  MU: 19.8
  FRACTION_EMPLOYED: 0.685
  RECOMMENDATION: Drug_A
- Do not refuse; give your best estimate."""

PHENOMENON_KEYWORDS = {
    "simpson": ["simpson", "confound", "stratif", "case mix"],
    "berkson": ["berkson", "collider", "selection bias", "dependent evidence"],
    "truncated": ["truncat", "inverse mills", "selection", "censor"],
    "survival": ["survivor", "survivorship", "selection bias", "truncat"],
}

IGNORE_GT_KEYS = {"explanation", "mechanism"}


def load_tasks(task_dir: str, category: Optional[str] = None, difficulty: Optional[str] = None) -> list[dict]:
    """Load task JSON files from a directory.

    Each file should contain one task object. For robustness, list-valued files
    are also accepted and flattened.
    """
    tasks: list[dict] = []
    task_path = Path(task_dir)
    for fpath in sorted(task_path.glob("*.json")):
        with open(fpath, encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        file_tasks = payload if isinstance(payload, list) else [payload]
        for task in file_tasks:
            if category and task.get("category") != category:
                continue
            if difficulty and task.get("difficulty") != difficulty:
                continue
            task["_source_file"] = str(fpath)
            tasks.append(task)
    console.print(f"[dim]Loaded {len(tasks)} tasks from {task_dir}[/dim]")
    return tasks


def load_existing_results(output_dir: str, model_name: str) -> dict[str, dict]:
    """Load the latest saved result for each task ID for a given model."""
    output_path = Path(output_dir)
    results_by_task_id: dict[str, dict] = {}
    safe_model = model_name.replace("/", "_").replace(":", "_")
    matched_files: list[Path] = []
    for pattern in (f"{safe_model}_*.json", f"checkpoint_{safe_model}_*.json"):
        matched_files.extend(output_path.glob(pattern))

    for fpath in sorted(set(matched_files), key=lambda path: path.name):
        try:
            with open(fpath, encoding="utf-8-sig") as handle:
                results = json.load(handle)
            for result in results:
                task_id = result.get("task_id")
                if task_id:
                    results_by_task_id[task_id] = result
        except Exception:
            continue

    return results_by_task_id


def load_completed_ids(output_dir: str, model_name: str) -> set[str]:
    """Load task IDs already completed in a previous partial run."""
    return set(load_existing_results(output_dir, model_name))


def build_prompt(task: dict) -> str:
    data_str = json.dumps(task["data"], indent=2)
    return f"""Scenario: {task['scenario']}

Data:
{data_str}

Question: {task['question']}

Please reason through this carefully, then provide your final answer with clear labels."""


def call_openai(model: str, prompt: str, max_tokens: int = 1024) -> str:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def call_anthropic(model: str, prompt: str, max_tokens: int = 1024) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def call_gemini(model: str, prompt: str) -> str:
    client = google_genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    last_exc: Optional[Exception] = None

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model,
                contents=SYSTEM_PROMPT + "\n\n" + prompt,
            )
            return response.text or ""
        except Exception as exc:
            last_exc = exc
            error_text = str(exc)
            transient = any(token in error_text for token in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"])
            if not transient or attempt == 4:
                raise

            backoff_seconds = min(30, 5 * (2**attempt))
            console.print(
                f"[yellow]Gemini transient error on attempt {attempt + 1}/5; retrying in {backoff_seconds}s...[/yellow]"
            )
            time.sleep(backoff_seconds)

    if last_exc:
        raise last_exc
    raise RuntimeError("Gemini request failed without an exception.")


def call_model(model_name: str, prompt: str) -> str:
    if model_name.startswith("gemini"):
        time.sleep(4)
    else:
        time.sleep(0.5)
    if model_name.startswith(("gpt", "o1", "o3")):
        return call_openai(model_name, prompt)
    if model_name.startswith("claude"):
        return call_anthropic(model_name, prompt)
    if model_name.startswith("gemini"):
        return call_gemini(model_name, prompt)
    raise ValueError(f"Unknown model provider for: {model_name}. Add routing in call_model().")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_all_numbers(text: str) -> list[float]:
    matches = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", text)
    values: list[float] = []
    for match in matches:
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def string_variants(value: str) -> set[str]:
    variants = {normalize_text(value)}
    variants.add(normalize_text(value.replace("_", " ")))
    variants.add(normalize_text(value.replace("_", "")))
    variants.add(normalize_text(value.replace("-", " ")))
    return {variant for variant in variants if variant}


def response_mentions_string(response: str, value: str) -> bool:
    response_norm = normalize_text(response)
    return any(variant in response_norm for variant in string_variants(value))


def response_mentions_bool(response: str, key: str, expected: bool) -> bool:
    response_norm = normalize_text(response)
    if "valid" in key:
        if expected:
            positive = ["valid", "appropriate", "sound", "not misleading"]
            return any(token in response_norm for token in positive)
        negative = ["invalid", "not valid", "misleading", "not appropriate", "not sound"]
        return any(token in response_norm for token in negative)
    if expected:
        return any(token in response_norm for token in ["yes", "true", "correct"])
    return any(token in response_norm for token in ["no", "false", "incorrect"])


def collect_expectations(task: dict) -> list[dict[str, Any]]:
    gt = task["ground_truth"]
    expectations: list[dict[str, Any]] = []

    if isinstance(gt, dict):
        for key, value in gt.items():
            if key in IGNORE_GT_KEYS:
                continue
            if isinstance(value, bool):
                expectations.append({"key": key, "type": "bool", "value": value})
            elif isinstance(value, (int, float)):
                expectations.append({"key": key, "type": "number", "value": float(value)})
            elif isinstance(value, str):
                expectations.append({"key": key, "type": "string", "value": value})
    elif isinstance(gt, bool):
        expectations.append({"key": "answer", "type": "bool", "value": gt})
    elif isinstance(gt, (int, float)):
        expectations.append({"key": "answer", "type": "number", "value": float(gt)})
    elif isinstance(gt, str):
        expectations.append({"key": "answer", "type": "string", "value": gt})

    return expectations


def expectation_matches(response: str, response_numbers: list[float], expectation: dict[str, Any], tolerance: float) -> bool:
    kind = expectation["type"]
    value = expectation["value"]
    key = expectation["key"]

    if kind == "number":
        return any(abs(number - value) <= tolerance for number in response_numbers)
    if kind == "string":
        return response_mentions_string(response, value)
    if kind == "bool":
        return response_mentions_bool(response, key, value)
    return False


def phenomenon_identified(task: dict, response: str) -> bool:
    response_norm = normalize_text(response)
    return any(keyword in response_norm for keyword in PHENOMENON_KEYWORDS.get(task.get("category", ""), []))


def detect_naive_answer(task: dict, response: str, matched_fraction: float) -> bool:
    if matched_fraction >= 1.0:
        return False

    response_norm = normalize_text(response)
    naive_text = task.get("naive_answer", "")
    if naive_text and normalize_text(naive_text) in response_norm:
        return True

    naive_numbers = extract_all_numbers(naive_text)
    response_numbers = extract_all_numbers(response)
    numeric_overlap = any(abs(actual - naive) <= task.get("tolerance", 0.02) for actual in response_numbers for naive in naive_numbers)

    if numeric_overlap and not phenomenon_identified(task, response):
        return True

    if task.get("category") == "simpson" and not phenomenon_identified(task, response):
        overall = task.get("data", {}).get("values", {}).get("overall", {})
        overall_rates = []
        for item in overall.values():
            if isinstance(item, dict) and "rate" in item:
                overall_rates.append(float(item["rate"]))
        if any(abs(actual - rate) <= task.get("tolerance", 0.02) for actual in response_numbers for rate in overall_rates):
            return True

    return False


def score_response(task: dict, response: str) -> dict:
    tolerance = float(task.get("tolerance", 0.02))
    expectations = collect_expectations(task)
    response_numbers = extract_all_numbers(response)

    matched_keys: list[str] = []
    missing_keys: list[str] = []
    for expectation in expectations:
        if expectation_matches(response, response_numbers, expectation, tolerance):
            matched_keys.append(expectation["key"])
        else:
            missing_keys.append(expectation["key"])

    total_expected = len(expectations)
    matched_fraction = (len(matched_keys) / total_expected) if total_expected else 0.0
    has_phenomenon = phenomenon_identified(task, response)

    score = 0.0
    note = ""
    if total_expected and len(matched_keys) == total_expected:
        score = 1.0
        note = "Matched all expected outputs"
    elif matched_keys:
        score = 0.5
        note = f"Matched {len(matched_keys)}/{total_expected} expected outputs"
    elif has_phenomenon:
        score = 0.5
        note = "Identified the right phenomenon but missed the required outputs"
    else:
        note = "Did not match the expected outputs"

    return {
        "task_id": task["task_id"],
        "score": score,
        "is_naive": detect_naive_answer(task, response, matched_fraction),
        "matched_keys": matched_keys,
        "missing_keys": missing_keys,
        "parsed_numbers": response_numbers,
        "note": note,
    }


def run_evaluation(model_name: str, tasks: list[dict], output_dir: str) -> list[dict]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing_results = load_existing_results(output_dir, model_name)
    completed_ids = set(existing_results)
    task_ids = {task["task_id"] for task in tasks}
    if completed_ids:
        console.print(f"[yellow]Resuming: {len(completed_ids)} tasks already done, skipping.[/yellow]")

    pending = [task for task in tasks if task["task_id"] not in completed_ids]
    console.print(f"[dim]Running {len(pending)} remaining tasks out of {len(tasks)} total.[/dim]")

    if not pending:
        console.print("[green]All tasks already completed.[/green]")
        return [existing_results[task_id] for task_id in sorted(task_ids) if task_id in existing_results]

    merged_results = {task_id: existing_results[task_id] for task_id in task_ids if task_id in existing_results}
    for task in track(pending, description=f"Evaluating {model_name}"):
        prompt = build_prompt(task)
        try:
            response = call_model(model_name, prompt)
        except Exception as exc:
            console.print(f"[red]Error on {task['task_id']}: {exc}[/red]")
            response = ""

        score_dict = score_response(task, response)
        score_dict["model"] = model_name
        score_dict["response"] = response
        score_dict["category"] = task["category"]
        score_dict["difficulty"] = task["difficulty"]
        score_dict["source_file"] = task.get("_source_file")
        merged_results[task["task_id"]] = score_dict

        newly_completed = len(merged_results) - len(existing_results)
        if newly_completed > 0 and newly_completed % 10 == 0:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model = model_name.replace("/", "_").replace(":", "_")
            checkpoint_file = output_path / f"checkpoint_{safe_model}_{ts}.json"
            with open(checkpoint_file, "w", encoding="utf-8") as handle:
                json.dump([merged_results[task_id] for task_id in sorted(merged_results)], handle, indent=2)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace(":", "_")
    out_file = output_path / f"{safe_model}_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as handle:
        json.dump([merged_results[task_id] for task_id in sorted(merged_results)], handle, indent=2)
    console.print(f"[green]Results saved to {out_file}[/green]")
    return [merged_results[task_id] for task_id in sorted(task_ids) if task_id in merged_results]


def compute_stats(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {}

    by_category: dict[str, list[float]] = {}
    by_difficulty: dict[str, list[float]] = {}
    naive_count = 0

    for result in results:
        category = result.get("category", "unknown")
        difficulty = result.get("difficulty", "unknown")
        by_category.setdefault(category, []).append(result["score"])
        by_difficulty.setdefault(difficulty, []).append(result["score"])
        if result.get("is_naive"):
            naive_count += 1

    return {
        "model": results[0]["model"],
        "n_tasks": total,
        "overall_acc": sum(result["score"] for result in results) / total,
        "naive_trap_rate": naive_count / total,
        "by_category": {key: sum(values) / len(values) for key, values in by_category.items()},
        "by_difficulty": {key: sum(values) / len(values) for key, values in by_difficulty.items()},
    }


def print_leaderboard(all_stats: list[dict]) -> None:
    if not all_stats:
        console.print("[yellow]No results to display.[/yellow]")
        return

    table = Table(title="PRISM Leaderboard", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Overall ACC", justify="center")
    table.add_column("Naive Trap Rate", justify="center")
    table.add_column("Simpson", justify="center")
    table.add_column("Berkson", justify="center")
    table.add_column("Truncated", justify="center")
    table.add_column("Survival", justify="center")
    table.add_column("Easy", justify="center")
    table.add_column("Medium", justify="center")
    table.add_column("Hard", justify="center")

    for stats in sorted(all_stats, key=lambda item: item["overall_acc"], reverse=True):
        by_category = stats.get("by_category", {})
        by_difficulty = stats.get("by_difficulty", {})
        table.add_row(
            stats["model"],
            f"{stats['overall_acc']:.1%}",
            f"{stats['naive_trap_rate']:.1%}",
            f"{by_category.get('simpson', math.nan):.1%}" if "simpson" in by_category else "-",
            f"{by_category.get('berkson', math.nan):.1%}" if "berkson" in by_category else "-",
            f"{by_category.get('truncated', math.nan):.1%}" if "truncated" in by_category else "-",
            f"{by_category.get('survival', math.nan):.1%}" if "survival" in by_category else "-",
            f"{by_difficulty.get('easy', math.nan):.1%}" if "easy" in by_difficulty else "-",
            f"{by_difficulty.get('medium', math.nan):.1%}" if "medium" in by_difficulty else "-",
            f"{by_difficulty.get('hard', math.nan):.1%}" if "hard" in by_difficulty else "-",
        )

    console.print(table)


def load_and_aggregate(results_dir: str) -> list[dict]:
    results_path = Path(results_dir)
    model_results: dict[str, list[dict]] = {}
    seen_task_ids: dict[str, set[str]] = {}

    for fpath in sorted(results_path.glob("*.json")):
        with open(fpath, encoding="utf-8-sig") as handle:
            results = json.load(handle)
        if not results:
            continue
        model = results[0]["model"]
        if model not in model_results:
            model_results[model] = []
            seen_task_ids[model] = set()

        for result in results:
            task_id = result.get("task_id")
            if task_id and task_id not in seen_task_ids[model]:
                model_results[model].append(result)
                seen_task_ids[model].add(task_id)
    return [compute_stats(values) for values in model_results.values()]


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM Benchmark Evaluation Harness")
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser("eval", help="Run evaluation on a model")
    eval_parser.add_argument("--model", required=True, help="Model ID")
    eval_parser.add_argument("--tasks", default="data/tasks", help="Task directory")
    eval_parser.add_argument("--output", default="results", help="Results directory")
    eval_parser.add_argument("--category", help="Filter by category")
    eval_parser.add_argument("--difficulty", help="Filter by difficulty")

    report_parser = subparsers.add_parser("report", help="Aggregate saved results")
    report_parser.add_argument("--results", default="results", help="Results directory")

    args = parser.parse_args()

    if args.command == "eval":
        tasks = load_tasks(args.tasks, category=args.category, difficulty=args.difficulty)
        if not tasks:
            console.print("[red]No tasks found. Check your --tasks path and filters.[/red]")
            return
        results = run_evaluation(args.model, tasks, args.output)
        print_leaderboard([compute_stats(results)])
        return

    if args.command == "report":
        print_leaderboard(load_and_aggregate(args.results))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
