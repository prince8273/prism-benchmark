"""
PRISM Evaluation Harness
Probabilistic Reasoning under Implicit Selection Mechanisms

Usage:
    python evaluate.py --model gpt-4o --tasks data/tasks/ --output results/
    python evaluate.py --model claude-3-5-sonnet-20241022 --tasks data/tasks/ --output results/
    python evaluate.py --report results/  # aggregate existing results into leaderboard
"""

import json
import os
import re
import time
import argparse
import datetime
from pathlib import Path
from typing import Optional

import anthropic
import openai
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

# ─────────────────────────────────────────────────────────────
# Task Loading
# ─────────────────────────────────────────────────────────────

def load_tasks(task_dir: str, category: Optional[str] = None, difficulty: Optional[str] = None) -> list[dict]:
    """Load all task JSON files from directory, with optional filters."""
    tasks = []
    task_path = Path(task_dir)
    for fpath in sorted(task_path.glob("*.json")):
        with open(fpath) as f:
            task = json.load(f)
        if category and task.get("category") != category:
            continue
        if difficulty and task.get("difficulty") != difficulty:
            continue
        tasks.append(task)
    console.print(f"[dim]Loaded {len(tasks)} tasks from {task_dir}[/dim]")
    return tasks


# ─────────────────────────────────────────────────────────────
# Prompt Construction
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a statistical reasoning assistant. You will be given a scenario involving
probabilistic data, and you must answer a precise quantitative question.

Important instructions:
- Reason step by step. Identify whether there is a data-generating process that might make the
  observed sample unrepresentative of the target population.
- Consider whether the data was collected under any selection mechanism (truncation, conditioning
  on a collider, case-mix confounding, or survivorship filtering).
- Give a final numerical answer in the format: ANSWER: [number]
  Or for comparative questions: RECOMMENDATION: [option]
- If you give multiple numerical values, label each one clearly before your ANSWER line.
- Do not refuse. Give your best estimate even under uncertainty."""

def build_prompt(task: dict) -> str:
    data_str = json.dumps(task["data"], indent=2)
    return f"""Scenario: {task['scenario']}

Data:
{data_str}

Question: {task['question']}

Please reason through this carefully, then provide your final answer."""


# ─────────────────────────────────────────────────────────────
# LLM Calls
# ─────────────────────────────────────────────────────────────

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
    return response.choices[0].message.content


def call_anthropic(model: str, prompt: str, max_tokens: int = 1024) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_model(model_name: str, prompt: str) -> str:
    """Route to the correct provider based on model name."""
    time.sleep(0.5)  # Basic rate limiting
    if model_name.startswith("gpt") or model_name.startswith("o1") or model_name.startswith("o3"):
        return call_openai(model_name, prompt)
    elif model_name.startswith("claude"):
        return call_anthropic(model_name, prompt)
    else:
        raise ValueError(f"Unknown model provider for: {model_name}. Add routing in call_model().")


# ─────────────────────────────────────────────────────────────
# Response Parsing
# ─────────────────────────────────────────────────────────────

def extract_numerical_answer(response: str) -> Optional[float]:
    """Extract the number after ANSWER:."""
    pattern = r"ANSWER:\s*([-+]?\d*\.?\d+)"
    matches = re.findall(pattern, response, re.IGNORECASE)
    if matches:
        try:
            return float(matches[-1])  # Take the last ANSWER if multiple
        except ValueError:
            return None
    # Fallback: look for the last standalone number in the response
    fallback = re.findall(r"(?<!\w)([-+]?\d+\.?\d*)(?!\w)", response)
    if fallback:
        try:
            return float(fallback[-1])
        except ValueError:
            return None
    return None


def extract_categorical_answer(response: str, options: list[str]) -> Optional[str]:
    """Extract categorical answer (RECOMMENDATION: X) or match from options."""
    rec_pattern = r"RECOMMENDATION:\s*(\w[\w\s]*?)(?:\n|$|\.)"
    matches = re.findall(rec_pattern, response, re.IGNORECASE)
    if matches:
        candidate = matches[-1].strip()
        for opt in options:
            if opt.lower() in candidate.lower():
                return opt
    # Fallback: find which option is mentioned last
    last_pos = -1
    best_opt = None
    for opt in options:
        pos = response.lower().rfind(opt.lower())
        if pos > last_pos:
            last_pos = pos
            best_opt = opt
    return best_opt


# ─────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────

def score_response(task: dict, response: str) -> dict:
    """
    Returns a scoring dict with keys:
        score: 0.0 | 0.5 | 1.0
        is_naive: bool
        parsed_answer: the extracted answer
        note: explanation of scoring decision
    """
    gt = task["ground_truth"]
    answer_type = task["answer_type"]
    tolerance = task.get("tolerance", 0.02)
    naive = task.get("naive_answer")

    result = {
        "task_id": task["task_id"],
        "score": 0.0,
        "is_naive": False,
        "parsed_answer": None,
        "note": "",
    }

    # ── Numerical answer ──────────────────────────────────────
    if answer_type == "numerical":
        parsed = extract_numerical_answer(response)
        result["parsed_answer"] = parsed
        if parsed is None:
            result["note"] = "Could not parse numerical answer"
            return result

        # Check exact match within tolerance
        if isinstance(gt, (int, float)):
            if abs(parsed - gt) <= tolerance:
                result["score"] = 1.0
                result["note"] = f"Correct: {parsed} within {tolerance} of {gt}"
            else:
                result["note"] = f"Wrong: got {parsed}, expected {gt}±{tolerance}"

        # Check if model gave the naive answer
        if isinstance(naive, (int, float)) and abs(parsed - naive) <= tolerance:
            result["is_naive"] = True

        # Partial credit: check if response mentions key phenomenon
        phenomenon_keywords = {
            "simpson": ["simpson", "confound", "stratif", "case mix"],
            "berkson": ["berkson", "collider", "selection bias", "conditional"],
            "truncated": ["truncat", "truncation", "selection", "censor"],
            "survival": ["survivor", "survivorship", "selection bias", "truncat"],
        }
        cat = task.get("category", "")
        keywords = phenomenon_keywords.get(cat, [])
        if result["score"] == 0.0 and any(kw in response.lower() for kw in keywords):
            result["score"] = 0.5
            result["note"] += " | Partial: correct phenomenon identified"

    # ── Comparative / categorical answer ─────────────────────
    elif answer_type in ("comparative", "categorical", "directional"):
        options = []
        if isinstance(gt, str):
            options = [gt]
        elif isinstance(gt, dict):
            options = [str(v) for v in gt.values() if isinstance(v, str)]

        # Expand with naive to detect naive trap
        if isinstance(naive, str):
            options.append(naive)

        parsed = extract_categorical_answer(response, list(set(options)))
        result["parsed_answer"] = parsed

        if parsed is None:
            result["note"] = "Could not parse categorical answer"
            return result

        correct_val = gt if isinstance(gt, str) else gt.get("recommendation", gt.get("association_type"))
        if parsed and correct_val and parsed.lower() == str(correct_val).lower():
            result["score"] = 1.0
            result["note"] = f"Correct: {parsed}"
        elif parsed and naive and parsed.lower() == str(naive).lower():
            result["is_naive"] = True
            result["note"] = f"Naive answer given: {parsed}"
        else:
            result["note"] = f"Wrong: got {parsed}, expected {correct_val}"

    return result


# ─────────────────────────────────────────────────────────────
# Evaluation Loop
# ─────────────────────────────────────────────────────────────

def run_evaluation(model_name: str, tasks: list[dict], output_dir: str) -> list[dict]:
    """Run all tasks against the model and save per-task results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    for task in track(tasks, description=f"Evaluating {model_name}"):
        prompt = build_prompt(task)
        try:
            response = call_model(model_name, prompt)
        except Exception as e:
            console.print(f"[red]Error on {task['task_id']}: {e}[/red]")
            response = ""

        score_dict = score_response(task, response)
        score_dict["model"] = model_name
        score_dict["response"] = response
        score_dict["category"] = task["category"]
        score_dict["difficulty"] = task["difficulty"]
        results.append(score_dict)

    # Save raw results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace(":", "_")
    out_file = output_path / f"{safe_model}_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]Results saved to {out_file}[/green]")
    return results


# ─────────────────────────────────────────────────────────────
# Leaderboard
# ─────────────────────────────────────────────────────────────

def compute_stats(results: list[dict]) -> dict:
    """Compute aggregate statistics for one model's results."""
    total = len(results)
    if total == 0:
        return {}

    by_category = {}
    by_difficulty = {}
    naive_count = 0

    for r in results:
        cat = r.get("category", "unknown")
        diff = r.get("difficulty", "unknown")
        by_category.setdefault(cat, []).append(r["score"])
        by_difficulty.setdefault(diff, []).append(r["score"])
        if r.get("is_naive"):
            naive_count += 1

    return {
        "model": results[0]["model"],
        "n_tasks": total,
        "overall_acc": sum(r["score"] for r in results) / total,
        "naive_trap_rate": naive_count / total,
        "by_category": {k: sum(v)/len(v) for k, v in by_category.items()},
        "by_difficulty": {k: sum(v)/len(v) for k, v in by_difficulty.items()},
    }


def print_leaderboard(all_stats: list[dict]):
    """Print a rich leaderboard table."""
    if not all_stats:
        console.print("[yellow]No results to display.[/yellow]")
        return

    all_stats_sorted = sorted(all_stats, key=lambda x: x["overall_acc"], reverse=True)

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

    for s in all_stats_sorted:
        bc = s.get("by_category", {})
        bd = s.get("by_difficulty", {})
        table.add_row(
            s["model"],
            f"{s['overall_acc']:.1%}",
            f"{s['naive_trap_rate']:.1%}",
            f"{bc.get('simpson', float('nan')):.1%}" if "simpson" in bc else "—",
            f"{bc.get('berkson', float('nan')):.1%}" if "berkson" in bc else "—",
            f"{bc.get('truncated', float('nan')):.1%}" if "truncated" in bc else "—",
            f"{bc.get('survival', float('nan')):.1%}" if "survival" in bc else "—",
            f"{bd.get('easy', float('nan')):.1%}" if "easy" in bd else "—",
            f"{bd.get('medium', float('nan')):.1%}" if "medium" in bd else "—",
            f"{bd.get('hard', float('nan')):.1%}" if "hard" in bd else "—",
        )

    console.print(table)


def load_and_aggregate(results_dir: str) -> list[dict]:
    """Load all result files and compute per-model stats for the leaderboard."""
    results_path = Path(results_dir)
    model_results = {}
    for fpath in results_path.glob("*.json"):
        with open(fpath) as f:
            results = json.load(f)
        if not results:
            continue
        model = results[0]["model"]
        if model not in model_results:
            model_results[model] = []
        model_results[model].extend(results)

    return [compute_stats(v) for v in model_results.values()]


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PRISM Benchmark Evaluation Harness")
    subparsers = parser.add_subparsers(dest="command")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("eval", help="Run evaluation on a model")
    eval_parser.add_argument("--model", required=True, help="Model ID (e.g. gpt-4o, claude-3-5-sonnet-20241022)")
    eval_parser.add_argument("--tasks", default="data/tasks", help="Directory containing task JSON files")
    eval_parser.add_argument("--output", default="results", help="Directory to save results")
    eval_parser.add_argument("--category", help="Filter by category (simpson|berkson|truncated|survival)")
    eval_parser.add_argument("--difficulty", help="Filter by difficulty (easy|medium|hard)")

    # Report subcommand
    report_parser = subparsers.add_parser("report", help="Generate leaderboard from results directory")
    report_parser.add_argument("--results", default="results", help="Directory containing result JSON files")

    args = parser.parse_args()

    if args.command == "eval":
        tasks = load_tasks(args.tasks, category=args.category, difficulty=args.difficulty)
        if not tasks:
            console.print("[red]No tasks found. Check your --tasks path and filters.[/red]")
            return
        results = run_evaluation(args.model, tasks, args.output)
        stats = compute_stats(results)
        print_leaderboard([stats])

    elif args.command == "report":
        all_stats = load_and_aggregate(args.results)
        print_leaderboard(all_stats)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
