"""Generate a small HTML leaderboard report from saved PRISM results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for fpath in sorted(results_dir.glob("*.json")):
        with open(fpath, encoding="utf-8") as handle:
            rows = json.load(handle)
        if not rows:
            continue
        model = rows[0]["model"]
        grouped.setdefault(model, []).extend(rows)
    return grouped


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    naive = sum(1 for row in rows if row.get("is_naive"))
    by_category: dict[str, list[float]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row["score"])
    return {
        "n_tasks": total,
        "overall_acc": (sum(row["score"] for row in rows) / total) if total else 0.0,
        "naive_trap_rate": (naive / total) if total else 0.0,
        "by_category": {key: sum(values) / len(values) for key, values in by_category.items()},
    }


def render_html(stats: dict[str, dict]) -> str:
    rows = []
    for model, summary in sorted(stats.items(), key=lambda item: item[1]["overall_acc"], reverse=True):
        by_category = summary["by_category"]
        rows.append(
            "<tr>"
            f"<td>{model}</td>"
            f"<td>{summary['overall_acc']:.1%}</td>"
            f"<td>{summary['naive_trap_rate']:.1%}</td>"
            f"<td>{by_category.get('simpson', 0.0):.1%}</td>"
            f"<td>{by_category.get('berkson', 0.0):.1%}</td>"
            f"<td>{by_category.get('truncated', 0.0):.1%}</td>"
            f"<td>{by_category.get('survival', 0.0):.1%}</td>"
            "</tr>"
        )

    table_rows = "\n".join(rows) if rows else "<tr><td colspan='7'>No results yet.</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PRISM Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>PRISM Leaderboard</h1>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Overall ACC</th>
        <th>Naive Trap Rate</th>
        <th>Simpson</th>
        <th>Berkson</th>
        <th>Truncated</th>
        <th>Survival</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an HTML PRISM report from results JSON files.")
    parser.add_argument("--results", default="results", help="Directory containing evaluation result JSON files")
    parser.add_argument("--output", default="results/report.html", help="HTML output path")
    args = parser.parse_args()

    grouped = load_results(Path(args.results))
    stats = {model: summarize(rows) for model, rows in grouped.items()}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(stats), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
