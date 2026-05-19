"""Verify all task ground-truth snippets in data/tasks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def main() -> int:
    tasks_dir = Path("data/tasks")
    passed = 0
    failed = 0

    for fpath in sorted(tasks_dir.glob("*.json")):
        with open(fpath, encoding="utf-8") as handle:
            task = json.load(handle)

        code = task.get("python_verification", "").strip()
        if not code:
            print(f"SKIP {task['task_id']} (no python_verification)")
            continue

        namespace = {
            "__builtins__": __builtins__,
            "np": np,
            "norm": norm,
            "brentq": brentq,
        }

        try:
            exec(code, namespace, namespace)
            print(f"PASS {task['task_id']}")
            passed += 1
        except Exception as exc:
            print(f"FAIL {task['task_id']}: {exc}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
