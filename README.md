# PRISM: Probabilistic Reasoning under Implicit Selection Mechanisms

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A benchmark for evaluating whether large language models can reason about data-generating processes when selection mechanisms make observed samples unrepresentative of the target population.

GitHub repo: [prince8273/prism-benchmark](https://github.com/prince8273/prism-benchmark)

---

## Motivation

Language models often do reasonably well on Bayes' theorem when probabilities are stated explicitly, but they fail much more often when correct inference depends on understanding how the data was collected. PRISM isolates that failure mode across four structured categories:

- **Simpson's paradox** - aggregate statistics reverse when conditioning on a confounder
- **Berkson's paradox** - conditioning on a collider induces spurious correlations
- **Truncated sampling** - inference when only part of a distribution is observable
- **Survivorship bias** - estimation when non-surviving units are systematically absent

In every task, the naive answer derived from the observed statistics is wrong or incomplete, and the model must reason about the selection mechanism to recover the correct answer. PRISM is designed to make that bias measurable: models that fail here will give confidently wrong answers in domains where data collection is selective: medical studies, hiring audits, financial risk models, and legal evidence.

---

## Task Format

Each task is a JSON object. The full schema is documented in `docs/schema.md`.

```json
{
  "task_id": "prism_simp_001",
  "category": "simpson",
  "difficulty": "easy | medium | hard",
  "scenario": "...",
  "data": {
    "description": "...",
    "values": {}
  },
  "question": "...",
  "answer_type": "numerical | comparative | directional",
  "ground_truth": {},
  "tolerance": 0.01,
  "naive_answer": "...",
  "naive_trap": "...",
  "reasoning_trace": "...",
  "python_verification": "..."
}
```

Every ground-truth answer is independently verifiable with a `python_verification` snippet.

---

## Baseline Results

Current best documented run: `claude-sonnet-4-5` with `76.1%` overall accuracy.

Raw result files:
- [claude-haiku-4-5-20251001_20260520_181547.json](https://github.com/prince8273/prism-benchmark/blob/main/results/claude-haiku-4-5-20251001_20260520_181547.json)
- [claude-sonnet-4-5_20260520_185050.json](https://github.com/prince8273/prism-benchmark/blob/main/results/claude-sonnet-4-5_20260520_185050.json)
- [gpt-4o-mini_20260527_003842.json](https://github.com/prince8273/prism-benchmark/blob/main/results/gpt-4o-mini_20260527_003842.json)

| Model | Overall ACC | Naive Trap Rate | Simpson | Berkson | Truncated | Survival |
|-------|-------------|-----------------|---------|---------|-----------|----------|
| claude-sonnet-4-5 | 76.1% | 0.4% | 97.7% | 86.2% | 57.4% | 58.3% |
| claude-haiku-4-5-20251001 | 75.0% | 0.0% | 96.9% | 85.3% | 57.4% | 55.6% |
| gpt-4o-mini | 65.9% | 9.6% | 82.0% | 76.7% | 49.1% | 51.9% |
| gemini-2.5-pro | - | - | - | - | - | - |
| gemini-2.5-flash | - | - | - | - | - | - |

Interpretation guide: if flagship models land around 40-60% overall ACC with a clearly non-trivial naive-trap rate, the benchmark is likely probing the intended failure mode. If all strong models score above 80%, the tasks are probably too easy. If all models score below 20%, inspect parsing and prompting before concluding the benchmark is too hard.

---

## Reproduce These Exact Results

This section differs from **Quickstart** only in the exact baseline inputs used to generate the published results:

- `git checkout 97d245c`
- Model IDs: `claude-haiku-4-5-20251001`, `claude-sonnet-4-5`, `gpt-4o-mini`
- Expected result files:
  - `results/claude-haiku-4-5-20251001_20260520_181547.json`
  - `results/claude-sonnet-4-5_20260520_185050.json`
  - `results/gpt-4o-mini_20260527_003842.json`

Use the **Quickstart** commands for the full evaluation flow; swap in those model IDs and the commit above when you need to reproduce the documented numbers exactly.

---

## Evaluation Settings (Affect Scores)

The current harness behavior in `evaluate.py` uses the following settings:

- **Prompting**:
  - Fixed `SYSTEM_PROMPT` (selection-bias-focused instructions) plus per-task scenario/data/question prompt.
- **Temperature**:
  - OpenAI calls use `temperature=0.0`.
  - Anthropic and Gemini calls do not set temperature explicitly (provider defaults apply).
- **Max tokens**:
  - OpenAI: `max_tokens=1024`
  - Anthropic: `max_tokens=1024`
  - Gemini: no explicit max token parameter in the current call.
- **Seed / deterministic sampling**:
  - No seed parameter is set for any provider in the current harness.
- **Retries / backoff**:
  - Gemini only: up to 5 attempts on transient errors (`503`, `UNAVAILABLE`, `429`, `RESOURCE_EXHAUSTED`) with exponential backoff (`5s`, `10s`, `20s`, `30s`, `30s` max).
  - OpenAI/Anthropic: no custom retry loop in this script.
- **Rate pacing**:
  - `0.5s` sleep before OpenAI/Anthropic requests.
  - `4s` sleep before Gemini requests.
- **Parsing & scoring rules**:
  - Numeric answers: match if any extracted response number is within per-task `tolerance` (default `0.02`).
  - String answers: case/whitespace-normalized substring matching with underscore/hyphen variants.
  - Boolean answers: keyword heuristics (e.g., yes/true/correct; valid/invalid patterns).
  - Keys `explanation` and `mechanism` in `ground_truth` are ignored for matching.
  - Score is `1.0` (all matched), `0.5` (partial match), `0.5` (phenomenon identified but outputs missed), else `0.0`.
  - `naive_trap_rate` is computed via naive-answer detection heuristics in `detect_naive_answer`.
- **Resume behavior**:
  - `eval` resumes by skipping `task_id`s already present in prior `results/{model}_*.json` files.
  - Checkpoint JSON is written every 10 tasks; final JSON is written at end.
- **Report aggregation**:
  - `report` de-duplicates by `(model, task_id)` and keeps first-seen task result per model across JSON files.

---

## Task Inventory

| Category | Count | Difficulty split |
|----------|-------|------------------|
| Simpson's paradox | 64 | 2 easy / 6 medium / 56 hard |
| Berkson's paradox | 58 | 0 easy / 7 medium / 51 hard |
| Truncated sampling | 54 | 0 easy / 12 medium / 42 hard |
| Survivorship bias | 54 | 0 easy / 7 medium / 47 hard |
| **Total** | **230** | |

---

## Repo Structure

```text
prism-benchmark/
|-- README.md
|-- requirements.txt
|-- evaluate.py
|-- data/
|   `-- tasks/                     # 230 total task files
|       |-- prism_simp_001.json
|       |-- prism_berk_002.json
|       |-- prism_trunc_001.json
|       |-- prism_surv_002.json
|       |-- prism_simp_181.json
|       `-- prism_surv_230.json
|-- docs/
|   |-- schema.md
|   |-- scoring.md
|   |-- eval_settings.md
|   |-- task_authoring.md
|   `-- CHANGELOG.md
|-- scripts/
|   |-- verify_tasks.py
|   `-- generate_report.py
|-- notebooks/
|   `-- .gitkeep
`-- results/
    |-- README.md
    |-- claude-haiku-4-5-20251001_20260520_181547.json
    |-- claude-sonnet-4-5_20260520_185050.json
    `-- gpt-4o-mini_20260527_003842.json
```

---

## Quickstart

```bash
git clone https://github.com/prince8273/prism-benchmark.git
cd prism-benchmark
pip install -r requirements.txt

# Verify all ground truth answers are correct
python scripts/verify_tasks.py

# Run evaluation on an OpenAI model
export OPENAI_API_KEY=sk-...
python evaluate.py eval --model gpt-4o --tasks data/tasks --output results

# Run evaluation on an Anthropic model
export ANTHROPIC_API_KEY=sk-ant-...
python evaluate.py eval --model claude-3-5-sonnet-20241022 --tasks data/tasks --output results

# Aggregate saved runs into a leaderboard
python evaluate.py report --results results

# Optional: generate a small HTML report
python scripts/generate_report.py --results results --output results/report.html
```

PowerShell users can set keys like this:
`$env:OPENAI_API_KEY="sk-..."` and `$env:ANTHROPIC_API_KEY="sk-ant-..."`.

---

## How To Contribute

New tasks should satisfy all of the following:

1. **Verifiable**: `python_verification` must assert the ground truth with runnable Python.
2. **Novel domain**: avoid duplicating the exact same scenario with renamed entities.
3. **Explainable failure mode**: `naive_trap` should clearly describe why the shortcut answer fails.
4. **Calibrated difficulty**: easy tasks should be solvable carefully by a strong undergraduate; hard tasks can require numerical solving or more subtle dependence reasoning.

When contributing, add the task JSON to `data/tasks/` and note the change in `docs/CHANGELOG.md`.

---

## Citation

If you use PRISM in your research, please cite:

```bibtex
@misc{prism2026,
  title  = {PRISM: Probabilistic Reasoning under Implicit Selection Mechanisms},
  author = {PRINCE KUMAR},
  year   = {2026},
  url    = {https://github.com/prince8273/prism-benchmark}
}
```

---

## License

Code (`evaluate.py`, `scripts/`, `requirements.txt`) is licensed under
the Apache License 2.0. See [LICENSE](LICENSE).

Benchmark tasks and data (`data/tasks/`) are licensed under
Creative Commons Attribution 4.0 International (CC BY 4.0).
See [LICENSE-DATA](LICENSE-DATA).

Copyright (c) 2026 PRINCE KUMAR
