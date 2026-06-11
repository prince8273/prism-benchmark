# PRISM: Probabilistic Reasoning under Implicit Selection Mechanisms

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A benchmark for evaluating whether large language models can reason about data-generating processes when selection mechanisms make observed samples unrepresentative of the target population.

Repository: [prince8273/prism-benchmark](https://github.com/prince8273/prism-benchmark)

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

| Model | Overall ACC | Naive Trap Rate | Simpson | Berkson | Truncated | Survival |
|-------|-------------|-----------------|---------|---------|-----------|----------|
| Claude Sonnet 4.5 | 76.1% | 0.4% | 97.7% | 86.2% | 57.4% | 58.3% |
| Claude Haiku 4.5 | 75.0% | 0.0% | 96.9% | 85.3% | 57.4% | 55.6% |
| Gemini 2.5 Flash | — | — | — | — | — | — |
| GPT-4o | — | — | — | — | — | — |
| GPT-4o mini | 65.9% | 9.6% | 82.0% | 76.7% | 49.1% | 51.9% |

**Key finding:** Both Claude models score near-identically on Simpson (97%+) and Berkson (85%+)
tasks but drop sharply on truncated sampling (57.4%) and survivorship bias (55–58%) tasks —
confirming that PRISM's hardest failure mode is numerical reasoning about selection mechanisms,
not pattern recognition of paradox names.

**Interpretation guide:** A model scoring 40–60% overall with a clearly non-trivial naive-trap
rate indicates the benchmark is probing the intended failure mode. If all strong models score
above 80%, tasks are too easy. If all score below 20%, inspect parsing before concluding
the benchmark is too hard.

---

## Evaluation Settings

For detailed parsing rules and API settings (temperature, rate limits, retries, and resume behavior), please see [`docs/eval_settings.md`](docs/eval_settings.md).

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
|   `-- tasks/                     # one JSON file per task (230 total)
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

## Quickstart & Reproducing Baselines

```bash
# Recommended Environment: Python 3.11, scipy 1.11+, anthropic 0.34+
git clone https://github.com/prince8273/prism-benchmark.git
cd prism-benchmark
pip install -r requirements.txt

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

PowerShell users can set keys like this, using the same variable names:
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
