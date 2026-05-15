# PRISM: Probabilistic Reasoning under Implicit Selection Mechanisms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A benchmark evaluating whether large language models can reason about data-generating processes when selection mechanisms make observed samples unrepresentative of the target population.

---

## Motivation

Language models perform reasonably on Bayes' theorem applied to stated probabilities, yet exhibit a systematic and explainable failure when probabilistic inference requires reasoning about *how the data was collected*. PRISM isolates this failure mode across four structured categories:

- **Simpson's paradox** — aggregate statistics reverse when conditioning on a confounder
- **Berkson's paradox** — conditioning on a collider induces spurious correlations
- **Truncated sampling** — inference when only part of the distribution is observable
- **Survivorship bias** — estimation when non-surviving units are systematically absent

In every task, the naive answer derived from reported statistics is numerically wrong — and the model must reason about the selection mechanism to recover the correct answer. The failure mode is not random noise; it is a consistent bias toward treating observed data as representative of the target distribution. PRISM makes this bias measurable.

---

## Task Format

Each task is a JSON object. Full schema documented in `docs/schema.md`.

```json
{
  "task_id": "prism_simp_001",
  "category": "simpson",
  "difficulty": "easy | medium | hard",
  "scenario": "...",
  "data": { "description": "...", "values": {} },
  "question": "...",
  "answer_type": "numerical | comparative | directional",
  "ground_truth": "...",
  "tolerance": 0.01,
  "naive_answer": "...",
  "naive_trap": "...",
  "reasoning_trace": "...",
  "python_verification": "..."
}
```

Every ground truth answer is independently verifiable with a `python_verification` field — a runnable Python snippet using `scipy.stats` that asserts the correct answer.

---

## Baseline Results

*Results will be updated as evaluations complete.*

| Model | Overall ACC | Naive Trap Rate | Simpson | Berkson | Truncated | Survival |
|-------|-------------|-----------------|---------|---------|-----------|----------|
| GPT-4o | — | — | — | — | — | — |
| Claude 3.5 Sonnet | — | — | — | — | — | — |
| Gemini 1.5 Pro | — | — | — | — | — | — |
| GPT-3.5 Turbo | — | — | — | — | — | — |

**Interpretation guide**: A model scoring 40–60% on overall ACC with a Naive Trap Rate above 30% indicates the benchmark is successfully probing the intended failure mode. If all models score above 80%, the tasks should be made harder. If all score below 20%, inspect whether prompting format is causing parsing failures before concluding the tasks are too difficult.

---

## Repo Structure

```
prism-benchmark/
├── README.md                  — this file
├── requirements.txt           — Python dependencies
├── evaluate.py                — evaluation harness (load, run, score, report)
├── data/
│   └── tasks/                 — one JSON file per task (40 total)
│       ├── prism_simp_001.json
│       ├── prism_simp_002.json
│       ├── prism_berk_001.json
│       ├── ...
├── results/                   — output directory for evaluation runs
│   └── .gitkeep
├── docs/
│   ├── schema.md              — full task JSON schema with field descriptions
│   ├── scoring.md             — rubric, partial credit rules, parsing details
│   └── task_authoring.md      — guide for writing new tasks (for contributors)
├── scripts/
│   ├── verify_tasks.py        — run all python_verification snippets to confirm GT
│   └── generate_report.py     — HTML report from results/ directory
└── notebooks/
    └── error_analysis.ipynb   — analysis of model failure modes by category
```

---

## Quickstart

```bash
git clone https://github.com/YOUR_HANDLE/prism-benchmark
cd prism-benchmark
pip install -r requirements.txt

# Verify all ground truth answers are correct
python scripts/verify_tasks.py

# Run evaluation on a model
export OPENAI_API_KEY=sk-...
python evaluate.py eval --model gpt-4o --tasks data/tasks/ --output results/

export ANTHROPIC_API_KEY=sk-ant-...
python evaluate.py eval --model claude-3-5-sonnet-20241022 --tasks data/tasks/ --output results/

# Generate leaderboard from all completed runs
python evaluate.py report --results results/
```

---

## How to Contribute

New tasks must satisfy all four criteria:

1. **Verifiable**: the `python_verification` field must be a runnable assert that confirms the ground truth using `scipy`.
2. **Novel domain**: don't duplicate the scenario of an existing task. Use a different field (law, ecology, economics, medicine, hiring, insurance).
3. **Explainable failure mode**: the `naive_trap` field must explain precisely why a model anchoring on surface statistics will fail, and which selection mechanism creates the distortion.
4. **Difficulty calibrated**: easy tasks should be solvable by a careful undergraduate. Hard tasks require setting up and numerically solving a truncated/censored model.

Submit a PR with your task JSON in `data/tasks/` and a line in `docs/CHANGELOG.md`.

---

## Citation

If you use PRISM in your research, please cite:

```bibtex
@misc{prism2025,
  title  = {PRISM: Probabilistic Reasoning under Implicit Selection Mechanisms},
  author = {YOUR NAME},
  year   = {2025},
  url    = {https://github.com/YOUR_HANDLE/prism-benchmark}
}
```

---

## License

MIT License. See `LICENSE`.
