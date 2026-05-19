# PRISM: Probabilistic Reasoning under Implicit Selection Mechanisms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
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

In every task, the naive answer derived from the observed statistics is wrong or incomplete, and the model must reason about the selection mechanism to recover the correct answer. PRISM is designed to make that bias measurable.

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

_Results will be updated as evaluations complete._

| Model | Overall ACC | Naive Trap Rate | Simpson | Berkson | Truncated | Survival |
|-------|-------------|-----------------|---------|---------|-----------|----------|
| GPT-4o | - | - | - | - | - | - |
| Claude 3.5 Sonnet | - | - | - | - | - | - |
| Gemini 1.5 Pro | - | - | - | - | - | - |
| GPT-3.5 Turbo | - | - | - | - | - | - |

Interpretation guide: if flagship models land around 40-60% overall ACC with a clearly non-trivial naive-trap rate, the benchmark is likely probing the intended failure mode. If all strong models score above 80%, the tasks are probably too easy. If all models score below 20%, inspect parsing and prompting before concluding the benchmark is too hard.

---

## Repo Structure

```text
prism-benchmark/
|-- README.md
|-- requirements.txt
|-- evaluate.py
|-- data/
|   `-- tasks/
|       |-- prism_simp_001.json
|       |-- prism_simp_002.json
|       |-- prism_berk_002.json
|       |-- prism_trunc_001.json
|       `-- prism_surv_002.json
|-- docs/
|   |-- schema.md
|   |-- scoring.md
|   |-- task_authoring.md
|   `-- CHANGELOG.md
|-- scripts/
|   |-- verify_tasks.py
|   `-- generate_report.py
|-- notebooks/
|   `-- .gitkeep
`-- results/
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

MIT License. See `LICENSE`.
