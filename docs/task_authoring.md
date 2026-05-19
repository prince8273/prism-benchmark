# Authoring PRISM Tasks

Use this checklist when adding new tasks.

## 1. Pick a clear selection mechanism

Every task should revolve around one of:

- Simpson's paradox / confounding by subgroup mix
- collider conditioning / shared-selection bias
- truncation or censoring
- survivorship filtering

## 2. Choose a realistic domain

Good domains:

- medicine
- education
- hiring
- ecology
- finance
- insurance
- operations

Avoid repeating the same scenario with only renamed entities.

## 3. Make the naive answer tempting

The task should have an obvious but wrong shortcut, for example:

- comparing aggregate rates directly
- treating observed means as population means
- assuming independent evidence when there is shared noise

Document that shortcut in `naive_answer` and `naive_trap`.

## 4. Keep the answer verifiable

Every task must include a short `python_verification` snippet that:

- runs without external files
- uses `scipy`, `numpy`, or basic Python only
- asserts the stored ground truth

If you cannot verify the answer programmatically, the task is not ready.

## 5. Write maintainable ground truth

- Store required outputs in `ground_truth`.
- Put prose explanations in `ground_truth.explanation`.
- Use stable key names so evaluation remains simple.

## 6. Difficulty calibration

- `easy`: Careful undergrad-level reasoning.
- `medium`: Multi-step reasoning or non-obvious correction.
- `hard`: Numerical solving, subtle dependence, or severe truncation.

## 7. Before opening a PR

Run:

```bash
python scripts/verify_tasks.py
```

The PR should include:

- the new task file in `data/tasks/`
- any README or docs updates needed to explain the addition
