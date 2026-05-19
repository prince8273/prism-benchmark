# PRISM Task Schema

Each PRISM task is a single JSON object stored in `data/tasks/<task_id>.json`.

## Required fields

```json
{
  "task_id": "prism_simp_001",
  "category": "simpson | berkson | truncated | survival",
  "difficulty": "easy | medium | hard",
  "scenario": "Short natural-language setup.",
  "data": {
    "description": "What observations are available.",
    "values": {}
  },
  "question": "What the model must compute or decide.",
  "answer_type": "numerical | comparative | directional",
  "ground_truth": {},
  "tolerance": 0.01,
  "naive_answer": "What a selection-blind model would likely say.",
  "naive_trap": "Why the naive answer is wrong.",
  "reasoning_trace": "Reference derivation for maintainers.",
  "python_verification": "Runnable Python assertion snippet."
}
```

## Field notes

- `task_id`: Unique stable identifier.
- `category`: The failure mode being tested.
- `difficulty`: Author-calibrated difficulty band.
- `data.values`: Structured inputs shown to the model.
- `answer_type`: High-level scoring mode; multi-part tasks are allowed.
- `ground_truth`: May be a scalar or an object with multiple required outputs.
- `tolerance`: Absolute tolerance for numerical matching.
- `naive_answer`: The expected incorrect answer from a model that ignores selection.
- `reasoning_trace`: Not shown to models; used for auditing and authoring.
- `python_verification`: Must confirm the ground truth programmatically.

## Ground truth conventions

- Use concise keys for multi-part answers such as `recommendation`, `true_mean_mu`, or `fraction_included`.
- Keep explanatory prose in `ground_truth.explanation`; the evaluator ignores it.
- Avoid mixing optional and required outputs in the same `ground_truth` object.
