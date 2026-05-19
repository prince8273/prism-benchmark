# PRISM Scoring

PRISM is designed to evaluate whether a model can recover the correct answer after reasoning about the selection mechanism behind the observed data.

## Primary metric

- `Overall ACC`: Mean task score across all tasks.

## Secondary metric

- `Naive Trap Rate`: Fraction of tasks where the model appears to give the selection-blind answer.

## Per-task rubric

- `1.0`: All required outputs in `ground_truth` are recovered within tolerance.
- `0.5`: The model gets only part of a multi-part task right, or correctly identifies the phenomenon but misses the full quantitative answer.
- `0.0`: The model misses the required outputs entirely.

## Numerical matching

- Numerical values are matched with absolute tolerance from the task's `tolerance` field.
- The evaluator searches across all numbers in the response, so models may answer in prose or with labeled lines.

## Comparative and directional matching

- String-valued targets such as `Drug_A`, `Method_Y`, or `independent` are matched case-insensitively.
- Boolean targets such as `aggregate_valid: false` are matched with lightweight language heuristics like `invalid`, `misleading`, or `not valid`.

## Partial-credit philosophy

PRISM tasks are often multi-part. A model that gets one piece right but fails to carry the reasoning through should not receive full credit. Partial credit exists to separate:

- complete success
- partial understanding
- pure naive failure

## Auditing

Saved result files include:

- full model response
- matched and missing ground-truth keys
- extracted numeric candidates
- naive-answer flag
