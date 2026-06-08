# Evaluation Settings

All published baseline results were produced with the settings below.
Any replication attempt should match these to get comparable numbers.

## Model Call Settings

| Setting | Value |
|---------|-------|
| Temperature | 0.0 |
| Max tokens | 1024 |
| Random seed | not set |

## Rate Limiting

| Provider | Delay between requests | Retries |
|----------|----------------------|---------|
| Anthropic | 0.5s | 0 |
| Google Gemini | 4.0s | 5 (exponential backoff) |
| OpenAI | 0.5s | 0 |

## Scoring Rules

| Rule | Detail |
|------|--------|
| Numerical answer | Extracted via `ANSWER: [number]` tag, fallback to last number in response |
| Tolerance | Per-task field, typically 0.005 for probabilities |
| Full credit | 1.0 — answer within tolerance of ground truth |
| Partial credit | 0.5 — correct phenomenon named but wrong number |
| No credit | 0.0 — naive answer, wrong direction, or parse failure |
| Naive trap | Detected if parsed answer matches naive_answer within tolerance |

## Evaluation Harness Version

Scores are tied to a specific harness commit.
Re-running against a newer evaluate.py may produce different numbers
if scoring logic changed.

| Run | Harness commit | Date |
|-----|---------------|------|
| Claude Sonnet 4.5 | 44fe280 | 2026-05-20 |
| Claude Haiku 4.5 | 44fe280 | 2026-05-20 |
