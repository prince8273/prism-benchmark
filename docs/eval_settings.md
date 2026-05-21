# Evaluation Settings

All baseline results were produced with these settings:

| Setting | Value |
|---------|-------|
| Temperature | 0.0 |
| Max tokens | 1024 |
| Retries | 3 (Gemini), 0 (others) |
| Rate limit | 4s/request (Gemini), 0.5s (others) |
| Parsing | regex on ANSWER: tag, fallback to last number |
| Partial credit | 0.5 if correct phenomenon named, wrong number |
| Scoring version | evaluate.py commit 44fe280 |
