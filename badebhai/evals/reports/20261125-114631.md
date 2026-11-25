## Eval run 20261125-114631

Repo `psf/requests` @ `v2.32.3` (0e322af877), 38 questions from `requests_v2_32_3.yaml`. Models: {'llm': 'ollama/qwen3:4b-instruct', 'fast': 'qwen3:4b-instruct', 'embedding': 'BAAI/bge-small-en-v1.5'}. Errors: {'naive': 0, 'agent': 0}.

| Metric | naive | agent |
|---|---|---|
| Retrieval hit@1 | 71.4% | 91.4% |
| Retrieval hit@3 | 91.4% | 97.1% |
| Retrieval hit@5 | 91.4% | 100.0% |
| Retrieval hit@10 | 91.4% | 100.0% |
| Expected source found anywhere in gathered evidence | 91.4% | 100.0% |
| Answer cites an expected file/commit/issue | 88.6% | 100.0% |
| Citation validity rate | 98.6% | 98.8% |
| LLM-judge answer correctness (0-1; abstaining when answerable = 0) | 0.74 | 0.79 |
| Says 'insufficient evidence' on unanswerable questions | 66.7% | 100.0% |
| Abstains entirely on answerable questions | 0.0% | 0.0% |
| Latency p50 | 16.8 s | 43.5 s |
| Latency p95 | 36.1 s | 555.5 s |
| Cost per question (mean) | $0.0000 | $0.0000 |
| LLM calls per question | 1.00 | 3.76 |
| MCP tool calls per question | 0.00 | 5.89 |

Judge correctness by category:

| Category | naive | agent |
|---|---|---|
| docs | 1.00 | 1.00 |
| history | 0.50 | 0.32 |
| how | 0.60 | 0.85 |
| issues | 0.88 | 1.00 |
| location | 0.80 | 0.84 |
| planning | 0.69 | 0.94 |
| unanswerable | 1.00 | 1.00 |
