## Eval run 20261126-135933

Repo `psf/requests` @ `v2.32.3` (0e322af877), 38 questions from `requests_v2_32_3.yaml`. Models: {'llm': 'ollama/qwen3:4b-instruct', 'fast': 'qwen3:4b-instruct', 'embedding': 'BAAI/bge-small-en-v1.5'}. Errors: {'naive': 0, 'agent': 0}.

| Metric | naive | agent |
|---|---|---|
| Retrieval hit@1 | 71.4% | 91.4% |
| Retrieval hit@3 | 91.4% | 97.1% |
| Retrieval hit@5 | 91.4% | 97.1% |
| Retrieval hit@10 | 91.4% | 97.1% |
| Expected source found anywhere in gathered evidence | 91.4% | 97.1% |
| Answer cites an expected file/commit/issue | 88.6% | 97.1% |
| Citation validity rate | 98.5% | 100.0% |
| LLM-judge answer correctness (0-1; abstaining when answerable = 0) | 0.76 | 0.81 |
| Says 'insufficient evidence' on unanswerable questions | 100.0% | 100.0% |
| Abstains entirely on answerable questions | 0.0% | 0.0% |
| Latency p50 | 16.3 s | 46.0 s |
| Latency p95 | 204.5 s | 228.7 s |
| Cost per question (mean) | $0.0000 | $0.0000 |
| LLM calls per question | 1.00 | 3.58 |
| MCP tool calls per question | 0.00 | 5.66 |

Judge correctness by category:

| Category | naive | agent |
|---|---|---|
| docs | 0.83 | 1.00 |
| history | 0.50 | 0.64 |
| how | 0.80 | 0.68 |
| issues | 0.75 | 1.00 |
| location | 0.84 | 0.82 |
| planning | 0.69 | 0.81 |
| unanswerable | 1.00 | 1.00 |
