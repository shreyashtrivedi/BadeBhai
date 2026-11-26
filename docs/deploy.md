# Deploy the public demo (free)

A live demo at a permanent link like `https://badebhai.onrender.com`, on free tiers only:

| Piece | Free service | Why |
|---|---|---|
| Web app (API + UI) | **Render** free web service (512 MB) | serves requests; nothing heavy runs on it |
| Database + vector search | **Neon** free Postgres (pgvector) | keeps the index across restarts |
| Model and embeddings | **Google Gemini** free API (OpenAI-compatible) | a 512 MB box cannot run a model |

Visitors get the full UI on two pre-indexed repositories (psf/requests and sindresorhus/ky): the agent,
verified citations, code viewer, codebase map and repo comparison. Demo mode fixes the repositories
and allows 10 questions per visitor per hour (300 per day in total), so the free API quota cannot be
drained. Rehearsed locally in a container capped at 512 MB: about 135 MB in use, an agent answer with
8/8 verified citations, and a restart that restored the git clone in seconds without re-indexing.

Check each service's free tier when you sign up; free tiers change.

## 1. Gemini API key (2 minutes)

[aistudio.google.com/apikey](https://aistudio.google.com/apikey) → sign in → **Create API key** → copy it.

## 2. Neon database (2 minutes)

[neon.tech](https://neon.tech) → sign up → create a project (any name, region close to you) → copy the
**connection string** (`postgresql://...neon.tech/neondb?sslmode=require`). BadeBhai creates the
`vector` extension and its tables itself.

## 3. Render web service (5 minutes)

1. The code must be on GitHub (this repository).
2. [dashboard.render.com](https://dashboard.render.com) → sign up with GitHub → **New → Blueprint** →
   choose the BadeBhai repository. Render reads [render.yaml](../render.yaml).
3. It asks for three values: `DATABASE_URL` (the Neon string), `OPENAI_API_KEY` (the Gemini key) and,
   optionally, `GITHUB_TOKEN` (a read-only fine-grained token, so live issue search is not limited to
   60 requests an hour). **Apply.**
4. Index the demo repositories into Neon **from your own machine**: the free instance (0.1 CPU,
   512 MB) runs out of memory embedding them itself. Put `DATABASE_URL=<the Neon string>` in a
   git-ignored `.env.neon` and run `bash scripts/index_into_neon.sh` (about 25 minutes on a laptop).
   On start Render then only restores the git clones; its own auto-ingest finds every file unchanged.
   The link is shown at the top of the service page.

That link goes on your resume and in the README.

## Good to know

* **Sleeping:** free Render services sleep after about 15 minutes without visitors; the next visit
  wakes it in about a minute. The index lives in Neon, so after a restart only the git clones are
  re-downloaded (seconds).
* **Model:** `LLM_MODEL=gemini-2.5-flash`. When it is overloaded (503) or out of daily quota, the
  next best Flash model the API lists answers instead for five minutes (or `OPENAI_FALLBACK_MODELS`,
  comma-separated). Embeddings: `BAAI/bge-small-en-v1.5` on the instance (ONNX, 384 dimensions); if
  you change the embedding model, re-run `scripts/index_into_neon.sh` with the same settings.
* **Other APIs:** any OpenAI-compatible API works: Groq (`https://api.groq.com/openai/v1`, chat only)
  or OpenRouter (`https://openrouter.ai/api/v1`); set `OPENAI_BASE_URL`, `LLM_MODEL` and the key.
* **Limits:** `DEMO_ASKS_PER_HOUR`, `DEMO_ASKS_PER_DAY` and `OPENAI_MAX_RPM` are environment variables.
* **Privacy:** demo questions go to the Gemini API (Google's free tier may use them to improve its
  models). The repositories are public. Your own copy can stay fully local with Ollama.
* **Other demo repositories:** change `AUTO_INGEST_REPO` (comma-separated `owner/name@ref`).

## Alternative: any Docker host

[deploy/huggingface/Dockerfile](../deploy/huggingface/Dockerfile) builds one self-contained image
(Postgres inside, demo repositories indexed at build time, port 7860) for hosts that run Docker images,
such as a paid Hugging Face Space or a VM: `HF_TOKEN=... bash deploy/huggingface/deploy.sh <user>/<space>`
for Hugging Face, or `docker build -f deploy/huggingface/Dockerfile .` elsewhere. Set `OPENAI_API_KEY`.
