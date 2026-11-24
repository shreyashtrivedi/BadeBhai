# Demo GIF script

Two recordings: a terminal one (scripted with [VHS](https://github.com/charmbracelet/vhs), see
[demo.tape](demo.tape)) and a browser one (record by hand, about 60 seconds).

## Before recording

```bash
cp .env.example .env              # set ANTHROPIC_API_KEY; GITHUB_TOKEN avoids GitHub rate limits
docker compose up --build -d      # wait until GET /repos shows psf/requests with a last_indexed_commit
```

Use a fresh user id such as `demo-user` so the memory panel starts empty.

## Browser recording (http://localhost:8000)

| Time | Action | What to point out |
|---|---|---|
| 0-5 s | Page open, repo `psf/requests @ v2.32.3` selected | Three panels: chat, citations, trace, memory |
| 5-20 s | Type: *I work on the transport layer. Why was the retry logic added, and is it related to any open issues?* | Trace streams live: sub-questions, MCP tool calls (search_code, search_commits, blame, search_issues), grading |
| 20-30 s | Answer streams in, then is replaced by the verified version | Badges: "n/n citations verified", "remembered: The user works on the transport layer" |
| 30-40 s | Click a `[n]` chip on a commit, then on a code citation | Citation panel jumps to the source; "show source" expands the exact lines; links open GitHub at the pinned commit |
| 40-48 s | Ask: *Where is the GraphQL resolver?* | Grader marks it weak, two rewrites, then "Insufficient evidence" with no citations |
| 48-55 s | Click **New thread**, ask: *Where are connect and read timeouts applied?* | The Memory tab still shows the transport-layer fact: long-term memory survives new sessions |
| 55-60 s | Open the Memory tab, click **Forget** | User control over memory |

## Live-update beat (optional, 20 s)

Index a repo you can push to (`uv run repomind ingest owner/your-repo`), add a function, push, and
ask about it. With the webhook (`POST /webhooks/github`, forwarded with smee.io) or
`uv run repomind sync --all --watch --interval 10`, it is answerable within seconds.
