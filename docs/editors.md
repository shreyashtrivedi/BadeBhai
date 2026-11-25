# Use BadeBhai from your editor or AI assistant

BadeBhai speaks [MCP](https://modelcontextprotocol.io), so Claude Code, Cursor, VS Code (Copilot agent
mode), Claude Desktop and other MCP clients can use it. There are two ways to connect, and you can
use both:

| Server | What your assistant gets | Needs |
|---|---|---|
| **`assistant`** (recommended) | `ask` (BadeBhai's full research loop: a cited, verified answer), `list_repos`, `add_repo` | BadeBhai running (`docker compose up` or `badebhai serve`) |
| **`code_search`, `git_history`, `docs`** | the raw search tools BadeBhai's own agent uses (16 tools such as `search_code`, `repo_overview`, `file_outline`, `search_commits`, `blame`, `search_issues`, `search_docs`) | the same; Docker publishes them on `localhost:8101-8103` |

`assistant` is the simplest: your assistant asks one question and gets an answer with file/line
citations. The raw servers suit assistants that want to do their own searching.

In the examples below, replace `C:/path/to/BadeBhai` with the folder you cloned BadeBhai into. If
`uv` isn't on your PATH, use `python -m uv` instead of `uv`.

## Claude Code

```bash
# The assistant (ask BadeBhai)
claude mcp add badebhai -- uv run --directory C:/path/to/BadeBhai badebhai mcp serve assistant --transport stdio

# Optional: the raw search servers
claude mcp add --transport http badebhai-code http://localhost:8101/mcp
claude mcp add --transport http badebhai-git  http://localhost:8102/mcp
claude mcp add --transport http badebhai-docs http://localhost:8103/mcp
```

Then ask, for example: *"Use badebhai to find where psf/requests handles redirects."*

Agent answers take 15-90 seconds on a local model. If answers get cut off, raise the tool timeout,
e.g. `MCP_TOOL_TIMEOUT=300000 claude` (milliseconds).

## Cursor

`~/.cursor/mcp.json` (all projects) or `.cursor/mcp.json` (one project):

```json
{
  "mcpServers": {
    "badebhai": {
      "command": "uv",
      "args": ["run", "--directory", "C:/path/to/BadeBhai", "badebhai", "mcp", "serve", "assistant", "--transport", "stdio"]
    },
    "badebhai-code": { "url": "http://localhost:8101/mcp" }
  }
}
```

## VS Code (Copilot agent mode)

`.vscode/mcp.json`:

```json
{
  "servers": {
    "badebhai": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "C:/path/to/BadeBhai", "badebhai", "mcp", "serve", "assistant", "--transport", "stdio"]
    },
    "badebhai-code": { "type": "http", "url": "http://localhost:8101/mcp" }
  }
}
```

## Claude Desktop

Settings → Developer → Edit Config, then add the same `badebhai` entry as in the Cursor example under
`"mcpServers"`.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `BADEBHAI_URL` | `http://localhost:8000` | where the `assistant` server finds BadeBhai's API |
| `BADEBHAI_API_KEY` | (none) | sent as `Authorization: Bearer ...` when your BadeBhai requires login |

Pass them in your client's `env` block (Cursor/VS Code/Claude Desktop) or with
`claude mcp add -e BADEBHAI_URL=... badebhai -- ...`.

## Safety

Everything is read-only for the repositories: no tool writes to a repo or runs its code. `add_repo`
only makes BadeBhai download and index a repository. Repository text (code, comments, issues) is
untrusted, and tool results say so, but treat answers the way you would any search result.
