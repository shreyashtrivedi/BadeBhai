# GitHub bot and Slack command

Both are **off by default**, because they post messages to outside services. Each needs BadeBhai to be
reachable from the internet (GitHub and Slack call it), e.g. through a reverse proxy, a tunnel such
as `cloudflared tunnel --url http://localhost:8000`, or a small server.

## GitHub: answer `@badebhai` in issues and pull requests

Someone comments `@badebhai where is the retry limit configured?` on an issue or PR; BadeBhai replies
with a cited answer (sources collapsed under "Sources") a minute or two later.

1. **Token that can comment.** Create a fine-grained token at
   [github.com/settings/personal-access-tokens/new](https://github.com/settings/personal-access-tokens/new)
   for the repositories in question, with **Issues: Read and write** and **Pull requests: Read and write**
   (plus Contents: Read-only for private repos). Replies are posted as the token's owner, so a separate
   bot account is a good idea.
2. **Webhook.** In the repository: Settings → Webhooks → Add webhook.
   * Payload URL: `https://<your-badebhai>/webhooks/github`
   * Content type: `application/json`
   * Secret: a long random string
   * Events: **Pushes** (keeps the index live) and **Issue comments**
3. **Configure and restart BadeBhai** (`.env`):
   ```bash
   GITHUB_TOKEN=github_pat_...
   GITHUB_WEBHOOK_SECRET=<the webhook secret>
   GITHUB_BOT_ENABLED=true
   # GITHUB_BOT_HANDLE=@badebhai        # the mention it answers
   # GITHUB_BOT_ALLOW_ANYONE=false      # default: only owners, members and collaborators
   ```
4. Make sure the repository is added to BadeBhai (the bot ignores repos it hasn't indexed).

Safeguards: every webhook is HMAC-verified; only new comments that mention the handle are answered;
by default only the repo's owners, members and collaborators can ask (so drive-by users can't make it
post); comments by bots, including its own, are ignored; each answer is one comment, never edits.

## Slack: `/badebhai`

`/badebhai where is authentication handled?` or `/badebhai psf/requests where are timeouts applied?`
(start with a repo name to pick one). BadeBhai replies "Looking into it…" privately, then posts the
answer in the channel.

1. Create a Slack app at [api.slack.com/apps](https://api.slack.com/apps) → *From scratch*.
2. **Slash Commands** → Create New Command: command `/badebhai`, request URL
   `https://<your-badebhai>/integrations/slack/command`, a short description and usage hint.
3. **Install to Workspace**.
4. From **Basic Information**, copy the **Signing Secret** into `.env` and restart BadeBhai:
   ```bash
   SLACK_SIGNING_SECRET=...
   # SLACK_DEFAULT_REPO=psf/requests   # used when the command doesn't start with a repo name
   ```

Safeguards: every request's Slack signature and timestamp are checked (stale requests over five
minutes old are rejected); replies only go to the `hooks.slack.com` URL Slack provides. With
accounts enabled (`AUTH_ENABLED`), Slack can only ask about repositories shared with everyone.
