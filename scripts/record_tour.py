"""Record a static tour of BadeBhai from a running instance, then build the site.

    python scripts/record_tour.py --base http://localhost:8000 [--live-url https://...] [--build-only]

Asks each question below through the real API, saves the full responses (answer, citations,
verification, trace), every cited file (for the code viewer) and each repo's codebase map into
tour/recorded.js, then builds tour/site/: the real web UI plus tour/tour.js, which answers the UI's API
calls from the recording. The site is static and can be hosted anywhere (GitHub Pages).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
TOUR = ROOT / "tour"
SOURCE_URL = "https://github.com/shreyashtrivedi/BadeBhai"

QUESTIONS = [
    ("psf/requests", "What are the main parts of this codebase and how do they fit together?", "agent"),
    ("psf/requests", "Which data structure stores HTTP headers case-insensitively?", "auto"),
    ("psf/requests", "How does a Session decide which transport adapter handles a URL?", "agent"),
    ("psf/requests", "What security issue was fixed in 2.32.0, and which code changed for it?", "agent"),
    ("psf/requests", "Why was the retry logic added, and is it related to any GitHub issues?", "agent"),
    ("psf/requests", "Where is the GraphQL client implemented in this repository?", "auto"),
    ("sindresorhus/ky", "How does ky decide how long to wait before retrying a request?", "agent"),
    ("sindresorhus/ky", "Why was the prefixUrl option renamed, and in which change?", "agent"),
]
COMPARE = [(["psf/requests", "sindresorhus/ky"], "How does each library retry failed requests?")]


def record(base: str, live_url: str | None) -> dict:
    http = httpx.Client(base_url=base.rstrip("/"), timeout=httpx.Timeout(1200.0, connect=10.0))
    setup = http.get("/setup").json()
    wanted = {r for r, _, _ in QUESTIONS} | {r for repos, _ in COMPARE for r in repos}
    repos = [r for r in http.get("/repos").json() if r["name"] in wanted and r.get("last_indexed_commit")]
    missing = wanted - {r["name"] for r in repos}
    if missing:
        sys.exit(f"these repos are not indexed on {base}: {sorted(missing)}")
    maps = {r["name"]: http.get("/overview", params={"repo": r["name"]}).json()["map"] for r in repos}
    answers, files = [], {}

    def keep_files(resp: dict, default_repo: str) -> None:
        for c in resp.get("citations", []):
            repo = c.get("repo") or default_repo
            if c.get("path") and c.get("type") in ("code", "doc") and f"{repo}::{c['path']}" not in files:
                r = http.get("/file", params={"repo": repo, "path": c["path"]})
                if r.status_code == 200:
                    files[f"{repo}::{c['path']}"] = r.json()

    for repo, question, mode in QUESTIONS:
        t = time.time()
        body = {
            "question": question,
            "repo": repo,
            "mode": mode,
            "user_id": "tour",
            "thread_id": f"tour-{uuid.uuid4().hex[:8]}",
        }
        resp = http.post("/ask", json=body)
        resp.raise_for_status()
        data = resp.json()
        data["thread_id"] = None
        keep_files(data, repo)
        answers.append({"repo": repo, "question": question, "response": data})
        v = data["verification"]
        print(
            f"[{len(answers)}] {repo}: {question[:60]} -> {v['valid_citations']}/{v['total_citations']} verified, {time.time() - t:.0f}s"
        )
    for repos_, question in COMPARE:
        t = time.time()
        resp = http.post(
            "/compare", json={"question": question, "repos": repos_, "mode": "agent", "user_id": "tour"}
        )
        resp.raise_for_status()
        data = resp.json()
        keep_files(data, repos_[0])
        answers.append({"repo": " vs ".join(repos_), "question": question, "response": data})
        print(f"[compare] {question[:60]} -> {time.time() - t:.0f}s")
    return {
        "recorded_at": time.strftime("%Y-%m-%d"),
        "setup": {"model": setup.get("model"), "github_token": False, "token_url": setup.get("token_url")},
        "source_url": SOURCE_URL,
        "live_url": live_url,
        "repos": repos,
        "maps": maps,
        "files": files,
        "answers": answers,
    }


def build() -> Path:
    site = TOUR / "site"
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)
    ui = ROOT / "badebhai" / "ui"
    for name in ("app.js", "styles.css"):
        shutil.copy(ui / name, site / name)
    shutil.copy(TOUR / "tour.js", site / "tour.js")
    shutil.copy(TOUR / "recorded.js", site / "recorded.js")
    html = (ui / "index.html").read_text(encoding="utf-8")
    html = html.replace("<title>BadeBhai</title>", "<title>BadeBhai: recorded tour</title>")
    html = html.replace(
        '<script src="app.js"></script>',
        '<script src="recorded.js"></script>\n  <script src="tour.js"></script>\n  <script src="app.js"></script>',
    )
    (site / "index.html").write_text(html, encoding="utf-8")
    (site / ".nojekyll").write_text("", encoding="utf-8")
    return site


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--live-url", default=None)
    ap.add_argument("--build-only", action="store_true", help="rebuild tour/site from tour/recorded.js")
    ap.add_argument("--out", default=None, help="write the recording here instead of tour/recorded.js (no site build)")
    args = ap.parse_args()
    if not args.build_only:
        data = record(args.base, args.live_url)
        out = Path(args.out) if args.out else TOUR / "recorded.js"
        out.write_text(
            "// Recorded by scripts/record_tour.py from a running BadeBhai. Repository content is untrusted data.\n"
            "window.BADEBHAI_TOUR = " + json.dumps(data, ensure_ascii=False) + ";\n",
            encoding="utf-8",
        )
        if args.out:
            print(f"recording: {out}")
            return
    elif args.live_url:
        rec = (TOUR / "recorded.js").read_text(encoding="utf-8")
        payload = json.loads(rec.split("window.BADEBHAI_TOUR = ", 1)[1].rstrip().rstrip(";"))
        payload["live_url"] = args.live_url
        (TOUR / "recorded.js").write_text(
            rec.split("window.BADEBHAI_TOUR = ", 1)[0]
            + "window.BADEBHAI_TOUR = "
            + json.dumps(payload, ensure_ascii=False)
            + ";\n",
            encoding="utf-8",
        )
    print(f"site: {build()}")


if __name__ == "__main__":
    main()
