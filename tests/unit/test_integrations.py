"""GitHub bot and Slack command parsing, answer formatting, compare renumbering."""

import time

from badebhai.app.compare import check_markers, renumber
from badebhai.app.schemas import AskResponse, Citation, VerificationReport
from badebhai.config import Settings
from badebhai.integrations import github_bot, slack
from badebhai.integrations.formatting import github_markdown, slack_mrkdwn


def _resp(answer="Auth is in **app/auth.py** [1]. Retries in [2].") -> AskResponse:
    cites = [
        Citation(
            id=1,
            evidence_key="a",
            type="code",
            label="app/auth.py#L1-L9",
            title="BasicAuth",
            url="https://x/a",
        ),
        Citation(id=2, evidence_key="b", type="code", label="app/retry.py#L3-L5", title="retry", url=None),
    ]
    return AskResponse(
        answer=answer,
        citations=cites,
        verification=VerificationReport(total_citations=2, valid_citations=2),
        trace=[],
        mode="naive",
        repo="acme/tiny",
    )


def test_github_comment_has_answer_sources_and_footer():
    md = github_markdown(_resp(), "Where is auth?")
    assert md.startswith("> Where is auth?") and "[app/auth.py#L1-L9](https://x/a)" in md
    assert "`app/retry.py#L3-L5`" in md and "2/2 citations verified" in md and "can be wrong" in md


def test_slack_text_uses_mrkdwn():
    text = slack_mrkdwn(_resp(), "acme/tiny")
    assert "*app/auth.py*" in text and "**" not in text and "<https://x/a|app/auth.py#L1-L9>" in text


def _comment(body, assoc="MEMBER", login="ana", user_type="User", action="created"):
    return {
        "action": action,
        "comment": {"body": body, "author_association": assoc, "user": {"login": login, "type": user_type}},
        "issue": {"number": 7, "title": "Login fails with SSO"},
        "repository": {"full_name": "acme/tiny"},
    }


def test_bot_answers_trusted_mentions_only():
    s = Settings(_env_file=None)
    q, why = github_bot.parse_mention(_comment("@badebhai where is the SSO login handled?"), s)
    assert q and q.question == "where is the SSO login handled?" and q.number == 7 and not why
    assert github_bot.parse_mention(_comment("no mention here"), s)[0] is None
    assert github_bot.parse_mention(_comment("@badebhai hi", assoc="NONE"), s)[0] is None
    assert (
        github_bot.parse_mention(_comment("@badebhai hi", login="badebhai[bot]", user_type="Bot"), s)[0]
        is None
    )
    assert github_bot.parse_mention(_comment("@badebhai hi", action="edited"), s)[0] is None
    assert github_bot.parse_mention(_comment("email me at x@badebhai.dev"), s)[0] is None
    anyone = Settings(_env_file=None, github_bot_allow_anyone=True)
    assert github_bot.parse_mention(_comment("@badebhai hi", assoc="NONE"), anyone)[0] is not None
    empty, _ = github_bot.parse_mention(_comment("@badebhai"), s)
    assert "Login fails with SSO" in empty.question  # falls back to the issue title


def test_slack_signatures_and_commands():
    body = b"text=psf%2Frequests+where+is+auth&user_id=U1&response_url=https%3A%2F%2Fhooks.slack.com%2Fcommands%2Fx"
    ts = str(int(time.time()))
    sig = slack.sign("shh", ts, body)
    assert slack.verify("shh", ts, body, sig)
    assert not slack.verify("shh", ts, body + b"x", sig)
    assert not slack.verify(
        "shh", str(int(time.time()) - 600), body, slack.sign("shh", str(int(time.time()) - 600), body)
    )
    cmd = slack.parse_command(body)
    assert cmd.text == "psf/requests where is auth" and cmd.user_id == "U1"
    assert slack.parse_command(b"text=x&response_url=https%3A%2F%2Fevil.example%2F") is None  # no SSRF
    assert slack.split_repo(cmd.text, ["psf/requests", "a/b"], None) == ("psf/requests", "where is auth")
    assert slack.split_repo("where is auth", ["psf/requests", "a/b"], "a/b") == ("a/b", "where is auth")
    assert slack.split_repo("where is auth", ["psf/requests", "a/b"], None) == (None, "where is auth")


def test_compare_renumbers_citations_and_flags_unknown_markers():
    text, cites = renumber(_resp(), "acme/tiny", offset=3)
    assert text == "Auth is in **app/auth.py** [4]. Retries in [5]."
    assert (
        [c.id for c in cites] == [4, 5]
        and cites[0].repo == "acme/tiny"
        and cites[0].label.startswith("acme/tiny: ")
    )
    fixed, bad = check_markers("A [4] and B [9].", {4, 5})
    assert fixed == "A [4] and B ⚠️(reference not found)." and bad[0]["marker"] == 9
