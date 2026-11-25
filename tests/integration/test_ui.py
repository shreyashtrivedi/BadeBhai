import pytest
from fastapi.testclient import TestClient

from badebhai.app.api import create_app
from badebhai.llm.fake import FakeLLM
from tests.integration.helpers import toolbox_factory

pytestmark = pytest.mark.db


def test_web_ui_is_served(db_settings):
    with TestClient(
        create_app(db_settings, llm=FakeLLM(), toolbox_factory=toolbox_factory(db_settings))
    ) as c:
        root = c.get("/", follow_redirects=False)
        assert root.status_code in (302, 307) and root.headers["location"] == "/ui/"
        page = c.get("/ui/")
        assert page.status_code == 200 and "<title>BadeBhai</title>" in page.text
        js = c.get("/ui/app.js")
        assert js.status_code == 200 and "/ask/stream" in js.text
        assert c.get("/ui/styles.css").status_code == 200
        tools = c.get("/tools").json()
        assert set(tools["servers"]) == {"code_search", "git_history", "docs"} and len(tools["tools"]) == 15
