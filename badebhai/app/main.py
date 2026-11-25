"""ASGI entry point: `uvicorn badebhai.app.main:app`."""

from badebhai.app.api import create_app

app = create_app()
