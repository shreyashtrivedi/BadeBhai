"""ASGI entry point: `uvicorn repomind.app.main:app`."""

from repomind.app.api import create_app

app = create_app()
