"""FastAPI server for the LLM Router & Gateway."""

from llm_router.server.app import create_app

__all__ = ["create_app"]
