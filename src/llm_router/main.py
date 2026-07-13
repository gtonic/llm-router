"""CLI entry point for llm-router."""

import sys

from llm_router import __version__


def main():
    """Main CLI entry point."""
    print(f"LLM Router & Gateway v{__version__}")
    print("Use 'llm-router-server' to start the FastAPI server.")
    sys.exit(0)


if __name__ == "__main__":
    main()
