"""
Application entry point.
Run with: python -m src.main or uvicorn src.main:app
"""

import asyncio
import signal
import sys
from typing import Optional

import uvicorn

from src.api.app import app, create_app
from src.config import settings
from src.observability import configure_logging, get_logger

logger = get_logger(__name__)


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "info"
) -> None:
    """
    Run the FastAPI server with Uvicorn.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
        workers: Number of worker processes
        log_level: Uvicorn log level
    """
    logger.info(
        "starting_server",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=log_level,
    )


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Router System")
    parser.add_argument(
        "--host",
        default=settings.api_host,
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.api_port,
        help="Port to bind to"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level.lower(),
        choices=["debug", "info", "warning", "error"],
        help="Log level"
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(level=args.log_level.upper())

    logger.info(
        "llm_router_starting",
        environment=settings.env,
        debug=settings.debug,
    )

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
