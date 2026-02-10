"""
FastAPI application initialization and configuration.
"""

import time
from contextlib import asynccontextmanager
from typing import Callable
import structlog
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.api.routes import router
from src.providers.health_monitor import start_health_monitoring, stop_health_monitoring

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Runs on startup and shutdown.
    """
    # Startup
    logger.info(
        "application_starting",
        env=settings.env,
        debug=settings.debug,
    )

    # Start health monitoring
    await start_health_monitoring()

    logger.info("application_started")

    yield

    # Shutdown
    logger.info("application_stopping")

    # Stop health monitoring
    await stop_health_monitoring()

    logger.info("application_stopped")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    app = FastAPI(
        title="LLM Router System",
        description="""
        Production-grade distributed LLM processing pipeline with intelligent routing.

        ## Features

        - **Multi-Provider Support**: Route requests across Groq, OpenAI, and Anthropic
        - **Intelligent Routing**: Automatic provider/model selection based on complexity
        - **Multi-Level Caching**: Exact match and semantic similarity caching
        - **Rate Limiting**: Token bucket rate limiting per client
        - **Circuit Breaker**: Automatic failover on provider failures
        - **Cost Optimization**: Track and optimize LLM costs

        ## Authentication

        Include your client ID in the `X-Client-ID` header for tracking.
        Use `X-Client-Tier` header to specify tier (standard/premium/enterprise).
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next: Callable) -> Response:
        """Log all requests with timing."""
        start_time = time.time()

        # Generate request ID if not provided
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])

        # Add request ID to state
        request.state.request_id = request_id

        # Log request
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)
        except Exception as e:
            logger.exception(
                "request_error",
                request_id=request_id,
                error=str(e),
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Internal server error",
                        "type": "internal_error",
                        "code": "internal_server_error",
                    }
                }
            )

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Add timing header
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        # Log response
        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        return response

    # Include routers
    app.include_router(router)

    return app


# Create the application instance
app = create_app()


# Additional endpoints at root level

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "name": "LLM Router System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/ping")
async def ping():
    """Simple ping endpoint."""
    return {"status": "pong"}
