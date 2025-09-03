"""Main application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from .config import settings
from .api.chat_stream import router as chat_stream_router
from .database.init import init_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    await init_database()
    yield
    # Shutdown
    print("Shutting down DrinkUp Agent...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="DrinkUp Agent",
        description="AI agent service for DrinkUp chatbot interactions",
        version="0.1.0",
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat_stream_router, prefix=f"{settings.api_prefix}")

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"name": "DrinkUp Agent", "version": "0.1.0", "status": "running"}

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "drinkup_agent.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug,
    )
