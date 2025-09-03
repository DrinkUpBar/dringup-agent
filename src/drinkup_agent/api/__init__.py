"""API routes for DrinkUp Agent."""

from .chat_stream import router as chat_stream_router

__all__ = ["chat_stream_router"]
