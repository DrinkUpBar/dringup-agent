"""Database initialization and health check."""

import asyncio
from sqlalchemy import text

from .base import get_async_engine, get_async_session
from ..config import settings


async def check_database_connection() -> bool:
    """
    Check if database connection is available.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        engine = get_async_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


async def init_database():
    """
    Initialize database connection and check health.
    """
    print(
        f"Connecting to MySQL database at {settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
    )

    if await check_database_connection():
        print("✓ Database connection successful")

        # Test prompt table access
        try:
            session_factory = get_async_session()
            async with session_factory() as session:
                result = await session.execute(
                    text("SELECT COUNT(*) FROM prompt_content")
                )
                count = result.scalar()
                print(f"✓ Found {count} prompt(s) in database")
        except Exception as e:
            print(f"⚠ Warning: Could not access prompt_content table: {e}")
            print("  The application will use default prompts as fallback")
    else:
        print("⚠ Warning: Database connection failed")
        print("  The application will work with default prompts only")


def run_init():
    """Run database initialization synchronously."""
    asyncio.run(init_database())
