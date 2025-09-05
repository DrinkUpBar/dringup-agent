"""Database connection and session management."""

from urllib.parse import quote_plus
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from ..config import settings

# Create base class for models
Base = declarative_base()

# Create database URL with properly encoded password
DATABASE_URL = (
    f"mysql+aiomysql://{settings.mysql_user}:{quote_plus(settings.mysql_password)}"
    f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
)

# Create async engine
engine = None


def get_async_engine():
    """Get or create async database engine."""
    global engine
    if engine is None:
        engine = create_async_engine(
            DATABASE_URL,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
    return engine


# Create async session factory
async_session = None


def get_async_session():
    """Get async session factory."""
    global async_session
    if async_session is None:
        async_session = async_sessionmaker(
            get_async_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return async_session
