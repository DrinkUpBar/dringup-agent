"""Configuration settings for DrinkUp Agent."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_base_url: Optional[str] = None

    # Mem0 Configuration (Optional - enables memory tools when provided)
    mem0_api_key: Optional[str] = None
    mem0_base_url: Optional[str] = "https://api.mem0.ai"

    # Milvus Configuration for Mem0 Vector Store
    milvus_url: str = "http://localhost:19530"  # Milvus server URL
    milvus_token: Optional[str] = (
        None  # Authentication token (optional for local setup)
    )
    milvus_collection_name: str = "mem0"  # Name of the Milvus collection
    milvus_db_name: str = ""  # Database name (empty for default)

    # Memgraph Configuration for Mem0 Graph Store (deprecated, use Milvus instead)
    memgraph_url: str = "bolt://localhost:7687"
    memgraph_username: str = "memgraph"
    memgraph_password: str = ""

    # OpenAI Embedding Configuration for Mem0
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 1536  # Dimensions for embedding model

    # Server Configuration
    server_host: str = "0.0.0.0"
    server_port: int = 8001

    # DrinkUp Backend Configuration
    drinkup_backend_url: str = "http://localhost:8080"

    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # MySQL Configuration
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_database: str = "drinkup"
    mysql_user: str = "root"
    mysql_password: str = ""

    # API Configuration
    api_prefix: str = "/api"
    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
