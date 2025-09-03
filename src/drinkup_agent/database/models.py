"""Database models."""

from sqlalchemy import Column, Integer, String, Text
from .base import Base


class PromptContent(Base):
    """PromptContent model matching Java entity."""

    __tablename__ = "prompt_content"

    id = Column(Integer, primary_key=True, autoincrement=True)
    system_prompt = Column(Text, nullable=True)
    type = Column(String(50), nullable=True)
    name = Column(String(255), nullable=True)
