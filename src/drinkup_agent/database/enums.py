"""Database enums."""

from enum import Enum


class PromptTypeEnum(str, Enum):
    """Prompt types matching Java PromptTypeEnum."""

    CHAT = "CHAT"
    IMAGE_RECOGNITION = "IMAGE_RECOGNITION"
    BARTENDER = "BARTENDER"
    TRANSLATE = "TRANSLATE"
    MATERIAL_ANALYSIS = "MATERIAL_ANALYSIS"
    CHAT_STREAM = "CHAT_STREAM"
