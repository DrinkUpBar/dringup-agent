"""Services package for DrinkUp Agent.

Avoid importing heavy submodules at package import time to prevent
side effects (e.g., optional dependencies not installed) when any
submodule is imported like `services.x`. Use lazy attribute loading
instead so `from drinkup_agent.services import ChatService` continues
to work without eagerly importing everything when importing
`drinkup_agent.services` as a package.
"""

__all__ = ["ChatService", "ConversationService", "AgentService"]


def __getattr__(name):
    if name == "ChatService":
        from .chat_service import ChatService as _ChatService

        return _ChatService
    if name == "ConversationService":
        from .conversation_service import ConversationService as _ConversationService

        return _ConversationService
    if name == "AgentService":
        from .agent_service import AgentService as _AgentService

        return _AgentService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
