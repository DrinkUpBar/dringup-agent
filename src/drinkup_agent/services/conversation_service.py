"""Conversation management service."""

import json
from typing import List, Dict, Any, Optional
import redis.asyncio as redis
from ..config import settings


class ConversationService:
    """Service for managing conversation history."""

    def __init__(self):
        self.redis_client = None
        self._init_redis()

    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True,
            )
        except Exception as e:
            print(f"Redis initialization failed: {e}")
            # Fallback to in-memory storage if Redis is not available
            self.conversations = {}

    async def get_conversation(
        self, conversation_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history."""
        if self.redis_client:
            try:
                data = await self.redis_client.get(f"conversation:{conversation_id}")
                return json.loads(data) if data else None
            except Exception as e:
                print(f"Error getting conversation from Redis: {e}")
                return None
        else:
            return self.conversations.get(conversation_id)

    async def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get messages for a conversation."""
        conversation = await self.get_conversation(conversation_id)
        return conversation or []

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to conversation with optional metadata."""
        messages = await self.get_messages(conversation_id)
        message = {"role": role, "content": content}
        if metadata:
            message["metadata"] = metadata
        messages.append(message)

        if self.redis_client:
            try:
                await self.redis_client.set(
                    f"conversation:{conversation_id}",
                    json.dumps(messages),
                    ex=86400,  # Expire after 24 hours
                )
            except Exception as e:
                print(f"Error saving to Redis: {e}")
                # Fallback to in-memory
                self.conversations[conversation_id] = messages
        else:
            self.conversations[conversation_id] = messages

    async def update_system_message(self, conversation_id: str, content: str) -> None:
        """Update the system message in a conversation."""
        messages = await self.get_messages(conversation_id)

        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = content

            if self.redis_client:
                try:
                    await self.redis_client.set(
                        f"conversation:{conversation_id}",
                        json.dumps(messages),
                        ex=86400,
                    )
                except Exception as e:
                    print(f"Error updating Redis: {e}")
                    self.conversations[conversation_id] = messages
            else:
                self.conversations[conversation_id] = messages

    async def add_ai_message_with_tool_calls(
        self, conversation_id: str, content: str, tool_calls: List[Dict[str, Any]]
    ) -> None:
        """Add an AI message with tool calls to conversation history."""
        await self.add_message(
            conversation_id, "assistant", content, metadata={"tool_calls": tool_calls}
        )

    async def add_tool_result(
        self, conversation_id: str, tool_call_id: str, tool_name: str, result: str
    ) -> None:
        """Add a tool result to conversation history."""
        await self.add_message(
            conversation_id,
            "tool",
            result,
            metadata={"tool_call_id": tool_call_id, "tool_name": tool_name},
        )

    async def clear_conversation(self, conversation_id: str) -> None:
        """Clear a conversation."""
        if self.redis_client:
            try:
                await self.redis_client.delete(f"conversation:{conversation_id}")
            except Exception as e:
                print(f"Error clearing conversation: {e}")
        elif conversation_id in self.conversations:
            del self.conversations[conversation_id]
