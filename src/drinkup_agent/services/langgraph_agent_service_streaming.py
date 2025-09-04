"""Enhanced LangGraph agent service with character-level streaming."""

import json
import uuid
import logging
from typing import Optional, List, Dict, Any, AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import BaseTool

# Tool execution helper: Try to import ToolNode; if unavailable, we fallback later
try:
    from langgraph.prebuilt import ToolNode  # type: ignore
except Exception:  # ModuleNotFoundError or others
    ToolNode = None  # Fallback to manual tool execution

from ..config import settings
from ..repositories import PromptRepository
from ..services.conversation_service import ConversationService
from ..tools.mem0_tools import create_mem0_tools
from ..tools.drinkup_backend_tools import create_drinkup_backend_tools

logger = logging.getLogger(__name__)


class StreamingLangGraphAgentService:
    """Service for managing LangGraph agents with character-level streaming."""

    def __init__(self):
        """Initialize the agent service."""
        self.prompt_repository = PromptRepository()
        self.conversation_service = ConversationService()
        self.model = settings.openai_model

        # Initialize streaming LLM with usage metadata enabled
        llm_params = {
            "model": self.model,
            "temperature": 0.7,
            "openai_api_key": settings.openai_api_key,
            "streaming": True,
            "stream_options": {
                "include_usage": True
            },  # Enable usage data in streaming response
        }

        if settings.openai_base_url:
            llm_params["base_url"] = settings.openai_base_url

        self.llm = ChatOpenAI(**llm_params)

    async def _get_tools(self, user_id: str) -> List[BaseTool]:
        """Get all available tools for the agent."""
        tools = []

        # Add Mem0 tools if configured
        if settings.openai_api_key:
            mem0_tools = create_mem0_tools(
                api_key=settings.openai_api_key, user_id=user_id
            )
            tools.extend(mem0_tools)

        # Add DrinkUp backend tools with user_id
        drinkup_tools = create_drinkup_backend_tools(user_id=user_id)
        tools.extend(drinkup_tools)

        return tools

    async def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        db_prompt = await self.prompt_repository.get_chat_prompt()

        default_prompt = """You are a knowledgeable and friendly bartender assistant for DrinkUp.
You help users with cocktail recommendations, recipes, and bar management.

You have access to these tools:

**Memory Management Tools:**
- Use 'add_memory' to store important information about the user
- Use 'search_memory' to recall relevant information when answering questions
- Use 'get_all_memories' to see everything you know about the user
- Use 'update_memory' to correct or update stored information
- Use 'delete_memory' to remove outdated information

**Cocktail Generation Tool:**
- Use 'generate_cocktail' when users ask for cocktail recommendations or custom recipes
- This tool creates detailed cocktail recipes based on user preferences, flavors, or ingredients

Always:
1. When users ask for cocktail recommendations, use the generate_cocktail tool
2. Search for relevant memories when starting a conversation or when context would be helpful
3. Store new important information shared by the user (preferences, restrictions, favorites, etc.)
4. Update memories when users correct or change their preferences
5. Provide personalized responses based on stored memories and generated cocktails

Current context:
- User Stock: {user_stock}
- User Info: {user_info}

Respond naturally without JSON formatting."""

        return db_prompt if db_prompt else default_prompt

    async def chat_with_agent_stream(
        self,
        user_message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        user_stock: Optional[str] = None,
        user_info: Optional[str] = None,
        image_attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a chat message with character-level streaming.
        """
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        try:
            # Get conversation history
            messages = await self.conversation_service.get_messages(conversation_id)

            # Convert to LangChain format
            chat_history = []
            for msg in messages:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    # Check if this message has tool calls
                    metadata = msg.get("metadata", {})
                    tool_calls = metadata.get("tool_calls", [])

                    if tool_calls:
                        # Create AIMessage with tool_calls
                        chat_history.append(
                            AIMessage(content=msg["content"], tool_calls=tool_calls)
                        )
                    else:
                        # Regular assistant message
                        chat_history.append(AIMessage(content=msg["content"]))
                elif msg["role"] == "tool":
                    # Tool result message
                    metadata = msg.get("metadata", {})
                    tool_call_id = metadata.get("tool_call_id")

                    if tool_call_id:
                        chat_history.append(
                            ToolMessage(
                                content=msg["content"], tool_call_id=tool_call_id
                            )
                        )

            # Handle image attachments
            if image_attachments:
                image_note = f" [User shared {len(image_attachments)} image(s)]"
                user_message = user_message + image_note

            # Add user message to history
            chat_history.append(HumanMessage(content=user_message))

            # Get tools and system prompt
            tools = await self._get_tools(user_id)
            system_prompt = await self._build_system_prompt()

            # Format system prompt with context
            formatted_prompt = system_prompt
            formatted_prompt = formatted_prompt.replace(
                "{userInfo}", user_info or "Not provided"
            )
            formatted_prompt = formatted_prompt.replace(
                "{userStock}", user_stock or "Not provided"
            )
            formatted_prompt = formatted_prompt.replace(
                "{user_info}", user_info or "Not provided"
            )
            formatted_prompt = formatted_prompt.replace(
                "{user_stock}", user_stock or "Not provided"
            )

            # Create messages with system prompt
            messages_with_system = [
                SystemMessage(content=formatted_prompt)
            ] + chat_history

            # Bind tools to LLM
            llm_with_tools = self.llm.bind_tools(tools) if tools else self.llm

            # Stream the response
            full_content = ""
            is_tool_call = False
            accumulated_message = None
            usage_data = None  # Store usage data from OpenAI
            finish_reason = None  # Track finish reason

            async for chunk in llm_with_tools.astream(messages_with_system):
                logger.info(f"Chunk: {chunk}")
                # Accumulate the message chunks
                if accumulated_message is None:
                    accumulated_message = chunk
                else:
                    accumulated_message += chunk

                # Check for finish_reason in response_metadata
                if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                    finish_reason = chunk.response_metadata.get("finish_reason")

                # Check for usage data in chunk (OpenAI returns this in the final chunk)
                if hasattr(chunk, "usage_metadata"):
                    usage_data = chunk.usage_metadata
                elif (
                    hasattr(chunk, "response_metadata")
                    and "token_usage" in chunk.response_metadata
                ):
                    usage_data = chunk.response_metadata["token_usage"]

                # Check if this is a tool call (but don't yield yet, wait for complete accumulation)
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    is_tool_call = True

                # Stream content chunks
                elif hasattr(chunk, "content") and chunk.content:
                    content_chunk = chunk.content
                    full_content += content_chunk

                    # Yield each character/chunk as streaming_content
                    yield {
                        "type": "streaming_content",
                        "data": {"content": content_chunk},
                    }

            # After streaming is complete, check finish_reason
            # Emit final_message if finish_reason is 'stop' or 'tool_calls' with content
            if finish_reason in ["stop", "tool_calls"] and full_content:
                # Send final_message with the accumulated content
                yield {
                    "type": "final_message",
                    "data": {"content": full_content},
                }

            # After streaming is complete, yield tool calls if any
            if (
                is_tool_call
                and accumulated_message
                and hasattr(accumulated_message, "tool_calls")
            ):
                # Now we have the complete tool calls with all arguments
                tool_calls_data = []
                for tc in accumulated_message.tool_calls:
                    tool_calls_data.append({"name": tc["name"], "args": tc["args"]})

                # Save AI message with tool calls to conversation
                await self.conversation_service.add_ai_message_with_tool_calls(
                    conversation_id,
                    "",  # No content when calling tools
                    accumulated_message.tool_calls,  # Save the raw tool_calls
                )

                if tool_calls_data:
                    yield {
                        "type": "agent_thinking",
                        "data": {"tool_calls": tool_calls_data},
                    }
                
                # Don't emit final_message when finish_reason is 'tool_calls' 
                # since there's no content to send

            # If tool calls were made, execute them
            if (
                is_tool_call
                and accumulated_message
                and hasattr(accumulated_message, "tool_calls")
            ):
                # Use the accumulated message which has proper tool_call_ids
                messages_with_system.append(accumulated_message)

                # Execute tools via ToolNode if available, otherwise do a manual fallback
                if ToolNode is not None:
                    tool_node = ToolNode(tools)
                    tool_results = await tool_node.ainvoke(
                        {"messages": messages_with_system}, {"configurable": {}}
                    )
                else:
                    # Manual execution fallback producing a similar shape to ToolNode output
                    tool_messages = []
                    for tc in getattr(accumulated_message, "tool_calls", []) or []:
                        tool_name = tc.get("name")
                        tool_args = tc.get("args", {})
                        tool_id = tc.get("id")
                        selected = None
                        for t in tools:
                            try:
                                if getattr(t, "name", None) == tool_name:
                                    selected = t
                                    break
                            except Exception:
                                pass
                        if selected is None:
                            result_text = f"Tool '{tool_name}' not found"
                        else:
                            try:
                                # Prefer ainvoke if available
                                if hasattr(selected, "ainvoke"):
                                    result = await selected.ainvoke(tool_args)
                                elif hasattr(selected, "arun"):
                                    result = await selected.arun(**tool_args)
                                else:
                                    # Fallback to sync run in thread if needed
                                    result = selected.run(**tool_args)
                                result_text = (
                                    result
                                    if isinstance(result, str)
                                    else json.dumps(result, ensure_ascii=False)
                                )
                            except Exception as e:
                                result_text = f"Error running tool '{tool_name}': {e}"
                        tool_messages.append(
                            ToolMessage(content=result_text, tool_call_id=tool_id)
                        )
                    tool_results = {"messages": tool_messages}

                # Extract and yield tool results
                if "messages" in tool_results:
                    for msg in tool_results["messages"]:
                        if isinstance(msg, ToolMessage):
                            # Find the corresponding tool call
                            tool_name = None
                            for tc in accumulated_message.tool_calls:
                                if tc.get("id") == msg.tool_call_id:
                                    tool_name = tc.get("name")
                                    break

                            # Save tool result to conversation history
                            await self.conversation_service.add_tool_result(
                                conversation_id,
                                msg.tool_call_id,  # Save the actual tool_call_id
                                tool_name or "unknown",
                                msg.content,
                            )

                            # Yield tool result event
                            yield {
                                "type": "tool_result",
                                "data": {
                                    "tool": tool_name or "unknown",
                                    "result": msg.content,
                                },
                            }

                # Add tool results to messages
                messages_with_system.extend(tool_results["messages"])

                # Get final response after tool execution
                full_content = ""
                final_usage_data = None  # Store usage data for final response
                final_finish_reason = None  # Track finish reason for final response
                
                async for chunk in self.llm.astream(messages_with_system):
                    # Check for finish_reason in response_metadata
                    if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                        final_finish_reason = chunk.response_metadata.get("finish_reason")
                    
                    # Check for usage data in chunk
                    if hasattr(chunk, "usage_metadata"):
                        final_usage_data = chunk.usage_metadata
                    elif (
                        hasattr(chunk, "response_metadata")
                        and "token_usage" in chunk.response_metadata
                    ):
                        final_usage_data = chunk.response_metadata["token_usage"]

                    if hasattr(chunk, "content") and chunk.content:
                        content_chunk = chunk.content
                        full_content += content_chunk

                        yield {
                            "type": "streaming_content",
                            "data": {"content": content_chunk},
                        }
                
                # Emit final_message after tool execution response
                if final_finish_reason == "stop":
                    yield {
                        "type": "final_message", 
                        "data": {"content": full_content},
                    }

                # Merge usage data from both responses if tool was called
                if usage_data and final_usage_data:
                    usage_data = {
                        "input_tokens": usage_data.get("input_tokens", 0)
                        + final_usage_data.get("input_tokens", 0),
                        "output_tokens": usage_data.get("output_tokens", 0)
                        + final_usage_data.get("output_tokens", 0),
                        "total_tokens": usage_data.get("total_tokens", 0)
                        + final_usage_data.get("total_tokens", 0),
                    }
                elif final_usage_data:
                    usage_data = final_usage_data

            # Save conversation
            await self.conversation_service.add_message(
                conversation_id, "user", user_message
            )
            await self.conversation_service.add_message(
                conversation_id, "assistant", full_content
            )

            # Use actual usage data from OpenAI if available, otherwise fall back to estimate
            if usage_data:
                # Use the actual token counts from OpenAI API
                usage = {
                    "prompt_tokens": usage_data.get("input_tokens", 0),
                    "completion_tokens": usage_data.get("output_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                }
            else:
                # Fall back to estimation if no usage data is available
                prompt_tokens = sum(
                    len(m.content.split()) * 1.3 for m in messages_with_system
                )
                completion_tokens = len(full_content.split()) * 1.3
                usage = {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens),
                }

            # Yield final response with usage
            yield {
                "type": "final_response",
                "data": {
                    "conversation_id": conversation_id,
                    "content": full_content,
                    "usage": usage,
                },
            }

        except Exception as e:
            import traceback

            logger.error("Error in streaming agent chat: %s", str(e))
            logger.error("Full traceback: %s", traceback.format_exc())

            yield {
                "type": "error",
                "data": {
                    "conversation_id": conversation_id,
                    "content": f"I encountered an error processing your request: {str(e)}",
                    "usage": None,
                },
            }
