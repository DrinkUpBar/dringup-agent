"""LangGraph-based agent service with streaming and intermediate tool execution display."""

import json
import uuid
from typing import (
    List,
    Optional,
    Dict,
    Any,
    TypedDict,
    Annotated,
    Sequence,
    AsyncIterator,
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import logging

from ..config import settings
from ..tools.mem0_tools import create_mem0_tools
from ..repositories import PromptRepository
from ..services.conversation_service import ConversationService

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State of the agent during execution."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_stock: Optional[str]
    user_info: Optional[str]
    current_tool: Optional[str]
    tool_results: List[Dict[str, Any]]
    token_usage: Dict[str, Any]


class LangGraphAgentService:
    """Service for managing LangGraph agents with streaming tool execution."""

    def __init__(self):
        """Initialize the agent service."""
        self.prompt_repository = PromptRepository()
        self.conversation_service = ConversationService()
        self.model = settings.openai_model

        # Initialize LLM
        llm_params = {
            "model": self.model,
            "temperature": 0.7,
            "openai_api_key": settings.openai_api_key,
            "streaming": True,  # Enable streaming for real-time character output
        }

        # Add base URL if configured
        if settings.openai_base_url:
            llm_params["base_url"] = settings.openai_base_url

        self.llm = ChatOpenAI(**llm_params)

        # Also create a non-streaming version for token counting
        non_streaming_params = llm_params.copy()
        non_streaming_params["streaming"] = False
        self.llm_non_streaming = ChatOpenAI(**non_streaming_params)

    async def _get_tools(self, user_id: str) -> List[BaseTool]:
        """Get all available tools for the agent."""
        tools = []

        # Add Mem0 tools if configured
        if settings.openai_api_key:
            mem0_tools = create_mem0_tools(
                api_key=settings.openai_api_key, user_id=user_id
            )
            tools.extend(mem0_tools)
            logger.info(f"Added {len(mem0_tools)} Mem0 tools for user {user_id}")

        return tools

    async def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        db_prompt = await self.prompt_repository.get_chat_prompt()

        default_prompt = """You are a knowledgeable and friendly bartender assistant for DrinkUp.
You help users with cocktail recommendations, recipes, and bar management.

You have access to memory tools to remember user preferences and past conversations:
- Use 'add_memory' to store important information about the user
- Use 'search_memory' to recall relevant information when answering questions
- Use 'get_all_memories' to see everything you know about the user
- Use 'update_memory' to correct or update stored information
- Use 'delete_memory' to remove outdated information

Always:
1. Search for relevant memories when starting a conversation or when context would be helpful
2. Store new important information shared by the user (preferences, restrictions, favorites, etc.)
3. Update memories when users correct or change their preferences
4. Provide personalized responses based on stored memories

Current context:
- User Stock: {user_stock}
- User Info: {user_info}

Always respond in JSON format with a 'response' field containing your message."""

        return db_prompt if db_prompt else default_prompt

    def create_graph(self, tools: List[BaseTool], system_prompt: str) -> StateGraph:
        """Create the LangGraph state machine."""

        # Bind tools to the LLM
        llm_with_tools = self.llm.bind_tools(tools)

        # Create tool node from the tools
        tool_node_executor = ToolNode(tools)

        # Define the agent node
        async def agent_node(state: AgentState) -> Dict[str, Any]:
            """Agent decides what to do next."""
            # Format system prompt with context
            # Use string replacement instead of format() to avoid issues with JSON examples in prompt
            formatted_prompt = system_prompt

            # Replace camelCase placeholders (from DB)
            formatted_prompt = formatted_prompt.replace(
                "{userInfo}", state.get("user_info", "Not provided")
            )
            formatted_prompt = formatted_prompt.replace(
                "{userStock}", state.get("user_stock", "Not provided")
            )

            # Replace snake_case placeholders (from code)
            formatted_prompt = formatted_prompt.replace(
                "{user_info}", state.get("user_info", "Not provided")
            )
            formatted_prompt = formatted_prompt.replace(
                "{user_stock}", state.get("user_stock", "Not provided")
            )

            # Create messages with system prompt
            messages = [SystemMessage(content=formatted_prompt)] + list(
                state["messages"]
            )

            # Get response from LLM
            response = await llm_with_tools.ainvoke(messages)

            # Extract token usage if available
            token_usage = {}
            if hasattr(response, "response_metadata"):
                metadata = response.response_metadata
                if "token_usage" in metadata:
                    token_usage = metadata["token_usage"]
                elif "usage" in metadata:
                    token_usage = metadata["usage"]

            return {"messages": [response], "token_usage": token_usage}

        # Use the built-in ToolNode for tool execution
        # We'll wrap it to add logging
        async def tool_node_wrapper(state: AgentState) -> Dict[str, Any]:
            """Wrapper around ToolNode to add logging."""
            messages = state["messages"]
            last_message = messages[-1]

            tool_results = []

            # Log tool calls
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    logger.info(
                        f"Executing tool: {tool_call['name']} with args: {tool_call['args']}"
                    )
                    tool_results.append(
                        {
                            "tool": tool_call["name"],
                            "args": tool_call["args"],
                            "status": "executing",
                        }
                    )

            # Execute tools using the built-in ToolNode
            # ToolNode.ainvoke expects (state, config)

            config = {"configurable": {}}
            result = await tool_node_executor.ainvoke(state, config)

            # Extract tool results from messages
            if "messages" in result:
                for msg in result["messages"]:
                    if isinstance(msg, ToolMessage):
                        # Find the corresponding tool call
                        for tr in tool_results:
                            if tr["status"] == "executing":
                                tr["result"] = msg.content
                                tr["status"] = (
                                    "success"
                                    if not msg.content.startswith("Error:")
                                    else "error"
                                )
                                break

            return {
                "messages": result.get("messages", []),
                "tool_results": state.get("tool_results", []) + tool_results,
            }

        # Define the conditional edge function
        def should_continue(state: AgentState) -> str:
            """Determine if we should continue or end."""
            messages = state["messages"]
            last_message = messages[-1]

            # If there are tool calls, execute them
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            # Otherwise, we're done
            return "end"

        # Build the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", agent_node)
        # Use ToolNode directly as a node - it handles execution properly
        workflow.add_node("tools", tool_node_executor)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add edges
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )

        # After tools, always go back to agent
        workflow.add_edge("tools", "agent")

        return workflow.compile()

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
        Process a chat message using LangGraph with streaming.
        Yields intermediate results including tool executions.
        """
        # Generate conversation ID if not provided
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
                    chat_history.append(AIMessage(content=msg["content"]))

            # Handle image attachments
            if image_attachments:
                image_note = f" [User shared {len(image_attachments)} image(s)]"
                user_message = user_message + image_note

            # Add user message to history
            chat_history.append(HumanMessage(content=user_message))

            # Get tools and system prompt
            tools = await self._get_tools(user_id)
            system_prompt = await self._build_system_prompt()

            # Create the graph
            app = self.create_graph(tools, system_prompt)

            # Initial state
            initial_state = {
                "messages": chat_history,
                "user_stock": user_stock,
                "user_info": user_info,
                "tool_results": [],
                "token_usage": {},
            }

            # Stream execution
            async for event in app.astream(initial_state):
                # Yield intermediate updates
                for node, state_update in event.items():
                    if node == "tools" and "tool_results" in state_update:
                        # Yield tool execution results
                        for tool_result in state_update["tool_results"]:
                            if tool_result not in initial_state.get("tool_results", []):
                                yield {"type": "tool_execution", "data": tool_result}

                    elif node == "agent" and "messages" in state_update:
                        # Check if this is the final response
                        last_message = state_update["messages"][-1]
                        if (
                            not hasattr(last_message, "tool_calls")
                            or not last_message.tool_calls
                        ):
                            # Final response from agent
                            content = last_message.content

                            # Save conversation
                            await self.conversation_service.add_message(
                                conversation_id, "user", user_message
                            )
                            await self.conversation_service.add_message(
                                conversation_id, "assistant", content
                            )

                            # Return raw content without JSON wrapper
                            # Parse JSON if it's already in JSON format, otherwise use raw content
                            try:
                                parsed_content = json.loads(content)
                                # If it's our old format with 'response' field, extract it
                                if (
                                    isinstance(parsed_content, dict)
                                    and "response" in parsed_content
                                ):
                                    response_content = parsed_content["response"]
                                else:
                                    response_content = content
                            except json.JSONDecodeError:
                                # Not JSON, use raw content
                                response_content = content

                            # Get token usage from state
                            token_usage = state_update.get("token_usage", {})
                            usage_data = {
                                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                                "completion_tokens": token_usage.get(
                                    "completion_tokens", 0
                                ),
                                "total_tokens": token_usage.get("total_tokens", 0),
                            }

                            yield {
                                "type": "final_response",
                                "data": {
                                    "conversation_id": conversation_id,
                                    "content": response_content,
                                    "usage": usage_data,
                                },
                            }
                        else:
                            # Agent is calling tools
                            yield {
                                "type": "agent_thinking",
                                "data": {
                                    "tool_calls": [
                                        {"name": tc["name"], "args": tc["args"]}
                                        for tc in last_message.tool_calls
                                    ]
                                },
                            }

        except Exception as e:
            import traceback

            logger.error("Error in agent chat: %s", str(e))
            logger.error("Full traceback: %s", traceback.format_exc())

            # Return error response
            error_response = json.dumps(
                {
                    "response": "I encountered an error processing your request. Please try again.",
                    "error": str(e),
                }
            )

            yield {
                "type": "error",
                "data": {
                    "conversation_id": conversation_id,
                    "content": error_response,
                    "usage": None,
                },
            }

    async def chat_with_agent(
        self,
        user_message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        user_stock: Optional[str] = None,
        user_info: Optional[str] = None,
        image_attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat message using LangGraph (non-streaming version).
        Returns the final response with tool execution details.
        """
        tool_executions = []
        final_response = None

        # Collect all events from the stream
        async for event in self.chat_with_agent_stream(
            user_message=user_message,
            user_id=user_id,
            conversation_id=conversation_id,
            user_stock=user_stock,
            user_info=user_info,
            image_attachments=image_attachments,
        ):
            if event["type"] == "tool_execution":
                tool_executions.append(event["data"])
            elif event["type"] == "final_response":
                final_response = event["data"]
            elif event["type"] == "error":
                return event["data"]

        # Add tool executions to the response
        if final_response and tool_executions:
            final_response["tool_executions"] = tool_executions

        return final_response
