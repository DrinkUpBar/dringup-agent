"""Agent service that uses LangGraph for better control and streaming."""

from .langgraph_agent_service import LangGraphAgentService

# Use LangGraphAgentService as the main AgentService
AgentService = LangGraphAgentService
