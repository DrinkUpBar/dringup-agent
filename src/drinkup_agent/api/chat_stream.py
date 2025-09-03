"""Streaming chat API endpoint for real-time tool execution display."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import Dict, Any
import json
import logging

from ..models.chat import ChatV2Request, ChatParams
from ..services.langgraph_agent_service_streaming import StreamingLangGraphAgentService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/workflow/chat/v2/stream")
async def chat_stream(request: Dict[str, Any]):
    """
    Stream chat responses with intermediate tool execution details.

    Returns Server-Sent Events (SSE) stream with:
    - tool_execution: When a tool is being executed
    - agent_thinking: When agent decides which tool to call
    - final_response: The final response from the agent
    - error: If an error occurs
    """
    agent_service = StreamingLangGraphAgentService()

    async def event_generator():
        """Generate SSE events from the agent stream."""
        try:
            # Log raw request to understand structure
            logger.info(f"Raw streaming request: {json.dumps(request, indent=2)}")

            # Convert to ChatV2Request model
            # Handle camelCase to snake_case conversion
            chat_request = ChatV2Request(
                user_message=request.get(
                    "userMessage", request.get("user_message", "")
                ),
                user_id=request.get("userId", request.get("user_id", "default_user")),
                conversation_id=request.get(
                    "conversationId", request.get("conversation_id")
                ),
                params=ChatParams(
                    user_stock=request.get("params", {}).get("userStock", ""),
                    user_info=request.get("params", {}).get("userInfo", ""),
                    image_attachment_list=request.get("params", {}).get(
                        "imageAttachmentList"
                    ),
                )
                if "params" in request
                else ChatParams(),
            )

            # Extract request parameters
            user_message = chat_request.user_message
            user_id = chat_request.user_id
            conversation_id = chat_request.conversation_id

            # Get user context from params
            params = chat_request.params
            user_stock = params.user_stock
            user_info = params.user_info

            # Get image attachments if any
            image_attachments = params.image_attachment_list

            # Stream responses from agent
            async for event in agent_service.chat_with_agent_stream(
                user_message=user_message,
                user_id=user_id,
                conversation_id=conversation_id,
                user_stock=user_stock,
                user_info=user_info,
                image_attachments=image_attachments,
            ):
                # Format as Server-Sent Event
                event_type = event["type"]
                event_data = json.dumps(event["data"], ensure_ascii=False)

                yield f"event: {event_type}\n"
                yield f"data: {event_data}\n\n"

                # If this is the final response or error, we're done
                if event_type in ["final_response", "error"]:
                    break

        except Exception as e:
            logger.error(f"Error in chat stream: {e}")
            error_event = {
                "type": "error",
                "data": {
                    "error": str(e),
                    "message": "An error occurred while processing your request",
                },
            }
            yield "event: error\n"
            yield f"data: {json.dumps(error_event['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )
