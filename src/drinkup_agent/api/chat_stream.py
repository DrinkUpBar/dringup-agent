"""Streaming chat API endpoint for real-time tool execution display."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
import json
import logging

from ..models.chat import ChatV2Request, ChatParams
from ..services.langgraph_agent_service_streaming import StreamingLangGraphAgentService
from ..services.conversation_service import ConversationService
from ..tools.mem0_tools import Mem0Manager
from ..config import settings

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
    # 在函数开始时就打印请求体信息
    logger.info("=" * 50)
    logger.info("Received POST request to /api/workflow/chat/v2/stream")
    logger.info(f"Request type: {type(request)}")
    logger.info(
        f"Request keys: {list(request.keys()) if isinstance(request, dict) else 'Not a dict'}"
    )
    logger.info(
        f"Raw request body: {json.dumps(request, indent=2, ensure_ascii=False)}"
    )
    logger.info("=" * 50)

    agent_service = StreamingLangGraphAgentService()

    async def event_generator():
        """Generate SSE events from the agent stream."""
        try:
            # 详细记录请求结构分析
            logger.info("Analyzing request structure:")
            logger.info(
                f"- userMessage/user_message: {request.get('userMessage', request.get('user_message', 'NOT_FOUND'))}"
            )
            logger.info(
                f"- userId/user_id: {request.get('userId', request.get('user_id', 'NOT_FOUND'))}"
            )
            logger.info(
                f"- conversationId/conversation_id: {request.get('conversationId', request.get('conversation_id', 'NOT_FOUND'))}"
            )
            logger.info(f"- params: {request.get('params', 'NOT_FOUND')}")

            # Convert to ChatV2Request model
            # Handle camelCase to snake_case conversion
            logger.info("Converting to ChatV2Request model...")
            try:
                chat_request = ChatV2Request(
                    user_message=request.get(
                        "userMessage", request.get("user_message", "")
                    ),
                    user_id=request.get(
                        "userId", request.get("user_id", "default_user")
                    ),
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
                logger.info("Successfully converted to ChatV2Request model")
                logger.info(f"Converted model: {chat_request.model_dump()}")
            except Exception as conversion_error:
                logger.error(
                    f"Error converting to ChatV2Request model: {conversion_error}"
                )
                logger.error(f"Conversion error type: {type(conversion_error)}")
                import traceback

                logger.error(f"Conversion traceback: {traceback.format_exc()}")
                raise

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
                logger.info(f"Event: {event}")
                # Format as Server-Sent Event
                event_type = event["type"]
                event_data = json.dumps(event["data"], ensure_ascii=False)

                yield f"event: {event_type}\n"
                yield f"data: {event_data}\n\n"

                # If this is the final response or error, we're done
                if event_type in ["final_response", "error"]:
                    break

        except Exception as e:
            import traceback

            logger.error("=" * 50)
            logger.error("ERROR in chat stream:")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error(
                f"Original request that caused error: {json.dumps(request, indent=2, ensure_ascii=False)}"
            )
            logger.error("=" * 50)

            error_event = {
                "type": "error",
                "data": {
                    "error": str(e),
                    "error_type": str(type(e)),
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


@router.post("/workflow/conversation/save-to-memory")
async def save_conversation_to_memory(request: Dict[str, Any]):
    """
    Save all messages from a conversation to user's mem0 memory.

    Request body:
    {
        "conversationId": "conversation_id",
        "userId": "user_id"
    }

    Returns:
    {
        "status": "success",
        "messages_saved": <number of messages>,
        "memory_ids": [<list of created memory IDs>]
    }
    """
    try:
        # Extract parameters
        conversation_id = request.get("conversationId", request.get("conversation_id"))
        user_id = request.get("userId", request.get("user_id"))

        if not conversation_id:
            raise HTTPException(status_code=400, detail="conversationId is required")
        if not user_id:
            raise HTTPException(status_code=400, detail="userId is required")

        logger.info(
            f"Saving conversation {conversation_id} to memory for user {user_id}"
        )

        # Get conversation history
        conversation_service = ConversationService()
        messages = await conversation_service.get_messages(conversation_id)

        if not messages:
            return {
                "status": "success",
                "messages_saved": 0,
                "memory_ids": [],
                "message": "No messages found in conversation",
            }

        # Initialize Mem0 manager
        # Prepare vector store configuration for Milvus
        vector_store_config = {
            "type": "milvus",
            "url": settings.milvus_url,
            "token": settings.milvus_token,
            "collection_name": settings.milvus_collection_name,
            "db_name": settings.milvus_db_name,
            "embedding_model": settings.embedding_model,
            "embedding_dims": settings.embedding_dims,
        }

        mem0_manager = Mem0Manager(settings.openai_api_key, vector_store_config)

        if not mem0_manager.is_enabled():
            raise HTTPException(
                status_code=500, detail="Memory service is not available"
            )

        # Add all messages directly to memory
        try:
            # Pass all messages directly to mem0
            result = mem0_manager.memory.add(messages=messages, user_id=user_id)

            # Extract memory IDs from result
            memory_ids = []
            if result:
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and "id" in item:
                            memory_ids.append(item["id"])
                elif isinstance(result, dict) and "id" in result:
                    memory_ids.append(result["id"])

            messages_saved = len(messages)

        except Exception as e:
            logger.error(f"Error adding messages to memory: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to save messages to memory: {str(e)}"
            )

        return {
            "status": "success",
            "messages_saved": messages_saved,
            "memory_ids": memory_ids,
            "message": f"Successfully saved {messages_saved} messages to memory",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in save_conversation_to_memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))
