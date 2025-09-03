# DrinkUp Agent

AI Agent service for DrinkUp chatbot and AI interactions. This service handles all AI-related functionality including chat conversations, memory management, and integration with OpenAI using LangGraph for advanced agent workflows.

## Features

- **Streaming Chat API v2** - Real-time streaming responses with Server-Sent Events (SSE)
- **LangGraph Integration** - Advanced agent workflows with tool execution
- **Memory Integration** - Mem0 integration for long-term memory and user preferences
- **OpenAI Integration** - Using OpenAI's latest models with streaming support
- **Tool Execution** - Backend API integration and memory management tools
- **MySQL Integration** - Reads prompts and configuration from DrinkUp database
- **Conversation Management** - Persistent conversation history

## Installation

1. Install dependencies using uv (recommended):
```bash
uv install
```

Or using Poetry:
```bash
poetry install
```

2. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

3. Edit `.env` with your configuration:
- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `OPENAI_BASE_URL` - Custom OpenAI API endpoint (optional, for using OpenAI-compatible services)
- `MEM0_API_KEY` - Your Mem0 API key (optional)
- `REDIS_HOST/PORT` - Redis configuration (optional, falls back to in-memory)
- `MYSQL_HOST/PORT/DATABASE/USER/PASSWORD` - MySQL configuration (uses same database as DrinkUp)

## Running the Service

### Using the run script:
```bash
./run.sh
```

### Using uv directly:
```bash
uv run uvicorn src.drinkup_agent.main:app --host 0.0.0.0 --port 8001 --reload
```

### Using Python module:
```bash
uv run python -m src.drinkup_agent.main
```

## API Endpoints

### Streaming Chat v2 (Real-time with Server-Sent Events)
- **POST** `/api/workflow/chat/v2/stream`
- **Content-Type**: `application/json`
- **Response**: Server-Sent Events (SSE) stream
- Request body:
```json
{
  "userMessage": "Recommend a cocktail for tonight",
  "userId": "user123",
  "conversationId": "optional-uuid",
  "params": {
    "userStock": "Gin, Vodka, Rum, Tequila, Triple sec, Lime juice, Lemon juice, Simple syrup",
    "userInfo": "user_id: 123"
  }
}
```

#### Response Events:
- `agent_thinking`: AI planning and tool selection
- `tool_result`: Tool execution results
- `streaming_content`: Real-time response content
- `final_response`: Complete response with usage statistics
- `error`: Error information

### Non-streaming Chat v2 (Legacy)
- **POST** `/api/workflow/chat/v2`
- Same request format, returns complete JSON response

### Health Check
- **GET** `/health`
- **GET** `/api/workflow/chat/health`

## Integration with DrinkUp Backend

To integrate with the Java backend, update the backend configuration to point to this service:

1. In the Java backend, update the chat service to call this API:
   - Streaming Chat v2 endpoint: `http://localhost:8001/api/workflow/chat/v2/stream`
   - Non-streaming endpoint: `http://localhost:8001/api/workflow/chat/v2`

2. The Java backend should handle:
   - User authentication
   - Business logic and data validation
   - Frontend WebSocket connections (for streaming)
   - Database operations for user data

3. This Python service handles:
   - LangGraph agent workflows
   - AI chat interactions with streaming
   - Tool execution (memory search, backend API calls)
   - OpenAI API calls with streaming support
   - Memory management (Mem0)
   - Prompt templates from MySQL database

## Development

### Code linting and formatting:
```bash
# Lint and auto-fix issues
uv run ruff check --fix

# Format code
uv run ruff format
```

### Running tests:
```bash
uv run pytest
```

### Interactive Testing:
Open `chat_test.html` in your browser to test the streaming API with a chat interface.

## Architecture

```
drinkup-agent/
├── src/drinkup_agent/
│   ├── api/                    # FastAPI routes and streaming endpoints
│   ├── database/              # Database models and connections
│   ├── models/                # Pydantic models
│   ├── repositories/          # Data access layer
│   ├── services/              # Business logic and LangGraph agents
│   ├── tools/                 # LangGraph tools (Mem0, backend API)
│   ├── config.py             # Configuration management
│   └── main.py               # FastAPI application entry
├── chat_test.html            # Interactive test interface
├── run.sh                    # Development server script
├── .env.example              # Environment template
├── pyproject.toml           # uv/Poetry configuration
└── README.md                # This file
```

## Key Components

- **LangGraph Agent Service**: Core agent with streaming support and tool execution
- **Mem0 Tools**: Memory search and storage integration
- **Backend Tools**: DrinkUp API integration
- **Streaming API**: Real-time Server-Sent Events responses
- **Database Integration**: MySQL connection for prompts and configuration

## License

Private - DrinkUp Team