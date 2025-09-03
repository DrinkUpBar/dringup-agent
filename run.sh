#!/usr/bin/env bash
set -Eeuo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
SKIP_SYNC="${SKIP_SYNC:-0}"

echo -e "${GREEN}Starting DrinkUp Agent...${NC}"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found!${NC}"
    echo "Creating .env from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}.env file created. Please update it with your configuration.${NC}"
        exit 1
    else
        echo -e "${RED}Error: .env.example not found!${NC}"
        exit 1
    fi
fi

# Set Python path to include src directory
export PYTHONPATH="${PYTHONPATH:-}:${PWD}/src"

# Optional checks (non-fatal)
if command -v redis-cli >/dev/null 2>&1; then
    if ! redis-cli ping >/dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Redis is not running. Conversation memory will not persist.${NC}"
        echo "To start Redis: redis-server"
    fi
else
    echo -e "${YELLOW}Note: redis-cli not found; skipping Redis check.${NC}"
fi

if grep -q "MEMGRAPH_PASSWORD=." .env 2>/dev/null; then
    if command -v nc >/dev/null 2>&1; then
        if ! nc -z localhost 7687 2>/dev/null; then
            echo -e "${YELLOW}Warning: Memgraph is configured but not running on port 7687.${NC}"
            echo "To start Memgraph with Docker:"
            echo "  docker run -p 7687:7687 memgraph/memgraph-mage:latest --schema-info-enabled=True"
        else
            echo -e "${GREEN}Memgraph detected on port 7687${NC}"
        fi
    else
        echo -e "${YELLOW}Note: 'nc' not found; skipping Memgraph port check.${NC}"
    fi
fi

run_with_uv() {
    echo "Using uv for dependency management and running."
    if [ "$SKIP_SYNC" != "1" ]; then
        echo "Syncing dependencies with uv... (set SKIP_SYNC=1 to skip)"
        uv sync --dev
    else
        echo "Skipping uv sync because SKIP_SYNC=1"
    fi
    echo -e "${GREEN}Starting server on http://${HOST}:${PORT}${NC}"
    echo "API docs: http://localhost:${PORT}/docs"
    uv run uvicorn drinkup_agent.main:app --host "${HOST}" --port "${PORT}" --reload
}

run_with_poetry() {
    echo "Using poetry for dependency management and running."
    if [ "$SKIP_SYNC" != "1" ]; then
        echo "Installing dependencies with poetry... (set SKIP_SYNC=1 to skip)"
        poetry install
    else
        echo "Skipping poetry install because SKIP_SYNC=1"
    fi
    echo -e "${GREEN}Starting server on http://${HOST}:${PORT}${NC}"
    echo "API docs: http://localhost:${PORT}/docs"
    poetry run uvicorn drinkup_agent.main:app --host "${HOST}" --port "${PORT}" --reload
}

run_with_venv() {
    local PY=python3
    if ! command -v python3 >/dev/null 2>&1 && command -v python >/dev/null 2>&1; then
        PY=python
    fi
    if [ ! -d .venv ]; then
        echo "Creating virtual environment in .venv ..."
        "$PY" -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    if [ "$SKIP_SYNC" != "1" ]; then
        echo "Installing dependencies with pip... (set SKIP_SYNC=1 to skip)"
        python -m pip install --upgrade pip
        # Prefer installing from pyproject (PEP 517)
        python -m pip install -e .
    else
        echo "Skipping pip install because SKIP_SYNC=1"
    fi
    echo -e "${GREEN}Starting server on http://${HOST}:${PORT}${NC}"
    echo "API docs: http://localhost:${PORT}/docs"
    python -m uvicorn drinkup_agent.main:app --host "${HOST}" --port "${PORT}" --reload
}

# Choose runner: uv -> poetry -> venv/pip
if command -v uv >/dev/null 2>&1; then
    run_with_uv
elif command -v poetry >/dev/null 2>&1; then
    run_with_poetry
else
    run_with_venv
fi
