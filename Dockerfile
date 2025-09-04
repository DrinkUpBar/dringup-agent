# syntax=docker/dockerfile:1.7-labs

# ---------- Builder stage ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for building wheels (uvicorn[standard] -> uvloop, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create a virtualenv for deterministic, relocatable deps
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:/root/.local/bin:$PATH"

# Copy only files needed for dependency resolution first for better cache
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install dependencies with binary wheels only
# 1) Export locked dependencies to requirements.txt
# 2) Install deps with --only-binary :all:
# 3) Install local project without deps (allows local wheel build)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev -o /tmp/requirements.txt \
 && uv pip install --python /opt/venv/bin/python --no-cache-dir --only-binary :all: -r /tmp/requirements.txt \
 && uv pip install --python /opt/venv/bin/python --no-cache-dir --no-deps .

# ---------- Runtime stage ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8001 \
    HOST=0.0.0.0

WORKDIR /app

# Copy venv with installed dependencies from builder
COPY --from=builder /opt/venv /opt/venv

# App code is already installed into /opt/venv during build; no source copy needed

# Create a non-root user
RUN useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# Default command (honors PORT env)
CMD ["/bin/sh", "-c", "uvicorn drinkup_agent.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
