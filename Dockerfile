FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ragas → scikit-network has no linux/aarch64 wheel; it compiles Cython + OpenMP.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY filing_rag/ filing_rag/
COPY config/ config/
RUN uv sync --frozen --no-dev

FROM python:3.14-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/filing_rag /app/filing_rag
COPY --from=builder /app/config /app/config

EXPOSE 8000

CMD ["filing-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
