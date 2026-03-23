FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install system dependencies for data science
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
