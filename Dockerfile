FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
COPY packages packages
RUN uv sync --frozen --no-dev --no-editable --package fish-audio-suite-proxy \
    && useradd --system --no-create-home --uid 10001 fish
ENV PATH="/app/.venv/bin:$PATH" \
    FISH_PROXY_HOST=0.0.0.0 \
    FISH_PROXY_PORT=8849
USER fish
EXPOSE 8849
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"FISH_PROXY_PORT\"]}/health', timeout=4)"]
CMD ["fish-audio-suite-proxy"]
