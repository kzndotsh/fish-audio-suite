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
# The proxy listens on every interface inside the container so a published
# port can reach it. It has no client auth unless FISH_PROXY_API_KEYS is set.
# Publish on loopback (-p 127.0.0.1:8849:8849) or set FISH_PROXY_API_KEYS
# before publishing on another address.
# On stop the proxy lets in-flight replies finish for up to
# FISH_PROXY_GRACEFUL_SHUTDOWN seconds (default 120). Docker kills a container
# after 10 seconds unless told otherwise, so run it with --stop-timeout 130
# (compose: stop_grace_period: 130s), which is that drain plus 10 seconds.
ENV PATH="/app/.venv/bin:$PATH" \
    FISH_PROXY_HOST=0.0.0.0 \
    FISH_PROXY_PORT=8849
USER fish
EXPOSE 8849
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; from fish_audio_suite_kit import env_int; p = env_int('FISH_PROXY_PORT', 8849); urllib.request.urlopen(f'http://127.0.0.1:{p if 1 <= p <= 65535 else 8849}/health', timeout=4)"]
CMD ["fish-audio-suite-proxy"]
