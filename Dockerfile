FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
COPY packages packages
RUN uv sync --frozen --no-dev --package fish-audio-suite-proxy
EXPOSE 8849
CMD ["uv", "run", "--frozen", "--no-dev", "--package", "fish-audio-suite-proxy", "fish-audio-suite-proxy"]
