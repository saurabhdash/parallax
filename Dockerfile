FROM ghcr.io/nvidia/jax:jax-2025-07-06@sha256:d1faa4714a40ab512184fb471fc039972b71f82dd3370f377d31fbcc37fabfe4 AS base

RUN python -m pip install --no-cache-dir uv==0.12.16

FROM base AS parallax

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app UV_LINK_MODE=hardlink

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY parallax/learner/pyproject.toml parallax/learner/uv.lock ./parallax/learner/
COPY parallax/sampler/pyproject.toml parallax/sampler/uv.lock ./parallax/sampler/
RUN uv sync --locked --no-install-project && \
    cd parallax/learner && uv sync --locked --no-install-project && \
    cd ../sampler && uv sync --locked --no-install-project && \
    uv cache clean

COPY . .

RUN uv sync --locked && \
    cd parallax/learner && uv sync --locked && \
    cd ../sampler && uv sync --locked && \
    uv cache clean

ENTRYPOINT ["/app/.venv/bin/python", "main.py"]
