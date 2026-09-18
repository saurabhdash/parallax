#!/usr/bin/env bash
# Create or update the isolated coordinator, learner, and sampler environments.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

uv sync --project "${HERE}"
uv sync --project "${HERE}/parallax/learner"
uv sync --project "${HERE}/parallax/sampler"
