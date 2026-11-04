#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ -x .venv-voice/bin/python ]; then
    exec .venv-voice/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${ROBOT_PORT:-8000}"
fi
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${ROBOT_PORT:-8000}"
