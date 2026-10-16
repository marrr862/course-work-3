#!/bin/bash
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python -m uvicorn fakenews.app:app --host 127.0.0.1 --port 8000
