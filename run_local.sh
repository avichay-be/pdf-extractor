#!/bin/bash
# Run the application locally
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

if [ -x ".venv/bin/python" ]; then
  exec .venv/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port 8002
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8002
fi

echo "Python environment not found. Create/install dependencies first, for example:" >&2
echo "  python3 -m venv .venv" >&2
echo "  .venv/bin/pip install -r requirements-dev.txt" >&2
exit 1
