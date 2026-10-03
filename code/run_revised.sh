#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
exec "${PYTHON:-python3}" "$PROJECT_ROOT/code/revised_pilot.py" "$@"
