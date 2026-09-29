#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
"${PYTHON:-python3}" - <<'PYDEPS'
import importlib.util
import sys
modules = ['numpy', 'pandas', 'scipy', 'sklearn', 'nibabel', 'yaml', 'matplotlib']
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    sys.exit('Missing Python dependencies: ' + ', '.join(missing) + '. Install requirements.txt with the selected Python interpreter.')
PYDEPS
exec "${PYTHON:-python3}" "$PROJECT_ROOT/code/pilot.py" "$@"
