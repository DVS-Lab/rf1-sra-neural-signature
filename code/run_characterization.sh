#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
# Avoid nested BLAS oversubscription in repeated small-sample, high-dimensional fits.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "${PYTHON:-python3}" "$PROJECT_ROOT/code/characterize_v4.py" "$@"
