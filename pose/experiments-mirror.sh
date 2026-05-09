#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

jupyter nbconvert \
  --to notebook \
  --execute pose/mirror.ipynb \
  --inplace
