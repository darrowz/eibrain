#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m apps.bootstrap_deployment --config "${1:-${ROOT_DIR}/config/eibrain.yaml}"
