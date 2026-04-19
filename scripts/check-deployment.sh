#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m apps.check_deployment --config "${ROOT_DIR}/config/eibrain.yaml"
