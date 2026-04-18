#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "config=${ROOT_DIR}/config/eibrain.yaml"
test -f "${ROOT_DIR}/config/eibrain.yaml"
test -f "${ROOT_DIR}/.env.example"
python3 -m apps.bootstrap_deployment --config "${ROOT_DIR}/config/eibrain.yaml" >/dev/null
echo "deployment-check=ok"
