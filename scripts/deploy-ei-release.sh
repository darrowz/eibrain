#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  deploy-ei-release.sh <project> [source_dir] [deploy_root]

Environment:
  COPY_VENV=1  Copy <current>/.venv into the new release before rsync.

Examples:
  deploy-ei-release.sh eibrain /dev-project/eibrain /opt/eibrain
  COPY_VENV=1 deploy-ei-release.sh eimemory /dev-project/eimemory /opt/eimemory
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

project="${1:?project is required}"
source_dir="${2:-/dev-project/${project}}"
deploy_root="${3:-/opt/${project}}"

if [[ ! -d "${source_dir}/.git" ]]; then
  echo "source_dir must be a git repository: ${source_dir}" >&2
  exit 2
fi

short_sha="$(git -C "${source_dir}" rev-parse --short=7 HEAD)"
target="${deploy_root}/releases/${short_sha}"

case "${deploy_root}" in
  /opt/*) ;;
  *)
    echo "deploy_root must stay under /opt: ${deploy_root}" >&2
    exit 3
    ;;
esac

case "${target}" in
  "${deploy_root}/releases/"*) ;;
  *)
    echo "computed target escaped releases directory: ${target}" >&2
    exit 4
    ;;
esac

mkdir -p "${deploy_root}/releases" "${deploy_root}/logs" "${deploy_root}/run"

current_path=""
if [[ -e "${deploy_root}/current" || -L "${deploy_root}/current" ]]; then
  current_path="$(readlink -f "${deploy_root}/current" || true)"
fi

mkdir -p "${target}"

if [[ "${COPY_VENV:-0}" == "1" && -n "${current_path}" && -d "${current_path}/.venv" && ! -d "${target}/.venv" ]]; then
  cp -a "${current_path}/.venv" "${target}/.venv"
fi

rsync -a --delete \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.claude' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  "${source_dir}/" "${target}/"

if [[ -e "${deploy_root}/current" && ! -L "${deploy_root}/current" ]]; then
  legacy="${deploy_root}/releases/legacy-current-$(date +%Y%m%d%H%M%S)"
  mv "${deploy_root}/current" "${legacy}"
  echo "moved non-symlink current to ${legacy}"
fi

ln -sfn "${target}" "${deploy_root}/current"

cat <<EOF
project=${project}
source=${source_dir}
release=${short_sha}
target=${target}
current=$(readlink -f "${deploy_root}/current")
EOF
