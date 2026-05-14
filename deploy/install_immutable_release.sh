#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/dev-project/eibrain}"
INSTALL_ROOT="${INSTALL_ROOT:-/opt/eibrain}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COMMIT="${1:-$(git -C "$REPO_DIR" rev-parse --short HEAD)}"
RELEASE_DIR="$INSTALL_ROOT/releases/$COMMIT"
CURRENT_LINK="$INSTALL_ROOT/current"
COPY_VENV="${COPY_VENV:-1}"

if [ ! -d "$REPO_DIR" ]; then
  echo "Repository path does not exist: $REPO_DIR" >&2
  exit 2
fi

if ! git -C "$REPO_DIR" rev-parse --verify "$COMMIT^{commit}" >/dev/null 2>&1; then
  echo "Unknown commit: $COMMIT" >&2
  exit 2
fi

mkdir -p "$INSTALL_ROOT/releases" "$INSTALL_ROOT/run" "$INSTALL_ROOT/logs" /etc/eibrain /var/lib/eibrain /var/log/eibrain

CURRENT_PATH=""
if [ -e "$CURRENT_LINK" ] || [ -L "$CURRENT_LINK" ]; then
  CURRENT_PATH="$(readlink -f "$CURRENT_LINK" || true)"
fi

if [ ! -d "$RELEASE_DIR" ]; then
  mkdir -p "$RELEASE_DIR"
  git -C "$REPO_DIR" archive "$COMMIT" | tar -C "$RELEASE_DIR" -xf -
fi

if [ "$COPY_VENV" = "1" ] && [ -n "$CURRENT_PATH" ] && [ -d "$CURRENT_PATH/.venv" ] && [ ! -d "$RELEASE_DIR/.venv" ]; then
  cp -a "$CURRENT_PATH/.venv" "$RELEASE_DIR/.venv"
fi

if [ ! -x "$RELEASE_DIR/.venv/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$RELEASE_DIR/.venv"
fi

"$RELEASE_DIR/.venv/bin/python" -m pip install --upgrade pip
"$RELEASE_DIR/.venv/bin/python" -m pip install "$RELEASE_DIR"

ln -sfn "$RELEASE_DIR" "$CURRENT_LINK.next"
mv -Tf "$CURRENT_LINK.next" "$CURRENT_LINK"

echo "release=$RELEASE_DIR"
echo "current=$CURRENT_LINK"
echo "commit=$COMMIT"
echo "copy_venv=$COPY_VENV"
