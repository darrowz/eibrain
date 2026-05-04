#!/usr/bin/env bash
set -eu
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MONITOR_BASE_URL="${EIHEAD_MONITOR_URL:-http://127.0.0.1:18080}"

SERVICES="
eibrain-body.service
eibrain-cognitive.service
eibrain-monitor.service
eibrain-vision-hailo.service
eihead-runtime.service
eihead-monitor.service
eihead-vision.service
eihead-audio.service
eihead-neck.service
"

section() {
  printf '\n## %s\n' "$1"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

run_command() {
  label="$1"
  shift
  section "$label"
  if [ "$#" -eq 0 ]; then
    echo "missing command arguments"
    return 0
  fi
  if ! have "$1"; then
    echo "missing command: $1"
    return 0
  fi
  "$@" 2>&1 || {
    code="$?"
    echo "non-fatal exit=${code}"
  }
}

check_device() {
  device_path="$1"
  if [ -e "$device_path" ]; then
    ls -l "$device_path" 2>&1 || echo "non-fatal: cannot stat ${device_path}"
  else
    echo "missing: ${device_path}"
  fi
}

curl_endpoint() {
  path="$1"
  if ! have curl; then
    echo "missing command: curl"
    return 0
  fi
  url="${MONITOR_BASE_URL}${path}"
  echo "\$ curl -fsS --max-time 3 ${url}"
  curl -fsS --max-time 3 "$url" 2>&1 || {
    code="$?"
    echo "non-fatal curl exit=${code}"
  }
}

section "baseline metadata"
run_command "hostname" hostname
run_command "date" date -Is
run_command "kernel" uname -a

section "git commit"
if have git && [ -d "${REPO_ROOT}/.git" ]; then
  git -C "$REPO_ROOT" rev-parse --short HEAD 2>&1 || echo "non-fatal: git commit unavailable"
  git -C "$REPO_ROOT" status --short 2>&1 || echo "non-fatal: git status unavailable"
else
  echo "missing command or repo: git"
fi

section "systemctl service discovery"
if have systemctl; then
  systemctl --user list-units --type=service --all 'eibrain*' 'eihead*' 2>&1 || echo "non-fatal: user service discovery unavailable"
  systemctl list-units --type=service --all 'eibrain*' 'eihead*' 2>&1 || echo "non-fatal: system service discovery unavailable"
else
  echo "missing command: systemctl"
fi

for service in $SERVICES; do
  run_command "systemctl --user status ${service}" systemctl --user status "$service" --no-pager --lines=20
  run_command "systemctl status ${service}" systemctl status "$service" --no-pager --lines=20
done

section "listen ports"
if have ss; then
  ss -ltnup 2>&1 || echo "non-fatal: ss listen probe failed"
elif have netstat; then
  netstat -ltnup 2>&1 || echo "non-fatal: netstat listen probe failed"
else
  echo "missing command: ss or netstat"
fi

section "device nodes"
check_device /dev/video0
check_device /dev/hailo0
check_device /dev/i2c-1
check_device /dev/snd

section "camera and audio discovery"
run_command "v4l2 devices" v4l2-ctl --list-devices
run_command "arecord devices" arecord -l
run_command "aplay devices" aplay -l

section "vision state files"
check_device /tmp/eibrain-vision/state.json
check_device /tmp/eibrain-vision/latest.jpg

section "web monitor health/status"
curl_endpoint /health
curl_endpoint /status
curl_endpoint /api/health
curl_endpoint /api/status
curl_endpoint /metrics

section "recent journal tails"
if have journalctl; then
  for service in $SERVICES; do
    echo "\$ journalctl --user -u ${service} -n 80 --no-pager"
    journalctl --user -u "$service" -n 80 --no-pager 2>&1 || echo "non-fatal: user journal unavailable for ${service}"
    echo "\$ journalctl -u ${service} -n 80 --no-pager"
    journalctl -u "$service" -n 80 --no-pager 2>&1 || echo "non-fatal: system journal unavailable for ${service}"
  done
else
  echo "missing command: journalctl"
fi

section "manual follow-up hints"
echo "Compare this output before and after eihead cutover."
echo "Record honxin source path: /dev-project/eibrain and future /dev-project/eihead."
echo "Record honjia deployment path from active service ExecStart and config deployment.root_dir."
