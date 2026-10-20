#!/usr/bin/env bash
set -uo pipefail

BRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_ROOT="$BRAIN_ROOT/02_仿真抓取"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/root/gpufree-data/RoboTwin}"
RUNS_ROOT="$DEMO_ROOT/最近运行结果/runs"
mkdir -p "$RUNS_ROOT"

exec 9>"$DEMO_ROOT/brain_twin_demo.lock"
if ! flock -n 9; then
    echo '已有 RoboBrain 仿真 Demo 正在运行，请先结束上一实例。' >&2
    exit 2
fi

RUN_DIR="$RUNS_ROOT/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$DEMO_ROOT/最近运行结果/latest"

REQUESTED_MODE=""
for arg in "$@"; do
    case "$arg" in
        --viewer|--headless) REQUESTED_MODE="$arg" ;;
    esac
done
ARGS=("--output-dir" "$RUN_DIR")
if [[ -n "$REQUESTED_MODE" ]]; then
    ARGS+=("$REQUESTED_MODE")
elif [[ "${ROBOTWIN_HEADLESS:-0}" == 1 ]]; then
    ARGS+=(--headless)
elif [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    ARGS+=(--viewer)
else
    ARGS+=(--headless)
fi
for arg in "$@"; do
    case "$arg" in
        --viewer|--headless) ;;
        *) ARGS+=("$arg") ;;
    esac
done

echo "RoboBrain 主框架 → RoboTwin 环境 → SAPIEN Viewer"
echo "运行结果：$RUN_DIR"
ROBOTWIN_ROOT="$ROBOTWIN_ROOT" \
  /opt/conda/envs/twin_env/bin/python -u \
  "$DEMO_ROOT/demo/brain_twin_pipeline.py" "${ARGS[@]}" 2>&1 | tee "$RUN_DIR/run.log"
STATUS=${PIPESTATUS[0]}
if [[ $STATUS == 0 ]]; then
    echo "验证通过：$RUN_DIR/report.json"
else
    echo "验证失败（退出码 $STATUS），请检查：$RUN_DIR/run.log" >&2
fi
exit "$STATUS"
