#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# FlashKDAGrad TTK entry. Activate CANN/Conda before invoking this script.
# Examples:
#   bash run_one.sh validate
#   bash run_one.sh -t kdag_stc_109
#   KDA_MANUAL_DATA=/path/to/data bash run_one.sh replay -t kdag_stc_109
#   KDA_PROF=true bash run_one.sh -t kdag_stc_109
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash run_one.sh [run|l2|validate|list|exec|dump|replay|xpu] [TTK arguments]
Default mode: run, using native cross_check L2 and all 130 STC cases.
KDA_CASES: CSV; KDA_CASE: case filter; KDA_DEVICES: device IDs (default 0).
KDA_OUT: result CSV; KDA_MANUAL_DATA: input/golden directory for dump/replay.
KDA_PY (or TTK_PY): Python from the active environment.
TTK_REPO: optional ops-test-kit source directory; otherwise use installed ttk.
KDA_PC=1, KDA_RUN=1, KDA_PROF=false, KDA_SEED=20261027, KDA_XPU_PORT=9090.
KDA_XPU_LOG: optional CPU peer log path; xpu services remain running afterward.
HELP
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    usage
    exit 0
fi
MODE=run
if [[ $# -gt 0 && "$1" != -* ]]; then
    MODE=$1
    shift
fi
case "$MODE" in
    run|l2|validate|list|exec|dump|replay|xpu) ;;
    *) echo "ERROR: unknown mode: $MODE" >&2; usage >&2; exit 2 ;;
esac

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KDA_ROOT=$(cd "${KDA_ROOT:-$HERE/../..}" && pwd)
KDA_ASSETS=$(cd "${KDA_ASSETS:-$KDA_ROOT/tests/ttk}" && pwd)
KDA_PY=$(command -v "${KDA_PY:-${TTK_PY:-python3}}") || {
    echo 'ERROR: select an available Python with KDA_PY.' >&2
    exit 2
}
KDA_CASES=${KDA_CASES:-$KDA_ASSETS/cases/flash_kda_grad_stc.csv}
KDA_OUT=${KDA_OUT:-$PWD/kda_result.csv}
KDA_MANUAL_DATA=${KDA_MANUAL_DATA:-$PWD/kda_manual_data}
# Resolve caller-relative paths before changing to a TTK source checkout.
[[ "$KDA_CASES" == /* ]] || KDA_CASES="$PWD/$KDA_CASES"
[[ "$KDA_OUT" == /* ]] || KDA_OUT="$PWD/$KDA_OUT"
[[ "$KDA_MANUAL_DATA" == /* ]] || KDA_MANUAL_DATA="$PWD/$KDA_MANUAL_DATA"
KDA_DEVICES=${KDA_DEVICES:-0}
KDA_SEED=${KDA_SEED:-20261027}
KDA_PC=${KDA_PC:-1}
KDA_RUN=${KDA_RUN:-1}
KDA_PROF=${KDA_PROF:-false}
KDA_XPU_PORT=${KDA_XPU_PORT:-9090}
KDA_XPU_LOG=${KDA_XPU_LOG:-${TMPDIR:-/tmp}/kda_xpu_${KDA_XPU_PORT}.log}
[[ "$KDA_XPU_LOG" == /* ]] || KDA_XPU_LOG="$PWD/$KDA_XPU_LOG"
[[ "$KDA_PROF" == true || "$KDA_PROF" == false ]] || {
    echo 'ERROR: KDA_PROF must be true or false.' >&2
    exit 2
}
[[ "$KDA_XPU_PORT" =~ ^[0-9]+$ ]] && ((10#$KDA_XPU_PORT >= 1 && 10#$KDA_XPU_PORT <= 65535)) || {
    echo 'ERROR: KDA_XPU_PORT must be 1..65535.' >&2
    exit 2
}
KDA_XPU_PORT=$((10#$KDA_XPU_PORT))
if [[ -z "${TTK_REPO:-}" && -f "$KDA_ROOT/../../../ops-test-kit/ttk/__init__.py" ]]; then
    TTK_REPO="$KDA_ROOT/../../../ops-test-kit"
fi
if [[ -n "${TTK_REPO:-}" ]]; then
    TTK_REPO=$(cd "$TTK_REPO" && pwd)
    [[ -f "$TTK_REPO/ttk/__init__.py" ]] || {
        echo "ERROR: TTK_REPO must contain ttk/: $TTK_REPO" >&2
        exit 2
    }
fi
export KDA_ROOT KDA_ASSETS KDA_PY
export PYTHONPATH="$KDA_ASSETS${TTK_REPO:+:$TTK_REPO}${PYTHONPATH:+:$PYTHONPATH}"
[[ -z "${TTK_REPO:-}" ]] || cd "$TTK_REPO"

peer_ready() {
    (exec 3<>/dev/tcp/127.0.0.1/"$KDA_XPU_PORT") 2>/dev/null
}
ensure_xpu() {
    if peer_ready; then
        echo "[xpu] reuse 127.0.0.1:$KDA_XPU_PORT"
        return 0
    fi
    mkdir -p "$(dirname "$KDA_XPU_LOG")"
    setsid nohup "$KDA_PY" -m ttk.remote.server.xpu_server \
        --port "$KDA_XPU_PORT" --devices cpu \
        > "$KDA_XPU_LOG" 2>&1 < /dev/null &
    local peer_pid=$!
    echo "[xpu] started pid=$peer_pid log=$KDA_XPU_LOG"
    for ((attempt=0; attempt<30; attempt++)); do
        if peer_ready; then return 0; fi
        kill -0 "$peer_pid" 2>/dev/null || break
        sleep 1
    done
    echo "ERROR: CPU peer did not start; see $KDA_XPU_LOG" >&2
    return 3
}

CONFIG_DIR=
cleanup() {
    if [[ -n "$CONFIG_DIR" ]]; then
        rm -f -- "$CONFIG_DIR/peer.yaml"
        rmdir -- "$CONFIG_DIR"
    fi
}
trap cleanup EXIT
peer_config() {
    CONFIG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kda_ttk.XXXXXX")
    printf 'remote:\n  endpoints:\n    - host: 127.0.0.1\n      port: %s\n' \
        "$KDA_XPU_PORT" > "$CONFIG_DIR/peer.yaml"
}
SPEC="$KDA_ASSETS/flash_kda_grad_ttk/spec.py"
TTK_ARGS=(
    -i "$KDA_CASES" --plugin "$SPEC" --device-whitelist "$KDA_DEVICES"
    --seed "$KDA_SEED" --pc "$KDA_PC" --proc-no-reuse --proc-timeout 3000
    --warmup=false --run "$KDA_RUN" --task-prof "$KDA_PROF" -o "$KDA_OUT"
)
[[ -z "${KDA_CASE:-}" ]] || TTK_ARGS+=(-t "$KDA_CASE")
run_ttk() {
    local status=0
    mkdir -p "$(dirname "$KDA_OUT")"
    printf '[ttk] mode=%s cases=%s result=%s\n' "$MODE" "$KDA_CASES" "$KDA_OUT"
    "$KDA_PY" -m ttk e2e "${TTK_ARGS[@]}" "$@" || status=$?
    [[ ! -f "$KDA_OUT" ]] || printf '[result] %s\n' "$KDA_OUT"
    return "$status"
}
case "$MODE" in
    validate)
        "$KDA_PY" -m ttk e2e -i "$KDA_CASES" --plugin "$SPEC" --validate "$@"
        ;;
    list)
        "$KDA_PY" -m ttk list -i "$KDA_CASES" "$@"
        ;;
    xpu)
        ensure_xpu
        ;;
    run|l2)
        ensure_xpu
        peer_config
        run_ttk --config "$CONFIG_DIR/peer.yaml" "$@"
        ;;
    exec)
        run_ttk --golden-mode Disable "$@"
        ;;
    dump)
        mkdir -p "$KDA_MANUAL_DATA"
        run_ttk --no-prof --dump in,golden --dump-format bin \
            --manual-data-dirs "$KDA_MANUAL_DATA" "$@"
        ;;
    replay)
        [[ -d "$KDA_MANUAL_DATA" ]] || {
            echo "ERROR: manual data directory does not exist: $KDA_MANUAL_DATA" >&2
            exit 2
        }
        ensure_xpu
        peer_config
        run_ttk --config "$CONFIG_DIR/peer.yaml" \
            --manual-data-dirs "$KDA_MANUAL_DATA" "$@"
        ;;
esac
