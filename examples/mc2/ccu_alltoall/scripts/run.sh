#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# run.sh — CCU AllToAll + MXFP8 MatMul 示例：数据(按需) + 构建 + 运行
#
# 流程: rank 数探测 -> 数据生成(按需, meta 不匹配自动重生成) -> cmake 构建 -> 多 rank 运行
#       （rank 数 = 本机 NPU 数，进程内每卡一线程；precision 位级校验 / perf event 计时）
#
# 用法:
#   bash scripts/run.sh                                      # 数据(按需)+构建+precision 默认 512 128 256 256
#   bash scripts/run.sh precision 8192 512 2048 3584         # 任意 case（meta 不匹配自动重生成）
#   bash scripts/run.sh perf 8192 512 2048 3584 10           # perf 直跑
#   bash scripts/run.sh --perf perf 8192 512 2048 3584 10    # perf + msprof 采集（prof/，parse_prof.py 解析）
#   bash scripts/run.sh --gen precision 8192 512 2048 3584   # 仅生成数据
#   bash scripts/run.sh --skip-build ...                     # 跳过编译
#   rank 数探测: npu-smi info -l（CCU_A2A_RANKS 覆盖，失败默认 4）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"     # 工程根（data/output/prof/build 落这里）
BUILD_DIR="${PROJ_DIR}/build"
DATA_DIR="${PROJ_DIR}/data"
PROF_DIR="${PROJ_DIR}/prof"
TARGET=ccu_alltoall

SKIP_BUILD=0; PERF_PROF=0; GEN_ONLY=0
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build)  SKIP_BUILD=1; shift ;;
        --perf)        PERF_PROF=1; shift ;;
        --gen)         GEN_ONLY=1; shift ;;
        -h|--help)     sed -n '2,14p' "$0"; exit 0 ;;
        *)             ARGS+=("$1"); shift ;;
    esac
done

# ---- 环境检查 ----
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1 || true
else
    echo "ERROR: ASCEND_HOME_PATH not set or set_env.sh missing"; exit 1
fi

export HCCL_OP_EXPANSION_MODE=CCU_SCHED

# ---- rank 数探测（CCU_A2A_RANKS 覆盖；npu-smi 失败默认 4）----
detect_ranks() {
    if [ -n "${CCU_A2A_RANKS:-}" ]; then
        echo "${CCU_A2A_RANKS}"
        return 0
    fi
    local n
    n="$(npu-smi info -l 2>/dev/null | sed -n 's/.*Total Count[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)"
    if [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null; then
        echo "$n"
    else
        echo "WARNING: cannot detect NPU count via npu-smi, defaulting to 4 (override with CCU_A2A_RANKS)" >&2
        echo 4
    fi
}
RANKS="$(detect_ranks)"

# ---- 位置参数（带默认值；mode 词必须合法）----
MODE="${ARGS[0]:-precision}"
M="${ARGS[1]:-512}"
TILEM="${ARGS[2]:-128}"
K="${ARGS[3]:-256}"
N="${ARGS[4]:-256}"
LOOP="${ARGS[5]:-10}"

if [ "$MODE" != "precision" ] && [ "$MODE" != "perf" ]; then
    echo "ERROR: unknown mode '$MODE' (precision | perf)"; exit 1
fi
if [ "$PERF_PROF" -eq 1 ] && [ "$MODE" != "perf" ]; then
    echo "WARNING: --perf only applies to perf mode, running without msprof"
fi

# ---- 数据（meta.txt 不匹配则调 gen_data.py 重生成）----
need_gen=0
if [ ! -f "${DATA_DIR}/meta.txt" ]; then
    need_gen=1
else
    read -r gM gK gN gR < "${DATA_DIR}/meta.txt" || need_gen=1
    if [ "${gM:-0}" != "$M" ] || [ "${gK:-0}" != "$K" ] || [ "${gN:-0}" != "$N" ] || [ "${gR:-0}" != "$RANKS" ]; then
        need_gen=1
    fi
fi
if [ "$need_gen" -eq 1 ]; then
    echo "[data] generating M=$M K=$K N=$N ranks=$RANKS -> ${DATA_DIR}"
    python3 "${SCRIPT_DIR}/gen_data.py" --M "$M" --K "$K" --N "$N" --ranks "$RANKS" --out "$DATA_DIR"
else
    echo "[data] ${DATA_DIR}/meta.txt matches (M=$M K=$K N=$N R=$RANKS), skip generation"
fi

if [ "$GEN_ONLY" -eq 1 ]; then
    echo "[done] data generated under ${DATA_DIR}"
    exit 0
fi

# ---- 构建 ----
if [ "$SKIP_BUILD" -eq 0 ]; then
    echo "[build] ${TARGET} ..."
    cmake -S "$PROJ_DIR" -B "$BUILD_DIR"
    cmake --build "$BUILD_DIR" --target "$TARGET" --parallel 8
fi
BIN="${BUILD_DIR}/${TARGET}"
[ -x "$BIN" ] || { echo "ERROR: binary missing at $BIN (use --skip-build only after a build)"; exit 1; }

# ---- 运行（cd 工程根：data/output 相对路径落工程根）----
cd "$PROJ_DIR"
if [ "$PERF_PROF" -eq 1 ] && [ "$MODE" = "perf" ]; then
    rm -rf "$PROF_DIR"; mkdir -p "$PROF_DIR"
    msprof --application="${BIN} ${MODE} ${M} ${TILEM} ${K} ${N} ${LOOP}" --output="${PROF_DIR}" --ai-core=on
    echo "perf data saved to: ${PROF_DIR}"
    echo "parse with: python3 ${SCRIPT_DIR}/parse_prof.py --latest"
elif [ "$MODE" = "perf" ]; then
    "$BIN" "$MODE" "$M" "$TILEM" "$K" "$N" "$LOOP"
else
    "$BIN" "$MODE" "$M" "$TILEM" "$K" "$N"
fi
