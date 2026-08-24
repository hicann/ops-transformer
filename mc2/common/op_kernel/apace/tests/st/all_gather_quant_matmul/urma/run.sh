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
# run.sh — apace AllGatherQuantMatmul Prefill ST
#
# 流程: 生成数据 -> 编译 -> 多 rank 运行 -> 精度比对
#
# 用法:
#   bash run.sh                        # 无参数: 列出 cases.csv 全部 case，依次运行
#   bash run.sh 1                      # 运行 CSV 第 1 行
#   bash run.sh 1 3 5                  # 运行 CSV 第 1/3/5 行
#   bash run.sh 2-4                    # 运行 CSV 第 2~4 行
#   bash run.sh all                    # 运行 CSV 全部行
#   bash run.sh --csv <file> 1 3       # 指定 csv 文件 + 行号
#   bash run.sh --cli m k n r          # 命令行模式(绕过 csv)
#   bash run.sh --skip-build ...       # 跳过编译
#   bash run.sh --gen-only ...         # 仅生成 CPU golden
#   bash run.sh --verify-only ...      # 仅精度比对
#   bash run.sh --perf ...             # 性能模式(msprof采集)
#   环境变量 KERNEL_TIMEOUT=<秒>       # 覆盖默认超时(600s)
#   perf 模式跑完后: python3 scripts/parse_prof.py --all  解析性能数据
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${ST_DIR}/build"
PROF_DIR="${SCRIPT_DIR}/prof"
KERNEL_TIMEOUT=${KERNEL_TIMEOUT:-600}  # 默认600秒(10分钟)，可用环境变量 KERNEL_TIMEOUT 覆盖

SKIP_BUILD=0; GEN_ONLY=0; VERIFY_ONLY=0; CLI_MODE=0; PERF_MODE=0
CSV_FILE="${SCRIPT_DIR}/cases.csv"

# ---- 解析参数 ----
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build)  SKIP_BUILD=1; shift ;;
        --gen-only)    GEN_ONLY=1; shift ;;
        --verify-only) VERIFY_ONLY=1; shift ;;
        --cli)         CLI_MODE=1; shift ;;
        --csv)         CSV_FILE="$2"; shift 2 ;;
        --perf)        PERF_MODE=1; shift ;;
        -h|--help)
            sed -n '1,20p' "$0"; exit 0 ;;
        *)
            ARGS+=("$1"); shift ;;
    esac
done

if [ "$GEN_ONLY" -eq 1 ] && [ "$VERIFY_ONLY" -eq 1 ]; then
    echo "ERROR: --gen-only and --verify-only are mutually exclusive"; exit 1
fi
if [ "$PERF_MODE" -eq 1 ] && [ "$VERIFY_ONLY" -eq 1 ]; then
    echo "ERROR: --perf and --verify-only are mutually exclusive"; exit 1
fi

# ---- 将行号参数展开为列表（支持 "1 3 5"、"2-4"、"all"）----
expand_rows() {
    local input=("$@")
    local result=()
    for item in "${input[@]}"; do
        if [ "$item" = "all" ]; then
            for ((i=1; i<=TOTAL_LINES; i++)); do result+=($i); done
        elif [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local start=${BASH_REMATCH[1]} end=${BASH_REMATCH[2]}
            for ((i=start; i<=end; i++)); do result+=($i); done
        elif [[ "$item" =~ ^[0-9]+$ ]]; then
            result+=($item)
        fi
    done
    echo "${result[@]}"
}

# ---- 环境检查（提前做，避免多行重复）----
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1 || true
else
    echo "ERROR: ASCEND_HOME_PATH not set or set_env.sh missing"; exit 1
fi
command -v python3 >/dev/null || { echo "ERROR: python3 not available"; exit 1; }

# ---- 获取 prof 目录列表（用于 msprof 前后对比找新增目录）----
get_prof_dirs() {
    find "$PROF_DIR" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | sort || true
}

# ---- perf 模式结束后提示用户解析性能数据 ----
print_perf_hint() {
    if [ "$PERF_MODE" -eq 1 ]; then
        echo ""
        echo "=========================================="
        echo "Performance data saved to: $PROF_DIR"
        echo "Parse with: python3 scripts/parse_prof.py --all"
        echo "=========================================="
    fi
}

# ---- 单 case 执行函数 ----
run_single() {
    local M=$1 K=$2 N=$3 RANK_NUM=$4
    echo ""
    echo "=========================================="
    echo "apace AllGatherQuantMatmul Prefill ST"
    echo "  M=$M K=$K N=$N rankNum=$RANK_NUM"
    echo "=========================================="

    cd "$SCRIPT_DIR"
    rm -rf input output

    # ---- 1. 生成数据 ----
    echo "[1/4] Generate CPU golden + input data..."
    python3 scripts/gen_data.py "$M" "$K" "$N" "$RANK_NUM"


    if [ "$GEN_ONLY" -eq 1 ]; then return 0; fi

    # ---- 2. 编译 ----
    if [ "$VERIFY_ONLY" -eq 0 ]; then
        if [ "$SKIP_BUILD" -eq 0 ]; then
            echo "[2/4] Build..."
            mkdir -p "$BUILD_DIR"
            cmake -S "$ST_DIR" -B "$BUILD_DIR" || { echo "ERROR: cmake failed"; return 1; }
            cmake --build "$BUILD_DIR" --target apace_ag_qmm_urma_st --parallel 4 || { echo "ERROR: build failed"; return 1; }
        else
            [ -x "$BUILD_DIR/all_gather_quant_matmul/urma/apace_ag_qmm_urma_st" ] || { echo "ERROR: --skip-build but binary missing"; return 1; }
            echo "[2/4] Skip build (--skip-build)"
        fi

        # ---- 3. 多 rank 运行 ----
        echo "[3/4] NPU run ($RANK_NUM ranks)..."
        local EXE_PATH="$BUILD_DIR/all_gather_quant_matmul/urma/apace_ag_qmm_urma_st"
        local EXE_DIR="$(dirname "$EXE_PATH")"
        rm -rf "$EXE_DIR/input" "$EXE_DIR/output"
        cp -r input "$EXE_DIR/input"
        mkdir -p "$EXE_DIR/output"
        for r in $(seq 0 $((RANK_NUM-1))); do
            mkdir -p "$EXE_DIR/output/$r"
        done

        MAX_RETRY=3
        RETRY=0
        local MODE="precision"
        [ "$PERF_MODE" -eq 1 ] && MODE="perf"
        while [ ${RETRY} -lt ${MAX_RETRY} ]; do
            if command -v fuser >/dev/null 2>&1; then
                fuser -k 8998/tcp 2>/dev/null || true
            else
                echo "  [WARN] fuser not installed, skip port cleanup"
            fi
            sleep 1

            if [ "$PERF_MODE" -eq 1 ]; then
                # ---- perf 模式: msprof 包裹 exe ----
                if ! command -v msprof >/dev/null 2>&1; then
                    echo "ERROR: msprof not found in PATH"; return 1
                fi
                mkdir -p "$PROF_DIR"
                local before_dirs after_dirs new_dirs
                before_dirs=$(get_prof_dirs)

                echo "  [msprof] Running $EXE_PATH $M $K $N $RANK_NUM $MODE ..."
                if command -v timeout >/dev/null 2>&1; then
                    set +e
                    timeout ${KERNEL_TIMEOUT}s msprof --output="$PROF_DIR" \
                        --application="$EXE_PATH $M $K $N $RANK_NUM $MODE" 2>&1
                    local MSPROF_RC=$?
                    set -e
                else
                    set +e
                    msprof --output="$PROF_DIR" \
                        --application="$EXE_PATH $M $K $N $RANK_NUM $MODE" 2>&1
                    local MSPROF_RC=$?
                    set -e
                fi

                if [ ${MSPROF_RC} -eq 124 ]; then
                    echo "ERROR: msprof timeout after ${KERNEL_TIMEOUT}s"
                    return 1
                fi

                after_dirs=$(get_prof_dirs)
                new_dirs=""
                for d in $after_dirs; do
                    if ! echo "$before_dirs" | grep -qFx "$d"; then
                        new_dirs="$new_dirs $d"
                    fi
                done
                new_dirs=$(echo "$new_dirs" | xargs)

                if [ -z "$new_dirs" ]; then
                    echo "  [retry ${RETRY}/${MAX_RETRY}] No new PROF dirs created"
                    RETRY=$((RETRY + 1))
                    sleep 2
                    continue
                fi

                for d in $new_dirs; do
                    echo "M=$M K=$K N=$N" > "$PROF_DIR/$d/case_info.txt"
                done

                # ---- 校验 PROF 数据有效性（op_summary CSV 是否能解析出 latency）----
                local parse_out latency
                set +e
                parse_out=$(python3 scripts/parse_prof.py --check-latest-threshold 999999999 \
                    --mkn "$M $K $N" --prof-dir "$PROF_DIR" 2>/dev/null)
                set -e
                latency=$(echo "$parse_out" | grep -E '^[0-9]+\.[0-9]+$' | head -1)

                if [ -z "$latency" ]; then
                    echo "  [retry ${RETRY}/${MAX_RETRY}] Cannot parse latency from PROF data"
                    for d in $new_dirs; do rm -rf "$PROF_DIR/$d"; done
                    RETRY=$((RETRY + 1))
                    sleep 2
                    continue
                fi
                echo "  [msprof] latency=${latency}us"

                cd "$SCRIPT_DIR"
                for r in $(seq 0 $((RANK_NUM-1))); do
                    mkdir -p "output/$r"
                    cp "$EXE_DIR/output/$r/npu_out.bin" "output/$r/npu_out.bin" 2>/dev/null || true
                done

                local precision="FAIL"
                if [ -d "$SCRIPT_DIR/output" ]; then
                    set +e
                    precision=$(python3 scripts/verify_result.py "$M" "$N" "$RANK_NUM" "$SCRIPT_DIR/output" --check 2>/dev/null)
                    [ -z "$precision" ] && precision="FAIL"
                    set -e
                fi

                for d in $new_dirs; do
                    echo "$precision" > "$PROF_DIR/$d/case_precision.txt"
                done

                if [ "$precision" != "PASS" ]; then
                    echo "  [retry ${RETRY}/${MAX_RETRY}] Precision FAILED"
                    for d in $new_dirs; do rm -rf "$PROF_DIR/$d"; done
                    RETRY=$((RETRY + 1))
                    sleep 2
                    continue
                fi

                echo "  [OK] msprof done, precision=PASS, dirs=$new_dirs"
                break
            else
                # ---- precision 模式: 直接调 exe ----
                if command -v timeout >/dev/null 2>&1; then
                    set +e
                    KERNEL_OUT=$(timeout ${KERNEL_TIMEOUT}s "$EXE_PATH" $M $K $N $RANK_NUM "$MODE" 2>&1)
                    KERNEL_RC=$?
                    set -e
                else
                    set +e
                    KERNEL_OUT=$("$EXE_PATH" $M $K $N $RANK_NUM "$MODE" 2>&1)
                    KERNEL_RC=$?
                    set -e
                fi
                echo "${KERNEL_OUT}"

                if [ ${KERNEL_RC} -eq 124 ]; then
                    echo "ERROR: Kernel timeout after ${KERNEL_TIMEOUT}s"
                    return 1
                fi

                if [ ${KERNEL_RC} -eq 0 ] && echo "${KERNEL_OUT}" | grep -q "Status: SUCCESS"; then
                    echo "  Kernel finished!"
                    break
                fi

                if echo "${KERNEL_OUT}" | grep -q "connect peers failed"; then
                    RETRY=$((RETRY + 1))
                    echo "  [retry ${RETRY}/${MAX_RETRY}] connect conflict..."
                    sleep 2
                    continue
                fi

                echo "ERROR: Kernel failed"
                return 1
            fi
        done

        if [ ${RETRY} -ge ${MAX_RETRY} ]; then
            echo "ERROR: Max retries exceeded"
            return 1
        fi

        if [ "$PERF_MODE" -eq 0 ]; then
            cd "$SCRIPT_DIR"
            for r in $(seq 0 $((RANK_NUM-1))); do
                mkdir -p "output/$r"
                cp "$EXE_DIR/output/$r/npu_out.bin" "output/$r/npu_out.bin" 2>/dev/null || true
            done
        fi
    fi

    # ---- 4. 精度比对（perf 模式已在步骤3内完成 verify，此处跳过）----
    if [ "$PERF_MODE" -eq 0 ]; then
        echo "[4/4] Verify..."
        python3 scripts/verify_result.py "$M" "$N" "$RANK_NUM" "./output"
        return $?
    fi
    echo "[4/4] Verify (done in perf loop)"
    return 0
}

# ---- 选择参数来源 ----
if [ "${CLI_MODE:-0}" -eq 1 ]; then
    M=2048; K=3584; N=4096; RANK_NUM=4
    idx=0
    [ $idx -lt ${#ARGS[@]} ] && { M=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { K=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { N=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { RANK_NUM=${ARGS[$idx]}; idx=$((idx+1)); }
    run_single "$M" "$K" "$N" "$RANK_NUM"
    rc=$?
    print_perf_hint
    exit $rc
fi

# ---- CSV 模式 ----
if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV not found: $CSV_FILE"; exit 1
fi

DATA_LINES=$(grep -v '^#' "$CSV_FILE" | grep -v '^$' | tail -n +2)
TOTAL_LINES=$(echo "$DATA_LINES" | grep -c .)

# 无参数: 列出全部 case 并全部运行
if [ ${#ARGS[@]} -eq 0 ]; then
    echo "==== cases.csv ($TOTAL_LINES cases) ===="
    echo "Row  M      K     N     rank"
    local_idx=0
    while IFS= read -r line; do
        local_idx=$((local_idx + 1))
        printf "%-5s %s\n" "$local_idx" "$(echo "$line" | awk -F, '{printf "%-6s %-5s %-5s %s", $1,$2,$3,$4}')"
    done <<< "$DATA_LINES"
    echo ""
    echo "Running all $TOTAL_LINES cases..."
    SELECTED_ROWS=($(seq 1 $TOTAL_LINES))
else
    SELECTED_ROWS=($(expand_rows "${ARGS[@]}"))
fi

# 校验行号范围
for row in "${SELECTED_ROWS[@]}"; do
    if [ "$row" -lt 1 ] || [ "$row" -gt "$TOTAL_LINES" ]; then
        echo "ERROR: row $row out of range (1-$TOTAL_LINES)"; exit 1
    fi
done

echo "==== Running ${#SELECTED_ROWS[@]} case(s): ${SELECTED_ROWS[*]} ===="

# ---- 逐行执行 ----
PASS_CNT=0; FAIL_CNT=0
RESULTS=()

for row in "${SELECTED_ROWS[@]}"; do
    SELECTED=$(echo "$DATA_LINES" | sed -n "${row}p")
    M=$(echo "$SELECTED" | awk -F, '{print $1}')
    K=$(echo "$SELECTED" | awk -F, '{print $2}')
    N=$(echo "$SELECTED" | awk -F, '{print $3}')
    RANK_NUM=$(echo "$SELECTED" | awk -F, '{print $4}')

    echo ""
    echo "########## CSV row ${row}/${TOTAL_LINES} ##########"

    if run_single "$M" "$K" "$N" "$RANK_NUM"; then
        PASS_CNT=$((PASS_CNT + 1))
        RESULTS+=("PASS  row$row  M=$M K=$K N=$N rank=$RANK_NUM")
    else
        FAIL_CNT=$((FAIL_CNT + 1))
        RESULTS+=("FAIL  row$row  M=$M K=$K N=$N rank=$RANK_NUM")
    fi
    # 首次编译后，后续 case 跳过编译
    SKIP_BUILD=1
done

# ---- 汇总 ----
echo ""
echo "=========================================="
echo "Summary: PASS=$PASS_CNT  FAIL=$FAIL_CNT  total=${#SELECTED_ROWS[@]}"
echo "=========================================="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done

print_perf_hint

if [ "$FAIL_CNT" -eq 0 ]; then exit 0; else exit 1; fi
