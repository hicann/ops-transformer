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
# run.sh - apace QuantMatmulReduceScatter ST
#
# 流程: 生成数据 -> 编译 -> 多 rank 运行 -> 精度比对
#
# 用法 (CSV 行号从表头后的第一个数据行开始计 1):
#   bash run.sh                        # 无参数: 列出 cases.csv 全部 case, 依次运行
#   bash run.sh 1                      # 运行 CSV 第 1 行
#   bash run.sh 1 3 5                  # 运行 CSV 第 1/3/5 行
#   bash run.sh 2-4                    # 运行 CSV 第 2~4 行
#   bash run.sh all                    # 运行 CSV 全部行
#   bash run.sh --csv <file> 1 3       # 指定 csv 文件 + 行号
#   注: rank_num=8 的用例需 8 卡环境, 4 卡机器请选择 rank_num<=4 的行
#   bash run.sh --cli [m k n r [h] [q]]  # 参数按顺序省略, 默认2048 3584 4096 2 512 1; h<=0自动, q=0~4
#   bash run.sh --skip-build ...       # 跳过编译
#   bash run.sh --gen-only ...         # 仅生成输入和 CPU golden
#   bash run.sh --verify-only ...      # 复用已有输出, 仅精度比对
#   bash run.sh --perf ...             # 性能模式(msprof采集)
#   bash run.sh --precision2 ...       # 连续两次 launch 并逐元素比较输出
#   环境变量 KERNEL_TIMEOUT=<秒>       # 覆盖默认超时(600s)
#   环境变量 MSPROF_TIMEOUT=<秒>       # 覆盖 perf 采集超时(300s)
#   环境变量 QMRS_L2_FLUSH=0           # perf 时关闭默认 L2 flush
#   perf 模式跑完后: python3 scripts/parse_prof.py --all  解析性能数据
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${ST_DIR}/build/matmul_reduce_scatter/urma"
PROF_DIR="${SCRIPT_DIR}/prof"
KERNEL_TIMEOUT=${KERNEL_TIMEOUT:-600}
MSPROF_TIMEOUT=${MSPROF_TIMEOUT:-300}

SKIP_BUILD=0; GEN_ONLY=0; VERIFY_ONLY=0; CLI_MODE=0; PERF_MODE=0; PRECISION2_MODE=0; BUILD_READY=0
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
        --precision2)  PRECISION2_MODE=1; shift ;;
        -h|--help)
            sed -n '1,32p' "$0"; exit 0 ;;
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
if [ "$PERF_MODE" -eq 1 ] && [ "$PRECISION2_MODE" -eq 1 ]; then
    echo "ERROR: --perf and --precision2 are mutually exclusive"; exit 1
fi

# ---- 将行号参数展开为列表(支持 "1 3 5", "2-4", "all")----
expand_rows() {
    local input=("$@")
    local result=()
    for item in "${input[@]}"; do
        if [ "$item" = "all" ]; then
            for ((i=1; i<=TOTAL_LINES; i++)); do result+=("$i"); done
        elif [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local start=${BASH_REMATCH[1]} end=${BASH_REMATCH[2]}
            if [ "$start" -gt "$end" ]; then
                echo "ERROR: invalid row range $item" >&2
                return 1
            fi
            for ((i=start; i<=end; i++)); do result+=("$i"); done
        elif [[ "$item" =~ ^[0-9]+$ ]]; then
            result+=("$item")
        else
            echo "ERROR: invalid row selector $item" >&2
            return 1
        fi
    done
    echo "${result[@]}"
}

# ---- 环境检查 ----
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1 || true
else
    echo "ERROR: ASCEND_HOME_PATH not set or set_env.sh missing"; exit 1
fi
command -v python3 >/dev/null || { echo "ERROR: python3 not available"; exit 1; }

# ---- 获取 prof 目录列表(用于 msprof 前后对比找新增目录)----
get_prof_dirs() {
    find "$PROF_DIR" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | sort || true
}

# ---- perf 模式结束后提示用户解析性能数据 ----
print_perf_hint() {
    if [ "$PERF_MODE" -eq 1 ] && [ "$GEN_ONLY" -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Performance data saved to: $PROF_DIR"
        echo "Parse with: python3 scripts/parse_prof.py --all"
        echo "=========================================="
    fi
}

# ---- 单 case 执行函数 ----
run_single() {
    local M=$1 K=$2 N=$3 RANK_NUM=$4 HEAD_M_SIZE=$5 QUANT_TYPE=${6:-1}
    echo ""
    echo "=========================================="
    echo "apace QuantMatmulReduceScatter ST"
    echo "  M=$M K=$K N=$N rankNum=$RANK_NUM headMSize=$HEAD_M_SIZE quantType=$QUANT_TYPE"
    echo "=========================================="

    cd "$SCRIPT_DIR"
    if [ "$VERIFY_ONLY" -eq 0 ]; then
        rm -rf input output
        # ---- 1. 生成数据 ----
        echo "[1/4] Generate CPU golden + input data..."
        python3 scripts/gen_data.py "$M" "$K" "$N" "$RANK_NUM" "$QUANT_TYPE" || { echo "ERROR: data generation failed"; return 1; }
    fi

    if [ "$GEN_ONLY" -eq 1 ]; then return 0; fi

    # ---- 2. 编译并校验 binary, 后续 case 复用 ----
    if [ "$VERIFY_ONLY" -eq 0 ]; then
        if [ "$BUILD_READY" -eq 0 ]; then
            if [ "$SKIP_BUILD" -eq 0 ]; then
                echo "[2/4] Build..."
                mkdir -p "$BUILD_DIR"
                cmake -S "$ST_DIR" -B "${ST_DIR}/build" || { echo "ERROR: cmake failed"; return 1; }
                cmake --build "${ST_DIR}/build" --target apace_qmm_rs_urma_st --parallel 4 || { echo "ERROR: build failed"; return 1; }
            fi
            [ -x "$BUILD_DIR/apace_qmm_rs_urma_st" ] || { echo "ERROR: build binary missing"; return 1; }
            BUILD_READY=1
        fi
        echo "[2/4] Build ready"

        # ---- 3. 多 rank 运行 ----
        echo "[3/4] NPU run ($RANK_NUM ranks)..."
        local EXE_PATH="$BUILD_DIR/apace_qmm_rs_urma_st"
        local EXE_DIR="$(dirname "$EXE_PATH")"
        rm -rf "$EXE_DIR/input" "$EXE_DIR/output"
        cp -r input "$EXE_DIR/input"
        mkdir -p "$EXE_DIR/output"
        for r in $(seq 0 $((RANK_NUM-1))); do
            mkdir -p "$EXE_DIR/output/$r"
        done

        # 等待上一用例释放 HCCL/AICPU 资源后再启动
        sleep 20

        MAX_RETRY=3
        RETRY=0
        local MODE="precision"
        [ "$PERF_MODE" -eq 1 ] && MODE="perf"
        [ "$PRECISION2_MODE" -eq 1 ] && MODE="precision2"
        while [ ${RETRY} -lt ${MAX_RETRY} ]; do
            sleep 1

            if [ "$PERF_MODE" -eq 1 ]; then
                # ---- perf 模式: msprof 包裹 exe ----
                if ! command -v msprof >/dev/null 2>&1; then
                    echo "ERROR: msprof not found in PATH"; return 1
                fi
                mkdir -p "$PROF_DIR"
                local before_dirs after_dirs new_dirs
                before_dirs=$(get_prof_dirs)

                echo "  [msprof] Running $EXE_PATH $M $K $N $RANK_NUM $MODE $HEAD_M_SIZE $QUANT_TYPE ..."
                # msprof 在 exe 目录直接拉起, 避免多 rank TCP 建链时序问题
                if command -v timeout >/dev/null 2>&1;
                then
                    set +e
                    (cd "$EXE_DIR" && timeout ${MSPROF_TIMEOUT}s msprof --output="$PROF_DIR" \
                        --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization \
                        --task-time=on --ascendcl=on \
                        ./"$(basename "$EXE_PATH")" $M $K $N $RANK_NUM "$MODE" "$HEAD_M_SIZE" "$QUANT_TYPE" 2>&1)
                    local MSPROF_RC=$?
                    set -e
                else
                    set +e
                    (cd "$EXE_DIR" && msprof --output="$PROF_DIR" \
                        --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization \
                        --task-time=on --ascendcl=on \
                        ./"$(basename "$EXE_PATH")" $M $K $N $RANK_NUM "$MODE" "$HEAD_M_SIZE" "$QUANT_TYPE" 2>&1)
                    local MSPROF_RC=$?
                    set -e
                fi

                after_dirs=$(get_prof_dirs)
                new_dirs=""
                for d in $after_dirs; do
                    if ! echo "$before_dirs" | grep -qFx "$d"; then
                        new_dirs="$new_dirs $d"
                    fi
                done
                new_dirs=$(echo "$new_dirs" | xargs)

                if [ ${MSPROF_RC} -eq 124 ]; then
                    echo "  [retry ${RETRY}/${MAX_RETRY}] msprof timeout after ${MSPROF_TIMEOUT}s"
                    for d in $new_dirs; do rm -rf "$PROF_DIR/$d"; done
                    RETRY=$((RETRY + 1))
                    sleep 2
                    continue
                fi

                if [ ${MSPROF_RC} -ne 0 ]; then
                    echo "  [retry ${RETRY}/${MAX_RETRY}] msprof exited with rc=${MSPROF_RC} (worker failed)"
                    for d in $new_dirs; do rm -rf "$PROF_DIR/$d"; done
                    RETRY=$((RETRY + 1))
                    sleep 2
                    continue
                fi

                if [ -z "$new_dirs" ]; then
                    echo "  [retry ${RETRY}/${MAX_RETRY}] No new PROF dirs created"
                    RETRY=$((RETRY + 1))
                    sleep 2
                    continue
                fi

                for d in $new_dirs; do
                    echo "M=$M K=$K N=$N R=$RANK_NUM headM=$HEAD_M_SIZE quantType=$QUANT_TYPE" > "$PROF_DIR/$d/case_info.txt"
                done

                # ---- 校验 PROF 数据有效性(op_summary CSV 是否能解析出 latency)----
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
                    precision=$(python3 scripts/verify_result.py "$M" "$N" "$RANK_NUM" "$SCRIPT_DIR/output" --quant-type "$QUANT_TYPE" --check 2>/dev/null)
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
                # ---- precision/precision2 模式: 直接调 exe ----
                # tee 实时输出并保存日志; PIPESTATUS[0] 取 exe 退出码
                local out_log="${SCRIPT_DIR}/kernel_out_r${RANK_NUM}.log"
                rm -f "$out_log"
                if command -v timeout >/dev/null 2>&1; then
                    set +e
                    timeout ${KERNEL_TIMEOUT}s "$EXE_PATH" $M $K $N $RANK_NUM "$MODE" "$HEAD_M_SIZE" "$QUANT_TYPE" 2>&1 | tee "$out_log"
                    KERNEL_RC=${PIPESTATUS[0]}
                    set -e
                else
                    set +e
                    "$EXE_PATH" $M $K $N $RANK_NUM "$MODE" "$HEAD_M_SIZE" "$QUANT_TYPE" 2>&1 | tee "$out_log"
                    KERNEL_RC=${PIPESTATUS[0]}
                    set -e
                fi
                KERNEL_OUT=$(cat "$out_log" 2>/dev/null || true)

                if [ ${KERNEL_RC} -eq 124 ]; then
                    echo "  [retry ${RETRY}/${MAX_RETRY}] kernel timeout after ${KERNEL_TIMEOUT}s (possible HCCL/URMA teardown race)"
                    RETRY=$((RETRY + 1))
                    sleep 20
                    continue
                fi

                if [ ${KERNEL_RC} -ne 0 ]; then
                    echo "  [warn] exe exited rc=${KERNEL_RC}, check ${out_log} for rank-level errors (e.g. CreateDeviceContext / HCCL setup)"
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

                if echo "${KERNEL_OUT}" | grep -q "CreateDeviceContext failed"; then
                    RETRY=$((RETRY + 1))
                    echo "  [retry ${RETRY}/${MAX_RETRY}] CreateDeviceContext failed (resource contention? check concurrent jobs)"
                    sleep 20
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

    # ---- 4. 精度比对(perf 模式已在步骤3内完成 verify, 此处跳过)----
    if [ "$PERF_MODE" -eq 0 ]; then
        echo "[4/4] Verify..."
        python3 scripts/verify_result.py "$M" "$N" "$RANK_NUM" "./output" --quant-type "$QUANT_TYPE"
        return $?
    fi
    echo "[4/4] Verify (done in perf loop)"
    return 0
}

# ---- 选择参数来源 ----
if [ "${CLI_MODE:-0}" -eq 1 ]; then
    M=2048; K=3584; N=4096; RANK_NUM=2; HEAD_M_SIZE=512
    idx=0
    [ $idx -lt ${#ARGS[@]} ] && { M=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { K=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { N=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { RANK_NUM=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { HEAD_M_SIZE=${ARGS[$idx]}; idx=$((idx+1)); }
    QUANT_TYPE=1
    [ $idx -lt ${#ARGS[@]} ] && { QUANT_TYPE=${ARGS[$idx]}; idx=$((idx+1)); }
    if [ $idx -lt ${#ARGS[@]} ]; then
        echo "ERROR: too many CLI arguments"; exit 1
    fi
    run_single "$M" "$K" "$N" "$RANK_NUM" "$HEAD_M_SIZE" "$QUANT_TYPE"
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
    echo "Row  M      K     N     rank  headM quant"
    local_idx=0
    while IFS= read -r line; do
        local_idx=$((local_idx + 1))
        printf "%-5s %s\n" "$local_idx" "$(echo "$line" | awk -F, '{q=$6; if (q == "") q=1; printf "%-6s %-5s %-5s %-5s %-5s %s", $1,$2,$3,$4,$5,q}')"
    done <<< "$DATA_LINES"
    echo ""
    echo "Running all $TOTAL_LINES cases..."
    SELECTED_ROWS=($(seq 1 $TOTAL_LINES))
else
    SELECTED_ROWS=($(expand_rows "${ARGS[@]}"))
fi
if [ ${#SELECTED_ROWS[@]} -eq 0 ]; then
    echo "ERROR: no valid cases selected"; exit 1
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
    HEAD_M_SIZE=$(echo "$SELECTED" | awk -F, '{print $5}')
    # 第 6 列 quant_type (可选, 缺省 1=E4M3E4M3)
    QUANT_TYPE=$(echo "$SELECTED" | awk -F, '{print $6}')
    [ -z "$QUANT_TYPE" ] && QUANT_TYPE=1

    echo ""
    echo "########## CSV row ${row}/${TOTAL_LINES} ##########"

    if run_single "$M" "$K" "$N" "$RANK_NUM" "$HEAD_M_SIZE" "$QUANT_TYPE"; then
        PASS_CNT=$((PASS_CNT + 1))
        RESULTS+=("PASS  row$row  M=$M K=$K N=$N rank=$RANK_NUM headM=$HEAD_M_SIZE quantType=$QUANT_TYPE")
    else
        FAIL_CNT=$((FAIL_CNT + 1))
        RESULTS+=("FAIL  row$row  M=$M K=$K N=$N rank=$RANK_NUM headM=$HEAD_M_SIZE quantType=$QUANT_TYPE")
    fi
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
