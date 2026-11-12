#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# quant_matmul_reduce_scatter (ubmem) 精度/性能测试入口。
#
# 前置（调用前自行完成）：
#   1) conda activate <已安装 torch、torch_npu 和算子 torch_extension wheel 的环境>
#   2) source <CANN_HOME>/set_env.sh
#   3) source <自定义算子包>/bin/set_env.bash（隔离安装时使用实际安装目录）
#
# 用法：
#   未指定输入/输出 dtype、B 格式或转置时，批跑 36 组组合。
#   bash run_test.sh                                                  # 四卡精度，36 组
#   bash run_test.sh --world-size 8                                    # 八卡精度，36 组
#   bash run_test.sh --mode precision --world-size 4                    # 四卡精度
#   bash run_test.sh --mode perf --world-size 4                         # 四卡性能
#   bash run_test.sh --mode perf --world-size 4 --with-baseline         # 性能及小算子对照
#   bash run_test.sh --mode precision --cases b4_mtp4,b20_mtp4 --world-size 4     # 指定内置用例
#   bash run_test.sh --mode precision --m 64 --k 2048 --n 3584           # 自定义形状
#   bash run_test.sh --list-cases                                      # 查看内置用例
#
# --m 为每卡输出行数，输入 M = m * world-size。
# ASCEND_RT_VISIBLE_DEVICES 由运行环境指定，须覆盖 world-size 个设备。
# NPU 任务请通过本机排队系统（如 qrun）执行。
# -----------------------------------------------------------------------------------------------------------
set -o pipefail

run_matrix=true
for arg in "$@"; do
    case "$arg" in
        --x1-dtype|--x1-dtype=*|--x2-dtype|--x2-dtype=*|--y-dtype|--y-dtype=*|\
        --x2-format|--x2-format=*|--x2-orient|--x2-orient=*|--list-cases|-h|--help)
            run_matrix=false
            break
            ;;
    esac
done

if "$run_matrix"; then
    failed=0
    run_one() {
        local x1="$1" x2="$2" out="$3" fmt="$4" orient="$5"
        shift 5
        echo "[MATRIX] $x1 x $x2 -> $out, $fmt, $orient"
        bash "$0" "$@" --x1-dtype "$x1" --x2-dtype "$x2" --y-dtype "$out" \
            --x2-format "$fmt" --x2-orient "$orient" || failed=$((failed + 1))
    }
    for out in bf16 fp16; do
        for orient in trans non-trans; do
            for x1 in e4m3 e5m2; do
                for x2 in e4m3 e5m2; do
                    for fmt in nd nz; do
                        run_one "$x1" "$x2" "$out" "$fmt" "$orient" "$@"
                    done
                done
            done
            run_one fp4 fp4 "$out" nd "$orient" "$@"
        done
    done
    echo "[MATRIX] $((36 - failed))/36 passed"
    exit "$((failed != 0))"
fi

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "[ERROR] 未检测到 ASCEND_HOME_PATH，请先 source CANN 的 set_env.sh"
    exit 1
fi

# 自定义包的环境脚本负责设置 ASCEND_CUSTOM_OPP_PATH 和动态库路径。
# 未指定隔离包时，补充 CANN 默认安装位置的算子 API 库路径。
if [ -z "${ASCEND_CUSTOM_OPP_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# 通信窗口容量沿用原测试配置；允许调用方通过环境变量覆盖。
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1000}

cd "$(dirname "$0")" || exit 1
echo "[ENV] python=$(command -v python3)"
echo "[ENV] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "[ENV] ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}"
echo "[ENV] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-未指定}"
# 临时捕获父/子进程输出；结束后与本次汇总保存在同一结果目录。
log_file=$(mktemp) || exit 1
python3 -u quant_matmul_reduce_scatter_test.py "$@" 2>&1 | tee "$log_file"
rc=${PIPESTATUS[0]}
output_dir=$(sed -n 's/^\[INFO\].* output=//p' "$log_file" | head -n 1)
if [ -z "$output_dir" ] || [ ! -d "$output_dir" ]; then
    output_dir="results/failed_$(date +%Y%m%d_%H%M%S)_$$"
    mkdir -p "$output_dir" || exit 1
fi
mv -- "$log_file" "$output_dir/run.log" || exit 1
exit "$rc"
