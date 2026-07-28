#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# 批量隔离执行脚本:
#   1. 获取指定路径下的所有 .pt 用例
#   2. 对每条用例单独拉起一个 pytest 进程执行, 进程间完全隔离
#      - 某条用例 device 越界/崩溃不会影响后续用例
#      - msprof 挂载在单用例进程上, 性能数据采集互不干扰
# 用法:
#   bash batch_isolated_run.sh [用例目录] [是否msprof采集: 0|1]
# 示例:
#   bash batch_isolated_run.sh                       # 默认 pt_path 目录, 不采集性能
#   bash batch_isolated_run.sh pt_path 1             # 启用 msprof 采集性能
# -----------------------------------------------------------------------------------------------------------

set -o pipefail

# Ctrl+C / SIGTERM 中断处理: 递归杀所有子进程后退出
_cleanup_on_interrupt() {
    echo -e "\n\n[中断] 收到终止信号，正在清理子进程..." | tee -a "$SUMMARY_LOG"
    pkill -TERM -P $$ 2>/dev/null
    sleep 2
    pkill -KILL -P $$ 2>/dev/null
    exit 130
}
trap _cleanup_on_interrupt SIGINT SIGTERM

TEST_SCRIPT="test_quant_compressor_batch.py"
TESTCASE_DIR="${1:-./pt_path}"
USE_MSPROF="${2:-0}"
RESULT_XLSX="result.xlsx"
SUMMARY_LOG="batch_summary.log"
FAIL_LOG="batch_fail_list.log"

# 清理旧文件
[ -f "$RESULT_XLSX" ] && rm -f "$RESULT_XLSX"
[ -f "${RESULT_XLSX%.xlsx}_perf.xlsx" ] && rm -f "${RESULT_XLSX%.xlsx}_perf.xlsx"
rm -f "${RESULT_XLSX%.xlsx}_perf.xlsx.tmp.xlsx"
: > "$SUMMARY_LOG"
: > "$FAIL_LOG"

# 清理旧的 PROF 文件夹, 避免与本次运行的数据混淆
for _prof_dir in PROF_*/; do
    [ -d "$_prof_dir" ] && rm -rf "$_prof_dir"
done

# 1. 获取用例列表: 优先从 excel 文件读取 Testcase_Name 列, 不存在或缺列时回退为扫描 .pt 文件
if [ ! -d "$TESTCASE_DIR" ]; then
    echo "错误: 用例目录不存在: $TESTCASE_DIR"
    exit 1
fi

EXCEL_DIR="./excel"
CASE_FILES=()
CASE_SOURCE=""

# 尝试从 excel 读取 Testcase_Name 列, 映射为 .pt 文件路径
if [ -d "$EXCEL_DIR" ]; then
    _excel_files=$(find "$EXCEL_DIR" -maxdepth 1 -name "*.xlsx" -o -name "*.xls" 2>/dev/null | head -1)
    if [ -n "$_excel_files" ]; then
        echo "找到 Excel 文件: $_excel_files, 尝试读取 Testcase_Name 列..."
        _names=$(python3 -c "
import sys, pandas as pd, glob, os
excel_dir = '$EXCEL_DIR'
files = glob.glob(os.path.join(excel_dir, '*.xlsx')) + glob.glob(os.path.join(excel_dir, '*.xls'))
if not files:
    sys.exit(1)
df = pd.read_excel(files[0])
if 'Testcase_Name' not in df.columns:
    sys.exit(1)
for name in df['Testcase_Name'].dropna().astype(str):
    print(name.strip())
" 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$_names" ]; then
            while IFS= read -r _name; do
                _pt_file="${TESTCASE_DIR}/${_name}.pt"
                if [ -f "$_pt_file" ]; then
                    CASE_FILES+=("$_pt_file")
                else
                    echo "  警告: Testcase_Name='$_name' 对应的 .pt 文件不存在: $_pt_file, 已跳过"
                fi
            done <<< "$_names"
            CASE_SOURCE="excel(${_excel_files})"
        fi
    fi
fi

# excel 无结果时回退为扫描 .pt 文件
if [ ${#CASE_FILES[@]} -eq 0 ]; then
    echo "未从 Excel 获取到有效用例, 回退为扫描 .pt 文件..."
    mapfile -t CASE_FILES < <(find "$TESTCASE_DIR" -maxdepth 1 -name "*.pt" | sort)
    CASE_SOURCE="scan"
fi

TOTAL=${#CASE_FILES[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "错误: 目录 $TESTCASE_DIR 下未找到任何 .pt 用例"
    exit 1
fi

echo "共发现 $TOTAL 条用例, 来源: $CASE_SOURCE , 目录: $TESTCASE_DIR , msprof采集: $USE_MSPROF"
echo "开始隔离批量执行..." | tee -a "$SUMMARY_LOG"

PASS=0
FAIL=0
FAIL_LIST=()

# 2. 对每条用例单独调用一次测试脚本, 独立进程
i=0
for case_file in "${CASE_FILES[@]}"; do
    i=$((i+1))
    case_name=$(basename "$case_file")
    echo -e "\n===== [$i/$TOTAL] 执行用例: $case_name =====" | tee -a "$SUMMARY_LOG"

    if [ "$USE_MSPROF" = "1" ]; then
        RUN_CMD="QUANT_COMPRESSOR_TESTCASE_PATH=\"${case_file}\" msprof python3 -m pytest -rA -s ${TEST_SCRIPT} -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning"
    else
        RUN_CMD="QUANT_COMPRESSOR_TESTCASE_PATH=\"${case_file}\" python3 -m pytest -rA -s ${TEST_SCRIPT} -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning"
    fi

    eval "$RUN_CMD" 2>&1 | grep -v "^ninja: no work to do\.$" | tee -a "$SUMMARY_LOG"
    status=${PIPESTATUS[0]}

    if [ "$status" -eq 0 ]; then
        PASS=$((PASS+1))
        echo "[PASS] $case_name" | tee -a "$SUMMARY_LOG"
        # 增量收集性能数据（每条用例跑完立即写入 result_perf.xlsx）
        if [ "$USE_MSPROF" = "1" ]; then
            sync
            python3 collect_perf_data.py --incremental --test_result_path "$RESULT_XLSX" 2>&1 | tee -a "$SUMMARY_LOG"
            # 重命名 PROF 文件夹，防止下一条用例的 msprof 覆盖
            _latest_prof=$(ls -dt PROF_*/ 2>/dev/null | head -1)
            if [ -n "$_latest_prof" ]; then
                _new_name="${_latest_prof%/}_${case_name%.pt}"
                mv "$_latest_prof" "$_new_name" 2>/dev/null
            fi
        fi
    else
        FAIL=$((FAIL+1))
        FAIL_LIST+=("$case_name")
        echo "[FAIL] $case_name" | tee -a "$SUMMARY_LOG"
        echo "$case_name" >> "$FAIL_LOG"
    fi
done

# 3. 最终汇总（批量模式兜底，确保所有用例的性能数据都已收集）
if [ "$USE_MSPROF" = "1" ]; then
    echo -e "\n========== 性能数据汇总校验 ==========" | tee -a "$SUMMARY_LOG"
    python3 collect_perf_data.py --test_result_path "$RESULT_XLSX" 2>&1 | tee -a "$SUMMARY_LOG"
fi

# 汇总
echo -e "\n========== 批量执行汇总 ==========" | tee -a "$SUMMARY_LOG"
echo "总计: $TOTAL  通过: $PASS  失败: $FAIL" | tee -a "$SUMMARY_LOG"
if [ "$FAIL" -gt 0 ]; then
    echo "失败用例:" | tee -a "$SUMMARY_LOG"
    for f in "${FAIL_LIST[@]}"; do
        echo "  - $f" | tee -a "$SUMMARY_LOG"
    done
fi
echo "详细日志:  $SUMMARY_LOG"
echo "失败清单:  $FAIL_LOG"
echo "结果表格:  $RESULT_XLSX"
if [ "$USE_MSPROF" = "1" ]; then
    echo "性能表格:  ${RESULT_XLSX%.xlsx}_perf.xlsx"
fi

exit 0
