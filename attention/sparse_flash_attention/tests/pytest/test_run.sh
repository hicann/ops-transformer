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

set -euo pipefail

# 脚本路径
PT_SAVE_SCRIPT="./batch/test_sparse_flash_attention_pt_save.py"
GEN_EXCEL_SCRIPT="./batch/gen_excel_from_paramset.py"
TEST_BATCH_SCRIPT="test_sparse_flash_attention_batch.py"
TEST_SINGLE_SCRIPT="test_sparse_flash_attention_single.py"
PT_SAVE_PATH="./pt_files/"

mkdir -p $PT_SAVE_PATH

# ====================== 执行区======================

# 单用例算子调测
run_single() {
    local paramset_file="$1"
    echo "===== 执行单用例算子调测 ====="
    if [ -n "$paramset_file" ]; then
        export PARAMSET_FILE="$paramset_file"
        echo "使用 paramset 文件: $paramset_file"
    fi
    python3 -m pytest -rA -s "$TEST_SINGLE_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 从 Excel 批量生成 pt 文件
run_batch_save() {
    local excel_path="$1"
    local excel_sheet="$2"
    local pt_path="$3"
    echo "===== 批量生成 pt 文件 ====="
    if [ -n "$excel_path" ]; then
        export EXCEL_PATH="$excel_path"
        echo "使用 Excel 文件: $excel_path"
    fi
    if [ -n "$excel_sheet" ]; then
        export EXCEL_SHEET="$excel_sheet"
        echo "使用 Sheet 页: $excel_sheet"
    fi
    if [ -n "$pt_path" ]; then
        echo "pt 文件保存路径: $pt_path"
        mkdir -p "$pt_path"
        sed -i "s|PT_SAVE_PATH = .*|PT_SAVE_PATH = \"$pt_path/\"|" "$PT_SAVE_SCRIPT"
    fi
    python3 -m pytest -rA -s "$PT_SAVE_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "生成 pt 文件失败，退出"
        exit 1
    fi
    echo "===== pt 文件生成完成 ====="
}

# 从 paramset 生成 Excel 文件
run_gen_excel_from_paramset() {
    local paramset_file="$1"
    local excel_output="$2"
    local excel_sheet="$3"
    echo "===== 从 paramset 生成 Excel 文件 ====="
    if [ -n "$paramset_file" ]; then
        export PARAMSET_FILE="$paramset_file"
        echo "使用 paramset 文件: $paramset_file"
    fi
    if [ -n "$excel_output" ]; then
        export EXCEL_OUTPUT_PATH="$excel_output"
        echo "输出 Excel 文件: $excel_output"
    fi
    if [ -n "$excel_sheet" ]; then
        export EXCEL_SHEET="$excel_sheet"
        echo "Sheet 名称: $excel_sheet"
    fi
    python3 "$GEN_EXCEL_SCRIPT"
    if [ $? -ne 0 ]; then
        echo "生成 Excel 文件失败，退出"
        exit 1
    fi
    echo "===== Excel 文件生成完成 ====="
}

# 从 pt 文件批量执行 NPU 测试
run_batch_exec() {
    local pt_files_path="$1"
    echo "===== 从 pt 文件批量执行 NPU 测试 ====="
    if [ -n "$pt_files_path" ]; then
        export PT_FILES_PATH="$pt_files_path"
        echo "使用 pt 文件路径: $pt_files_path"
    fi
    python3 -m pytest -rA -s "$TEST_BATCH_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "批量执行失败"
        exit 1
    fi
    echo "===== 批量执行完成 ====="
}

show_help() {
    echo "用法: $0 <模式> [-E excel_path] [-S sheet] [-P path]"
    echo ""
    echo "模式说明："
    echo "  single        执行单算子用例调测（含 CPU golden + NPU + 精度对比）"
    echo "                选项: -P paramset_file (指定 paramset 文件名，默认: sparse_flash_attention_paramset)"
    echo ""
    echo "  batch_save    从 Excel 批量生成 pt 文件"
    echo "                选项: -E excel_path (指定 Excel 文件路径，默认: ./excel/example.xlsx)"
    echo "                      -S sheet (指定 Sheet 页名，默认: Sheet1)"
    echo "                      -P pt_path (指定 pt 文件保存路径，默认: ./pt_files/)"
    echo ""
    echo "  gen_excel_from_paramset  从 paramset 生成 Excel 文件"
    echo "                选项: -P paramset_file (指定 paramset 文件名，默认: sparse_flash_attention_paramset)"
    echo "                      -E excel_output (指定输出 Excel 文件路径，默认: ./excel/example.xlsx)"
    echo "                      -S sheet (指定 Sheet 页名，默认: Sheet1)"
    echo ""
    echo "  batch_exec    从 pt 文件批量执行 NPU 测试"
    echo "                选项: -P pt_files_path (指定 pt 文件路径，目录或单个 pt 文件，默认: ./pt_files/)"
    echo ""
    echo "  help          显示本帮助信息"
    echo ""
    echo "注意事项："
    echo "  1. -E、-S、-P 选项可省略，使用默认值"
    echo "  2. 选项顺序可任意，如 batch_save -S Sheet1 -E ./test.xlsx"
    echo ""
    echo "示例："
    echo "  $0 single"
    echo "  $0 single -P my_paramset"
    echo "  $0 batch_save"
    echo "  $0 batch_save -E ./test.xlsx"
    echo "  $0 batch_save -E ./test.xlsx -S Sheet1"
    echo "  $0 batch_save -E ./test.xlsx -S Sheet1 -P ./output_pt/"
    echo "  $0 batch_save -S Sheet1 -E ./test.xlsx                # 参数顺序可任意"
    echo "  $0 gen_excel_from_paramset"
    echo "  $0 gen_excel_from_paramset -P my_paramset"
    echo "  $0 gen_excel_from_paramset -P my_paramset -E ./output/example.xlsx"
    echo "  $0 gen_excel_from_paramset -P my_paramset -E ./output/example.xlsx -S decode"
    echo "  $0 batch_exec"
    echo "  $0 batch_exec -P ./pt_files/test.pt"
    echo "  $0 batch_exec -P ./custom_pt_dir/"
}

parse_options() {
    local excel_path=""
    local excel_sheet=""
    local pt_path=""
    local paramset_file=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -E)
                excel_path="$2"
                shift 2
                ;;
            -S)
                excel_sheet="$2"
                shift 2
                ;;
            -P)
                pt_path="$2"
                paramset_file="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    EXCEL_PATH="$excel_path"
    EXCEL_SHEET="$excel_sheet"
    PT_PATH="$pt_path"
    PARAMSET_FILE="$paramset_file"
}

if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

mode="$1"
shift

EXCEL_PATH=""
EXCEL_SHEET=""
PT_PATH=""
PARAMSET_FILE=""

parse_options "$@"

case "$mode" in
    single)
        run_single "$PT_PATH"
        ;;
    batch_save)
        run_batch_save "$EXCEL_PATH" "$EXCEL_SHEET" "$PT_PATH"
        ;;
    gen_excel_from_paramset)
        run_gen_excel_from_paramset "$PARAMSET_FILE" "$EXCEL_PATH" "$EXCEL_SHEET"
        ;;
    batch_exec)
        run_batch_exec "$PT_PATH"
        ;;
    help)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
