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

# ====================== 配置区======================
PATH1="./csv/stem_indexer_generalized_cases.csv"
PATH2="./pt_path"

STEM_INDEXER_PT_SAVE_SCRIPT="./batch/stem_indexer_pt_save.py"
TEST_STEM_INDEXER_SINGLE_SCRIPT="test_stem_indexer_single.py"
TEST_STEM_INDEXER_BATCH_SCRIPT="test_stem_indexer_batch.py"

# ====================== 执行区======================

# 单用例算子调测
# 用法: run_single [eager|graph]
run_single() {
    local exec_mode="${1:-eager}"
    local pytest_mark="ci"
    local mode_label="eager"
    if [ "$exec_mode" == "graph" ]; then
        pytest_mark="graph"
        mode_label="graph"
    fi
    echo "===== 执行单用例算子调测（${mode_label}模式） ====="
    STEM_INDEXER_MODE="$exec_mode" python3 -m pytest -rA -s $TEST_STEM_INDEXER_SINGLE_SCRIPT -v -m ${pytest_mark} -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 用例批量生成调试
# 用法: run_batch [eager|graph]
run_batch() {
    local exec_mode="${1:-eager}"
    local pytest_mark="ci"
    local mode_label="eager"
    if [ "$exec_mode" == "graph" ]; then
        pytest_mark="graph"
        mode_label="graph"
    fi
    echo "===== 执行用例批量生成测试（${mode_label}模式） ====="

    rm -f result.csv

    # 第一步：生成pt文件（不再按BatchSize分轮次）
    echo "生成pt文件..."
    python3 $STEM_INDEXER_PT_SAVE_SCRIPT $PATH1 $PATH2
    if [ $? -ne 0 ]; then
        echo "pt_save.py 执行失败，退出"
        exit 1
    fi

    # 第二步：执行pytest
    echo "执行pytest [${mode_label}]..."
    STEM_INDEXER_PT_DIR="$PATH2" \
    STEM_INDEXER_RESULT_PATH="result.csv" \
    STEM_INDEXER_MODE="$exec_mode" \
    python3 -m pytest -rA -s $TEST_STEM_INDEXER_BATCH_SCRIPT -v -m ${pytest_mark} \
        -W ignore::UserWarning -W ignore::DeprecationWarning
    pytest_status=$?
    if [ $pytest_status -ne 0 ]; then
        echo "pytest有失败用例"
    fi

    echo -e "\n=====全部执行完成！[${mode_label}]模式====="
    echo "结果文件: result.csv"
    if [ $pytest_status -ne 0 ]; then
        exit 1
    fi
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single       执行单算子用例调测（eager）"
    echo "  single_graph 执行单算子用例调测（graph图模式）"
    echo "  batch        执行用例批量生成调测（eager模式）"
    echo "  batch_graph  执行用例批量生成调测（graph图模式）"
    echo "  help         显示本帮助信息"
    echo "示例："
    echo "  $0 single        # 执行single模式（eager）"
    echo "  $0 single_graph  # 执行single模式（graph图模式）"
    echo "  $0 batch         # 执行batch模式（eager）"
    echo "  $0 batch_graph   # 执行batch模式（graph图模式）"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -ne 1 ]; then
    echo "错误：必须传入且仅传入一个参数（single/single_graph/batch/batch_graph/help）"
    show_help
    exit 1
fi

# 根据参数执行对应函数
case "$1" in
    single)
        run_single eager
        ;;
    single_graph)
        run_single graph
        ;;
    batch)
        run_batch eager
        ;;
    batch_graph)
        run_batch graph
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/single_graph/batch/batch_graph/help"
        show_help
        exit 1
        ;;
esac

exit 0
