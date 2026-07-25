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
PATH1="./csv/stem_indexer_cases.csv"
PATH2="./pt_path"

STEM_INDEXER_PT_SAVE_SCRIPT="./batch/stem_indexer_pt_save.py"
TEST_STEM_INDEXER_SINGLE_SCRIPT="test_stem_indexer_single.py"
TEST_STEM_INDEXER_BATCH_SCRIPT="test_stem_indexer_batch.py"

# ====================== 执行区======================

# 单用例算子调测
run_single() {
    echo "===== 执行单用例算子调测 ====="
    python3 -m pytest -rA -s $TEST_STEM_INDEXER_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 用例批量生成调试
run_batch() {
    echo "===== 执行用例批量生成测试 ====="

    echo -e "\n===== 第一步：执行stem_indexer_pt_save.py ====="
    python3 $STEM_INDEXER_PT_SAVE_SCRIPT $PATH1 $PATH2
    if [ $? -ne 0 ]; then
        echo "stem_indexer_pt_save.py 执行失败，退出"
        exit 1
    fi

    echo -e "\n===== 第二步：执行pytest命令 ====="
    STEM_INDEXER_PT_DIR="$PATH2" python3 -m pytest -rA -s $TEST_STEM_INDEXER_BATCH_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    pytest_status=$?
    if [ $pytest_status -ne 0 ]; then
        echo "pytest执行失败"
        exit 1
    fi

    echo -e "\n=====执行完成！====="
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single    执行单算子用例调测"
    echo "  batch     执行用例批量生成调测"
    echo "  help      显示本帮助信息"
    echo "示例："
    echo "  $0 single  # 执行single模式"
    echo "  $0 batch   # 执行batch模式"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -ne 1 ]; then
    echo "错误：必须传入且仅传入一个参数（single/batch/help）"
    show_help
    exit 1
fi

# 根据参数执行对应函数
case "$1" in
    single)
        run_single
        ;;
    batch)
        run_batch
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/batch/help"
        show_help
        exit 1
        ;;
esac

exit 0
