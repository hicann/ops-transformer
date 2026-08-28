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

# 脚本路径
TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT="test_recurrent_gated_delta_rule_single.py"

# ====================== 执行区======================

# 算子调测
run_single() {
    echo "===== 执行单算子用例调测 ====="
    TEST_MODE=single python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# RDV测试
run_rdv() {
    echo "===== 执行RDV参数集测试 ====="
    TEST_MODE=rdv python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 随机用例测试
run_random() {
    local count="${1:-100}"
    echo "===== 执行随机用例调测 ($count 条) ====="
    TEST_MODE=random RANDOM_CASE_COUNT=$count python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single       执行单算子用例调测"
    echo "  rdv          执行RDV参数集测试"
    echo "  random [N]   随机生成并执行N条用例（默认100）"
    echo "  help         显示本帮助信息"
    echo "示例："
    echo "  $0 single     # 执行single模式"
    echo "  $0 rdv        # 执行rdv模式"
    echo "  $0 random 100  # 随机执行100条用例"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "错误：参数数量错误，用法 $0 {single|rdv|random [N]|help}"
    show_help
    exit 1
fi

# 根据参数执行对应函数
case "$1" in
    single)
        run_single
        ;;
    rdv)
        run_rdv
        ;;
    random)
        run_random "$2"
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/rdv/random/help"
        show_help
        exit 1
        ;;
esac

exit 0
