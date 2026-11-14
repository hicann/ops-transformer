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

# QuantSparseLightningIndexer（两级 TopK 第二级 consumer）pytest 驱动脚本。
# 用例全部集中在 test_quant_sparse_lightning_indexer_consumer.py：
#   - 默认执行 38 条「吃 golden 候选」的算子级精度用例；
#   - E2E_IDS 内的用例额外执行「吃第一级 NPU 候选」的链路一致性检查。
# 运行前需 source CANN 的 set_env.sh（否则拿不到含本算子的 cann_ops_transformer 包），
# 并用 ASCEND_RT_VISIBLE_DEVICES 指定空闲卡。

TEST_FILE="test_quant_sparse_lightning_indexer_consumer.py"

# 全量用例
run_all() {
    echo "===== 执行全量用例（golden 候选 + E2E 链路） ====="
    python3 -m pytest -rA -s "${TEST_FILE}" -v -m ci \
        -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 指定用例（按 -k 匹配用例 id）
run_single() {
    local case_names="$1"
    if [ -z "${case_names}" ]; then
        echo "错误：single 需要指定用例 id"
        exit 1
    fi
    echo "===== 执行指定用例: ${case_names} ====="
    python3 -m pytest -rA -s "${TEST_FILE}" -v -m ci -k "${case_names}" \
        -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 显示帮助信息
show_help() {
    echo "用法: $0 <command> [options]"
    echo "命令说明："
    echo "  all                          执行全量用例（38 条 golden 候选 + E2E 链路）"
    echo "  single -k <id>               执行指定用例（-k 支持 pytest 的匹配表达式）"
    echo "示例："
    echo "  $0 all"
    echo "  $0 single -k pin_overflow"
    echo "  $0 single -k \"decode or prefill\""
}

# ====================== 主逻辑 ======================
if [ $# -lt 1 ]; then
    echo "错误：必须传入至少一个参数"
    show_help
    exit 1
fi

COMMAND="$1"
shift

case "$COMMAND" in
    all)
        run_all
        ;;
    single)
        case_names=""
        while [ $# -gt 0 ]; do
            case "$1" in
                -k|--cases)
                    if [ $# -lt 2 ]; then
                        echo "错误：$1 缺少参数值"
                        exit 1
                    fi
                    case_names="$2"
                    shift 2
                    ;;
                *)
                    echo "错误：未知 single 参数 '$1'"
                    show_help
                    exit 1
                    ;;
            esac
        done
        run_single "$case_names"
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知命令 '$COMMAND'"
        show_help
        exit 1
        ;;
esac

exit 0
