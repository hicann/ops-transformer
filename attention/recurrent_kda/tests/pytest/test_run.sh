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

set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TEST_RECURRENT_KDA_SINGLE_SCRIPT="test_accuracy.py"

run_single() {
    echo "===== 执行RecurrentKda单算子精度测试 ====="
    cd "$SCRIPT_DIR" || exit 1
    python3 -m pytest -rA -s "$TEST_RECURRENT_KDA_SINGLE_SCRIPT" -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single    执行单算子精度测试"
    echo "  help      显示本帮助信息"
}

if [ "$#" -ne 1 ]; then
    echo "错误：必须传入且仅传入一个参数（single/help）"
    show_help
    exit 1
fi

case "$1" in
    single)
        run_single
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/help"
        show_help
        exit 1
        ;;
esac
