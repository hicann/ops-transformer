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

# 脚本路径
QSMLA_PT_SAVE_SCRIPT="./batch/test_quant_sparse_flash_mla_pt_save.py"
TEST_QSMLA_PT_BATCH_SCRIPT="test_quant_sparse_flash_mla_batch.py"
TEST_QSMLA_SINGLE_SCRIPT="test_quant_sparse_flash_mla_single.py"
RUN_GRAPH=0
# 默认参数
PT_SAVE_DIR="qsmla_testcase"
EXCEL_FILE="./excel/example.xlsx"
SHEET_NAME="decode"
KEEP_PT=false
BATCH_CONSISTENCY_POLICY="auto"

# ====================== 执行区 ======================

# 单用例算子调测
run_single() {
    echo "===== 执行单用例算子调测 ====="
    export RUN_GRAPH="$RUN_GRAPH"
    export QSMLA_BATCH_CONSISTENCY="$BATCH_CONSISTENCY_POLICY"
    python3 -m pytest -rA -s $TEST_QSMLA_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning --show-capture=no
}

# 批量从excel读取用例、golden计算并保存pt文件
run_batch_save() {
    echo "===== 执行batch_save：从excel读取用例并保存pt文件 ====="
    echo "  excel: $EXCEL_FILE"
    echo "  sheet: $SHEET_NAME"
    echo "  pt目录: $PT_SAVE_DIR"
    export QSMLA_EXCEL="$EXCEL_FILE"
    export QSMLA_SHEET="$SHEET_NAME"
    export QSMLA_PT_DIR="$PT_SAVE_DIR"
    export QSMLA_BATCH_CONSISTENCY="$BATCH_CONSISTENCY_POLICY"
    python3 -m pytest -rA -s $QSMLA_PT_SAVE_SCRIPT -x -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning --show-capture=no
    if [ $? -ne 0 ]; then
        echo "batch_save 执行失败，退出"
        exit 1
    fi
    echo -e "\n===== batch_save 完成！pt文件保存在 $PT_SAVE_DIR 目录 ====="
}

# 批量读取pt文件并执行NPU测试
run_batch_exec() {
    echo "===== 执行batch_exec：读取pt文件并执行NPU测试 ====="

    if [ ! -d "$PT_SAVE_DIR" ]; then
        echo "错误: pt文件目录不存在: $PT_SAVE_DIR，请先执行 batch_save 生成pt文件"
        exit 1
    fi

    pt_count=$(ls -1 $PT_SAVE_DIR/*.pt 2>/dev/null | wc -l)
    if [ $pt_count -eq 0 ]; then
        echo "错误: 目录中没有找到.pt文件: $PT_SAVE_DIR，请先执行 batch_save 生成pt文件"
        exit 1
    fi

    echo "找到 $pt_count 个pt文件，开始执行NPU测试"
    export QSMLA_PT_DIR="$PT_SAVE_DIR"
    export RUN_GRAPH="$RUN_GRAPH"
    export QSMLA_BATCH_CONSISTENCY="$BATCH_CONSISTENCY_POLICY"
    python3 -m pytest -rA -s $TEST_QSMLA_PT_BATCH_SCRIPT -x -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning --show-capture=no
    if [ $? -ne 0 ]; then
        echo "batch_exec 执行失败"
        exit 1
    fi

    echo -e "\n===== batch_exec 完成！====="
}

# 批量从excel读取用例、生成pt文件并执行NPU测试
run_batch() {
    echo "===== 执行batch：从excel批量生成pt并执行NPU测试 ====="

    echo -e "\n===== 第一步：batch_save - 从excel生成pt文件 ====="
    run_batch_save
    if [ $? -ne 0 ]; then
        exit 1
    fi

    echo -e "\n===== 第二步：batch_exec - 读取pt文件执行NPU测试 ====="
    run_batch_exec
    if [ $? -ne 0 ]; then
        exit 1
    fi

    # 根据KEEP_PT决定是否清理pt文件
    if [ "$KEEP_PT" = false ]; then
        echo -e "\n===== 清理pt文件（KEEP_PT=false） ====="
        rm -rf $PT_SAVE_DIR
        echo "pt文件已清理"
    else
        echo -e "\n===== 保留pt文件（KEEP_PT=true）保存在 $PT_SAVE_DIR ====="
    fi

    echo -e "\n===== batch 全流程执行完成！====="
}

# 显示帮助信息
show_help() {
    echo "用法: $0 <命令> [选项]"
    echo ""
    echo "命令说明："
    echo "  single       执行单算子用例调测"
    echo "  batch_save   从excel读取用例，golden计算并保存pt文件"
    echo "  batch_exec   批量读取pt文件并执行NPU测试"
    echo "  batch        全流程：从excel生成pt + NPU测试"
    echo "  help         显示本帮助信息"
    echo ""
    echo "选项（batch_save/batch_exec/batch 命令支持）："
    echo "  --excel <路径>    指定excel文件路径（默认: ./excel/example.xlsx）"
    echo "  --sheet <名称>    指定excel sheet名（默认: decode）"
    echo "  --pt-dir <目录>   指定pt文件保存/读取目录（默认: qsmla_testcase）"
    echo "  --keep-pt         执行完成后保留pt文件（默认清理，仅batch命令）"
    echo "  --run-graph       执行aclgraph模式（默认不开启）"
    echo "  --batch-consistency <auto|on|off>  batch一致性策略（默认: auto）"
    echo ""
    echo "示例："
    echo "  $0 single                              # eager模式下执行single模式"
    echo "  $0 single                              # aclgraph模式下执行single模式"
    echo "  $0 batch_save                          # 用默认参数生成pt文件"
    echo "  $0 batch_save --excel my.xlsx --sheet prefill --pt-dir my_pt"
    echo "  $0 batch_exec                          # 用默认pt目录执行NPU测试"
    echo "  $0 batch_exec --pt-dir my_pt           # 从指定目录读取pt文件"
    echo "  $0 batch                               # 全流程，完成后清理pt文件"
    echo "  $0 batch --keep-pt                     # 全流程，完成后保留pt文件"
    echo "  $0 batch --excel my.xlsx --sheet prefill --pt-dir my_pt --keep-pt"
}

# ====================== 主逻辑 ======================

# 解析参数
if [ $# -lt 1 ]; then
    echo "错误：必须传入至少一个命令参数"
    show_help
    exit 1
fi

COMMAND="$1"
shift

# 解析选项（batch_save/batch_exec/batch 共用）
if [ "$COMMAND" = "batch_save" ] || [ "$COMMAND" = "batch_exec" ] || [ "$COMMAND" = "batch" ]; then
    while [ $# -gt 0 ]; do
        case "$1" in
            --excel)
                if [ $# -lt 2 ]; then
                    echo "错误：--excel 需要参数值"
                    exit 1
                fi
                EXCEL_FILE="$2"
                shift 2
                ;;
            --sheet)
                if [ $# -lt 2 ]; then
                    echo "错误：--sheet 需要参数值"
                    exit 1
                fi
                SHEET_NAME="$2"
                shift 2
                ;;
            --pt-dir)
                if [ $# -lt 2 ]; then
                    echo "错误：--pt-dir 需要参数值"
                    exit 1
                fi
                PT_SAVE_DIR="$2"
                shift 2
                ;;
            --keep-pt)
                if [ "$COMMAND" != "batch" ]; then
                    echo "错误：--keep-pt 仅适用于 batch 命令"
                    exit 1
                fi
                KEEP_PT=true
                shift
                ;;
            --run-graph)
                if [ "$COMMAND" == "batch_save" ]; then
                    echo "错误：--run-graph 不适用于batch_save"
                    exit 1
                fi
                RUN_GRAPH=1
                shift
                ;;
            --batch-consistency)
                if [ $# -lt 2 ] || { [ "$2" != "auto" ] && [ "$2" != "on" ] && [ "$2" != "off" ]; }; then
                    echo "错误：--batch-consistency 需要 auto、on 或 off"
                    exit 1
                fi
                BATCH_CONSISTENCY_POLICY="$2"
                shift 2
                ;;
            *)
                echo "错误：未知选项 '$1'"
                show_help
                exit 1
                ;;
        esac
    done
fi


if [ "$COMMAND" = "single" ] ; then
    while [ $# -gt 0 ]; do
        case "$1" in
            --run-graph)
                RUN_GRAPH=1
                shift
                ;;
            --batch-consistency)
                if [ $# -lt 2 ] || { [ "$2" != "auto" ] && [ "$2" != "on" ] && [ "$2" != "off" ]; }; then
                    echo "错误：--batch-consistency 需要 auto、on 或 off"
                    exit 1
                fi
                BATCH_CONSISTENCY_POLICY="$2"
                shift 2
                ;;
            *)
                echo "错误：未知选项 '$1'"
                show_help
                exit 1
                ;;
        esac
    done
fi

# 根据命令执行对应函数
case "$COMMAND" in
    single)
        run_single
        ;;
    batch_save)
        run_batch_save
        ;;
    batch_exec)
        run_batch_exec
        ;;
    batch)
        run_batch
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
