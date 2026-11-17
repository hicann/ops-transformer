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

# Keep the switch available to pytest subprocesses and isolated batch workers.
super_kernel_args=()
for super_kernel_arg in "$@"; do
    if [[ "$super_kernel_arg" == "--super-kernel" ]]; then
        export LIV2_SUPER_KERNEL=1
    else
        super_kernel_args+=("$super_kernel_arg")
    fi
done
set -- "${super_kernel_args[@]}"

# ====================== 配置区======================
# 默认读取的用例excel表格路径（须为具体文件，不支持通配符），如下：
DEFAULT_EXCEL="./excel/test_cases.xlsx"
# 用例pt的文件存放路径，如下：
DEFAULT_PT_PATH="./pt_path"
# 默认Sheet页名，如下：
DEFAULT_SHEET="Sheet1"

# 脚本路径
LIV2_PT_SAVE_SCRIPT="./batch/lightning_indexer_v2_pt_save.py"
LIST_PT_FROM_EXCEL_SCRIPT="./batch/list_pt_from_excel.py"
TEST_LIV2_BATCH_SCRIPT="test_lightning_indexer_v2_batch.py"
TEST_LIV2_SINGLE_SCRIPT="test_lightning_indexer_v2_single.py"
TEST_LIV2_CANDIDATE_SCRIPT="test_lightning_indexer_v2_candidate.py"
TEST_LIV2_CANDIDATE_REJECT_SCRIPT="test_lightning_indexer_v2_candidate_reject.py"

# ====================== 执行区======================

# 单用例算子调测
run_single() {
    local paramset="${1:-default}"
    local case_names="$2"
    local run_mode="${3:-both}"
    echo "===== 执行单用例算子调测 ====="
    LIV2_PARAMSET="${paramset}" LIV2_CASE_NAMES="${case_names}" \
    LIV2_RUN_MODE="${run_mode}" \
        python3 -m pytest -rA -s $TEST_LIV2_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# candidate（两级 TopK source）功能用例
run_candidate() {
    local case_names="$1"
    echo "===== 执行 candidate 功能用例 ====="
    LIV2_CAND_CASE_NAMES="${case_names}" \
        python3 -m pytest -rA -s $TEST_LIV2_CANDIDATE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# candidate host 校验拒绝用例
run_candidate_reject() {
    echo "===== 执行 candidate host 校验拒绝用例 ====="
    python3 -m pytest -rA -s $TEST_LIV2_CANDIDATE_REJECT_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 校验运行模式
check_run_mode() {
    local cmd="$1"
    local run_mode="$2"
    if [[ "$run_mode" != "eager" && "$run_mode" != "graph" ]]; then
        echo "错误：${cmd} 模式仅支持 eager/graph，当前值: $run_mode"
        exit 1
    fi
}

# 用例批量生成调试（先生成pt，再执行NPU测试和精度对比）
# 用法: run_batch <excel> <sheet> <pt_path> <eager|graph>
run_batch() {
    local excel_path="${1:-$DEFAULT_EXCEL}"
    local sheet_name="${2:-$DEFAULT_SHEET}"
    local pt_path="${3:-$DEFAULT_PT_PATH}"
    local run_mode="${4:-eager}"
    echo "===== 执行用例批量生成测试 (模式: ${run_mode}) ====="

    if [ ! -f "$excel_path" ]; then
        echo "错误：Excel文件不存在: $excel_path（可通过 -E 指定，默认 $DEFAULT_EXCEL）"
        exit 1
    fi

    echo -e "\n===== 第一步：执行lightning_indexer_v2_pt_save.py ====="
    python3 $LIV2_PT_SAVE_SCRIPT "$excel_path" "$pt_path" --sheet "$sheet_name"
    if [ $? -ne 0 ]; then
        echo "lightning_indexer_v2_pt_save.py 执行失败，退出"
        exit 1
    fi

    echo -e "\n===== 第二步：执行pytest命令 (LIV2_RUN_MODE=${run_mode}) ====="
    LIV2_TESTCASE_DIR="${pt_path}" LIV2_RUN_MODE="${run_mode}" \
        python3 -m pytest -rA -s $TEST_LIV2_BATCH_SCRIPT -v -m ci
    if [ $? -ne 0 ]; then
        echo "pytest执行失败"
        exit 1
    fi

    echo -e "\n=====执行完成！====="
}

# batch_exec：按Excel的Testcase_Name筛选已有pt文件，跳过pt生成直接执行NPU测试和精度对比
# 用法: run_batch_exec <excel> <sheet> <pt_path> <eager|graph>
run_batch_exec() {
    local excel_path="${1:-$DEFAULT_EXCEL}"
    local sheet_name="${2:-$DEFAULT_SHEET}"
    local pt_path="${3:-$DEFAULT_PT_PATH}"
    local run_mode="${4:-eager}"
    echo "===== 执行batch_exec（按Excel筛选已有pt，模式: ${run_mode}） ====="

    if [ ! -f "$excel_path" ]; then
        echo "错误：Excel文件不存在: $excel_path（可通过 -E 指定，默认 $DEFAULT_EXCEL）"
        exit 1
    fi
    if [ ! -d "$pt_path" ]; then
        echo "错误：pt目录不存在: $pt_path（可通过 -P 指定，默认 $DEFAULT_PT_PATH）"
        exit 1
    fi

    echo -e "\n===== 第一步：执行list_pt_from_excel.py 筛选pt文件 ====="
    local file_list
    file_list=$(python3 $LIST_PT_FROM_EXCEL_SCRIPT "$excel_path" "$pt_path" --sheet "$sheet_name")
    if [ $? -ne 0 ]; then
        echo "筛选pt文件失败，退出"
        exit 1
    fi

    echo -e "\n===== 第二步：执行pytest命令 (LIV2_RUN_MODE=${run_mode}) ====="
    LIV2_PT_FILE_LIST="${file_list}" LIV2_TESTCASE_DIR="${pt_path}" LIV2_RUN_MODE="${run_mode}" \
        python3 -m pytest -rA -s $TEST_LIV2_BATCH_SCRIPT -v -m ci
    if [ $? -ne 0 ]; then
        echo "pytest执行失败"
        exit 1
    fi

    echo -e "\n=====执行完成！====="
}

# 显示帮助信息
show_help() {
    echo "用法: $0 <command> [options]"
    echo "命令说明："
    echo "  single [default|stc] [eager|graph|both]"
    echo "  single [--paramset default|stc] [-C CASE[,CASE...]] [-M eager|graph|both]"
    echo "  candidate [-C CASE[,CASE...]]         执行 candidate（两级 TopK source）功能用例"
    echo "  candidate_reject                      执行 candidate host 校验拒绝用例"
    echo "  batch [options]      执行用例批量生成调试（生成pt后执行NPU测试）"
    echo "  batch_exec [options] 按Excel的Testcase_Name筛选已有pt直接执行NPU测试（跳过pt生成）"
    echo ""
    echo "  --super-kernel      Enable ACLGraph SuperKernel with full-core debug (single/batch/batch_exec)"
    echo ""
    echo "batch/batch_exec 选项："
    echo "  -E, --excel FILE     Excel用例表格路径（默认 $DEFAULT_EXCEL，须为具体文件）"
    echo "  -S, --sheet NAME     Excel Sheet页名（默认 $DEFAULT_SHEET）"
    echo "  -P, --pt-path DIR    pt文件存放目录（默认 $DEFAULT_PT_PATH）"
    echo "  -M, --run-mode MODE  运行模式 eager|graph（默认 eager）"
    echo ""
    echo "示例："
    echo "  $0 single        # 执行single模式"
    echo "  $0 single stc eager # 以eager模式执行STC白盒参数集"
    echo "  $0 single --paramset stc -C S2_16_003 -M graph"
    echo "  $0 candidate     # 全量 candidate 用例"
    echo "  $0 candidate -C LIV2_CAND_BS_2,LIV2_CAND_TND_TND"
    echo "  $0 candidate_reject"
    echo "  $0 batch         # 使用配置区默认值执行batch模式（默认eager）"
    echo "  $0 batch graph   # 使用graph模式（兼容旧用法，直接跟run_mode）"
    echo "  $0 batch -E ./excel/test_cases.xlsx -P ./pt_path -M graph"
    echo "  $0 batch_exec -E ./excel/test_cases.xlsx -P ./pt_path -M graph"
}

# 解析batch/batch_exec的选项
parse_batch_options() {
    local cmd="$1"
    shift
    while [ $# -gt 0 ]; do
        case "$1" in
            -E|--excel)
                if [ $# -lt 2 ]; then
                    echo "错误：$1 缺少参数值"
                    exit 1
                fi
                EXCEL_PATH="$2"
                shift 2
                ;;
            -S|--sheet)
                if [ $# -lt 2 ]; then
                    echo "错误：$1 缺少参数值"
                    exit 1
                fi
                SHEET_NAME="$2"
                shift 2
                ;;
            -P|--pt-path)
                if [ $# -lt 2 ]; then
                    echo "错误：$1 缺少参数值"
                    exit 1
                fi
                PT_PATH="$2"
                shift 2
                ;;
            -M|--run-mode)
                if [ $# -lt 2 ]; then
                    echo "错误：$1 缺少参数值"
                    exit 1
                fi
                RUN_MODE="$2"
                shift 2
                ;;
            -*)
                echo "错误：未知 ${cmd} 参数 '$1'"
                show_help
                exit 1
                ;;
            *)
                # 兼容旧用法：bash test_run.sh batch [eager|graph]
                RUN_MODE="$1"
                shift
                ;;
        esac
    done
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -lt 1 ]; then
  	     echo "错误：必须传入至少一个参数"
    show_help
    exit 1
fi

COMMAND="$1"
EXCEL_PATH=""
SHEET_NAME="$DEFAULT_SHEET"
PT_PATH="$DEFAULT_PT_PATH"
RUN_MODE="eager"

# 根据参数执行对应函数
case "$COMMAND" in
    single)
        shift
        paramset="default"
        case_names=""
        run_mode="both"
        if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
            paramset="$1"
            shift
            if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
                run_mode="$1"
                shift
            fi
        fi
        while [ $# -gt 0 ]; do
            case "$1" in
                --paramset)
                    if [ $# -lt 2 ]; then
                        echo "错误：--paramset 缺少参数值"
                        exit 1
                    fi
                    paramset="$2"
                    shift 2
                    ;;
                -C|--cases)
                    if [ $# -lt 2 ]; then
                        echo "错误：$1 缺少参数值"
                        exit 1
                    fi
                    case_names="$2"
                    shift 2
                    ;;
                -M|--run-mode)
                    if [ $# -lt 2 ]; then
                        echo "错误：$1 缺少参数值"
                        exit 1
                    fi
                    run_mode="$2"
                    shift 2
                    ;;
                *)
                    echo "错误：未知 single 参数 '$1'"
                    show_help
                    exit 1
                    ;;
            esac
        done
        if [[ "$run_mode" != "eager" && "$run_mode" != "graph" && "$run_mode" != "both" ]]; then
            echo "错误：single 模式仅支持 eager/graph/both，当前值: $run_mode"
            exit 1
        fi
        run_single "$paramset" "$case_names" "$run_mode"
        ;;
    candidate)
        shift
        case_names=""
        while [ $# -gt 0 ]; do
            case "$1" in
                -C|--cases)
                    if [ $# -lt 2 ]; then
                        echo "错误：$1 缺少参数值"
                        exit 1
                    fi
                    case_names="$2"
                    shift 2
                    ;;
                *)
                    echo "错误：未知 candidate 参数 '$1'"
                    show_help
                    exit 1
                    ;;
            esac
        done
        run_candidate "$case_names"
        ;;
    candidate_reject)
        run_candidate_reject
        ;;
    batch)
        shift
        parse_batch_options batch "$@"
        check_run_mode batch "$RUN_MODE"
        run_batch "$EXCEL_PATH" "$SHEET_NAME" "$PT_PATH" "$RUN_MODE"
        ;;
    batch_exec)
        shift
        parse_batch_options batch_exec "$@"
        check_run_mode batch_exec "$RUN_MODE"
        run_batch_exec "$EXCEL_PATH" "$SHEET_NAME" "$PT_PATH" "$RUN_MODE"
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
