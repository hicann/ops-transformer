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

# 显示帮助信息
show_help() {
    cat << EOF
使用方法: $0 {single|save|load|load_graph} [参数]

脚本选项:
  single         执行单跑功能
    -R           结果保存路径
    示例1: bash $0 single
    示例2: bash $0 single -R "./result/smla_result.xlsx"

  save           执行将excel文件中用例批量转化为PT文件的功能
    -E           excel表地址
    -S           sheet名
    -P           pt文件保存地址
    示例1: bash $0 save
    示例2: bash $0 save -E "./excel/example.xlsx" -S "Sheet1" -P "./data"

  load           执行批量执行PT形式保存的用例的功能
    -P           pt文件读取地址
    -R           结果保存路径
    -E           excel表地址（指定后仅跑表格中涉及的用例）
    -S           sheet名（配合-E使用）
    示例1: bash $0 load
    示例2: bash $0 load -P "./data" -R "./result/smla_result.xlsx"
    示例3: bash $0 load -P "./data" -E "./excel/example.xlsx" -S "CSA"

  load_graph     执行批量执行PT形式保存的用例的功能(Graph aclgraph模式)
    -P           pt文件读取地址
    -R           结果保存路径
    -E           excel表地址（指定后仅跑表格中涉及的用例）
    -S           sheet名（配合-E使用）
    示例1: bash $0 load_graph
    示例2: bash $0 load_graph -P "./data" -R "./result/smla_result.xlsx"
    示例3: bash $0 load_graph -P "./data" -E "./excel/example.xlsx" -S "CSA"

通用选项:
  --batch-consistency <auto|on|off>
               batch一致性策略（默认auto：paramset/PT有配置时开启）
  -h, --help   显示此帮助信息
EOF
}

BATCH_CONSISTENCY_POLICY="auto"

set_batch_consistency_policy() {
    if [[ "$1" != "auto" && "$1" != "on" && "$1" != "off" ]]; then
        echo "错误: --batch-consistency 仅支持 auto、on、off"
        exit 1
    fi
    BATCH_CONSISTENCY_POLICY="$1"
}

# 检查参数数量
if [ $# -lt 1 ]; then
    echo "错误: 需要至少一个参数"
    show_help
    exit 1
fi

# 获取脚本类型
SCRIPT_TYPE=$1
shift  # 移除第一个参数，剩下的都是选项

# 处理 help 请求
if [[ "$SCRIPT_TYPE" == "-h" ]] || [[ "$SCRIPT_TYPE" == "--help" ]]; then
    show_help
    exit 0
fi

# 运行 单跑 脚本的函数
run_script_single() {
    echo "准备运行 单跑 脚本..."

    # 解析 test_sparse_flash_mla_single.py 的参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -R)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -R 参数需要值"
                    exit 1
                fi
                R_VALUE="$2"
                shift 2
                ;;
            --batch-consistency)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: --batch-consistency 参数需要值"
                    exit 1
                fi
                set_batch_consistency_policy "$2"
                shift 2
                ;;
            *)
                echo "错误: 未知参数 '$1'"
                echo "test_sparse_flash_mla_single.py 脚本支持的参数: -R"
                exit 1
                ;;
        esac
    done

    # 打印参数信息
    echo "==============================="
    echo "参数配置:"

    if [ -n "$R_VALUE" ]; then
        echo "  结果存储至文件 $R_VALUE"
        export RESULT_SAVE_PATH="$R_VALUE"
    else
        echo "  结果存储至文件 ./result/smla_result.xlsx"
    fi

    echo "==============================="

    # 运行 Python 脚本
    export SMLA_BATCH_CONSISTENCY="$BATCH_CONSISTENCY_POLICY"
    if [ "$VERBOSE" = true ]; then
        echo "正在运行 test_sparse_flash_mla_single.py..."
    fi

    # 检查脚本是否存在
    if [ ! -f "test_sparse_flash_mla_single.py" ]; then
        echo "错误: 找不到 test_sparse_flash_mla_single.py 脚本"
        exit 1
    fi
    # 运行脚本，传递环境变量
    python3 -m pytest -rA -s test_sparse_flash_mla_single.py  -v -m ci
}

# 运行 批跑生成pt文件 脚本的函数
run_script_save() {
    echo "准备运行 sparse_flash_mla_pt_save.py 脚本..."

    # 解析 sparse_flash_mla_pt_save.py 的参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -E)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -E 参数需要值"
                    exit 1
                fi
                E_VALUE="$2"
                shift 2
                ;;
            -S)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -S 参数需要值"
                    exit 1
                fi
                S_VALUE="$2"
                shift 2
                ;;
            -P)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -P 参数需要值"
                    exit 1
                fi
                P_VALUE="$2"
                shift 2
                ;;
            --batch-consistency)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: --batch-consistency 参数需要值"
                    exit 1
                fi
                set_batch_consistency_policy "$2"
                shift 2
                ;;
            *)
                echo "错误: 未知参数 '$1'"
                echo "sparse_flash_mla_pt_save.py 脚本支持的参数: -E, -S, -P"
                exit 1
                ;;
        esac
    done

    # 打印参数信息
    echo "==============================="
    echo "脚本: sparse_flash_mla_pt_save.py"
    echo "参数配置:"

    if [ -n "$E_VALUE" ]; then
        echo "  输入excel文件路径 $E_VALUE"
        export SMLA_EXCEL_PATH="$E_VALUE"
    else
        echo "  默认输入excel文件路径 excel/example.xlsx"
    fi

    if [ -n "$S_VALUE" ]; then
        echo "  使用sheet名 $S_VALUE"
        export SMLA_EXCEL_SHEET="$S_VALUE"
    else
        echo "  默认使用 Sheet1"
    fi

    if [ -n "$P_VALUE" ]; then
        echo "  PT文件保存地址 $P_VALUE"
        export SMLA_PT_SAVE_PATH="$P_VALUE"
    else
        echo "  默认PT文件保存地址 ./data"
    fi

    echo "==============================="
    # 检查脚本是否存在
    if [ ! -f "batch/sparse_flash_mla_pt_save.py" ]; then
        echo "错误: 找不到 sparse_flash_mla_pt_save.py 脚本"
        exit 1
    fi

    # 运行脚本，传递环境变量
    export SMLA_BATCH_CONSISTENCY="$BATCH_CONSISTENCY_POLICY"
    python3 -m pytest -rA -s batch/sparse_flash_mla_pt_save.py  -v -m ci
}

# 运行 批跑执行pt文件 脚本的函数
run_script_load() {
    echo "准备运行 test_sparse_flash_mla_batch.py 脚本..."

    # 解析 test_sparse_flash_mla_batch.py 的参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -P)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -P 参数需要值"
                    exit 1
                fi
                P_VALUE="$2"
                shift 2
                ;;
            -R)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -R 参数需要值"
                    exit 1
                fi
                R_VALUE="$2"
                shift 2
                ;;
            -E)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -E 参数需要值"
                    exit 1
                fi
                E_VALUE="$2"
                shift 2
                ;;
            -S)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -S 参数需要值"
                    exit 1
                fi
                S_VALUE="$2"
                shift 2
                ;;
            --batch-consistency)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: --batch-consistency 参数需要值"
                    exit 1
                fi
                set_batch_consistency_policy "$2"
                shift 2
                ;;
            *)
                echo "错误: 未知参数 '$1'"
                echo "test_sparse_flash_mla_batch.py 脚本支持的参数: -P, -R, -E, -S"
                exit 1
                ;;
        esac
    done

    # 打印参数信息
    echo "==============================="
    export SMLA_BATCH_CONSISTENCY="$BATCH_CONSISTENCY_POLICY"
    echo "脚本: test_sparse_flash_mla_batch.py"
    echo "参数配置:"

    if [ -n "$P_VALUE" ]; then
        echo "  PT文件读取地址 $P_VALUE"
        export SMLA_PT_LOAD_PATH="$P_VALUE"
    else
        echo "  默认PT文件读取地址 ./data"
    fi

    if [ -n "$R_VALUE" ]; then
        echo "  结果存储至文件 $R_VALUE"
        export SMLA_RESULT_SAVE_PATH="$R_VALUE"
    else
        echo "  结果存储至文件 ./result/smla_result.xlsx"
    fi

    if [ -n "$E_VALUE" ]; then
        echo "  Excel文件路径 $E_VALUE"
        export SMLA_EXCEL_PATH="$E_VALUE"
        export SMLA_BATCH_TEST_MODE=1
        if [ -n "$S_VALUE" ]; then
            echo "  使用sheet名 $S_VALUE"
            export SMLA_EXCEL_SHEET="$S_VALUE"
        fi
    else
        echo "  未指定Excel，全量批跑目录下所有.pt文件"
    fi

    echo "==============================="

    # 检查脚本是否存在
    if [ ! -f "test_sparse_flash_mla_batch.py" ]; then
        echo "错误: 找不到 test_sparse_flash_mla_batch.py 脚本"
        exit 1
    fi

    # 获取用例目录
    LOAD_DIR="${SMLA_PT_LOAD_PATH:-./data}"
    if [ ! -d "$LOAD_DIR" ]; then
        echo "错误: 用例目录不存在: $LOAD_DIR"
        exit 1
    fi

    if [ "${SMLA_BATCH_TEST_MODE}" = "1" ] && [ -n "$E_VALUE" ]; then
        SHEET_NAME="${S_VALUE:-CSA}"
        echo "按表格筛选模式: Excel=$E_VALUE, Sheet=$SHEET_NAME"
        TARGET_NAMES=$(python3 -c "
import pandas as pd
df = pd.read_excel('$E_VALUE', sheet_name='$SHEET_NAME')
names = [str(n) for n in df['testcase_name'].dropna().tolist() if str(n) != 'None']
print('\n'.join(names))
")
        if [ -z "$TARGET_NAMES" ]; then
            echo "错误: 表格中没有有效的testcase_name"
            exit 1
        fi
        CASE_FILES=()
        while IFS= read -r target_name; do
            while IFS= read -r matched_file; do
                if [ -n "$matched_file" ] && [[ ! " ${CASE_FILES[*]} " =~ " ${matched_file} " ]]; then
                    CASE_FILES+=("$matched_file")
                fi
            done < <(find "$LOAD_DIR" -maxdepth 1 -name "*.pt" | grep "$target_name" | sort)
        done <<< "$TARGET_NAMES"
        echo "从表格中读取到目标用例名, 筛选后共 ${#CASE_FILES[@]} 个.pt文件"
    else
        mapfile -t CASE_FILES < <(find "$LOAD_DIR" -maxdepth 1 -name "*.pt" | sort)
    fi

    TOTAL=${#CASE_FILES[@]}
    if [ "$TOTAL" -eq 0 ]; then
        echo "错误: 目录 $LOAD_DIR 下未找到匹配的 .pt 用例"
        exit 1
    fi

    echo "共 $TOTAL 条用例待执行, 目录: $LOAD_DIR"
    echo "开始隔离批量执行（每条用例独立 pytest 进程）..."

    PASS=0
    FAIL=0
    FAIL_LIST=()
    SUMMARY_LOG="batch_summary.log"
    FAIL_LOG="batch_fail_list.log"
    : > "$SUMMARY_LOG"
    : > "$FAIL_LOG"

    i=0
    for case_file in "${CASE_FILES[@]}"; do
        i=$((i+1))
        case_name=$(basename "$case_file")
        echo -e "\n===== [$i/$TOTAL] 执行用例: $case_name =====" | tee -a "$SUMMARY_LOG"

        QSAS_TESTCASE_PATH="$case_file" python3 -m pytest -rA -s test_sparse_flash_mla_batch.py -v -m ci 2>&1 | tee -a "$SUMMARY_LOG"
        status=${PIPESTATUS[0]}

        if [ "$status" -eq 0 ]; then
            PASS=$((PASS+1))
            echo "[PASS] $case_name" | tee -a "$SUMMARY_LOG"
        else
            FAIL=$((FAIL+1))
            FAIL_LIST+=("$case_name")
            echo "[FAIL] $case_name" | tee -a "$SUMMARY_LOG"
            echo "$case_name" >> "$FAIL_LOG"
        fi
    done

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
    echo "结果表格:  ${SMLA_RESULT_SAVE_PATH:-./result/smla_result.xlsx}"
}

# 运行 批跑执行pt文件(Graph模式) 脚本的函数
run_script_load_graph() {
    echo "准备运行 test_sparse_flash_mla_batch_graph.py 脚本(Graph模式)..."

    # 解析 test_sparse_flash_mla_batch_graph.py 的参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -P)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -P 参数需要值"
                    exit 1
                fi
                P_VALUE="$2"
                shift 2
                ;;
            -R)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -R 参数需要值"
                    exit 1
                fi
                R_VALUE="$2"
                shift 2
                ;;
            -E)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -E 参数需要值"
                    exit 1
                fi
                E_VALUE="$2"
                shift 2
                ;;
            -S)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -S 参数需要值"
                    exit 1
                fi
                S_VALUE="$2"
                shift 2
                ;;
            --batch-consistency)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: --batch-consistency 参数需要值"
                    exit 1
                fi
                set_batch_consistency_policy "$2"
                shift 2
                ;;
            *)
                echo "错误: 未知参数 '$1'"
                echo "test_sparse_flash_mla_batch_graph.py 脚本支持的参数: -P, -R, -E, -S"
                exit 1
                ;;
        esac
    done

    # 打印参数信息
    echo "==============================="
    export SMLA_BATCH_CONSISTENCY="$BATCH_CONSISTENCY_POLICY"
    echo "脚本: test_sparse_flash_mla_batch_graph.py (Graph模式)"
    echo "参数配置:"

    if [ -n "$P_VALUE" ]; then
        echo "  PT文件读取地址 $P_VALUE"
        export SMLA_PT_LOAD_PATH="$P_VALUE"
    else
        echo "  默认PT文件读取地址 ./data"
    fi

    if [ -n "$R_VALUE" ]; then
        echo "  结果存储至文件 $R_VALUE"
        export SMLA_RESULT_SAVE_PATH="$R_VALUE"
    else
        echo "  结果存储至文件 ./result/smla_result.xlsx"
    fi

    if [ -n "$E_VALUE" ]; then
        echo "  Excel文件路径 $E_VALUE"
        export SMLA_EXCEL_PATH="$E_VALUE"
        export SMLA_BATCH_TEST_MODE=1
        if [ -n "$S_VALUE" ]; then
            echo "  使用sheet名 $S_VALUE"
            export SMLA_EXCEL_SHEET="$S_VALUE"
        fi
    else
        echo "  未指定Excel，全量批跑目录下所有.pt文件"
    fi

    echo "==============================="

    # 检查脚本是否存在
    if [ ! -f "test_sparse_flash_mla_batch_graph.py" ]; then
        echo "错误: 找不到 test_sparse_flash_mla_batch_graph.py 脚本"
        exit 1
    fi

    # 获取用例目录
    LOAD_DIR="${SMLA_PT_LOAD_PATH:-./data}"
    if [ ! -d "$LOAD_DIR" ]; then
        echo "错误: 用例目录不存在: $LOAD_DIR"
        exit 1
    fi

    if [ "${SMLA_BATCH_TEST_MODE}" = "1" ] && [ -n "$E_VALUE" ]; then
        SHEET_NAME="${S_VALUE:-CSA}"
        echo "按表格筛选模式: Excel=$E_VALUE, Sheet=$SHEET_NAME"
        TARGET_NAMES=$(python3 -c "
import pandas as pd
df = pd.read_excel('$E_VALUE', sheet_name='$SHEET_NAME')
names = [str(n) for n in df['testcase_name'].dropna().tolist() if str(n) != 'None']
print('\n'.join(names))
")
        if [ -z "$TARGET_NAMES" ]; then
            echo "错误: 表格中没有有效的testcase_name"
            exit 1
        fi
        CASE_FILES=()
        while IFS= read -r target_name; do
            while IFS= read -r matched_file; do
                if [ -n "$matched_file" ] && [[ ! " ${CASE_FILES[*]} " =~ " ${matched_file} " ]]; then
                    CASE_FILES+=("$matched_file")
                fi
            done < <(find "$LOAD_DIR" -maxdepth 1 -name "*.pt" | grep "$target_name" | sort)
        done <<< "$TARGET_NAMES"
        echo "从表格中读取到目标用例名, 筛选后共 ${#CASE_FILES[@]} 个.pt文件"
    else
        mapfile -t CASE_FILES < <(find "$LOAD_DIR" -maxdepth 1 -name "*.pt" | sort)
    fi

    TOTAL=${#CASE_FILES[@]}
    if [ "$TOTAL" -eq 0 ]; then
        echo "错误: 目录 $LOAD_DIR 下未找到匹配的 .pt 用例"
        exit 1
    fi

    echo "共 $TOTAL 条用例待执行, 目录: $LOAD_DIR"
    echo "开始隔离批量执行(Graph模式, 每条用例独立 pytest 进程)..."

    PASS=0
    FAIL=0
    FAIL_LIST=()
    SUMMARY_LOG="batch_graph_summary.log"
    FAIL_LOG="batch_graph_fail_list.log"
    : > "$SUMMARY_LOG"
    : > "$FAIL_LOG"

    i=0
    for case_file in "${CASE_FILES[@]}"; do
        i=$((i+1))
        case_name=$(basename "$case_file")
        echo -e "\n===== [$i/$TOTAL] 执行用例(Graph模式): $case_name =====" | tee -a "$SUMMARY_LOG"

        QSAS_TESTCASE_PATH="$case_file" python3 -m pytest -rA -s test_sparse_flash_mla_batch_graph.py -v -m graph 2>&1 | tee -a "$SUMMARY_LOG"
        status=${PIPESTATUS[0]}

        if [ "$status" -eq 0 ]; then
            PASS=$((PASS+1))
            echo "[PASS] $case_name" | tee -a "$SUMMARY_LOG"
        else
            FAIL=$((FAIL+1))
            FAIL_LIST+=("$case_name")
            echo "[FAIL] $case_name" | tee -a "$SUMMARY_LOG"
            echo "$case_name" >> "$FAIL_LOG"
        fi
    done

    echo -e "\n========== 批量执行汇总(Graph模式) ==========" | tee -a "$SUMMARY_LOG"
    echo "总计: $TOTAL  通过: $PASS  失败: $FAIL" | tee -a "$SUMMARY_LOG"
    if [ "$FAIL" -gt 0 ]; then
        echo "失败用例:" | tee -a "$SUMMARY_LOG"
        for f in "${FAIL_LIST[@]}"; do
            echo "  - $f" | tee -a "$SUMMARY_LOG"
        done
    fi
    echo "详细日志:  $SUMMARY_LOG"
    echo "失败清单:  $FAIL_LOG"
    echo "结果表格:  ${SMLA_RESULT_SAVE_PATH:-./result/smla_result.xlsx}"
}

# 根据脚本类型调用相应的函数
case "$SCRIPT_TYPE" in
    single)
        run_script_single "$@"
        ;;
    save)
        run_script_save "$@"
        ;;
    load)
        run_script_load "$@"
        ;;
    load_graph)
        run_script_load_graph "$@"
        ;;
    *)
        echo "错误: 未知的脚本类型 '$SCRIPT_TYPE'"
        echo "可用类型: single, save, load, load_graph"
        show_help
        exit 1
        ;;
esac

echo "==============================="
echo "所有任务完成"
