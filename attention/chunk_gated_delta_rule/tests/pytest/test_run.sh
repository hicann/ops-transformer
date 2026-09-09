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
TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT="test_chunk_gated_delta_rule_single.py"
OP_NAME="ChunkGatedDeltaRule"
PROF_RUNS=5
PROF_WARMUP=1

# ====================== 结果输出目录 ======================
RESULT_DIR="output"
mkdir -p "${RESULT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULT_DIR}/run_${TIMESTAMP}.log"
CSV_FILE="${RESULT_DIR}/result_${TIMESTAMP}.csv"

# ====================== msprof profile (一次跑所有用例) ======================

# 从 op_summary 提取算子耗时并回填 CSV
_backfill_durations() {
    local prof_dir="$1"
    local csv_file="$2"
    echo "===== 提取耗时并回填CSV ====="
    local summary_file=$(find "${prof_dir}" -name "op_summary_*.csv" | head -1)
    if [ -z "${summary_file}" ]; then
        echo "错误: 未找到 op_summary CSV 文件"
        return 1
    fi

    python3 -c "
import csv

OP_NAME = '${OP_NAME}'
PROF_RUNS = ${PROF_RUNS}
PROF_WARMUP = ${PROF_WARMUP}
CSV_FILE = '${csv_file}'
summary_file = '${summary_file}'

# 从 op_summary 提取所有 ChunkGatedDeltaRule 的 Task Duration
times = []
with open(summary_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('OP Type', '').strip() == OP_NAME:
            t = row['Task Duration(us)'].strip()
            if t:
                times.append(float(t))

print(f'op_summary 中 {OP_NAME} 记录数: {len(times)}')

# 每用例 PROF_WARMUP+PROF_RUNS 条, 丢弃前 PROF_WARMUP 条, 取后 PROF_RUNS 条平均
group_size = PROF_WARMUP + PROF_RUNS
durations = []
for i in range(0, len(times), group_size):
    group = times[i+PROF_WARMUP:i+group_size]
    avg = sum(group) / len(group)
    durations.append(f'{avg:.3f}')

print(f'用例数: {len(durations)}')

# 回填到 CSV
rows = []
with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for i, row in enumerate(reader):
        if i < len(durations):
            row['durations'] = durations[i]
        rows.append(row)

with open(CSV_FILE, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print('durations 回填完成')
"
}

run_profile() {
    local test_mode="$1"
    echo "===== 执行 ${test_mode} 模式性能profiling (USE_GRAPH=${USE_GRAPH}) ====="

    # 1. 正常跑一遍生成 CSV (含 status, durations 留空)
    echo "===== 第1步: 正常精度测试 ====="
    TEST_MODE=${test_mode} USE_GRAPH=${USE_GRAPH} SAVE_PT=${SAVE_PT} LOAD_PT=${LOAD_PT} CSV_FILE="${CSV_FILE}" \
        python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
        -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}

    # 2. 一次 msprof 跑所有用例 (ENABLE_PROF=true, 每用例跑5次算子)
    echo "===== 第2步: msprof 性能采集 (每用例${PROF_RUNS}次) ====="
    local prof_dir="${RESULT_DIR}/prof/prof_${TIMESTAMP}"
    mkdir -p "${prof_dir}"
    TEST_MODE=${test_mode} USE_GRAPH=${USE_GRAPH} SAVE_PT=${SAVE_PT} LOAD_PT=${LOAD_PT} ENABLE_PROF=true \
        msprof --output="${prof_dir}" \
        --summary-format=csv --export=on \
        python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
        -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee -a "${LOG_FILE}"

    # 3. 从 op_summary 提取算子耗时并回填 CSV
    _backfill_durations "${prof_dir}" "${CSV_FILE}"

    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    echo "Profiling 数据: ${prof_dir}"
    return ${exit_code}
}

# ====================== 执行区======================

# 算子调测
run_single() {
    echo "===== 执行单算子用例调测 (USE_GRAPH=${USE_GRAPH}, SAVE_PT=${SAVE_PT}, LOAD_PT=${LOAD_PT}) ====="
    TEST_MODE=single USE_GRAPH=${USE_GRAPH} SAVE_PT=${SAVE_PT} LOAD_PT=${LOAD_PT} CSV_FILE="${CSV_FILE}" \
        python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
        -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# RDV测试
run_rdv() {
    echo "===== 执行RDV参数集测试 (USE_GRAPH=${USE_GRAPH}, SAVE_PT=${SAVE_PT}, LOAD_PT=${LOAD_PT}) ====="
    TEST_MODE=rdv USE_GRAPH=${USE_GRAPH} SAVE_PT=${SAVE_PT} LOAD_PT=${LOAD_PT} CSV_FILE="${CSV_FILE}" \
        python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
        -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# 随机用例测试
run_random() {
    local count="${1:-100}"
    echo "===== 执行随机用例调测 ($count 条) ====="
    TEST_MODE=random RANDOM_CASE_COUNT=$count USE_GRAPH=${USE_GRAPH} CSV_FILE="${CSV_FILE}" \
        python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
        -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# 随机用例测试（不跑golden，仅NPU执行，加快速度）
run_random_npu() {
    local count="${1:-100}"
    echo "===== 执行随机用例调测-仅NPU ($count 条) ====="
    TEST_MODE=random SKIP_GOLDEN=1 RANDOM_CASE_COUNT=$count USE_GRAPH=${USE_GRAPH} CSV_FILE="${CSV_FILE}" \
        python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
        -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# 自定义单条用例执行（指定pt或指定参数）
run_custom() {
    local pt_file=""
    local c_B="" c_seqlen="" c_nk="" c_nv="" c_dk="" c_dv=""
    local c_chunk_size=64
    local c_dtype="bfloat16"
    local c_state_dtype="bfloat16"
    local c_has_g="true"
    local c_contig="true"

    while [ $# -gt 0 ]; do
        case "$1" in
            --pt)          pt_file="$2"; shift 2;;
            --B)           c_B="$2"; shift 2;;
            --seqlen)      c_seqlen="$2"; shift 2;;
            --nk)          c_nk="$2"; shift 2;;
            --nv)          c_nv="$2"; shift 2;;
            --dk)          c_dk="$2"; shift 2;;
            --dv)          c_dv="$2"; shift 2;;
            --chunk_size)  c_chunk_size="$2"; shift 2;;
            --dtype)       c_dtype="$2"; shift 2;;
            --state_dtype) c_state_dtype="$2"; shift 2;;
            --no_g)        c_has_g="false"; shift;;
            --no_contig)   c_contig="false"; shift;;
            graph)         USE_GRAPH=true; shift;;
            prof)          ENABLE_PROF=true; shift;;
            save)          SAVE_PT=true; shift;;
            load)          LOAD_PT=true; shift;;
            *)             echo "错误：未知参数 '$1'"; show_help; exit 1;;
        esac
    done

    local custom_case=""
    local load_pt_file=""
    if [ -n "${pt_file}" ]; then
        if [ ! -f "${pt_file}" ]; then
            echo "错误：pt 文件不存在: ${pt_file}"
            exit 1
        fi
        custom_case=$(python3 -c "
import torch, json
m = torch.load('${pt_file}', map_location='cpu')['meta']
def dt(x): return str(x).replace('torch.', '')
print(json.dumps({'_name':'custom','B':m['B'],'seqlen':m['seqlen'],'nk':m['nk'],'nv':m['nv'],
                  'dk':m['dk'],'dv':m['dv'],'chunk_size':m['chunk_size'],
                  'data_type':dt(m['data_type']),'state_data_type':dt(m['state_data_type']),
                  'has_g':m['has_g'],'is_contiguous':m['is_contiguous']}))
")
        LOAD_PT=true
        load_pt_file="${pt_file}"
        echo "===== 自定义用例(从pt加载): ${pt_file} ====="
    else
        if [ -z "${c_B}" ] || [ -z "${c_seqlen}" ] || [ -z "${c_nk}" ] || [ -z "${c_nv}" ] \
           || [ -z "${c_dk}" ] || [ -z "${c_dv}" ]; then
            echo "错误：需指定 --pt <file> 或 --B --seqlen --nk --nv --dk --dv 等参数"
            exit 1
        fi
        local py_has_g="True"
        [ "${c_has_g}" == "false" ] && py_has_g="False"
        local py_contig="True"
        [ "${c_contig}" == "false" ] && py_contig="False"
        local py_seqlen="${c_seqlen}"
        if echo "${c_seqlen}" | grep -q ","; then
            py_seqlen="[${c_seqlen}]"
        fi
        custom_case=$(python3 -c "
import json
print(json.dumps({'_name':'custom','B':${c_B},'seqlen':${py_seqlen},'nk':${c_nk},'nv':${c_nv},
                  'dk':${c_dk},'dv':${c_dv},'chunk_size':${c_chunk_size},
                  'data_type':'${c_dtype}','state_data_type':'${c_state_dtype}',
                  'has_g':${py_has_g},'is_contiguous':${py_contig}}))
")
        echo "===== 自定义用例(参数生成): B=${c_B} seqlen=${c_seqlen} nk=${c_nk} nv=${c_nv} dk=${c_dk} dv=${c_dv} ====="
    fi

    if [ "${SAVE_PT}" == "true" ] && [ "${LOAD_PT}" == "true" ]; then
        echo "错误：save 与 load 不可同时使用"
        exit 1
    fi

    if [ "${ENABLE_PROF}" == "true" ]; then
        echo "===== msprof 性能采集 (每用例${PROF_RUNS}次) ====="
        local prof_dir="${RESULT_DIR}/prof/prof_${TIMESTAMP}"
        mkdir -p "${prof_dir}"
        CUSTOM_CASE="${custom_case}" USE_GRAPH=${USE_GRAPH} SAVE_PT=${SAVE_PT} \
        LOAD_PT=${LOAD_PT} LOAD_PT_FILE="${load_pt_file}" ENABLE_PROF=true \
        CSV_FILE="${CSV_FILE}" \
            msprof --output="${prof_dir}" --summary-format=csv --export=on \
            python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
            -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee -a "${LOG_FILE}"
        local exit_code=${PIPESTATUS[0]}
        _backfill_durations "${prof_dir}" "${CSV_FILE}"
        echo "Profiling 数据: ${prof_dir}"
    else
        CUSTOM_CASE="${custom_case}" USE_GRAPH=${USE_GRAPH} SAVE_PT=${SAVE_PT} \
        LOAD_PT=${LOAD_PT} LOAD_PT_FILE="${load_pt_file}" \
        CSV_FILE="${CSV_FILE}" \
            python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
            -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
        local exit_code=${PIPESTATUS[0]}
    fi
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# 显示帮助信息
show_help() {
    echo "用法:"
    echo "  $0 <single|rdv|random [N]|random_npu [N]> [graph] [prof] [save] [load]"
    echo "  $0 run --pt <pt文件> [graph] [prof] [save]"
    echo "  $0 run --B <B> --seqlen <S> --nk <Nk> --nv <Nv> --dk <Dk> --dv <Dv> [选项]"
    echo ""
    echo "模式:"
    echo "  single          执行单算子用例调测"
    echo "  rdv             执行RDV参数集测试"
    echo "  random [N]      随机生成并执行N条用例（默认100，含golden精度对比）"
    echo "  random_npu [N]  随机生成并执行N条用例，不跑golden仅NPU执行（默认100）"
    echo "  run             自定义单条用例（指定pt或指定参数生成）"
    echo "  help            显示本帮助信息"
    echo ""
    echo "可选标志 (single/rdv/random/random_npu 模式):"
    echo "  graph     启用aclgraph模式"
    echo "  prof      启用msprof性能采集"
    echo "  save      保存输入数据为 .pt (output/pt/)"
    echo "  load      从 output/pt/ 加载 .pt 执行"
    echo ""
    echo "run 模式参数:"
    echo "  --pt <file>          指定pt文件加载执行（自动启用load）"
    echo "  --B <n>              batch size"
    echo "  --seqlen <n>         序列长度（支持逗号分隔变长，如 64,128）"
    echo "  --nk <n>             key头数"
    echo "  --nv <n>             value头数"
    echo "  --dk <n>             key维度"
    echo "  --dv <n>             value维度"
    echo "  --chunk_size <n>     chunk大小（默认64）"
    echo "  --dtype <str>        数据类型（默认bfloat16）"
    echo "  --state_dtype <str>  状态数据类型（默认bfloat16）"
    echo "  --no_g               不使用g门控（默认使用）"
    echo "  --no_contig          state非连续（默认连续）"
    echo "  graph/prof/save      同上可选标志"
    echo ""
    echo "示例："
    echo "  $0 single                           # single模式"
    echo "  $0 single save                      # 保存输入pt"
    echo "  $0 single load                      # 从pt加载执行"
    echo "  $0 run --pt output/pt/xxx.pt        # 指定pt执行"
    echo "  $0 run --B 1 --seqlen 64 --nk 4 --nv 4 --dk 128 --dv 128   # 指定参数生成执行"
    echo "  $0 run --B 1 --seqlen 64 --nk 4 --nv 4 --dk 128 --dv 128 save  # 生成并保存pt"
    echo "  $0 rdv graph                        # rdv模式+aclgraph"
    echo "  $0 random 100                       # 随机执行100条用例"
    echo "  $0 random_npu 100                   # 随机执行100条用例（仅NPU）"
}

# ====================== 主逻辑 ======================
# 检查参数数量
if [ $# -lt 1 ]; then
    echo "错误：缺少参数"
    show_help
    exit 1
fi

# 解析第一个参数：模式
TEST_MODE_ARG="$1"
USE_GRAPH=false
ENABLE_PROF=false
SAVE_PT=false
LOAD_PT=false

# run 模式有独立参数解析，单独处理
if [ "${TEST_MODE_ARG}" == "run" ]; then
    shift
    run_custom "$@"
    exit $?
fi

# random / random_npu 模式：第二个参数为可选的用例数量
if [ "${TEST_MODE_ARG}" == "random" ] || [ "${TEST_MODE_ARG}" == "random_npu" ]; then
    RANDOM_COUNT=""
    if [ $# -ge 2 ]; then
        RANDOM_COUNT="$2"
    fi
    if [ "${TEST_MODE_ARG}" == "random" ]; then
        run_random "${RANDOM_COUNT}"
    else
        run_random_npu "${RANDOM_COUNT}"
    fi
    exit $?
fi

# 其余模式参数数量校验
if [ $# -gt 5 ]; then
    echo "错误：参数数量不正确（需1-5个参数）"
    show_help
    exit 1
fi

# 解析剩余参数
for arg in "${@:2}"; do
    case "$arg" in
        graph)
            USE_GRAPH=true
            ;;
        prof)
            ENABLE_PROF=true
            ;;
        save)
            SAVE_PT=true
            ;;
        load)
            LOAD_PT=true
            ;;
        *)
            echo "错误：未知参数 '$arg'，仅支持 'graph' / 'prof' / 'save' / 'load'"
            show_help
            exit 1
            ;;
    esac
done

# save 和 load 互斥
if [ "${SAVE_PT}" == "true" ] && [ "${LOAD_PT}" == "true" ]; then
    echo "错误：save 与 load 不可同时使用"
    exit 1
fi

# 根据参数执行对应函数
case "$TEST_MODE_ARG" in
    single)
        if [ "${ENABLE_PROF}" == "true" ]; then
            run_profile "single"
        else
            run_single
        fi
        ;;
    rdv)
        if [ "${ENABLE_PROF}" == "true" ]; then
            run_profile "rdv"
        else
            run_rdv
        fi
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$TEST_MODE_ARG'，仅支持 single/rdv/random/random_npu/run/help"
        show_help
        exit 1
        ;;
esac

exit 0
