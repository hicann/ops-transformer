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

# ====================== 结果输出目录 ======================
RESULT_DIR="output"
mkdir -p "${RESULT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULT_DIR}/run_${TIMESTAMP}.log"
CSV_FILE="${RESULT_DIR}/result_${TIMESTAMP}.csv"

# ====================== 执行区======================

# 算子调测
run_single() {
    echo "===== 执行单算子用例调测 ====="
    TEST_MODE=single CSV_FILE="${CSV_FILE}" python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# RDV测试
run_rdv() {
    echo "===== 执行RDV参数集测试 ====="
    TEST_MODE=rdv CSV_FILE="${CSV_FILE}" python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# 随机用例测试
run_random() {
    local count="${1:-100}"
    echo "===== 执行随机用例调测 ($count 条) ====="
    TEST_MODE=random RANDOM_CASE_COUNT=$count CSV_FILE="${CSV_FILE}" python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# 随机用例测试（不跑golden，仅NPU执行，加快速度）
run_random_npu() {
    local count="${1:-100}"
    echo "===== 执行随机用例调测-仅NPU ($count 条) ====="
    TEST_MODE=random SKIP_GOLDEN=1 RANDOM_CASE_COUNT=$count CSV_FILE="${CSV_FILE}" python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# mssanitizer 检测结论回填 CSV（mss_check 列；mssanitizer 报错时 pytest 仍 PASSED，必须由日志解析回填）
_backfill_mss_check() {
    local csv_file="$1"
    local log_file="$2"
    local mss_status="$3"
    python3 -c "
import csv

csv_file = '${csv_file}'
mss_status = '${mss_status}'
err_count = 0
with open('${log_file}', errors='ignore') as f:
    err_count = sum(1 for line in f if '====== ERROR' in line)

rows = []
with open(csv_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for row in reader:
        row['mss_check'] = mss_status if err_count == 0 else f'{mss_status}(errors={err_count})'
        rows.append(row)

with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f'mss_check 回填完成: {mss_status}, ERROR块数: {err_count}')
"
}

# 合并多个批次 CSV 为一个（按批次顺序追加行）
_merge_batch_csvs() {
    local output="$1"
    shift
    python3 -c "
import csv, sys

output = sys.argv[1]
inputs = sys.argv[2:]
all_rows = []
fields = None
for f in inputs:
    try:
        with open(f, 'r', encoding='utf-8-sig') as fh:
            reader = csv.DictReader(fh)
            if fields is None:
                fields = reader.fieldnames
            for row in reader:
                all_rows.append(row)
    except Exception:
        pass
if fields and all_rows:
    with open(output, 'w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'CSV合并完成: {output} ({len(all_rows)} 行)')
else:
    print('CSV合并: 无数据可合并')
" "$output" "$@"
}

# 单批 mssanitizer 检测（内部函数，供 run_mss 分批调用）
_run_mss_single_batch() {
    local count="$1"
    local mss_tool="$2"
    local mss_bin="$3"
    local kernel_opts="$4"
    local mss_kernel="$5"
    local batch_seed="$6"
    local batch_csv="$7"
    local batch_log="$8"
    local csv_append="${9:-0}"

    TEST_MODE=random SKIP_GOLDEN=1 RANDOM_CASE_COUNT=$count \
        RANDOM_SEED=$batch_seed CSV_FILE="${batch_csv}" CSV_APPEND=${csv_append} MSS_TOOL=${mss_tool} \
        "$mss_bin" --tool=$mss_tool $kernel_opts $MSS_EXTRA_OPTS -- \
        python3 -m pytest -rA -s $TEST_RECURRENT_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci \
        -W ignore::UserWarning -W ignore::DeprecationWarning \
        2>&1 | tee -a "$batch_log"
    local pytest_status=${PIPESTATUS[0]}

    local mss_status="FAIL"
    if grep -q "====== ERROR" "$batch_log"; then
        echo "===== FAIL: mssanitizer 检测到问题 ====="
        grep "====== ERROR" "$batch_log"
    elif [ "$pytest_status" -ne 0 ]; then
        mss_status="CRASH"
        echo "===== CRASH: pytest 退出码 $pytest_status ====="
    else
        mss_status="PASS"
        echo "===== PASS: mssanitizer 未检测到问题 ====="
    fi
    # 分批追加模式下回填到每批的临时 CSV 片段再追加到主 CSV
    if [ -f "${batch_csv}" ]; then
        _backfill_mss_check "${batch_csv}" "$batch_log" "$mss_status"
    fi
    echo "$mss_status"
}

# mssanitizer 检测（随机用例，仅NPU不跑golden；tool 可选 memcheck/racecheck/initcheck/synccheck）
# MSS_BATCH 环境变量 > 0 时启用分批模式：每批重启 mssanitizer 避免 host 内存累积
run_mss() {
    local count="${1:-100}"
    local mss_tool="${MSS_TOOL:-memcheck}"
    local mss_bin="${MSSANITIZER_BIN:-mssanitizer}"
    # 默认只检本算子 kernel，跳过 ZerosLike/ViewCopy/TensorMove/rand 等旁路 kernel；置空 MSS_KERNEL 关闭过滤
    local mss_kernel="${MSS_KERNEL:-RecurrentGatedDeltaRule}"
    local kernel_opts=""
    if [ -n "$mss_kernel" ]; then
        kernel_opts="--kernel-name=$mss_kernel"
    fi
    local batch_size="${MSS_BATCH:-0}"

    # 分批模式：每批重启 mssanitizer，避免 host 内存累积导致 OOM
    # CSV/log 统一追加到单一文件，不产生每批独立文件
    if [ "$batch_size" -gt 0 ] && [ "$count" -gt "$batch_size" ]; then
        local num_batches=$(( (count + batch_size - 1) / batch_size ))
        local base_seed="${RANDOM_SEED:-$(python3 -c 'import random; print(random.randrange(2**31))')}"
        local unified_log="${RESULT_DIR}/mss_${TIMESTAMP}.log"
        local total_passed=0
        local total_failed=0
        local total_mss_fail=0

        # 首批前清空统一 CSV/log
        > "${CSV_FILE}"
        > "${unified_log}"

        echo "===== 执行 mssanitizer($mss_tool) 分批检测: $count 条, 每批 $batch_size, 共 $num_batches 批 =====" | tee -a "$unified_log"
        echo "base_seed=$base_seed, kernel过滤: ${mss_kernel:-无}, CSV: ${CSV_FILE}, 日志: ${unified_log}" | tee -a "$unified_log"

        for ((i=0; i<num_batches; i++)); do
            local remaining=$(( count - i * batch_size ))
            local cur=$(( remaining < batch_size ? remaining : batch_size ))
            local batch_seed=$(( base_seed + i ))
            local batch_csv="${CSV_FILE}"
            local batch_log="${unified_log}"

            echo "----- 批次 $((i+1))/$num_batches: $cur 条, seed=$batch_seed -----" | tee -a "$unified_log"
            local status
            status=$(_run_mss_single_batch "$cur" "$mss_tool" "$mss_bin" "$kernel_opts" \
                "$mss_kernel" "$batch_seed" "$batch_csv" "$batch_log" "1")

            local bp bf
            bp=$(python3 -c "
import csv
with open('${CSV_FILE}', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
p = sum(1 for r in rows if r.get('result')=='PASSED')
n = sum(1 for r in rows if r.get('result') in ('FAILED','ERROR'))
print(f'{p} {n}')
" 2>/dev/null)
            bf=${bp#* }
            bp=${bp%% *}
            total_passed=$((total_passed + ${bp:-0}))
            total_failed=$((total_failed + ${bf:-0}))

            if [ "$status" = "FAIL" ]; then
                total_mss_fail=$((total_mss_fail + 1))
                echo "!!!!! 批次 $((i+1)) mssanitizer 检出 ERROR，终止 !!!!!" | tee -a "$unified_log"
                break
            fi
            echo "  批次 $((i+1))/$num_batches: status=$status, passed=${bp:-0}, failed=${bf:-0} (累计 passed=$total_passed, failed=$total_failed)" | tee -a "$unified_log"
        done

        echo "===== 分批检测完成: $total_passed passed, $total_failed failed, mss_fail=$total_mss_fail =====" | tee -a "$unified_log"
        echo "CSV: ${CSV_FILE}"
        echo "日志: ${unified_log}"
        [ "$total_mss_fail" -gt 0 ] && exit 1
        [ "$total_failed" -gt 0 ] && exit 1
        return 0
    fi

    # 单进程模式（原始行为）
    local log_file="${RESULT_DIR}/mss_${TIMESTAMP}.log"
    echo "===== 执行 mssanitizer($mss_tool) 随机用例检测 ($count 条), kernel过滤: ${mss_kernel:-无}, 日志: $log_file ====="
    local status
    status=$(_run_mss_single_batch "$count" "$mss_tool" "$mss_bin" "$kernel_opts" \
        "$mss_kernel" "${RANDOM_SEED:-0}" "${CSV_FILE}" "$log_file")
    echo "执行日志: ${log_file}"
    echo "CSV: ${CSV_FILE}"
    [ "$status" != "PASS" ] && exit 1
    return 0
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single       执行单算子用例调测"
    echo "  rdv          执行RDV参数集测试"
    echo "  random [N]   随机生成并执行N条用例（默认100）"
    echo "  random_npu [N]  随机生成并执行N条用例，不跑golden仅NPU执行（默认100）"
    echo "  mss [N]   随机生成并执行N条用例，仅NPU执行并用mssanitizer检测（默认100）"
    echo "  help         显示本帮助信息"
    echo "示例："
    echo "  $0 single       # 执行single模式"
    echo "  $0 rdv          # 执行rdv模式"
    echo "  $0 random 100   # 随机执行100条用例"
    echo "  $0 random_npu 100  # 随机执行100条用例（仅NPU）"
    echo "  $0 mss 10          # 随机执行10条用例并用mssanitizer memcheck检测"
    echo "mss 模式环境变量："
    echo "  MSS_BATCH=10               # 分批大小，每批重启mssanitizer避免host内存累积（0=不分批）"
    echo "  MSS_TOOL=racecheck         # 检测工具，默认memcheck，可选racecheck/initcheck/synccheck"
    echo "  MSS_KERNEL=''              # kernel过滤，默认RecurrentGatedDeltaRule只检本算子；置空检测全部kernel"
    echo "  MSS_EXTRA_OPTS='--leak-check=yes --full-backtrace=yes'  # 追加mssanitizer参数"
    echo "  MSSANITIZER_BIN=<PATH>     # mssanitizer路径，默认取PATH"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "错误：参数数量错误，用法 $0 {single|rdv|random [N]|random_npu [N]|mss [N]|help}"
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
    random_npu)
        run_random_npu "$2"
        ;;
    mss)
        run_mss "$2"
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/rdv/random/random_npu/mss/help"
        show_help
        exit 1
        ;;
esac

exit 0
