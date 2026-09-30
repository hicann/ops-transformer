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
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=4

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

# 导出环境变量
SHORT=v:,h:,c:,
LONG=dtype:,head-num:,chunk-size:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in
        # float16, float, int32
        (-v | --dtype)
            DTYPE="$2"
            shift 2;;
        (-h | --head-num)
            HEAD_NUM="$2"
            shift 2;;
        (-c | --chunk-size)
            CHUNK_SIZE="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

if [ ! $ASCEND_HOME_DIR ]; then
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        export ASCEND_HOME_DIR=$HOME/Ascend/ascend-toolkit/latest
    else
        export ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
export DDK_PATH=$ASCEND_HOME_DIR
arch=$(uname -m)
export NPU_HOST_LIB=$ASCEND_HOME_DIR/${arch}-linux/lib64
export CMAKE_PREFIX_PATH=${HOME}/.local

curlen=$1
HEAD_NUM=$2
CHUNK_SIZE=$3
# echo $curlen
# echo $seqlen
function main {
    # # 1. 清除遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*
    rm ./input/*.bin
    rm ./output/*.bin

    # 2. 生成输入数据和真值数据
    cd $CURRENT_DIR
    # python3 scripts/gen_data_case3.py --seqlen-kv $curlen --cu-seqlen-q $seqlen
    # python3 scripts/gen_gqa_i8_data.py --seqlen-kv $curlen --cu-seqlen-q $seqlen
    python3 scripts/gen_gqa_data.py --cu-seqlens $curlen --head-num $HEAD_NUM --chunk-size $CHUNK_SIZE
    # python3 scripts/gen_gqa_i8_data_nz_ntd.py --seqlen-kv $curlen --cu-seqlen-q $seqlen
    # python3 scripts/gen_tnd_mqa_data.py --seqlen-kv $curlen --cu-seqlen-q $seqlen
    if [ $? -ne 0 ]; then
        echo "ERROR: generate input data failed!"
        return 1
    fi
    echo "INFO: generate input data success!"

    # 3. 编译acl可执行文件
    cd $CURRENT_DIR; rm -rf build; mkdir -p build; cd build
    cmake ../src
    if [ $? -ne 0 ]; then
        echo "ERROR: cmake failed!"
        return 1
    fi
    echo "INFO: cmake success!"
    make -j16
    if [ $? -ne 0 ]; then
        echo "ERROR: make failed!"
        return 1
    fi
    echo "INFO: make success!"

    # 4. 运行可执行文件
    cd $CURRENT_DIR/output
    echo "INFO: execute op!"
    # mssanitizer -t memcheck -t racecheck ./execute_gqa_op $curlen  > aa.txt
    ./execute_gqa_op $curlen $HEAD_NUM $CHUNK_SIZE
    # msprof op  simulator --output=. ./execute_gqa_op $curlen
    # msprof op  --output=. ./execute_gqa_op $curlen

    # ./execute_gqa_op $curlen $seqlen
    if [ $? -ne 0 ]; then
        echo "ERROR: acl executable run failed! please check your project!"
        return 1
    fi
    echo "INFO: acl executable run success!"

    # 5. 比较真值文件
    cd $CURRENT_DIR
    python3 scripts/verify_result.py output/output1.bin output/golden_output.bin
    # python3 scripts/verify_result.py output/output_tmp.bin output/tmp_golden.bin
}

main
