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
set +e

REPOSITORY_NAME="ops-transformer"
export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
if [[ "${GIT_TARGET_BRANCH}" = "master" ]]; then
    sudo update-alternatives --set gcc /usr/bin/gcc-15
else
    sudo update-alternatives --set gcc /usr/bin/gcc-14
fi

if gcc --version | head -n1 | grep -q "15\."; then
    rm -rf /home/jenkins/opensource/lib_cache
    if [ -d  /home/jenkins/opensource/gcc15 ];then
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15/lib_cache /home/jenkins/opensource/lib_cache
    elif [ -d  /home/jenkins/opensource/gcc15x86 ];then
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15x86/lib_cache /home/jenkins/opensource/lib_cache
    fi
else
    gcc --version
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi
gcc --version
rm -rf /home/jenkins/opensource/json

if [ -z "${ASCEND_3RD_LIB_PATH}" ]; then
    export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
fi

if [ -f /home/jenkins/Ascend/cann/bin/setenv.bash ]; then
    source /home/jenkins/Ascend/cann/bin/setenv.bash
fi

LOG_HEAD()
{
    echo "========================================"
    echo "  $1"
    echo "========================================"
}

LOG_DO()
{
    echo "[LOG_DO] $*"
    "$@"
}

DP_ASSERT_EQUAL()
{
    local actual="$1"
    local expected="$2"
    local msg="$3"
    if [ "${actual}" != "${expected}" ]; then
        echo "::error::ASSERT FAILED: ${msg} (expected=${expected}, actual=${actual})"
        exit 1
    fi
}

cd "${WORKSPACE}/" || exit 1
non_skip_count=$(grep -vE '(\.md$)' "${WORKSPACE}/pr_filelist.txt" | grep -cv '^$')
if [ "${non_skip_count}" -eq 0 ]; then
    LOG_HEAD "pr_filelist.txt only contains .md  files, skip ut"
    exit 0
fi
sudo rm -rf /home/jenkins/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910_93

if [ "${GIT_TARGET_BRANCH}" = "master" ]; then
    LOG_DO sh build.sh --PR_UT "pr_filelist.txt" --cov --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
else
    LOG_DO sh build.sh --PR_UT "pr_filelist.txt" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
fi
DP_ASSERT_EQUAL "$?" "0" "Run UT TESTCASE"

echo "ut_process=coverage" >> "${ATOMGIT_OUTPUT}"
