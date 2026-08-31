#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set +e

REPOSITORY_NAME="ops-transformer"
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
if [[ "${task_name}" == *ubuntu24* ]]; then
    if [[ "${GIT_TARGET_BRANCH}" = "master" ]]; then
        sudo update-alternatives --set gcc /usr/bin/gcc-15
    else
        sudo update-alternatives --set gcc /usr/bin/gcc-14
    fi
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
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

if [ -z "${OS_TYPE}" ]; then
    OS_TYPE=$(uname -m)
fi

if [ -f /home/jenkins/Ascend/cann/bin/setenv.bash ]; then
    source /home/jenkins/Ascend/cann/bin/setenv.bash
fi
LOG_INFO() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "[INFO] ${date_time} ${assert_msg}"
}
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
DP_ASSERT_CHECK_SKIP() {
    local actual_value=${1}
    local assert_msg=${2}
    if [ "${actual_value}" != "0" ] && [ "${actual_value}" != "200" ]; then
        LOG_ERROR "${assert_msg} is failed."
        exit 1
    else
        LOG_INFO "${assert_msg} is success."
    fi
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

LOG_HEAD "Build ${REPOSITORY_NAME}."
cd "${WORKSPACE}/" || exit 1

non_skip_count=$(grep -vE '(\.md$|^tests/)' "${WORKSPACE}/pr_filelist.txt" | grep -cv '^$')
if [ "${non_skip_count}" -eq 0 ]; then
    LOG_HEAD "pr_filelist.txt only contains .md or tests/ files, skip build"
    mkdir -p build_out
    touch build_out/skip_build.run
    touch single.tar.gz
    echo "api-check=continue" >> "${ATOMGIT_OUTPUT}"
    exit 0
fi
if [[ "${task_name}" =~ Compile_Ascend_X86_ubuntu24 ]]; then
    sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
    echo "api-check=compile" >> "${ATOMGIT_OUTPUT}"
else
    echo "api-check=continue" >> "${ATOMGIT_OUTPUT}"
fi
if [ "${task_name}" == "Pre_Compile" ]; then
    LOG_DO bash build.sh --PR_PKG ./pr_filelist.txt -j32 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}
    ret=$?
    if [ $ret -eq 200 ]; then
        LOG_HEAD "NOTE: changed files are not supported in pre_smoke"
        exit 0
    fi
    DP_ASSERT_EQUAL "$ret" "0" "Build ${REPOSITORY_NAME}"
else
    if [ "${GE_ST_RT2}X" == "kirinx90X" ]; then
        if [ "${GIT_TARGET_BRANCH}" = "master" ]; then
            LOG_DO bash build.sh --pkg --soc=kirinx90 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
            DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
        else
            echo "not need build mobile_station"
            mkdir build_out
            touch build_out/cann-ops-transformer-kirinx90_linux-x86_64.run
            exit 0
        fi
    elif [ "${GE_ST_RT2}X" == "kirin9030X" ];then
        if [ "${GIT_TARGET_BRANCH}" = "master" ];then
            wget -nv https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/asc-devkit/package/5396/cann-asc-devkit_linux-x86_64_ubuntu24.run
            chmod +x *.run
            sudo chmod 777 /home/jenkins/Ascend
            yes "y" | sudo bash cann-asc-devkit_linux-x86_64_ubuntu24.run --full --install-path=/home/jenkins/Ascend
            LOG_DO bash build.sh --pkg --soc=kirin9030 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
            DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
        else
            echo "not need build mobile_station"
            mkdir build_out
            touch build_out/cann-ops-transformer-kirin9030_linux-x86_64.run
            exit 0
        fi
    elif [ "${GE_ST_RT2}X" == "experimentalX" ]; then
        if [ "${GIT_TARGET_BRANCH}" = "master" ]; then
            LOG_DO bash build.sh --experimental --PR_PKG "pr_filelist.txt" --soc=ascend910b -j16 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}
            DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
        else
            echo "not need build experimental"
            mkdir build_out
            touch build_out/cann-ops-math-experimental_linux-${OS_TYPE}.run
            exit 0
        fi
    elif [ "${GE_ST_RT2}X" == "experimental_950X" ]; then
        if [ "${GIT_TARGET_BRANCH}" = "master" ]; then
            LOG_DO bash build.sh --experimental --PR_PKG "pr_filelist.txt" --soc=ascend950 --PR_PKG "pr_filelist.txt" -j16 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}
            BUILD_EXIT_CODE=$?
            DP_ASSERT_CHECK_SKIP "$?" "0" "Build ${REPOSITORY_NAME}"
        else
            echo "not need build experimental_950"
            mkdir build_out
            touch build_out/cann-ops-math-experimental_linux-${OS_TYPE}.run
            exit 0
        fi
    elif [[ "${task_name}" =~ monitor ]]; then
        if [ "${GIT_TARGET_BRANCH}" = "master" ]; then
            if [[ "${task_name}" =~ "910c" ]]; then
                LOG_DO bash build.sh --pkg --jit --PR_PKG "pr_filelist.txt" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16 --soc=ascend910_93
                BUILD_EXIT_CODE=$?
                DP_ASSERT_CHECK_SKIP "$?" "0" "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend910_93]"
            elif [[ "${task_name}" =~ "950" ]]; then
                LOG_DO bash build.sh --pkg --jit --PR_PKG "pr_filelist.txt" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16 --soc=ascend950
                BUILD_EXIT_CODE=$?
                DP_ASSERT_CHECK_SKIP "$?" "0" "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend950]"
            else
                LOG_DO bash build.sh --pkg --jit --PR_PKG "pr_filelist.txt" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16 --soc=ascend910b
                BUILD_EXIT_CODE=$?
                DP_ASSERT_CHECK_SKIP "$?" "0" "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend910b]"
            fi
        else
            echo "not need build monitor"
            mkdir build_out
            touch build_out/cann-ops-transformer_linux-x86_64.run
            exit 0
        fi
    else
        LOG_DO bash build.sh --pkg --jit --PR_PKG "pr_filelist.txt" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
        BUILD_EXIT_CODE=$?
        DP_ASSERT_CHECK_SKIP "$?" "0" "Build ${REPOSITORY_NAME}"
    fi
fi
compile_package_name=$(ls "${WORKSPACE}/build_out/" |grep -E "*.run$"|head -n1)
if [[ -z "${compile_package_name}" ]]; then
    if [[ "${BUILD_EXIT_CODE}" == "200" ]]; then
        echo "not need compile"
        mkdir build_out
        touch build_out/cann-ops-transformer_linux-${OS_TYPE}.run
        echo "api-check=continue" >> "${ATOMGIT_OUTPUT}"
        exit 0
    else
        echo "ERROR: Not find *.run in  ${WORKSPACE}/output/package/!"
        exit 1
    fi
fi
