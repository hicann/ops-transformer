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
    sudo update-alternatives --set gcc /usr/bin/gcc-14
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
fi
gcc --version

if [ -z "${ASCEND_3RD_LIB_PATH}" ]; then
    export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
fi

if [ -z "${OS_TYPE}" ]; then
    OS_TYPE=$(uname -m)
fi

if [ "${GE_ST_RT2}X" != "kirinx90X" ];then
    whoami
    ls -ld /home/jenkins/Ascend/cann-9.0.0
    su - jenkins -c "sh /home/jenkins/Ascend/cann/share/info/ops_transformer/script/uninstall.sh"
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

LOG_HEAD "Build ${REPOSITORY_NAME}."
cd "${WORKSPACE}/" || exit 1

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
            wget -nv https://kiri-obs.obs.cn-north-4.myhuaweicloud.com/Cann%20Large%20Model%20Foundation%208.5.0.beta005/cann-bisheng-compiler_9.0.0_linux-x86_64.run
            chmod +x *.run
            wget -nv https://kiri-obs.obs.cn-north-4.myhuaweicloud.com/Cann%20Large%20Model%20Foundation%208.5.0.beta005/cann-bisheng-compiler_9.0.0_linux-x86_64.run || { echo "::error::wget failed"; exit 1; }
            chmod +x *.run
            sudo -u jenkins ./cann-bisheng-compiler*.run --full --quiet --install-path=/home/jenkins/Ascend
            LOG_DO bash build.sh --pkg --soc=kirinx90,kirin9030 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
            DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
            cd build_out
            tar -zcf kiri.tar.gz *.run
        else
            echo "not need build mobile_station"
            mkdir build_out
            touch build_out/cann-ops-transformer-kirinx90_linux-x86_64.run
            exit 0
        fi
    elif [ "${GE_ST_RT2}X" == "kirin9030X" ];then
        if [ "${GIT_TARGET_BRANCH}" = "master" ];then
            wget -nv https://kiri-obs.obs.cn-north-4.myhuaweicloud.com/Cann%20Large%20Model%20Foundation%208.5.0.beta005/cann-bisheng-compiler_9.0.0_linux-x86_64.run
            chmod +x *.run
            sudo -u jenkins ./cann-bisheng-compiler*.run --full --quiet --install-path=/home/jenkins/Ascend
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
            LOG_DO bash build.sh --experimental --soc=ascend950 --PR_PKG "pr_filelist.txt" -j16 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}
            DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
        else
            echo "not need build experimental_950"
            mkdir build_out
            touch build_out/cann-ops-math-experimental_linux-${OS_TYPE}.run
            exit 0
        fi
    elif [[ "${task_name}" =~ monitor ]]; then
        if [ "${GIT_TARGET_BRANCH}" = "master" ]; then
            if [[ "${task_name}" =~ "910c" ]]; then
                LOG_DO bash build.sh --pkg --jit --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16 --soc=ascend910_93
                DP_ASSERT_EQUAL "$?" "0" "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend910_93]"
            elif [[ "${task_name}" =~ "950" ]]; then
                LOG_DO bash build.sh --pkg --jit --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16 --soc=ascend950
                DP_ASSERT_EQUAL "$?" "0" "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend950]"
            else
                LOG_DO bash build.sh --pkg --jit --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16 --soc=ascend910b
                DP_ASSERT_EQUAL "$?" "0" "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend910b]"
            fi
        else
            echo "not need build monitor"
            mkdir build_out
            touch build_out/cann-ops-transformer_linux-x86_64.run
            exit 0
        fi
    else
        LOG_DO bash build.sh --pkg --jit --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
        DP_ASSERT_EQUAL "$?" "0" "Build ${REPOSITORY_NAME}"
    fi
fi
