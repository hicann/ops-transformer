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

set -e
RELEASE_TARGETS=("ophost" "opapi" "opgraph" "onnxplugin")
UT_TARGETS=()
########################################################################################################################
# 预定义变量
########################################################################################################################

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR=${CURRENT_DIR}/build
OUTPUT_DIR=${CURRENT_DIR}/output
BUILD_OUT_DIR=${CURRENT_DIR}/build_out
USER_ID=$(id -u)
PARENT_JOB="false"
HOST_TILING="false"
CHECK_COMPATIBLE="true"
ASAN="false"
UBSAN="false"
COV="false"
CLANG="false"
VERBOSE="false"
OOM="false"
ENABLE_MSSANITIZER=FALSE
ENABLE_DUMP_CCE=FALSE
THREAD_NUM=$(grep -c ^processor /proc/cpuinfo)
ENABLE_VALGRIND=FALSE
ENABLE_CREATE_LIB=FALSE
ENABLE_OPKERNEL=FALSE
ENABLE_BUILD_PKG=FALSE
ENABLE_BUILT_IN=FALSE
ENABLE_BUILT_JIT=FALSE
ENABLE_AICPU=TRUE
ENABLE_BUILT_CUSTOM=FALSE
ENABLE_STATIC=FALSE
ENABLE_EXPERIMENTAL=FALSE
ENABLE_TORCH_EXTENSION_ONLY=FALSE
PACKAGE_TYPE="run"
PACKAGE_TYPE_SET=FALSE
KERNEL_TEMPLATE_INPUT=""
ASCEND_SOC_UNITS="ascend910b"
# 标记用户是否显式指定了 --soc; 未指定时由 detect_soc_from_npu_smi 自动探测 NPU 芯片型号
SOC_USER_SPECIFIED=false
# 支持的 SoC 列表(校验用, 单一数据源): 取自 cmake/scripts/util/const_var.py 的 SOC_MAP_EXT keys(出包遍历的 kernel 编译 soc 集合)
# + CMakeLists.txt SOC_VERSION_LIST 中的 mc62。不在 CMakeLists.txt SOC_VERSION_LIST 的 soc 会走空包(cpack_empty_package)
SUPPORT_COMPUTE_UNIT_SHORT=("ascend910b" "ascend910_93" "ascend950" "ascend350" "ascend310p" "ascend310b" "ascend910" "ascend610lite" "kirinx90" "kirin9030" "mc62")
CMAKE_BUILD_MODE=""
BUILD_TYPE=""
VERSION=""
BUILD_LIBS=()
OP_API_UT=FALSE
OP_HOST_UT=FALSE
OP_GRAPH_UT=FALSE
OP_KERNEL_UT=FALSE
OP_API=FALSE
OP_HOST=FALSE
OP_GRAPH=FALSE
ONNX_PLUGIN=FALSE
OP_KERNEL=FALSE
SOC_ARRAY=()
ENABLE_UT_EXEC=TRUE
ENABLE_RULE_LAUNCH=""
ENABLE_GENOP=FALSE
ENABLE_GENOP_AICPU=FALSE
GENOP_TYPE=""
GENOP_NAME=""
INCREMENTAL_BUILD=FALSE
PR_CHANGED_FILES=""  # PR场景, 修改文件清单, 可用于标识是否PR场景
UT_SOC_ARRAY=()
UT_TEST_CNT=0
PR_UT_FLAG=FALSE
CI_MODE=FALSE
SPECIFIC_EXAMPLE_NAME=""  # 指定运行特定的 example 文件名

if [ "${USER_ID}" != "0" ]; then
    DEFAULT_TOOLKIT_INSTALL_DIR="${HOME}/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="${HOME}/Ascend/latest"
else
    DEFAULT_TOOLKIT_INSTALL_DIR="/usr/local/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="/usr/local/Ascend/latest"
fi
BISHENG_FLAGS=""
ENABLE_AICPU_KERNEL=FALSE
CANN_3RD_LIB_PATH="${CURRENT_DIR}/third_party"
CUSTOM_OPTION="-DBUILD_OPEN_PROJECT=ON"

dotted_line="---------------------------------------------------------------------------------------------------------------------"
########################################################################################################################
# 预定义函数
########################################################################################################################

function help_info() {
    local specific_help="$1"

    if [[ -n "$specific_help" ]]; then
        case "$specific_help" in
            package)
                echo "Package Build Options:"
                echo $dotted_line
                echo "    --pkg                  Build run package with kernel bin"
                echo "    --pkg-type=<TYPE>      Specify package type(TYPE options: run/rpm/deb/all), Default: run"
                echo "    --jit                  Build run package without kernel bin"
                echo "    --soc=soc_version      Compile for specified Ascend SoC (comma-separated for multiple)"
                echo "    --vendor_name=name     Specify custom operator package vendor name"
                echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
                echo "    --module=module1,...   Compile specified modules (comma-separated, supported: mc2,attention,moe,ffn,mhc,posembedding,gmm)"
                echo "    --incremental          Reuse existing build/output directories without default clean"
                echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
                echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
                echo "    --experimental         Build experimental version"
                echo "    --cann_3rd_lib_path=<PATH>"
                echo "                           Set ascend third_party package install path, default ./third_party"
                echo "    --oom                  Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
                echo "    --asan                 Enable ASAN (Address Sanitizer) on the host side"
                echo "    --kernel_template_input=args0,args1"
                echo "                           Specify kernel template input arguments (comma-separated for multiple)"
                echo "    --bisheng_flags        Specify bisheng compiler flags (comma-separated for multiple)"
                echo "    --mssanitizer          Build with mssanitizer mode on the kernel side, with options: '-g --cce-enable-sanitizer'"
                echo "    --dump_cce             Dump kernel precompiled files"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --pkg --soc=ascend910b --vendor_name=customize -j16 -O3"
                echo "    bash build.sh --pkg --pkg-type=deb --soc=ascend910b"
                echo "    bash build.sh --pkg --pkg-type=rpm --soc=ascend910b"
                echo "    bash build.sh --pkg --pkg-type=all --soc=ascend910b"
                echo "    bash build.sh --pkg --ops=add,sub"
                echo "    bash build.sh --pkg --experimental --soc=ascend910b"
                echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=abs --oom"
                echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=abs --bisheng_flags=dumc_cce"
                echo "    bash build.sh --pkg --soc=ascend910b --ops=add_example --opkernel_aicpu -j16"
                return
                ;;
            test)
                echo "Test Options:"
                echo $dotted_line
                echo "    -u|--test              Build and run all unit tests"
                echo "    --noexec               Only compile ut, do not execute"
                echo "    --cov                  Enable code coverage for unit tests"
                echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
                echo "    --soc=soc_version      Run unit tests for specified Ascend SoC"
                echo "    --incremental          Reuse existing build/output directories without default clean"
                echo "    --valgrind             Run unit tests with valgrind (disables ASAN and noexec)"
                echo "    --ophost_test          Build and run ophost unit tests"
                echo "    --opapi_test           Build and run opapi unit tests"
                echo "    --opgraph_test         Build and run opgraph unit tests"
                echo "    --ophost -u            Same as --ophost_test"
                echo "    --opapi -u             Same as --opapi_test"
                echo "    --opgraph -u           Same as --opgraph_test"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh -u --noexec --cov"
                echo "    bash build.sh --ophost_test --opapi_test --noexec"
                echo "    bash build.sh --ophost --opapi --opgraph -u --cov"
                return
                ;;
            clean)
                echo "Clean Options:"
                echo $dotted_line
                echo "    --make_clean           Clean build artifacts"
                echo $dotted_line
                return
                ;;
            valgrind)
                echo "Valgrind Options:"
                echo $dotted_line
                echo "    --valgrind             Run unit tests with valgrind (disables ASAN and noexec)"
                echo $dotted_line
                return
                ;;
            ophost)
                echo "Ophost Build Options:"
                echo $dotted_line
                echo "    --ophost               Build ophost library"
                echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
                echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --ophost -j16 -O3"
                return
                ;;
            opapi)
                echo "Opapi Build Options:"
                echo $dotted_line
                echo "    --opapi                Build opapi library"
                echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
                echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --opapi -j16 -O3"
                return
                ;;
            opgraph)
                echo "Opgraph Build Options:"
                echo $dotted_line
                echo "    --opgraph              Build opgraph library"
                echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
                echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --opgraph -j16 -O3"
                return
                ;;
            onnxplugin)
                echo "ONNXPlugin Build Options:"
                echo $dotted_line
                echo "    --onnxplugin           Build onnxplugin library"
                echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
                echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
                echo "    --debug                Build with debug mode"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --onnxplugin -j16 -O3"
                echo "    bash build.sh --onnxplugin --debug"
                return
                ;;
            opkernel)
                echo "Opkernel Build Options:"
                echo $dotted_line
                echo "    --opkernel             Build binary kernel"
                echo "    --soc=soc_version      Compile for specified Ascend SoC (comma-separated for multiple)"
                echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
                echo "    --oom                  Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
                echo "    --kernel_template_input=args0,args1"
 	            echo "                           Specify kernel template input arguments (comma-separated for multiple)"
                echo "    --bisheng_flags        Specify bisheng compiler flags (comma-separated for multiple)"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --opkernel --soc=ascend310p --ops=add,sub"
                echo "    bash build.sh --opkernel --soc=ascend310p --ops=add,sub --oom"
                echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=abs --bisheng_flags=dumc_cce"
                return
                ;;
            opkernel_aicpu)
                echo "AICPU Opkernel Build Options:"
                echo $dotted_line
                echo "    --opkernel_aicpu       Build AICPU kernel"
                echo "    --soc=soc_version      Compile for specified Ascend SoC"
                echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
                echo "    --build-type=<Type>    Specify build-type (Type options: Release/Debug), Default:Release"
                echo "    --mssanitizer          Build with mssanitizer mode on the kernel side"
                echo "    --oom                  Build with oom mode on the kernel side"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=add_example"
                echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=add_example --build-type=Debug"
                return
                ;;
            ophost_test)
                echo "Ophost Test Options:"
                echo $dotted_line
                echo "    --ophost_test          Build and run ophost unit tests"
                echo "    --noexec               Only compile ut, do not execute"
                echo "    --cov                  Enable code coverage for unit tests"
                echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
                echo "    --soc=soc_version      Run unit tests for specified Ascend SoC"
                echo "    --valgrind             Run unit tests with valgrind (disables ASAN and noexec)"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --ophost_test --noexec --cov"
                return
                ;;
            opapi_test)
                echo "Opapi Test Options:"
                echo $dotted_line
                echo "    --opapi_test           Build and run opapi unit tests"
                echo "    --noexec               Only compile ut, do not execute"
                echo "    --cov                  Enable code coverage for unit tests"
                echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
                echo "    --soc=soc_version      Run unit tests for specified Ascend SoC"
                echo "    --valgrind             Run unit tests with valgrind (disables ASAN and noexec)"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --opapi_test --noexec --cov"
                return
                ;;
            opgraph_test)
                echo "Opgraph Test Options:"
                echo $dotted_line
                echo "    --opgraph_test         Build and run opgraph unit tests"
                echo "    --noexec               Only compile ut, do not execute"
                echo "    --cov                  Enable code coverage for unit tests"
                echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
                echo "    --soc=soc_version      Run unit tests for specified Ascend SoC"
                echo "    --valgrind             Run unit tests with valgrind (disables ASAN and noexec)"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --opgraph_test --noexec --cov"
                return
                ;;
            run_example)
                echo "Run examples Options:"
                echo $dotted_line
                echo "    --run_example op_type  mode[eager:graph] [pkg_mode --vendor_name=name --example_name=name --soc=soc_version]     Compile and execute the test_aclnn_xxx.cpp/test_geir_xxx.cpp"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --run_example abs eager"
                echo "    bash build.sh --run_example abs eager --soc=ascend950"
                echo "    bash build.sh --run_example abs graph"
                echo "    bash build.sh --run_example abs eager --example_name=name"
                echo "    bash build.sh --run_example abs eager cust"
                echo "    bash build.sh --run_example abs eager cust --vendor_name=custom"
                return
                ;;
            genop)
                echo "Gen Op Directory Options:"
                echo $dotted_line
                echo "    --genop=op_class/op_name      Create the initial directory for op_name undef op_class"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --genop=example/add"
                return
                ;;
            genop_aicpu)
                echo "Gen Op Directory Options:"
                echo $dotted_line
                echo "    --genop_aicpu=op_class/op_name      Create the initial directory for op_name undef op_class"
                echo $dotted_line
                echo "Examples:"
                echo "    bash build.sh --genop_aicpu=examples/add_example"
                return
                ;;
        esac
    fi
    echo "build script for ops-transformer repository"
    echo "Usage:"
    echo "    bash build.sh [-h] [-j[n]] [-v] [-O[n]] [-u] "
    echo ""
    echo ""
    echo "Options:"
    echo $dotted_line
    echo "    Build parameters "
    echo $dotted_line
    echo "    -h Print usage"
    echo "    -j[n] Compile thread nums, default is 8, eg: -j8"
    echo "    -v Cmake compile verbose"
    echo "    -O[n] Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
    echo "    -u Compile all ut"
    echo $dotted_line
    echo "    examples, Build ophost_test with O3 level compilation optimization and do not execute."
    echo "    ./build.sh --ophost_test --noexec -O3"
    echo $dotted_line
    echo "    The following are all supported arguments:"
    echo $dotted_line
    echo "    --build-type=<TYPE> Specify build type (TYPE options:Release/Debug), Default: Release"
    echo "    --version Specify version"
    echo "    --cov When building uTest locally, count the coverage."
    echo "    --noexec Only compile ut, do not execute the compiled executable file"
    echo "    --incremental Reuse existing build/output directories and skip the default clean step"
    echo "    --make_clean Clean build artifacts"
    echo "    --asan Enable asan on the host side"
    echo "    --valgrind run ut with valgrind. This option will disable asan, noexec and run utest by valgrind"
    echo "    --ops Compile specified operator, use snake name, like: --ops=add,add_lora, use ',' to separate different operator"
    echo "    --module Compile specified module, like: --module=mc2,attention, use ',' to separate different modules (supported: mc2,attention,moe,ffn,mhc,posembedding,gmm)"
    echo "    --soc Compile binary with specified Ascend SoC, like: --soc=ascend910b,ascend910_93,ascend950 use ',' to separate different SoC"
    echo "    --soc supported parameters must only in [$(get_supported_soc_text)], A3(--soc=ascend910_93)"
    echo "    --list_soc List all supported Ascend SoC"
    echo "    --vendor_name Specify the custom operator package vendor name, like: --vendor_name=customize, default to custom"
    echo "    --opgraph build graph_plugin_transformer.so"
    echo "    --onnxplugin build oponnx_plugin_transformer.so"
    echo "    --opapi build opapi_transformer.so"
    echo "    --ophost build ophost_transformer.so"
    echo "    --opkernel build binary kernel"
    echo "    --jit build run package without kernel bin"
    echo "    --pkg build run package with kernel bin"
    echo "    --pkg-type=<TYPE> Specify package type(TYPE options: run/rpm/deb/all), Default: run"
    echo "    --torch_extension           Build torch_extension whl only, support --ops for single op packaging"
    echo "    --torch_extension_only       Build torch_extension whl package only"
    echo "    --experimental build experimental version"
    echo "    --opkernel_aicpu build aicpu kernel"
    echo "    --mssanitizer Build with mssanitizer mode on the kernel side, with options: '-g --cce-enable-sanitizer'"
    echo "    --dump_cce Dump kernel precompiled files"
    echo "    --opapi_test build and run opapi unit tests"
    echo "    --ophost_test build and run ophost unit tests"
    echo "    --opgraph_test build and run opgraph unit tests"
    echo "    --opkernel_test build and run opkernel unit tests"
    echo "    --run_example Compile and execute the test_aclnn_xxx.cpp/test_geir_xxx.cpp"
    echo "    --simulator    Enable simulator mode for run_example (requires --soc parameter)"
    echo "    --genop Create the initial directory for op"
    echo "    --genop_aicpu Create the initial directory for AI CPU op"
    echo "to be continued ..."
}


export BASE_PATH=$(
    cd "$(dirname $0)"
    pwd
)
export BUILD_PATH="${BASE_PATH}/build"
# source SoC 名称校验库, 并复用 SUPPORT_COMPUTE_UNIT_SHORT 作为唯一支持列表(单一数据源)
if [[ -f "${BASE_PATH}/scripts/util/soc_validator.sh" ]]; then
    source "${BASE_PATH}/scripts/util/soc_validator.sh"
else
    echo "[ERROR] scripts/util/soc_validator.sh not found, please check the repository integrity."
    exit 1
fi
SUPPORTED_SOC_LIST=("${SUPPORT_COMPUTE_UNIT_SHORT[@]}")
function log() {
    local current_time=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[$current_time] "$1
}

check_pkg_type() {
  local pkg_type="$1"
  if [[ "$pkg_type" != "run" && "$pkg_type" != "rpm" && "$pkg_type" != "deb" && "$pkg_type" != "all" ]]; then
    echo "[ERROR] --pkg-type only supports run/rpm/deb/all, got: $pkg_type"
    exit 1
  fi
}

function set_env()
{
    source $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash || echo "0"

    export BISHENG_REAL_PATH=$(which bisheng || true)

    if [ -z "${BISHENG_REAL_PATH}" ];then
        if [[ "$ENABLE_BUILT_JIT" == "TRUE" ]] && [[ "$ENABLE_AICPU" == "FALSE" ]] ; then
            log "Warning: bisheng compilation tool not found, but --jit --noaicpu is enabled, so continue."
            return
        fi
        log "Error: bisheng compilation tool not found, Please check whether the cann package or environment variables are set."
        exit 1
    fi
}

function clean()
{
    if [ -n "${BUILD_DIR}" ];then
        rm -rf ${BUILD_DIR}
    fi

    if [ -z "${TEST}" ] && [ -z "${EXAMPLE}" ];then
        if [ -n "${OUTPUT_DIR}" ];then
            rm -rf ${OUTPUT_DIR}
        fi
    fi

    mkdir -p ${BUILD_DIR} ${OUTPUT_DIR}
}

function ensure_build_dirs()
{
    mkdir -p ${BUILD_DIR} ${OUTPUT_DIR}
}

function clean_build_out()
{
    if [ -n "${BUILD_OUT_DIR}" ];then
        rm -rf ${BUILD_OUT_DIR}
    fi

    mkdir -p ${BUILD_OUT_DIR}
}

function clean_third_party()
{
    THIRD_PARTY_PATH=${BASE_PATH}/third_party
    if [ -d "${THIRD_PARTY_PATH}" ]; then
        rm -rf ${THIRD_PARTY_PATH}/abseil-cpp
        rm -rf ${THIRD_PARTY_PATH}/ascend_protobuf
    fi
}

function cmake_config()
{
    local extra_option="$1"
    log "Info: cmake config ${CUSTOM_OPTION} ${extra_option} ."
    cmake ..  ${CUSTOM_OPTION} ${extra_option}
}

function ci_print_compile_failed_ops_info()
{
    local failed_files=$(find . -type f -name "failed_ops.log")
    local success_files=$(find . -type f -name "success_ops.log")

    declare -A failed_ops_map
    declare -A success_ops_map

    if [[ -n "$failed_files" ]]; then
        while IFS= read -r file; do
            [[ -s "$file" ]] || continue
            while IFS= read -r line; do
                [[ -z "$line" ]] && continue
                local op_name=$(echo "$line" | awk '{print $1}')
                local bin_name=$(echo "$line" | cut -d' ' -f2-)
                if [[ -n "$op_name" && -n "$bin_name" ]]; then
                    if [[ -z "${failed_ops_map[$op_name]}" ]]; then
                        failed_ops_map["$op_name"]="$bin_name"
                    else
                        failed_ops_map["$op_name"]="${failed_ops_map[$op_name]}\\n$bin_name"
                    fi
                fi
            done < "$file"
        done <<< "$failed_files"
    fi

    if [[ -n "$success_files" ]]; then
        while IFS= read -r file; do
            [[ -s "$file" ]] || continue
            while IFS= read -r line; do
                [[ -z "$line" ]] && continue
                local op_name=$(echo "$line" | awk '{print $1}')
                local bin_name=$(echo "$line" | cut -d' ' -f2-)
                if [[ -n "$op_name" && -n "$bin_name" ]]; then
                    if [[ -z "${success_ops_map[$op_name]}" ]]; then
                        success_ops_map["$op_name"]="$bin_name"
                    else
                        success_ops_map["$op_name"]="${success_ops_map[$op_name]}\\n$bin_name"
                    fi
                fi
            done < "$file"
        done <<< "$success_files"
    fi

    if [[ ${#success_ops_map[@]} -gt 0 ]]; then
        echo "All CI compile success ops:"
        for op_name in "${!success_ops_map[@]}"; do
            local bin_list="${success_ops_map[$op_name]}"
            echo "ops name: $op_name, success bin: $bin_list"
        done
    fi
    if [[ ${#failed_ops_map[@]} -gt 0 ]]; then
        echo "All CI compile failed ops:"
        for op_name in "${!failed_ops_map[@]}"; do
            local bin_list="${failed_ops_map[$op_name]}"
            echo "ops name: $op_name, failed bin: $bin_list"
        done
        echo "[ERROR] build failed!"
        exit 1
    fi
}

function build()
{
    local target="$1"
    if [ "${VERBOSE}" == "true" ];then
        local option="--verbose"
    fi
    export LD_LIBRARY_PATH=${BUILD_DIR}:$LD_LIBRARY_PATH
    if [[ "$CI_MODE" == "TRUE" ]]; then
        export CI_MODE=TRUE
        set +e
        cmake --build . --target ${target} ${JOB_NUM} ${option}
        set -e
    else
        cmake --build . --target ${target} ${JOB_NUM} ${option}
    fi
    ci_print_compile_failed_ops_info
}

ARCH_INFO=$(uname -m)

export INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export ACLNN_INCLUDE_PATH="${INCLUDE_PATH}/aclnn"
export COMPILER_INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export GRAPH_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/graph"
export GE_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/ge"
export INC_INCLUDE_PATH="${ASCEND_OPP_PATH}/built-in/op_proto/inc"
export LINUX_INCLUDE_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include"
export EAGER_LIBRARY_OPP_PATH="${ASCEND_OPP_PATH}/lib64"
export EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
export GRAPH_LIBRARY_STUB_PATH="${ASCEND_HOME_PATH}/lib64/stub"
export GRAPH_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"

export EAGER_INCLUDE_OPP_ACLNNOP_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include/aclnnop"


function build_example()
{
    log "Start to run example,name:${EXAMPLE_NAME} mode:${EXAMPLE_MODE}"

    local verbose_option=""
    if [ "${VERBOSE}" == "true" ];then
        verbose_option="-v"
    fi

    if [ ! -d "${BUILD_PATH}" ]; then
    	mkdir -p ${BUILD_PATH}
    fi

    # 清理CMake缓存
    # clean_cmake_cache
    clean

    cd "${BUILD_PATH}"
    if [[ "${EXAMPLE_MODE}" == "eager" ]]; then
        pattern="test_aclnn_"
    elif [[ "${EXAMPLE_MODE}" == "graph" ]]; then
        pattern="test_geir_"
    fi
    # No soc provided.
    if [[ -z "$ASCEND_SOC_UNITS" ]]; then
        ASCEND_SOC_UNITS="ascend910b"
    fi
    is_soc_support=""
    for support_unit in "${SUPPORT_COMPUTE_UNIT_SHORT[@]}"; do
        if [[ "$support_unit" == "$ASCEND_SOC_UNITS" ]]; then
            is_soc_support="true"
            break
        fi
    done
    if [[ -z "$is_soc_support" ]]; then
        echo "Currently $ASCEND_SOC_UNITS is not supported, please input a valid soc."
        return 1
    fi
    # 非 experimental 模式下排除 experimental 目录,避免 built-in 算子修改误触发 experimental 算子的 example
    local exclude_experimental=()
    if [[ "$ENABLE_EXPERIMENTAL" != "TRUE" ]]; then
        exclude_experimental=(-not -path "*/experimental/*")
    fi
    # Obtain the example file corresponding to the input soc unit.
    if [[ "$ASCEND_SOC_UNITS" == "ascend950" ]]; then
        # 1. ascend950/ascend950 example is independent of other soc units.
        files=($(find ../ -path "*/${EXAMPLE_NAME}/examples/arch35/${pattern}*.cpp" "${exclude_experimental[@]}"))
        if [[ -z "$files" ]]; then
            # 2. Example is shared with other soc units, or the current operator only supports ascend950/ascend950.
            files=($(find ../ -path "*/${EXAMPLE_NAME}/examples/${pattern}*.cpp" "${exclude_experimental[@]}"))
        fi
    else
        # Except for ascend950/ascend950, the examples of other soc units are temporarily shared.
        # If you need to add independent examples, you can refer to the method of adding a directory for isolation.
        files=($(find ../ -path "*/${EXAMPLE_NAME}/examples/${pattern}*.cpp" "${exclude_experimental[@]}"))
    fi
    # Compile and Execute
    if [[ "${EXAMPLE_MODE}" == "eager" ]]; then
        if [ -z "$files" ]; then
            echo "${EXAMPLE_NAME} do not have eager example"
            return 2
        fi
        ABSOLUTE_MC2_PATH=$(realpath ${BUILD_PATH}/../mc2) # mc2目录绝对路径
        ABSOLUTE_EXAMPLES_MC2_PATH=$(realpath ${BUILD_PATH}/../examples/mc2)
        ABSOLUTE_EXPERIMENTAL_MC2_PATH=$(realpath ${BUILD_PATH}/../experimental/mc2)
        found_specific_example=false
        for file in "${files[@]}"; do
            # 如果指定了特定的 example 文件名，只运行该文件
            if [[ -n "${SPECIFIC_EXAMPLE_NAME}" ]]; then
                file_basename=$(basename "$file" .cpp)
                specific_name_without_suffix="${SPECIFIC_EXAMPLE_NAME%.cpp}"
                # 如果不以 pattern 开头，自动加上 pattern
                if [[ ! "${specific_name_without_suffix}" =~ ^${pattern} ]]; then
                    specific_name_without_suffix="${pattern}${specific_name_without_suffix}"
                fi
                if [[ "${file_basename}" != "${specific_name_without_suffix}" ]]; then
                    continue
                fi
                found_specific_example=true
            fi
            echo "Start compile and run example file: $file"
            REAL_FILE_PATH=$(realpath "$file")
            MC2_APPEND_INCLUDE_AND_LIBRARY=""
            if [[ "$REAL_FILE_PATH" == "${ABSOLUTE_MC2_PATH}"* || "$REAL_FILE_PATH" == "${ABSOLUTE_EXAMPLES_MC2_PATH}"* || "$REAL_FILE_PATH" == "${ABSOLUTE_EXPERIMENTAL_MC2_PATH}"* ]]; then
                MC2_APPEND_INCLUDE_AND_LIBRARY="-lpthread -Wl,--no-as-needed -lhccl -lhccl_fwk"
            fi
            if [[ "${PKG_MODE}" == "" ]]; then
                g++ ${verbose_option} ${file} \
                    -I ${INCLUDE_PATH} -I ${ACLNN_INCLUDE_PATH} -I ${EAGER_INCLUDE_OPP_ACLNNOP_PATH} \
                    -L ${EAGER_LIBRARY_OPP_PATH} -L ${EAGER_LIBRARY_PATH} \
                    -lopapi_math -lopapi_transformer -lascendcl -lnnopbase \
                    -lc_sec ${MC2_APPEND_INCLUDE_AND_LIBRARY} \
                    -o test_aclnn_${EXAMPLE_NAME}
            elif [[ "${PKG_MODE}" == "cust" ]]; then
                if [[ "${vendor_name}" == "" ]]; then
                    vendor_name="custom"
                fi
                echo "pkg_mode:${PKG_MODE} vendor_name:${vendor_name}"
                export CUST_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/${vendor_name}_transformer/op_api/lib"     # 仅自定义算子需要
                export CUST_INCLUDE_PATH="${ASCEND_OPP_PATH}/vendors/${vendor_name}_transformer/op_api/include" # 仅自定义算子需要
                if [[ -n "${ASCEND_CUSTOM_OPP_PATH}" ]]; then
                    CUST_VENDORS_PATH=$(dirname "${ASCEND_CUSTOM_OPP_PATH%%:*}")
                    CUST_LIBRARY_PATH="${CUST_VENDORS_PATH}/${vendor_name}_transformer/op_api/lib"
                    CUST_INCLUDE_PATH="${CUST_VENDORS_PATH}/${vendor_name}_transformer/op_api/include"
                fi
                g++ ${verbose_option} ${file} \
                    -I ${CUST_INCLUDE_PATH} -I ${INCLUDE_PATH} \
                    -L ${CUST_LIBRARY_PATH} -L ${EAGER_LIBRARY_PATH} \
                    -lopapi_math -lcust_opapi -lascendcl -lnnopbase \
                    -I ${EAGER_INCLUDE_OPP_ACLNNOP_PATH} \
                    -lc_sec ${MC2_APPEND_INCLUDE_AND_LIBRARY} \
                    -o test_aclnn_${EXAMPLE_NAME} \
                    -Wl,-rpath=${CUST_LIBRARY_PATH}
            else
                echo "Error: pkg_mode(${PKG_MODE}) must be cust."
                help_info "run_example"
                return 1
            fi
            if [[ "${SIMULATOR}" == "camodel" && "${ASCEND_SOC_UNITS}" == "ascend950" ]]; then
                cannsim record -s Ascend950 ./test_aclnn_${EXAMPLE_NAME} --gen-report
            else
                ./test_aclnn_${EXAMPLE_NAME}
            fi
            run_result=$?
            if [ $run_result -ne 0 ]; then
                echo "run test_aclnn_${EXAMPLE_NAME}, execute samples failed"
                return $run_result
            else
                echo "run test_aclnn_${EXAMPLE_NAME}, execute samples success"
            fi
        done
        # 如果指定了特定的 example 文件名但没有找到，给出错误提示
        if [[ -n "${SPECIFIC_EXAMPLE_NAME}" && "${found_specific_example}" == "false" ]]; then
            echo "Error: Specified example '${SPECIFIC_EXAMPLE_NAME}' not found in ${EXAMPLE_NAME}/examples/"
            return 2
        fi
    elif [[ "${EXAMPLE_MODE}" == "graph" ]]; then
        if [ -z "$files" ]; then
            echo "${EXAMPLE_NAME} do not have graph example"
            return 2
        fi
        found_specific_example=false
        for file in "${files[@]}"; do
            # 如果指定了特定的 example 文件名，只运行该文件
            if [[ -n "${SPECIFIC_EXAMPLE_NAME}" ]]; then
                file_basename=$(basename "$file" .cpp)
                specific_name_without_suffix="${SPECIFIC_EXAMPLE_NAME%.cpp}"
                # 如果不以 pattern 开头，自动加上 pattern (支持 operator name 格式)
                if [[ ! "${specific_name_without_suffix}" =~ ^${pattern} ]]; then
                    specific_name_without_suffix="${pattern}${specific_name_without_suffix}"
                fi
                if [[ "${file_basename}" != "${specific_name_without_suffix}" ]]; then
                    continue
                fi
                found_specific_example=true
            fi
            echo "Start compile and run example file: $file"
            g++ ${verbose_option} ${file} -I ${GRAPH_INCLUDE_PATH} -I ${GE_INCLUDE_PATH} -I ${LINUX_INCLUDE_PATH} -I ${INC_INCLUDE_PATH} -L ${GRAPH_LIBRARY_STUB_PATH} -L ${GRAPH_LIBRARY_PATH} -lgraph -lge_runner -lgraph_base -lge_compiler -o test_geir_${EXAMPLE_NAME}
            ./test_geir_${EXAMPLE_NAME}
            run_result=$?
            if [ $run_result -ne 0 ]; then
                echo "run test_geir_${EXAMPLE_NAME}, execute samples failed"
                return $run_result
            else
                echo "run test_geir_${EXAMPLE_NAME}, execute samples success"
            fi
        done
        # 如果指定了特定的 example 文件名但没有找到，给出错误提示
        if [[ -n "${SPECIFIC_EXAMPLE_NAME}" && "${found_specific_example}" == "false" ]]; then
            echo "Error: Specified example '${SPECIFIC_EXAMPLE_NAME}' not found in ${EXAMPLE_NAME}/examples/"
            return 2
        fi
    else
        help_info "run_example"
        return 1
    fi
    return 0
}

function find_rpm_deb_package() {
    if [[ "$PACKAGE_TYPE" == "run" ]]; then
        return 0
    fi

    find "${BUILD_PATH}" -type f -name "cann-*-ops-transformer*.${PACKAGE_TYPE}" | sort
}

function clean_rpm_deb_package() {
    if [[ "$PACKAGE_TYPE" == "run" ]]; then
        return 0
    fi

    local package_files=()
    while IFS= read -r package_file; do
        package_files+=("${package_file}")
    done < <(find_rpm_deb_package)

    if [[ ${#package_files[@]} -eq 0 ]]; then
        return 0
    fi

    for package_file in "${package_files[@]}"; do
        rm -f "${package_file}"
        echo "[INFO] Removed stale package artifact: ${package_file}"
    done
}

function collect_rpm_deb_package() {
    if [[ "$PACKAGE_TYPE" == "run" ]]; then
        return 0
    fi

    local package_files=()
    while IFS= read -r package_file; do
        package_files+=("${package_file}")
    done < <(find_rpm_deb_package)

    if [[ ${#package_files[@]} -eq 0 ]]; then
        echo "[ERROR] No .${PACKAGE_TYPE} package found in ${BUILD_PATH}"
        exit 1
    fi

    if [ ! -d "${BUILD_OUT_DIR}" ]; then
        mkdir -p "${BUILD_OUT_DIR}"
    fi
    for package_file in "${package_files[@]}"; do
        cp -f "${package_file}" "${BUILD_OUT_DIR}/"
        echo "[INFO] Package artifact copied to ${BUILD_OUT_DIR}/$(basename "${package_file}")"
    done
}

function build_package(){
    if [ -f "${BUILD_PATH}/no_ops.flag" ]; then
        echo "[INFO] No operator can be compiled (all requested ops skipped by CANN version check), skip package generation."
        return 0
    fi
    if [[ "${PACKAGE_TYPE}" == "all" ]]; then
        local saved_pkg_type="${PACKAGE_TYPE}"
        local option=""
        if [ "${VERBOSE}" == "true" ]; then
            option="--verbose"
        fi
        for PACKAGE_TYPE in run rpm deb; do
            clean_rpm_deb_package
            cmake -DPACKAGE_TYPE="${PACKAGE_TYPE}" "${BUILD_PATH}" > /dev/null 2>&1
            cmake --build . --target package ${JOB_NUM} ${option}
            if [ $? -ne 0 ]; then
                echo "[ERROR] target:package (${PACKAGE_TYPE}) build failed!"
                exit 1
            fi
            collect_rpm_deb_package
        done
        PACKAGE_TYPE="${saved_pkg_type}"
    else
        clean_rpm_deb_package
        build package
        collect_rpm_deb_package
    fi
}

function build_host(){
    build_package
}

function build_kernel(){
    if [ -f "${BUILD_PATH}/no_ops.flag" ]; then
        echo "[INFO] No operator can be compiled (all requested ops skipped by CANN version check), skip kernel build."
        return 0
    fi
    build ops_transformer_kernel
}

function build_torch_extension_whl() {
    local torch_ext_dir="${CURRENT_DIR}/torch_extension"
    local original_dir=$(pwd)
    if [ -d "${torch_ext_dir}" ]; then
        log "[INFO] Building torch_extension whl package..."
        cd "${torch_ext_dir}"

        # 检查 build 模块是否可用
        if ! python3 -c "import build" 2>/dev/null; then
            log "[WARNING] Python build module not found, skipping whl build"
            cd "${original_dir}"
            return 0
        fi

        if [[ -n "${ascend_op_name}" ]]; then
            export TORCH_EXTENSION_OPS="${ascend_op_name}"
            if [[ -n "${vendor_name}" ]]; then
                export TORCH_EXTENSION_VENDOR="${vendor_name}"
            else
                export TORCH_EXTENSION_VENDOR="custom"
            fi
            log "[INFO] Building torch_extension whl with ops: ${ascend_op_name}, vendor: ${TORCH_EXTENSION_VENDOR}"
        else
            unset TORCH_EXTENSION_OPS
            unset TORCH_EXTENSION_VENDOR
        fi

        python3 -m build --wheel -n 2>&1 || {
            log "[ERROR] Failed to build torch_extension whl package"
            cd "${original_dir}"
            return 1
        }
        cd "${original_dir}"
        log "[INFO] torch_extension whl package built successfully"
    fi
}

build_lib() {
  echo $dotted_line
  echo "Start to build libs ${BUILD_LIBS[@]}"
  clean

  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi

  cd "${BUILD_PATH}" && cmake .. ${CUSTOM_OPTION} -DENABLE_BUILT_IN=ON

  for lib in "${BUILD_LIBS[@]}"; do
    echo "Building target ${lib}"
    cmake --build . --target ${lib} ${JOB_NUM}
  done

  echo $dotted_line
  echo "Build libs ${BUILD_LIBS[@]} success"
  echo $dotted_line
}

build_static_lib() {
    local unit="$1"
    echo $dotted_line
    echo "Start to build static lib."

    local option=""
    if [ "${VERBOSE}" == "true" ]; then
        option="--verbose"
    fi

    cd "${BUILD_PATH}" && cmake ${CUSTOM_OPTION} .. -DENABLE_STATIC=ON -DASCEND_COMPUTE_UNIT=${unit}
    local all_targets=$(cmake --build . --target help)
    rm -fr ${BUILD_PATH}/bin_tmp
    mkdir -p ${BUILD_PATH}/bin_tmp
    if echo "${all_targets}" | grep -wq "ophost_transformer_static"; then
        cmake --build . --target ophost_transformer_static ${JOB_NUM} ${option}
    fi
    if echo "${all_targets}" | grep -wq "opapi_transformer_static"; then
        cmake --build . --target opapi_transformer_static ${JOB_NUM} ${option}
    fi
    local jit_command=""
    if [[ "$ENABLE_BUILT_JIT" == "TRUE" ]]; then
        jit_command="-j"
    fi

    rm -fr ${BUILD_PATH}/autogen/${unit}
    python3 "${BASE_PATH}/scripts/util/build_opp_kernel_static.py" GenStaticOpResourceIni -s ${unit} -b ${BUILD_PATH} ${jit_command}
    python3 "${BASE_PATH}/scripts/util/build_opp_kernel_static.py" StaticCompile -s ${unit} -b ${BUILD_PATH} -n=0 -a=${ARCH_INFO} ${jit_command}

    cd "${BUILD_PATH}" && cmake ${CUSTOM_OPTION} .. -DENABLE_STATIC=ON -DASCEND_COMPUTE_UNIT=${unit}
    if echo "${all_targets}" | grep -wq "cann_transformer_static"; then
        cmake --build . --target cann_transformer_static ${JOB_NUM} ${option}
    fi
    echo "Build static lib success!"
}

package_static() {
    local unit="$1"
    if [[ "$ENABLE_BUILT_CUSTOM" == "TRUE" ]]; then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_BUILT_IN=OFF -DENABLE_OPS_HOST=ON -DENABLE_BUILD_PKG=ON"
    else
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_BUILT_IN=ON -DENABLE_OPS_HOST=ON -DENABLE_BUILD_PKG=ON"
    fi
    if [[ "$ENABLE_BUILT_JIT" == "TRUE" ]]; then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_OPS_KERNEL=OFF"
    else
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_OPS_KERNEL=ON"
    fi
    cmake_config -DASCEND_COMPUTE_UNIT=${unit}
    build_package
    # Check weather BUILD_OUT_DIR directory exists
    if [ ! -d "$BUILD_OUT_DIR" ]; then
        echo "Error: Directory $BUILD_OUT_DIR does not exist."
        return 1
    fi

    # Check weather *.run is exists and verify the file numbers
    local run_files=("$BUILD_OUT_DIR"/*.run)
    if [ ${#run_files[@]} -eq 0 ]; then
        echo "Error: No .run files found in $BUILD_OUT_DIR directory."
        return 1
    fi
    if [ ${#run_files[@]} -gt 1 ]; then
        echo "Error: Multiple .run files found in $BUILD_OUT_DIR directory."
        printf '%s\n' "${run_files[@]}"
        return 1
    fi
    # Get filename of *.run file and set new directory name
    local run_file=$(basename "${run_files[0]}")
    if [[ "$run_file" != *"ops-transformer"* ]]; then
        echo "Error: Filename '$run_file' does not contain 'ops-transformer'."
        return 1
    fi
    local static_name="${run_file/ops-transformer/ops-transformer-static}"
    static_name="${static_name%.run}"

    # Check weather $BUILD_PATH/static_library_files directory exists and not empty
    local static_files_dir="$BUILD_PATH/static_library_files"
    if [ ! -d "$static_files_dir" ]; then
        return 0
    fi
    if [ -z "$(ls -A "$static_files_dir")" ]; then
        echo "Error: Directory $static_files_dir is empty."
        return 1
    fi

    # Rename directory
    local new_dir_path="$BUILD_PATH/$static_name"
    if mv "$static_files_dir" "$new_dir_path"; then
        echo "Preparing for packaging: renamed $static_files_dir to $new_dir_path"
    else
        echo "Packaging preparation failed: directory rename failed ($static_files_dir -> $new_dir_path)"
        return 1
    fi

    # Create compressed package and restore directory name
    local new_filename="${static_name}.tar.gz"
    if tar -czf "$BUILD_OUT_DIR/$new_filename" -C "$BUILD_PATH" "$static_name"; then
        echo "[SUCCESS] Build static lib success!"
        echo "Successfully created compressed package: $BUILD_OUT_DIR/$new_filename"
        # Restore original directory name
        echo "Restoring original directory name: $new_dir_path -> $static_files_dir"
        mv "$new_dir_path" "$static_files_dir"
        return 0
    else
        echo "Error: Failed to create compressed package."
        # Attempt to restore original directory name
        mv "$new_dir_path" "$static_files_dir"
        return 1
    fi
}

function process_soc_input(){
    local input_string="$1"
    local value_part="${input_string#*=}"
    ASCEND_SOC_UNITS="${value_part//,/;}"

    # 校验 --soc 输入的每个 SoC 名称, 任一非法立即报错退出
    if ! validate_soc_list "${ASCEND_SOC_UNITS}"; then
        if [[ -z "${INVALID_SOC_ITEM}" ]]; then
            echo "[ERROR] The input soc is empty or contains empty items. Please use one of: $(get_supported_soc_text)"
        else
            echo "[ERROR] The input soc ${INVALID_SOC_ITEM} is not supported. Please use one of: $(get_supported_soc_text)"
        fi
        exit 1
    fi

    # 归一化为小写, 与 CMakeLists.txt 中 SOC_VERSION_LIST 的大小写敏感精确匹配保持一致
    ASCEND_SOC_UNITS="${ASCEND_SOC_UNITS,,}"

    declare -A SOC_HARDWARE_MAP=(
        # 命名依据: README.md(Atlas A2/A3系列产品, Ascend 950PR/Ascend 950DT, KirinX90系列产品)
        # 与 cmake/scripts/util/const_var.py 的 SOC_MAP_EXT
        [ascend910b]="Atlas A2"
        [ascend910_93]="Atlas A3"
        [ascend310p]="Ascend310P3"
        [ascend950]="Ascend 950PR/Ascend 950DT"
        [kirinx90]="KirinX90"
        [kirin9030]="Kirin9030"
        # mc62 在本仓 README/docs/const_var.py 及 ops-nn 仓中均无官方产品命名, 暂不配置, 走通用提示
    )

    if [[ ${SOC_HARDWARE_MAP[$ASCEND_SOC_UNITS]} ]]; then
        echo "Warning: The current environment is configured for $ASCEND_SOC_UNITS, Please use ${SOC_HARDWARE_MAP[$ASCEND_SOC_UNITS]} series hardware for optimal performance."
    else
        echo "Warning: Hardware type '$ASCEND_SOC_UNITS' detected. Please ensure you are using compatible hardware."
    fi

    # 标记用户已显式指定 --soc, 后续 detect_soc_from_npu_smi 据此跳过自动探测
    SOC_USER_SPECIFIED=true
}

# detect_soc_from_npu_smi - 在用户未显式指定 --soc 时, 通过 npu-smi info 自动探测 NPU 芯片型号,
# 映射为编译用 soc_version 并赋值给 ASCEND_SOC_UNITS; 探测失败则保持默认值(ascend910b)。
# 映射依据: cmake/scripts/util/const_var.py 的 SOC_MAP_EXT 与 opdesc_parser.py 的 SOC_TO_SHORT_SOC_MAP。
function detect_soc_from_npu_smi() {
    # 用户已显式指定 --soc, 跳过自动探测
    if [[ "$SOC_USER_SPECIFIED" == "true" ]]; then
        return 0
    fi

    # npu-smi 不可用时, 保持默认 soc
    if ! command -v npu-smi &>/dev/null; then
        echo "[INFO] npu-smi not found, using default soc: $ASCEND_SOC_UNITS"
        return 0
    fi

    # 从 npu-smi info 输出提取首个 NPU 的芯片名(第三字段)
    # 输出行格式: "| 0     910B3               | OK  | ..." -> $3 为芯片名
    local chip_name
    chip_name=$(npu-smi info 2>/dev/null | grep -E '^\| [0-9]+\s+[0-9A-Za-z]+\s+\|' | head -1 | awk '{print $3}')

    if [[ -z "$chip_name" ]]; then
        echo "[INFO] No NPU detected by npu-smi, using default soc: $ASCEND_SOC_UNITS"
        return 0
    fi

    # 芯片名 -> soc_version 映射(大小写不敏感: ${chip_name^^} 转大写后匹配; * 通配匹配同系列变体)
    local detected_soc="unknown"
    case "${chip_name^^}" in
        910A|910)
            detected_soc="ascend910"
            ;;
        910B*)
            detected_soc="ascend910b"
            ;;
        910_93*)
            detected_soc="ascend910_93"
            ;;
        310P*)
            detected_soc="ascend310p"
            ;;
        310B*)
            detected_soc="ascend310b"
            ;;
        950*)
            detected_soc="ascend950"
            ;;
        *)
            detected_soc="unknown"
            ;;
    esac

    if [[ "$detected_soc" != "unknown" ]]; then
        ASCEND_SOC_UNITS="$detected_soc"
        echo "[INFO] Auto-detected NPU chip: $chip_name -> soc_version: $detected_soc"
    else
        echo "[INFO] Unrecognized NPU chip name '$chip_name', using default soc: $ASCEND_SOC_UNITS"
    fi
}

process_genop() {
    local opt_name=$1
    local genop_value=$2

    if [[ "$opt_name" == "genop" ]]; then
        ENABLE_GENOP=TRUE
    elif [[ "$opt_name" == "genop_aicpu" ]]; then
        ENABLE_GENOP_AICPU=TRUE
    else
        usage "genop"
        exit 1
    fi

    if [[ "$genop_value" != *"/"* ]] || [[ "$genop_value" == *"/" ]]; then
        usage "$opt_name"
        exit 1
    fi

    GENOP_NAME=${genop_value##*/}
    local remaining=${genop_value%/*}

    if [[ "$remaining" != *"/"* ]]; then
        GENOP_TYPE=$remaining
        GENOP_BASE=${BASE_PATH}
    else
        GENOP_TYPE=${remaining##*/}
        GENOP_BASE=${remaining%/*}
        if [[ ! "$GENOP_BASE" =~ ^/ && ! "$GENOP_BASE" =~ ^[a-zA-Z]: ]]; then
        GENOP_BASE="${BASE_PATH}/${GENOP_BASE}"
        fi
    fi
}

gen_op() {
  if [[ -z "$GENOP_NAME" ]] || [[ -z "$GENOP_TYPE" ]]; then
    echo "Error: op_class or op_name is not set."
    usage "genop"
  fi

  echo $dotted_line
  echo "Start to create the initial directory for ${GENOP_NAME} under ${GENOP_TYPE}"

  # 检查 python 或 python3 是否存在
  local python_cmd=""
  if command -v python3 &> /dev/null; then
      python_cmd="python3"
  elif command -v python &> /dev/null; then
      python_cmd="python"
  fi

  if [ -n "${python_cmd}" ]; then
    ${python_cmd} "${BASE_PATH}/scripts/opgen/opgen_standalone.py" -t ${GENOP_TYPE} -n ${GENOP_NAME} -p ${GENOP_BASE}
    return $?
  fi
}

gen_aicpu_op() {
  if [[ -z "$GENOP_NAME" ]] || [[ -z "$GENOP_TYPE" ]]; then
    echo "Error: op_class or op_name is not set."
    usage "genop_aicpu"
  fi

  echo $dotted_line
  echo "Start to create the AI CPU initial directory for ${GENOP_NAME} under ${GENOP_TYPE}"

  # 检查 python 或 python3 是否存在
  local python_cmd=""
  if command -v python3 &>/dev/null; then
    python_cmd="python3"
  elif command -v python &>/dev/null; then
    python_cmd="python"
  fi

  if [ -n "${python_cmd}" ]; then
    ${python_cmd} "${BASE_PATH}/scripts/opgen/opgen_standalone.py" -t ${GENOP_TYPE} -n ${GENOP_NAME} -p ${GENOP_BASE} -v aicpu
    echo "Create the AI CPU initial directory for ${GENOP_NAME} under ${GENOP_TYPE} success"
    return $?
  else
    echo "Please install Python to generate op project framework."
  fi
}

# function gen_op() {
#   if [[ -z "$GENOP_NAME" ]] || [[ -z "$GENOP_TYPE" ]]; then
#     echo "Error: op_class or op_name is not set."
#     help_info "genop"
#   fi

#   echo $dotted_line
#   echo "Start to create the initial directory for ${GENOP_NAME} under ${GENOP_TYPE}"

#   if [ ! -d "${GENOP_TYPE}" ]; then
#     mkdir -p "${GENOP_TYPE}"
#     cp examples/CMakeLists.txt "${GENOP_TYPE}/CMakeLists.txt"
#     sed -i '/list(APPEND OP_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}\/ffn\/ffn)/a add_subdirectory('"${GENOP_TYPE}"')' CMakeLists.txt
#   fi

#   BASE_DIR=${GENOP_TYPE}/${GENOP_NAME}
#   mkdir -p "${BASE_DIR}"

#   cp -r examples/add_example/* "${BASE_DIR}/"

#   rm -rf "${BASE_DIR}/examples"
#   rm -rf "${BASE_DIR}/op_host/config"

#   for file in $(find "${BASE_DIR}" -name "*.h" -o -name "*.cpp"); do
#     head -n 14 "$file" >"${file}.tmp"
#     cat "${file}.tmp" >"$file"
#     rm "${file}.tmp"
#   done

#   for file in $(find "${BASE_DIR}" -type f); do
#     sed -i "s/add_example/${GENOP_NAME}/g" "$file"
#   done

#   cd ${BASE_DIR}
#   for file in $(find ./ -name "add_example*"); do
#     new_file=$(echo "$file" | sed "s/add_example/${GENOP_NAME}/g")
#     mv "$file" "$new_file"
#   done

#   echo "Create the initial directory for ${GENOP_NAME} under ${GENOP_TYPE} success"
# }


set_ut_mode() {
  REPOSITORY_NAME="transformer"
  if [[ "$ENABLE_TEST" != "TRUE" ]]; then
    return
  fi
  if [ -n "${PR_CHANGED_FILES}" ]; then
    OP_HOST_UT=TRUE
    OP_API_UT=TRUE
    UT_TEST_ALL=FALSE
    UT_TARGETS+=("${REPOSITORY_NAME}_op_host_ut")
    UT_TARGETS+=("${REPOSITORY_NAME}_op_api_ut")
    return
  fi
  UT_TEST_ALL=TRUE
  if [[ "$OP_HOST" == "TRUE" ]]; then
    OP_HOST_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_API" == "TRUE" ]]; then
    OP_API_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_GRAPH" == "TRUE" ]]; then
    OP_GRAPH_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_KERNEL" == "TRUE" ]]; then
    OP_KERNEL_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_HOST_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_host_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_API_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_api_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_GRAPH_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_graph_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_KERNEL_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_kernel_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] && [[ "$ENABLE_BUILT_CUSTOM" != "TRUE" ]]; then
    OP_HOST_UT=TRUE
    OP_API_UT=TRUE
    OP_GRAPH_UT=TRUE
    UT_TEST_ALL=FALSE
    OP_KERNEL_UT=FALSE
  fi
  if [[ "$OP_KERNEL_UT" == "TRUE" ]] && [[ -z "$ascend_op_name" ]]; then
    echo "Warning: Please use --ops when running Opkernel UT."
    ascend_op_name="recurrent_gated_delta_rule,chunk_gated_delta_rule"
  fi
}
parse_changed_files() {
    if [[ -z "$PR_CHANGED_FILES" ]]; then
        return
    fi

    if [[ "$PR_CHANGED_FILES" != /* ]]; then
        PR_CHANGED_FILES=$PWD/$PR_CHANGED_FILES
    fi

    echo "changed files is" $PR_CHANGED_FILES
    echo $dotted_line
    cat $PR_CHANGED_FILES
    ops_names=$(python3 scripts/ci/parse_changed_ops.py $PR_CHANGED_FILES "$ENABLE_EXPERIMENTAL" "$ASCEND_SOC_UNITS")
}
set_example_opt() {
  if [[ -n $1 && $1 != -* ]]; then
    EXAMPLE_NAME=$1
    step=$((step + 1))
  fi
  if [[ -n $2 && $2 != -* ]]; then
    EXAMPLE_MODE=$2
    step=$((step + 1))
  fi
  if [[ -n $3 && $3 != -* ]]; then
    PKG_MODE=$3
    step=$((step + 1))
  fi
}

# 所有支持的长选项（不含值），用于参数合法性校验
SUPPORTED_LONG_OPTS=(
  "help" "list_soc" "pkg" "pkg-type=" "static" "jit" "noaicpu" "opkernel_aicpu" "opkernel"
  "experimental" "noexec" "torch_extension" "torch_extension_only" "oom" "make_clean"
  "mssanitizer" "dump_cce"
  "ophost" "opapi" "opgraph" "onnxplugin"
  "ophost_test" "opapi_test" "opgraph_test" "opkernel_test"
  "asan" "valgrind" "ubsan" "cov" "clang" "run_example"
  "parent_job" "enable_host_tiling" "disable-check-compatible" "disable-check-compatiable"
  "ops=" "module=" "aicpu=" "bisheng_flags=" "soc=" "build-type=" "version="
  "rule_launch=" "simulator=" "example_name=" "tiling_key=" "tiling-key="
  "kernel_template_input=" "vendor_name=" "genop=" "genop_aicpu="
  "cann_3rd_lib_path=" "op_build_tool" "ascend_cmake_dir"
  "ccache" "PR_UT" "PR_PKG" "op_debug_config" "ops-compile-options"
  "op-name" "compute-unit" "package-path" "build" "changed_list"
  "test" "example" "verbose" "incremental"
)

check_option_validity() {
  for arg in "$@"; do
    if [[ "$arg" =~ ^- ]]; then
      if [[ "$arg" =~ ^-- ]]; then
        local long_opt="${arg:2}"
        local opt_name="${long_opt%%=*}"
        local found=false
        for supported_opt in "${SUPPORTED_LONG_OPTS[@]}"; do
          if [[ "$supported_opt" =~ =$ ]]; then
            local base_opt="${supported_opt%=}"
            if [[ "$opt_name" == "$base_opt" ]]; then
              found=true
              break
            fi
          else
            if [[ "$opt_name" == "$supported_opt" ]]; then
              found=true
              break
            fi
          fi
        done
        if [[ "$found" == "false" ]]; then
          echo "[ERROR] Invalid option: --$opt_name"
          echo "Use 'bash build.sh --help' for more information."
          exit 1
        fi
      fi
    fi
  done
}

check_param() {
  if [[ "$ENABLE_MSSANITIZER" == "TRUE" && "$OOM" == "true" ]]; then
    echo "[ERROR] --mssanitizer cannot be used with --oom"
    exit 1
  fi

  if [[ "$ENABLE_MSSANITIZER" == "TRUE" && "$ENABLE_DUMP_CCE" == "TRUE" ]]; then
    echo "[ERROR] --mssanitizer cannot be used with --dump_cce"
    exit 1
  fi

  if [ -n "$BISHENG_FLAGS" ]; then
    if [[ "$ENABLE_MSSANITIZER" == "TRUE" || "$OOM" == "true" || "$ENABLE_DUMP_CCE" == "TRUE" ]]; then
      echo "[ERROR] --bisheng_flags= cannot be used with --mssanitizer, --oom, --dump_cce"
      exit 1
    fi
  fi

  if [[ "$BUILD_TYPE" == "Debug" ]]; then
    if [[ "$ENABLE_MSSANITIZER" == "TRUE" || "$OOM" == "true" || "$ENABLE_DUMP_CCE" == "TRUE" ]]; then
      echo "[ERROR] --build-type=Debug cannot be used with --mssanitizer, --oom, --dump_cce"
      exit 1
    fi
  fi

  if [ -n "$KERNEL_TEMPLATE_INPUT" ]; then
    if [[ -z "${ascend_op_name}" || "$ascend_op_name" == *","* ]]; then
      echo "[ERROR] --kernel_template_input must be used with --ops= and can only specify a single operator"
      exit 1
    fi
  fi
}

assemble_cmake_args() {
  if [ -n "${vendor_name}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DVENDOR_NAME=${vendor_name}"
  fi

  if [ -n "${VERSION}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DVERSION=${VERSION}"
  fi

  if [[ "$ENABLE_EXPERIMENTAL" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_EXPERIMENTAL=TRUE"
  fi

  if [ -n "${ascend_compute_unit}" ];then
    # 校验 -c/--compute-unit 输入的 SoC 名称, 任一非法立即报错退出
    if ! validate_soc_list "${ascend_compute_unit}"; then
      if [[ -z "${INVALID_SOC_ITEM}" ]]; then
        echo "[ERROR] The input soc is empty or contains empty items. Please use one of: $(get_supported_soc_text)"
      else
        echo "[ERROR] The input soc ${INVALID_SOC_ITEM} is not supported. Please use one of: $(get_supported_soc_text)"
      fi
      exit 1
    fi
    # 归一化为小写, 与 CMakeLists.txt 中 SOC_VERSION_LIST 的大小写敏感精确匹配保持一致
    ascend_compute_unit="${ascend_compute_unit,,}"
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_COMPUTE_UNIT=${ascend_compute_unit}"
  fi

  if [ -n "${ascend_op_name}" ];then
    if [[ "${ascend_op_name}" == *"moe_distribute_combine_v2"* ]] && [[ "${ascend_op_name}" != *"moe_distribute_combine_v3"* ]]; then
      ascend_op_name="${ascend_op_name};moe_distribute_combine_v3"
    fi
    if [[ "${ascend_op_name}" == *"moe_distribute_dispatch_v2"* ]] && [[ "${ascend_op_name}" != *"moe_distribute_dispatch_v3"* ]]; then
      ascend_op_name="${ascend_op_name};moe_distribute_dispatch_v3"
    fi
    if [[ "${ascend_op_name}" == *"distribute_barrier"* ]] && [[ "${ascend_op_name}" != *"distribute_barrier_extend"* ]]; then
      ascend_op_name="${ascend_op_name};distribute_barrier_extend"
    fi
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_OP_NAME=${ascend_op_name}"
    if [[ "${ascend_op_name}" != *"fused_infer_attention_score"* ]] && [[ "${ascend_op_name}" != *"incre_flash_attention"* ]]; then
      CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_TILING_SINK=OFF"
    fi
  fi

  if [ -n "${ascend_module_name}" ];then
    if [[ "$ascend_module_name" == *mc2* ]]; then
      if [[ "$ascend_module_name" != *gmm* ]]; then
        ascend_module_name="${ascend_module_name},gmm"
      fi
    fi
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_MODULE_NAME=${ascend_module_name}"
    if [[ "${ascend_module_name}" != *"attention"* ]]; then
      CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_TILING_SINK=OFF"
    fi
  fi

  if [ -n "${op_build_tool}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_BUILD_TOOL=${op_build_tool}"
  fi
  if [ -n "${ascend_cmake_dir}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_CMAKE_DIR=${ascend_cmake_dir}"
  fi
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    if [ -z "$ascend_op_name" ]; then
      TEST="all"
    else
      TEST="$ascend_op_name"
    fi
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_TEST=TRUE"
  fi
  if [[ $UT_TEST_CNT -eq 0 ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DUT_INFERSHAPE_FLAG=TRUE"
  fi
  if [[ "$OP_HOST_UT" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_HOST_UT=TRUE"
  fi
  if [[ "$OP_API_UT" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_API_UT=TRUE"
  fi
  if [[ "$OP_GRAPH_UT" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_GRAPH_UT=TRUE"
  fi
  if [[ "$OP_KERNEL_UT" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_KERNEL_UT=TRUE"
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DUT_TEST_ALL=TRUE"
  fi
  if [[ "$ENABLE_UT_EXEC" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_UT_EXEC=TRUE"
  fi
  if [[ "x$ENABLE_RULE_LAUNCH" != "x" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DRULE_LAUNCH=${ENABLE_RULE_LAUNCH}"
  fi

  if [ -n "${TEST}" ];then
    if [ -n "${PR_CHANGED_FILES}" ];then
      TEST=$(python3 "$CURRENT_DIR"/cmake/scripts/parse_changed_files.py -c "$CURRENT_DIR"/tests/test_config.yaml -f "$PR_CHANGED_FILES" get_related_ut)
      echo "Operators that need to run UT: $TEST"
      if [ -z "${TEST}" ];then
        log "Info: This PR didn't trigger any UTest."
        exit 0
      fi
      if [ "$TEST" != "all" ];then
        TEST="${TEST%;}"
        TEST="${TEST//;/,}"
        CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_OP_NAME=${TEST}"
      fi
      CUSTOM_OPTION="${CUSTOM_OPTION} -DTESTS_UT_OPS_TEST_CI_PR=ON"
    fi
    CUSTOM_OPTION="${CUSTOM_OPTION} -DTESTS_UT_OPS_TEST=${TEST}"
    if [ "${CLANG}" == "true" ];then
      CLANG_C_COMPILER="$(which clang)"
      if [ ! -f "${CLANG_C_COMPILER}" ];then
        log "Error: Can't find clang path ${CLANG_C_COMPILER}"
        exit 1
      fi

      CLANG_PATH=$(dirname "${CLANG_C_COMPILER}")
      CLANG_CXX_COMPILER="${CLANG_PATH}/clang++"
      CLANG_LINKER="${CLANG_PATH}/lld"
      CLANG_AR="${CLANG_PATH}/llvm-ar"
      CLANG_STRIP="${CLANG_PATH}/llvm-strip"
      CLANG_OBJCOPY="${CLANG_PATH}/llvm-objcopy"

      CUSTOM_OPTION="${CUSTOM_OPTION} -DCMAKE_C_COMPILER=${CLANG_C_COMPILER} -DCMAKE_CXX_COMPILER=${CLANG_CXX_COMPILER}"
      CUSTOM_OPTION="${CUSTOM_OPTION} -DCMAKE_LINKER=${CLANG_LINKER} -DCMAKE_AR=${CLANG_AR} -DCMAKE_STRIP=${CLANG_STRIP}"
      CUSTOM_OPTION="${CUSTOM_OPTION} -DCMAKE_OBJCOPY=${CLANG_OBJCOPY}"
    fi

    if [ "${ASAN}" == "true" ];then
      CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_ASAN=TRUE"
    fi

    if [ "${ENABLE_VALGRIND}" == "TRUE" ];then
      CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_VALGRIND=TRUE"
    fi

    if [ "${UBSAN}" == "true" ];then
      CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_UBSAN=true"
    fi

    BUILD=ops_test_utest
  fi

  if [ "${ASAN}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_ASAN=TRUE"
  fi

  if [ "${COV}" == "true" ];then
    if [ "${CLANG}" == "true" ];then
      log "Warning: GCOV only supported in gnu compiler."
    else
      CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_GCOV=true"
    fi
  fi

  if [ "${OOM}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_OOM=ON"
  fi

  if [[ "$ENABLE_MSSANITIZER" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_MSSANITIZER=TRUE"
  fi

  if [[ "$ENABLE_DUMP_CCE" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_DUMP_CCE=TRUE"
  fi

  if [ -n "${EXAMPLE}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DTESTS_EXAMPLE_OPS_TEST=${EXAMPLE}"
    BUILD=ops_test_example
  fi

  if [ -n "${TILING_KEY}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DTILING_KEY=${TILING_KEY}"
  fi

  if [ -n "${KERNEL_TEMPLATE_INPUT}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DKERNEL_TEMPLATE_INPUT=${KERNEL_TEMPLATE_INPUT}"
  fi

  if [ -n "${OP_DEBUG_CONFIG}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_DEBUG_CONFIG=${OP_DEBUG_CONFIG}"
  fi

  if [ -n "${OPS_COMPILE_OPTIONS}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOPS_COMPILE_OPTIONS=${OPS_COMPILE_OPTIONS}"
  fi

  if [ "${HOST_TILING}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_HOST_TILING=ON"
  fi

  if [ "${ENABLE_DEBUG}" == "TRUE" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_DEBUG=ON"
  fi

  if [ -n "${BUILD_TYPE}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  fi

  if [ -n "${VERSION}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DVERSION=${VERSION}"
  fi

  if [ -n "${CMAKE_BUILD_MODE}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DCMAKE_BUILD_MODE=${CMAKE_BUILD_MODE}"
  fi
  CUSTOM_OPTION="${CUSTOM_OPTION} -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"

  if [ -n "${BISHENG_FLAGS}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DBISHENG_FLAGS=${BISHENG_FLAGS}"
  fi

  if [ -n "${ENABLE_AICPU_KERNEL}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_AICPU_KERNEL=${ENABLE_AICPU_KERNEL}"
  fi

  if [[ "$ENABLE_STATIC" == "TRUE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_STATIC=${ENABLE_STATIC}"
  fi

  if [[ "$ENABLE_BUILD_PKG" == "TRUE" ]]; then
    cmake_pkg_type="${PACKAGE_TYPE}"
    [[ "${PACKAGE_TYPE}" == "all" ]] && cmake_pkg_type="run"
    CUSTOM_OPTION="${CUSTOM_OPTION} -DPACKAGE_TYPE=${cmake_pkg_type}"
  fi

  if [[ "$ENABLE_AICPU" == "FALSE" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_AICPU=OFF -DENABLE_TILING_SINK=OFF "
  fi

  if [ -n "${ascend_package_path}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ascend_package_path}
  elif [ -n "${ASCEND_HOME_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
  elif [ -n "${ASCEND_OPP_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=$(dirname ${ASCEND_OPP_PATH})
  elif [ -d "${DEFAULT_TOOLKIT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_TOOLKIT_INSTALL_DIR}
  elif [ -d "${DEFAULT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_INSTALL_DIR}
  else
    log "Error: Please set the toolkit package installation directory through parameter -p|--package-path."
    exit 1
  fi

  CUSTOM_OPTION="${CUSTOM_OPTION} -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH} -DCHECK_COMPATIBLE=${CHECK_COMPATIBLE}"
}

########################################################################################################################
# 参数解析处理
########################################################################################################################
set -o pipefail
{
if [[ $# -eq 0 ]]; then
    help_info
    exit 1
fi
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        SHOW_HELP="general"
        # 检查 --help 前面的命令
        for prev_arg in "$@"; do
            case "$prev_arg" in
            --pkg) SHOW_HELP="package" ;;
            --opkernel) SHOW_HELP="opkernel" ;;
            --opkernel_aicpu) SHOW_HELP="opkernel_aicpu" ;;
            -u|--test) SHOW_HELP="test" ;;
            --make_clean) SHOW_HELP="clean" ;;
            --valgrind) SHOW_HELP="valgrind" ;;
            --ophost) SHOW_HELP="ophost" ;;
            --opapi) SHOW_HELP="opapi" ;;
            --opgraph) SHOW_HELP="opgraph" ;;
            --onnxplugin) SHOW_HELP="onnxplugin" ;;
            --ophost_test) SHOW_HELP="ophost_test" ;;
            --opapi_test) SHOW_HELP="opapi_test" ;;
            --opgraph_test) SHOW_HELP="opgraph_test" ;;
            --run_example) SHOW_HELP="run_example" ;;
            --genop) SHOW_HELP="genop" ;;
            --genop_aicpu) SHOW_HELP="genop_aicpu" ;;
            esac
        done
      help_info "$SHOW_HELP"
      exit 0
    fi
  done

check_option_validity "$@"

while [[ $# -gt 0 ]]; do
    case $1 in
    -h|--help)
        help_info
        exit
        ;;
    --list_soc)
        # 输出当前支持的 SoC 列表(来自单一数据源 SUPPORTED_SOC_LIST), 供用户选择正确的芯片型号
        echo "Supported SoC list: $(get_supported_soc_text)"
        exit 0
        ;;
    --pkg)
        ENABLE_BUILD_PKG=TRUE
        ENABLE_BUILT_IN=TRUE            # 只输入--pkg时编builtin包
        shift
        ;;
    --static)
        ENABLE_STATIC=TRUE
        shift
        ;;
    --jit)
        ENABLE_BUILT_JIT=TRUE
        shift
        BUILD="jit"
        ;;
    --noaicpu)
        ENABLE_AICPU=FALSE
        shift
        ;;
    -n|--op-name)
        ascend_op_name="$2"
        shift 2
        ;;
    --ops=*)
        OPTARG=$1
        ascend_op_name=${OPTARG#*=}
        ENABLE_BUILT_CUSTOM=TRUE
        ENABLE_BUILT_IN=FALSE
        shift
        ;;
    --module=*)
        OPTARG=$1
        ascend_module_name=${OPTARG#*=}
        shift
        ;;
    --aicpu=*)
        OPTARG=$1
        shift
        ;;
    --bisheng_flags=*)
        OPTARG=$1
        BISHENG_FLAGS=${OPTARG#*=}
        shift
        ;;
    -c|--compute-unit)
        ascend_compute_unit="$2"
        shift 2
        ;;
    --soc=*)
        process_soc_input "$1"
        shift 1
    ;;
    --ccache)
        CCACHE_PROGRAM="$2"
        shift 2
        ;;
    -p|--package-path)
        ascend_package_path="$2"
        shift 2
        ;;
    -b|--build)
        BUILD="$2"
        shift 2
        ;;
    -u|--test)
        ENABLE_TEST=TRUE
        shift
        ;;
    --rule_launch=*)
        OPTARG=$1
        ENABLE_RULE_LAUNCH=${OPTARG#*=}
        shift
        ;;
    --simulator=*)
        OPTARG=$1
        SIMULATOR=${OPTARG#*=}
        shift
        ;;
    --run_example)
        ENABLE_RUN_EXAMPLE=TRUE
        step=1
        set_example_opt $2 $3 $4
        shift $step
        ;;
    --example_name=*)
        OPTARG=$1
        SPECIFIC_EXAMPLE_NAME=${OPTARG#*=}
        shift
        ;;
    --experimental)
        ENABLE_EXPERIMENTAL=TRUE
        shift
        ;;
    -e|--example)
        shift
        if [ -n "$1" ];then
            _parameter=$1
            first_char=${_parameter:0:1}
            if [ "${first_char}" == "-" ];then
                EXAMPLE="all"
            else
                EXAMPLE="${_parameter}"
                shift
            fi
        else
            EXAMPLE="all"
        fi
        ;;
    -f|--changed_list)
        PR_CHANGED_FILES="$2"
        ENABLE_SMOKE=TRUE
        PKG_MODE="cust"
        vendor_name="custom"
        CI_MODE=TRUE
        shift 2
        ;;
    --PR_UT)
        PR_CHANGED_FILES="$2"
        ENABLE_TEST=TRUE
        PR_UT_FLAG=TRUE
        TEST_MC2=$(python3 "$CURRENT_DIR"/cmake/scripts/parse_changed_files.py -c "$CURRENT_DIR"/tests/test_config.yaml -f "$PR_CHANGED_FILES" get_related_ut_mc2)
        TEST_EXCLUDE_MC2=$(python3 "$CURRENT_DIR"/cmake/scripts/parse_changed_files.py -c "$CURRENT_DIR"/tests/test_config.yaml -f "$PR_CHANGED_FILES" get_related_ut_exclude_mc2)
        ut_soc_version=$(python3 "$CURRENT_DIR"/cmake/scripts/get_soc_version.py -c "$CURRENT_DIR"/tests/test_soc_config.yaml -f "$PR_CHANGED_FILES" get_related_soc)
        ut_soc_version="ascend${ut_soc_version#*ascend}"
        IFS=',' read -ra UT_SOC_ARRAY <<< "$ut_soc_version"
        echo "UT_SOC_ARRAY = ${UT_SOC_ARRAY[@]}"

        CI_MODE=TRUE
        shift 2
        ;;
    --PR_PKG)
        PR_CHANGED_FILES="$2"
        if [[ "$ENABLE_EXPERIMENTAL" == "TRUE" ]]; then
            parse_changed_files
        else
            ops_names=$(python3 "$CURRENT_DIR"/cmake/scripts/parse_changed_files.py -c "$CURRENT_DIR"/tests/test_config.yaml -f "$PR_CHANGED_FILES" get_related_examples)
        fi
        echo "Operators that need custom package compilation:$ops_names"
        if [ -z "${ops_names}" ]; then
            log "Info: No custom packages to build for this PR."
            # ops_names="incre_flash_attention"
            exit 200
        fi
        ops_names="${ops_names%;}"
        ops_names="${ops_names//;/,}"
        ascend_op_name="$ops_names"
        ENABLE_BUILD_PKG=TRUE
        ENABLE_BUILT_CUSTOM=TRUE
        ENABLE_BUILT_IN=FALSE
        CI_MODE=TRUE
        shift 2
        ;;
    --parent_job)
        PARENT_JOB="true"
        shift
        ;;
    --enable_host_tiling)
        HOST_TILING="true"
        shift
        ;;
    --disable-check-compatible|--disable-check-compatiable)
        CHECK_COMPATIBLE="false"
        shift
        ;;
    --op_build_tool)
        op_build_tool="$2"
        shift 2
        ;;
    --ascend_cmake_dir)
        ascend_cmake_dir="$2"
        shift 2
        ;;
    -v|--verbose)
        VERBOSE="true"
        shift
        ;;
    --asan)
        ASAN="true"
        shift
        ;;
    --valgrind)
        ENABLE_VALGRIND=TRUE
        ENABLE_UT_EXEC=FALSE
        ASAN="false"
        shift
        ;;
    --ubsan)
        UBSAN="true"
        shift
        ;;
    --cov)
        COV="true"
        shift
        ;;
    --clang)
        CLANG="true"
        shift
        ;;
    --tiling-key|--tiling_key)
        TILING_KEY="$2"
        shift 2
        ;;
    --tiling_key=*)
        OPTARG=$1
        TILING_KEY=${OPTARG#*=}
        shift
        ;;
    --kernel_template_input=*)
        OPTARG=$1
        KERNEL_TEMPLATE_INPUT=${OPTARG#*=}
        shift
        ;;
    --op_debug_config)
        OP_DEBUG_CONFIG="$2"
        shift 2
        ;;
    --ops-compile-options)
        OPS_COMPILE_OPTIONS="$2"
        shift 2
        ;;
    --ophost_test)
        ENABLE_TEST=TRUE
        OP_HOST=TRUE
        shift
        ;;
    --opapi_test)
        ENABLE_TEST=TRUE
        OP_API=TRUE
        shift
        ;;
    --opgraph_test)
        ENABLE_TEST=TRUE
        OP_GRAPH=TRUE
        shift
        ;;
    --opkernel_test)
        ENABLE_TEST=TRUE
        OP_KERNEL=TRUE
        shift
        ;;
    --vendor_name=*)
        OPTARG=$1
        vendor_name=${OPTARG#*=}
        ENABLE_BUILT_CUSTOM=TRUE
        ENABLE_BUILT_IN=FALSE
        shift
        ;;
    --opgraph)
        BUILD_LIBS+=("opgraph_transformer")
        ENABLE_CREATE_LIB=TRUE
        OP_GRAPH=TRUE
        shift
        ;;
    --onnxplugin)
        BUILD_LIBS+=("oponnx_plugin_transformer")
        ENABLE_CREATE_LIB=TRUE
        ONNX_PLUGIN=TRUE
        shift
        ;;
    --opapi)
        BUILD_LIBS+=("opapi_transformer")
        ENABLE_CREATE_LIB=TRUE
        OP_API=TRUE
        shift
        ;;
    --ophost)
        BUILD_LIBS+=("ophost_transformer")
        ENABLE_CREATE_LIB=TRUE
        OP_HOST=TRUE
        shift
        ;;
    --opkernel)
        ENABLE_OPKERNEL=TRUE
        OP_KERNEL=TRUE
        shift
        ;;
    --noexec)
        ENABLE_UT_EXEC=FALSE
        shift
        ;;
    --torch_extension)
        ENABLE_TORCH_EXTENSION_ONLY=TRUE
        shift
        ;;
    --torch_extension_only)
        ENABLE_TORCH_EXTENSION_ONLY=TRUE
        shift
        ;;
    -j*)
        OPTARG=$1
        if [[ "$OPTARG" =~ ^-j[0-9]+$ ]]; then
            INPUT_JOB_NUM="${OPTARG#*-j}"
            # 可选：添加范围检查
            if [ "$INPUT_JOB_NUM" -le 0 ]; then
                echo "Error: Job number must be positive: $OPTARG" >&2
                exit 1
            fi
        else
            echo "Error: Invalid job number format: $OPTARG" >&2
            echo "Expected: -j[n]" >&2
            exit 1
        fi
        shift 1
        ;;
    --build-type=*)
        OPTARG=$1
        BUILD_TYPE=${OPTARG#*=}
        shift
        ;;
    --pkg-type)
        echo "[ERROR] --pkg-type requires a value: run/rpm/deb/all"
        exit 1
        ;;
    --pkg-type=*)
        OPTARG=$1
        PACKAGE_TYPE=${OPTARG#*=}
        check_pkg_type "${PACKAGE_TYPE}"
        PACKAGE_TYPE_SET=TRUE
        shift
        ;;
    --version=*)
        OPTARG=$1
        VERSION=${OPTARG#*=}
        shift
        ;;
    -O[0-3])
        CMAKE_BUILD_MODE=$1
        shift
        ;;
    --genop=*)
        OPTARG=$1
        process_genop "genop" "${OPTARG#*=}"
        shift
        ;;
    --genop_aicpu=*)
        OPTARG=$1
        process_genop "genop_aicpu" "${OPTARG#*=}"
        shift
        ;;
    --make_clean)
        clean
        clean_build_out
        clean_third_party
        shift
        ;;
    --incremental)
        INCREMENTAL_BUILD=TRUE
        shift
        ;;
    --cann_3rd_lib_path=*)
        OPTARG=$1
        CANN_3RD_LIB_PATH="$(realpath ${OPTARG#*=})"
        shift
        ;;
    --mssanitizer)
        ENABLE_MSSANITIZER=TRUE
        shift
        ;;
    --dump_cce)
        ENABLE_DUMP_CCE=TRUE
        shift
        ;;
    --opkernel_aicpu)
        ENABLE_AICPU_KERNEL=TRUE
        shift
        ;;
    --oom)
        OOM="true"
        shift
        ;;
    *)
        help_info
        exit 1
        ;;
    esac
done
set_ut_mode

check_param

if [[ "$PACKAGE_TYPE_SET" == "TRUE" && "$ENABLE_BUILD_PKG" != "TRUE" ]]; then
    echo "[ERROR] --pkg-type can only be used with --pkg"
    exit 1
fi

if [[ "$PACKAGE_TYPE" != "run" && "$PACKAGE_TYPE" != "all" ]]; then
    if [[ "$ENABLE_STATIC" == "TRUE" ]]; then
        echo "[ERROR] --pkg-type=${PACKAGE_TYPE} cannot be used with --static"
        exit 1
    fi
    if [[ "$ENABLE_BUILT_JIT" == "TRUE" ]]; then
        echo "[ERROR] --pkg-type=${PACKAGE_TYPE} cannot be used with --jit"
        exit 1
    fi
    if [[ "$ENABLE_BUILT_CUSTOM" == "TRUE" || "$ENABLE_EXPERIMENTAL" == "TRUE" ]]; then
        echo "[ERROR] --pkg-type=${PACKAGE_TYPE} only supports built-in ops-transformer packages; do not use --ops, --vendor_name, or --experimental"
        exit 1
    fi
fi

assemble_cmake_args

function get_cpu_num() {
    CPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)*2))
    if [ -n "${OPS_CPU_NUMBER}" ]; then
        if [[ "${OPS_CPU_NUMBER}" =~ ^[0-9]+$ ]]; then
            CPU_NUM="${OPS_CPU_NUMBER}"
        fi
    fi
}

if [ "${PARENT_JOB}" == "false" ]; then
    get_cpu_num
    JOB_NUM="-j${CPU_NUM}"
fi

if [ -n "${INPUT_JOB_NUM}" ]; then
    get_cpu_num
    JOB_NUM="-j${INPUT_JOB_NUM}"
fi

# 非打包命令调用，打包模式会打进同一个包里
function set_compute_unit_option() {
    # 使用共享校验库逐项校验 SoC 列表(支持逗号/分号分隔的多个 soc), 替换原有的整串精确匹配
    if ! validate_soc_list "${ASCEND_SOC_UNITS}"; then
        echo "[ERROR] The input soc ${INVALID_SOC_ITEM} is not supported. Please use one of: $(get_supported_soc_text)"
        exit 1
    fi
    CUSTOM_OPTION="$CUSTOM_OPTION -DASCEND_COMPUTE_UNIT=$ASCEND_SOC_UNITS"
}

# 上面的set_compute_unit_option修改成只能传入一个soc后导致ut有问题，现在复制一份旧的支持多个soc传入，用于ut
function set_compute_unit_option_ut() {
    IFS=';' read -ra SOC_ARRAY <<< "$ASCEND_SOC_UNITS"  # 分割字符串为数组
    local COMPUTE_UNIT_SHORT=""
    local soc=""
    local trimmed_soc=""
    for soc in "${SOC_ARRAY[@]}"; do
        # 裁剪前后空白后, 通过共享校验库逐项精确匹配(替换原有错误的子串匹配逻辑)
        trimmed_soc="${soc#"${soc%%[![:space:]]*}"}"
        trimmed_soc="${trimmed_soc%"${trimmed_soc##*[![:space:]]}"}"
        if soc_is_supported "${trimmed_soc}"; then
            COMPUTE_UNIT_SHORT="${COMPUTE_UNIT_SHORT}${trimmed_soc,,};"
        fi
    done
    if [[ -z "${COMPUTE_UNIT_SHORT}" ]]; then
        echo "[ERROR] The input soc ${ASCEND_SOC_UNITS} is not supported. Please use one of: $(get_supported_soc_text)"
        exit 1
    fi
    CUSTOM_OPTION="$CUSTOM_OPTION -DASCEND_COMPUTE_UNIT=$COMPUTE_UNIT_SHORT"
 }

########################################################################################################################
# 处理流程
########################################################################################################################

set_env

if [[ "${INCREMENTAL_BUILD}" == "TRUE" ]]; then
    log "Info: incremental build enabled, keep existing build/output directories."
    ensure_build_dirs
else
    clean
fi

if [ -n "${CCACHE_PROGRAM}" ]; then
    if [ "${CCACHE_PROGRAM}" == "false" ] || [ "${CCACHE_PROGRAM}" == "off" ]; then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_CCACHE=OFF"
    elif [ -f "${CCACHE_PROGRAM}" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_CCACHE=ON -DCUSTOM_CCACHE=${CCACHE_PROGRAM}"
        export ASCENDC_CCACHE_EXECUTABLE=${CCACHE_PROGRAM}
    fi
else
    # 判断有无默认的ccache 如果有则使用
    ccache_system=$(which ccache || true)
    if [ -n "${ccache_system}" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_CCACHE=ON -DCUSTOM_CCACHE=${ccache_system}"
        export ASCENDC_CCACHE_EXECUTABLE=${ccache_system}
    fi
fi
build_ut() {
  CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l)
  dotted_line="----------------------------------------------------------------"
  echo $dotted_line
  echo "Start to build ut"

  if [ ! -d "${BUILD_DIR}" ]; then
    mkdir -p "${BUILD_DIR}"
  fi
  cd "${BUILD_DIR}" && cmake ${CUSTOM_OPTION} ..

  local target="$1"
  if [ "${VERBOSE}" == "true" ]; then
    local option="--verbose"
  fi
  if [ $(cmake -LA -N . | grep 'UTEST_FRAMEWORK_OLD:BOOL=' | cut -d'=' -f2) == "TRUE" ]; then
    cmake --build . --target ${target} ${JOB_NUM} ${option}
  fi

  if [ $(cmake -LA -N . | grep 'UTEST_FRAMEWORK_NEW:BOOL=' | cut -d'=' -f2) == "TRUE" ]; then
    local has_valid_target="FALSE"
    for UT_TARGET in ${UT_TARGETS[@]} ; do
        if cmake --build . --target help | grep -w "$UT_TARGET"; then
            echo "Building target: $UT_TARGET."
            if ! cmake --build . --target ${UT_TARGET} -j $CORE_NUMS; then
                echo "[ERROR] Build failed for target: $UT_TARGET."
                exit 1
            fi
            has_valid_target="TRUE"
        else
            echo "Target $UT_TARGET not found, skipping build."
        fi
    done
    if [[ "$COV" == "true" && "$ENABLE_UT_EXEC" == "TRUE" && "$has_valid_target" == "TRUE" ]]; then
        python3 ${BASE_PATH}/cmake/scripts/utest/gen_coverage.py \
            -s=${BASE_PATH} \
            -c=${BUILD_PATH} \
            -f="/tmp/*" \
            -f="/usr/include/*" \
            -f="$(realpath $ASCEND_HOME_PATH/../)/*" \
            -y=${BASE_PATH}/tests/test_config.yaml
        local gen_coverage_result=$?
        if [ $gen_coverage_result -ne 0 ]; then
            echo "Error: Gen coverage failed with exit code: $gen_coverage_result"
            exit $gen_coverage_result
        fi
    fi
  fi
}

function build_pkg_for_single_soc() {
    local single_soc_option="$1"
    local original_option="${CUSTOM_OPTION}"
    if [[ "$ENABLE_BUILT_JIT" == "TRUE" ]]; then
        CUSTOM_OPTION="${CUSTOM_OPTION}  -DENABLE_BUILT_IN=ON -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=OFF"
        cmake_config ${single_soc_option}
        build_package
        CUSTOM_OPTION="${original_option}"
    elif [[ "$ENABLE_BUILT_IN" == "TRUE" ]]; then
        CUSTOM_OPTION="${CUSTOM_OPTION}  -DENABLE_BUILT_IN=ON -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON"
        cmake_config ${single_soc_option}
        build_package
        CUSTOM_OPTION="${original_option}"
    fi
}

function build_example_for_ci()
{
    EXAMPLE_NAME="$1"
    PKG_MODE="cust"

    EXAMPLE_MODE="eager"
    local eager_result=0
    build_example || eager_result=$? # 避免函数随build_example一起退出
    if [ $eager_result -ne 0 ] && [ $eager_result -ne 2 ]; then
        echo "Error: Eager Example failed with exit code: $eager_result"
        exit $eager_result
    fi

    EXAMPLE_MODE="graph"
    local geir_result=0
    build_example || geir_result=$? # 避免函数随build_example一起退出
    if [ $geir_result -ne 0 ] && [ $geir_result -ne 2 ]; then
        echo "Error: Graph Example failed with exit code: $geir_result"
        exit $geir_result
    fi

    if [ $eager_result -eq 2 ] && [ $geir_result -eq 2 ]; then
        echo "Error: Neither eager nor graph examples provided for $EXAMPLE_NAME"
        exit $geir_result
    fi
    return 0
}

# 冒烟任务只跑examples
function process_ci_smoke_with_changed_list()
{
    TEST=$(python3 "$CURRENT_DIR"/cmake/scripts/parse_changed_files.py -c "$CURRENT_DIR"/tests/test_config.yaml -f "$PR_CHANGED_FILES" get_related_examples)
    echo "Operators that need to run examples: $TEST"
    if [[ -z "$TEST" ]];then
        echo "No related unit tests found. Skipping CI test execution."
        exit 0
    fi
    IFS=';' read -ra OPS_ARRAY <<< "$TEST"
    for op in "${OPS_ARRAY[@]}";do
        op=$(echo "$op" | xargs)
        echo "Running example test for operator: $op"
        if [[ -n "$op" ]];then
            build_example_for_ci "$op"
        fi
    done
}

# 参数解析完成后, 若用户未显式指定 --soc, 则通过 npu-smi info 自动探测 NPU 芯片型号
detect_soc_from_npu_smi

if [[ "$ENABLE_SMOKE" == "TRUE" ]]; then
    process_ci_smoke_with_changed_list
fi

cd ${BUILD_DIR}

if [[ "$ENABLE_RUN_EXAMPLE" == "TRUE" ]];then
    build_example
    example_result=$?
    if [ $example_result -eq 2 ]; then
        echo "Error: ${EXAMPLE_NAME} do not have ${EXAMPLE_MODE} example"
    elif [ $example_result -ne 0 ]; then
        echo "Example failed with exit code: $example_result"
    else
        echo "Example completed successfully"
    fi
    exit $example_result
fi


function build_pr_ut_mc2()
{
    echo "Operators mc2 that need to run UT: $TEST_MC2"
    if [ -z "${TEST_MC2}" ];then
            log "Info: This PR didn't trigger any mc2 UTest."
            return
    fi
    if [ "$TEST_MC2" != "all" ];then
            TEST_MC2="${TEST_MC2%;}"
            TEST_MC2="${TEST_MC2//;/,}"
            CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_OP_NAME=${TEST_MC2}"
    fi
    CUSTOM_OPTION="${CUSTOM_OPTION} -DTESTS_UT_OPS_TEST_CI_PR=ON"
    CUSTOM_OPTION="${CUSTOM_OPTION} -DTESTS_UT_OPS_TEST=${TEST_MC2}"
    for element in "${UT_SOC_ARRAY[@]}"; do
            if [ $UT_TEST_CNT -eq 0 ]; then
                CUSTOM_OPTION="${CUSTOM_OPTION} -DUT_INFERSHAPE_FLAG=TRUE"
                CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_API_UT=TRUE"
            else
                CUSTOM_OPTION="${CUSTOM_OPTION} -DUT_INFERSHAPE_FLAG=FALSE"
                CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_API_UT=FALSE"
            fi
            echo "start to test $element"
            process_soc_input "$element"
            set_compute_unit_option_ut
            build_ut ${BUILD}
            UT_TEST_CNT=$((UT_TEST_CNT +1))
    done
}

function build_pr_ut_exclude_mc2()
{
    echo "Operators exclude mc2 that need to run UT: $TEST_EXCLUDE_MC2"
    if [ -z "${TEST_EXCLUDE_MC2}" ];then
            log "Info: This PR didn't trigger any exclude mc2 UTest."
            return
    fi
    if [ "$TEST_EXCLUDE_MC2" != "all" ];then
            TEST_EXCLUDE_MC2="${TEST_EXCLUDE_MC2%;}"
            TEST_EXCLUDE_MC2="${TEST_EXCLUDE_MC2//;/,}"
            CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_OP_NAME=${TEST_EXCLUDE_MC2}"
    fi
    CUSTOM_OPTION="${CUSTOM_OPTION} -DTESTS_UT_OPS_TEST_CI_PR=ON"
    CUSTOM_OPTION="${CUSTOM_OPTION} -DTESTS_UT_OPS_TEST=${TEST_EXCLUDE_MC2}"
    process_soc_input "ascend310p,ascend910b,ascend950,ascend350"
    set_compute_unit_option_ut
    build_ut ${BUILD}
}

main() {

if [[ "$ENABLE_GENOP" == "TRUE" ]]; then
    gen_op
fi
if [[ "$ENABLE_GENOP_AICPU" == "TRUE" ]]; then
    gen_aicpu_op
fi

if [[ "$ENABLE_TORCH_EXTENSION_ONLY" == "TRUE" ]]; then
    build_torch_extension_whl || exit 1
    mkdir -p "${BUILD_OUT_DIR}"
    cp "${CURRENT_DIR}/torch_extension/dist/"*.whl "${BUILD_OUT_DIR}/"
    log "[INFO] torch_extension whl package copied to ${BUILD_OUT_DIR}"
    exit 0
fi

if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    if [[ "$PR_UT_FLAG" == "TRUE" ]]; then
        build_pr_ut_exclude_mc2
        build_pr_ut_mc2
    else
        set_compute_unit_option_ut
        build_ut ${BUILD}
    fi
    exit 0
elif [[ "$ENABLE_CREATE_LIB" == "TRUE" ]]; then
    build_lib
elif [[ "$ENABLE_STATIC" == "TRUE" ]]; then
    IFS=';' read -ra SOC_ARRAY <<< "$ASCEND_SOC_UNITS"  # 分割字符串为数组
    for soc in "${SOC_ARRAY[@]}"; do
        soc=$(echo "${soc}" | xargs)  # 去除前后空格
        if [[ -n "${soc}" ]]; then  # 检查非空
            if [[ "$ENABLE_BUILT_JIT" == "FALSE" ]]; then
                cmake_config -DASCEND_COMPUTE_UNIT=${soc}
                build_kernel
            fi
            build_static_lib ${soc}
            if [[ "$ENABLE_BUILD_PKG" == "TRUE" ]]; then
                package_static ${soc}
            fi
        fi
        make clean
    done
elif [[ "$ENABLE_OPKERNEL" == "TRUE" ]]; then
    set_compute_unit_option
    cmake_config -DENABLE_HOST_TILING=ON
    build_kernel
elif [[ "$ENABLE_AICPU_KERNEL" == "TRUE" && "$ENABLE_BUILD_PKG" != "TRUE" && "$ENABLE_BUILT_CUSTOM" != "TRUE" ]]; then
    set_compute_unit_option
    cmake_config -DENABLE_HOST_TILING=ON -DENABLE_AICPU_KERNEL=TRUE
    build_kernel
elif [[ "$ENABLE_BUILT_CUSTOM" == "TRUE" ]]; then      # --ops, --vendor 新命令新使用
    set_compute_unit_option
    if [[ "$ENABLE_BUILT_JIT" == "TRUE" ]]; then
        ops_kernel_value="OFF"
    else
        ops_kernel_value="ON"
    fi
    CUSTOM_OPTION="${CUSTOM_OPTION}  -DENABLE_BUILT_IN=OFF -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=${ops_kernel_value}"
    if [[ "$ENABLE_BUILD_PKG" == "TRUE" ]]; then      # --pkg 新命令新使用
        cmake_config " -DENABLE_BUILD_PKG=ON"
    else
        cmake_config " -DENABLE_BUILD_PKG=OFF"
    fi
    build_package
elif [[ "$ENABLE_BUILD_PKG" == "TRUE" ]]; then      # --pkg 新命令新使用
    # 构建 torch_extension whl 包
    build_torch_extension_whl || exit 1
    IFS=';' read -ra SOC_ARRAY <<< "$ASCEND_SOC_UNITS"  # 分割字符串为数组
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_BUILD_PKG=ON"
    for soc in "${SOC_ARRAY[@]}"; do
        soc=$(echo "${soc}" | xargs)  # 去除前后空格
        soc_options=" -DASCEND_COMPUTE_UNIT=${soc}"
        if [[ -n "${soc}" ]]; then  # 检查非空
            build_pkg_for_single_soc ${soc_options} && make clean
        fi
    done
else
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_BUILD_PKG=ON"
    if [ "${BUILD}" == "host" ];then
        cmake_config -DENABLE_OPS_KERNEL=OFF -DENABLE_OPS_HOST=ON
        build_host
        # TO DO
        rm -rf ${CURRENT_DIR}/output
        mkdir -p ${CURRENT_DIR}/output
        cp ${BUILD_DIR}/*.run ${CURRENT_DIR}/output
    elif [ "${BUILD}" == "kernel" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_OPS_HOST=OFF -DENABLE_OPS_KERNEL=ON -DBUILD_OPS_RTY_KERNEL=ON"
        cmake_config
        build_kernel
    elif [ "${BUILD}" == "package" ];then
        # 构建 torch_extension whl 包
        build_torch_extension_whl || exit 1
        CUSTOM_OPTION="${CUSTOM_OPTION}  -DENABLE_BUILT_IN=ON -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON"
        build_package
    elif [ -n "${BUILD}" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION}  -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON"
        cmake_config
        build ${BUILD}
    fi
fi
}

main
} | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done
