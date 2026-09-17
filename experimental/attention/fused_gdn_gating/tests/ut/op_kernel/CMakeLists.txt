# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# NOTE: experimental 嵌套目录（experimental/attention/<op>）不在 AddOpTestCase 的
# KernelFile GLOB 与 gen_tiling_head_file.py 的算子搜索范围内，此处参照
# quant_block_sparse_attn 的做法手工构建 cases_obj（kernel源 + 测试用例），
# 并使用与 AddOpTestCase 等价的编译配置（tikicpulib/gtest/include路径/宏）。
# tiling 数据由测试文件内 InitTilingData 手动构造，不依赖生成的 tiling 头。

if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(opName "fused_gdn_gating")

    # kernel入口 + 测试用例（ICPU_RUN_KF 需要 kernel 符号，故同时编译）
    set(fused_gdn_gating_kernel_src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/${opName}.cpp
    )
    file(GLOB OPKERNEL_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_${opName}*.cpp)

    foreach(oriSocVersion "Ascend910B1" "Ascend310P1")
        string(REPLACE "ascend" "Ascend" socVersion "${oriSocVersion}")

        add_library(${opName}_${socVersion}_cases_obj OBJECT
            ${fused_gdn_gating_kernel_src} ${OPKERNEL_CASES_SRC})
        target_compile_options(
            ${opName}_${socVersion}_cases_obj PRIVATE
            -g -DUT_SOC_VERSION="${socVersion}"
            -include ${CMAKE_CURRENT_SOURCE_DIR}/${opName}_tiling.h
            )
        target_include_directories(
            ${opName}_${socVersion}_cases_obj
            PRIVATE ${ASCEND_DIR}/include/base/context_builder
                    ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/op_kernel
                    ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common
                    ${PROJECT_SOURCE_DIR}/common/include/op_kernel
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel
                    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/impl/basic_api
                    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc
                    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
                    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/basic_api
                    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/adv_api
                    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/ascendc/highlevel_api
            )
        target_link_libraries(
            ${opName}_${socVersion}_cases_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> tikicpulib::${socVersion}
                                                      gtest
            )

        # 挂到框架聚合目标：transformer_op_kernel_ut_<soc>_cases_obj
        if(NOT TARGET ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj)
            add_library(
                ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj OBJECT
                $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
                )
        else()
            target_sources(
                ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj PRIVATE
                $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
                )
        endif()
        add_dependencies(
            ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj ${opName}_${socVersion}_cases_obj
            )
        target_link_libraries(
            ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
                                                                    $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
            )

        list(FIND fastOpTestSocVersions "${socVersion}" index)
        if(index EQUAL -1)
            set(fastOpTestSocVersions
                ${fastOpTestSocVersions} ${socVersion}
                CACHE STRING "fastOp Test SocVersions" FORCE
                )
        endif()
    endforeach()
endif()
