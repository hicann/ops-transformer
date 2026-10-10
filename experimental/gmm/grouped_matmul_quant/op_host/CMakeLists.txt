# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        grouped_matmul_quant_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME GroupedMatmulQuant
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
)

# The new UT framework's repository-level discovery scans category/op/tests,
# while experimental operators live under experimental/category/op/tests.
# Add this operator's tests locally when op_host is selected by --ops.
if(ENABLE_TEST OR BENCHMARK)
    get_property(GROUPED_MATMUL_QUANT_TESTS_ADDED GLOBAL PROPERTY GROUPED_MATMUL_QUANT_TESTS_ADDED)
    if(NOT GROUPED_MATMUL_QUANT_TESTS_ADDED)
        set_property(GLOBAL PROPERTY GROUPED_MATMUL_QUANT_TESTS_ADDED TRUE)
        add_subdirectory(
            "${CMAKE_CURRENT_SOURCE_DIR}/../tests"
            "${CMAKE_CURRENT_BINARY_DIR}/../tests"
        )
    endif()
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE grouped_matmul_quant ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
