# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(UT_TEST_ALL OR OP_GRAPH_UT)
    add_modules_ut_sources(UT_NAME ${OP_GRAPH_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(QGMM_INPLACE_ADD_TRANSPOSE_PASS_CSV
        ${CMAKE_CURRENT_SOURCE_DIR}/test_quant_grouped_matmul_inplace_add_transpose_fusion_pass.csv)
    configure_file(
        ${QGMM_INPLACE_ADD_TRANSPOSE_PASS_CSV}
        ${CMAKE_BINARY_DIR}/tests/ut/op_graph/test_quant_grouped_matmul_inplace_add_transpose_fusion_pass.csv
        COPYONLY)
    if(TARGET ${OP_GRAPH_MODULE_NAME}_cases_obj)
        target_compile_definitions(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
            QGMM_INPLACE_ADD_TRANSPOSE_PASS_CSV_PATH="${QGMM_INPLACE_ADD_TRANSPOSE_PASS_CSV}")
    endif()
endif()
