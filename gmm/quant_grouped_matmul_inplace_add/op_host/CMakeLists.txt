# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        set(quant_grouped_matmul_inplace_add_apt_depends gmm/common CACHE INTERNAL "Dependencies for quant_grouped_matmul_inplace_add_apt")
    elseif("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend350")
        set(quant_grouped_matmul_inplace_add_apt_depends gmm/common CACHE INTERNAL "Dependencies for quant_grouped_matmul_inplace_add_apt")
    endif()
    target_sources(op_host_aclnnExc PRIVATE
        quant_grouped_matmul_inplace_add_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    add_ops_compile_options(
        OP_NAME QuantGroupedMatmulInplaceAdd
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )

    if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        set(quant_grouped_matmul_inplace_add_apt_depends gmm/common CACHE INTERNAL "Dependencies for quant_grouped_matmul_inplace_add_apt")
        add_ops_compile_options(
            OP_NAME QuantGroupedMatmulInplaceAdd
            COMPUTE_UNIT Ascend950PR_9599
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
        )
    elseif("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend350")
        set(quant_grouped_matmul_inplace_add_apt_depends gmm/common CACHE INTERNAL "Dependencies for quant_grouped_matmul_inplace_add_apt")
        add_ops_compile_options(
            OP_NAME QuantGroupedMatmulInplaceAdd
            COMPUTE_UNIT Ascend350_355e
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
        )
    endif()

else()
    add_modules_sources(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE quant_grouped_matmul_inplace_add ACLNNTYPE aclnn_exclude)
endif()

if((UT_TEST_ALL OR OP_HOST_UT) AND TARGET ${OPHOST_NAME}_tiling_obj)
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp)
    target_link_libraries(${OPHOST_NAME}_tiling_obj PRIVATE
        $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>)
endif()

if((UT_TEST_ALL OR OP_API_UT) AND TARGET ${OPHOST_NAME}_opapi_obj)
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../grouped_matmul/op_api/grouped_matmul_950_checker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../grouped_matmul/op_api/grouped_matmul_util.cpp)
    target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../grouped_matmul/op_api)
endif()
