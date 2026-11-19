# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

set(QUANT_FLASH_ATTN_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging header for quant_flash_attn")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        quant_flash_attn_def.cpp
    )
endif()

if(ASCEND_COMPUTE_UNIT STREQUAL "ascend910b" OR ASCEND_COMPUTE_UNIT STREQUAL "ascend910_93" OR
   ASCEND_COMPUTE_UNIT STREQUAL "ascend950" OR ASCEND_COMPUTE_UNIT STREQUAL "ascend350" OR
   ASCEND_COMPUTE_UNIT STREQUAL "ascend960dt" OR ASCEND_COMPUTE_UNIT STREQUAL "ascend960pr")
    set(quant_flash_attn_depends attention/common CACHE INTERNAL "Dependencies for quant_flash_attn")
endif()
set(QFA_AUTO_SYNC_OPTION --cce-auto-sync=off)
set(QFA_ARCH22_INCLUDE_OPTION)
if(ASCEND_COMPUTE_UNIT STREQUAL "ascend910b" OR ASCEND_COMPUTE_UNIT STREQUAL "ascend910_93")
    set(QFA_FIA_ARCH22_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../fused_infer_attention_score/op_kernel/arch22")
    set(QFA_ARCH22_INCLUDE_DIR "${CMAKE_BINARY_DIR}/qfa_arch22_kernel_include")
    file(COPY "${QFA_FIA_ARCH22_DIR}/attn_infra" DESTINATION "${QFA_ARCH22_INCLUDE_DIR}")
    file(COPY "${QFA_FIA_ARCH22_DIR}"
        DESTINATION "${QFA_ARCH22_INCLUDE_DIR}/attention/fused_infer_attention_score/op_kernel")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../../common" DESTINATION "${QFA_ARCH22_INCLUDE_DIR}/attention")
    set(QFA_ARCH22_INCLUDE_OPTION -I${QFA_ARCH22_INCLUDE_DIR})
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME QuantFlashAttn
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
elseif("${CONDITION_UNIT}" STREQUAL "ascend350")
    add_ops_compile_options(
        OP_NAME QuantFlashAttn
        COMPUTE_UNIT Ascend350_355e
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

file(GLOB_RECURSE ARCH35_CHECKER_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/checkers/*checker*.cpp
)

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME QuantFlashAttn
            OPTIONS ${QFA_AUTO_SYNC_OPTION}
                    -Wno-deprecated-declarations
                    -Werror
                    ${QFA_ARCH22_INCLUDE_OPTION}
    )
else()
    add_ops_compile_options(
            OP_NAME QuantFlashAttn
            OPTIONS ${QFA_AUTO_SYNC_OPTION}
                    -Wno-deprecated-declarations
                    -Werror
                    ${QFA_ARCH22_INCLUDE_OPTION}
    )
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
    ${ARCH35_CHECKER_SRC_FILES}
    qfa_tiling_info_parser.cpp
    arch22/quant_flash_attn_tiling_int8.cpp
    arch22/quant_flash_attn_tiling_a8c8.cpp
    ../../common/op_host/split_core.cpp
    )
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE quant_flash_attn
        ACLNNTYPE aclnn_inner
    )
endif()
