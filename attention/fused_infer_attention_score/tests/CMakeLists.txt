# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


########################################################################################################################
# 调用编译方法, 生成对应编译目标
########################################################################################################################

set(_FIA_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp
        ${OPS_TRANSFORMER_DIR}/attention/incre_flash_attention/op_host/incre_flash_attention_tiling_check.cpp
        ${OPS_TRANSFORMER_DIR}/attention/incre_flash_attention/op_host/incre_flash_attention_tiling_register.cpp
        ${OPS_TRANSFORMER_DIR}/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp
        ${OPS_TRANSFORMER_DIR}/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_register.cpp
        ${OPS_TRANSFORMER_DIR}/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp
        ${OPS_TRANSFORMER_DIR}/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_register.cpp
        ${OPS_TRANSFORMER_DIR}/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_v3.cpp
        ${OPS_TRANSFORMER_DIR}/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_check.cpp
        ${OPS_TRANSFORMER_DIR}/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_check_single_para.cpp
        ${OPS_TRANSFORMER_DIR}/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_check_existence.cpp
        ${OPS_TRANSFORMER_DIR}/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_check_feature.cpp
        ${OPS_TRANSFORMER_DIR}/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_check_consistency.cpp
        ${OPS_TRANSFORMER_DIR}/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.cpp
        ${OPS_TRANSFORMER_DIR}/attention/common/op_host/fia_tiling_info.cpp
        ${OPS_TRANSFORMER_DIR}/attention/common/op_host/fia_tiling_shape.cpp
        ${OPS_TRANSFORMER_DIR}/attention/common/op_host/arch32/fia_tiling_nonquant_mla.cpp
)

set(_FIA_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/attention/incre_flash_attention/op_kernel/incre_flash_attention_tiling.h
        ${CMAKE_SOURCE_DIR}/attention/prompt_flash_attention/op_kernel/prompt_flash_attention_tiling_data.h
)
set(_FIA_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/attention/incre_flash_attention/op_host/
        ${CMAKE_SOURCE_DIR}/attention/incre_flash_attention/op_host/op_api/
        ${CMAKE_SOURCE_DIR}/attention/prompt_flash_attention/op_host/
        ${CMAKE_SOURCE_DIR}/attention/prompt_flash_attention/op_host/op_api/
)

set(_FIA_UTestCommonPrivateIncludeExt
        ${OPBASE_INC_DIRS}
        ${OPS_TRANSFORMER_DIR}/common/include
)

set(_FIA_OpKernelPrivateIncludesExt
        ${OPS_TRANSFORMER_DIR}/common/include/kernel
)

set(_FIA_TargetPrivateLinkLibrariesExt)

set(_FIA_UTestSourcesExtForce)

if (UT_TEST_ALL)
        message(STATUS "+++++ DEBUG: all ut")
        file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cc")
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cpp")
elseif(OP_HOST_UT)
        message(STATUS "+++++ DEBUG: only host")
        file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/op_host/*tiling.cpp")
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/op_host/*infershape.cpp")
elseif(OP_KERNEL_UT)
        message(STATUS "+++++ DEBUG: only kernel")
        file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/op_kernel/*kernel.cpp")
elseif(OP_API_UT)
        message(STATUS "+++++ DEBUG: only opapi")
        file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/op_api/*opapi.cpp")
endif()

list(APPEND _FIA_UTestSourcesExtForce ${_Src1} ${_Src2})
message(STATUS "+++++ DEBUG: .cpp files: ${_FIA_UTestSourcesExtForce}")

if (_FIA_UTestSourcesExtForce)
    set(UTEST_FORCE_FLAG ON)
else()
    set(UTEST_FORCE_FLAG OFF)
endif()

aux_source_directory(${CMAKE_SOURCE_DIR}/attention/fused_infer_attention_score/op_kernel _FIA_KernelSourcesExt)
set(_FIA_CompileDefintions
        KernelCtrlParam incre_flash_attention,incre_flash_attention_FIAS,prompt_flash_attention,prompt_flash_attention_FIAS fia PFA_UT
)
OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   Fia
        SNAKE                                   fused_infer_attention_score
        KERNEL_TILING_DATA_DEF_H                ${_FIA_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_FIA_KernelSourcesExt}
        TILING_SOURCES_EXT                      ${_FIA_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_FIA_KernelTilingExtInclude}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_FIA_CompileDefintions}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_FIA_TargetPrivateLinkLibrariesExt}
        KERNEL_PRIVATE_INCLUDES_EXT             ${_FIA_OpKernelPrivateIncludesExt}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_FIA_UTestCommonPrivateIncludeExt}
        UTEST_SOURCES_EXT                       ${_FIA_UTestSourcesExtForce}
        UTEST_SOURCES_EXT_FORCE                 ${UTEST_FORCE_FLAG}
)
