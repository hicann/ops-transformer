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

set(_IFA_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp
        ${OPS_TRANSFORMER_DIR}/attention/incre_flash_attention/op_host/incre_flash_attention_tiling_check.cpp
        ${OPS_TRANSFORMER_DIR}/attention/incre_flash_attention/op_host/incre_flash_attention_tiling_register.cpp
        ${OPS_TRANSFORMER_DIR}/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp
        ${OPS_TRANSFORMER_DIR}/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_register.cpp
)

set(_IFA_KernelTilingDataDefH
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/attention/incre_flash_attention/op_kernel/incre_flash_attention_tiling.h
        ${CMAKE_SOURCE_DIR}/attention/prompt_flash_attention/op_kernel/prompt_flash_attention_tiling_data.h
)

set(_IFA_TargetPrivateLinkLibrariesExt)

set(_IFA_OpKernelPrivateIncludesExt
        ${OPS_TRANSFORMER_DIR}/common/include/kernel
)

aux_source_directory(${CMAKE_SOURCE_DIR}/attention/incre_flash_attention/op_kernel _IFA_KernelSourcesExt)

set(_IFA_KernelPrivateCompileDefinitionsExt
        KernelCtrlParam incre_flash_attention,incre_flash_attention_FIAS fp16_fp16 ORIG_DTYPE_QUERY=DT_FLOAT16 ORIG_DTYPE_ATTENTION_OUT=DT_FLOAT16
        KernelCtrlParam incre_flash_attention,incre_flash_attention_FIAS fp16_int8 ORIG_DTYPE_QUERY=DT_FLOAT16 ORIG_DTYPE_ATTENTION_OUT=DT_INT8
        KernelCtrlParam incre_flash_attention,incre_flash_attention_FIAS bf16_bf16 ORIG_DTYPE_QUERY=DT_BF16 ORIG_DTYPE_ATTENTION_OUT=DT_BF16
        KernelCtrlParam incre_flash_attention,incre_flash_attention_FIAS bf16_int8 ORIG_DTYPE_QUERY=DT_BF16 ORIG_DTYPE_ATTENTION_OUT=DT_INT8
)

set(_IFA_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/attention/prompt_flash_attention/op_host/
        ${CMAKE_SOURCE_DIR}/attention/prompt_flash_attention/op_host/op_api
)

set(_IFA_UTestCommonPrivateIncludeExt
        ${OPBASE_INC_DIRS}
        ${OPS_TRANSFORMER_DIR}/common/include
)

set(_IFA_UTestSourcesExtForce)

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

list(APPEND _IFA_UTestSourcesExtForce ${_Src1} ${_Src2})
message(STATUS "+++++ DEBUG: .cpp files: ${_IFA_UTestSourcesExtForce}")

if (_IFA_UTestSourcesExtForce)
    set(UTEST_FORCE_FLAG ON)
else()
    set(UTEST_FORCE_FLAG OFF)
endif()

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   Ifa
        SNAKE                                   incre_flash_attention
        KERNEL_TILING_DATA_DEF_H                ${_IFA_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_IFA_KernelSourcesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_IFA_KernelPrivateCompileDefinitionsExt}
        TILING_SOURCES_EXT                      ${_IFA_TilingSourcesExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_IFA_TargetPrivateLinkLibrariesExt}
        KERNEL_PRIVATE_INCLUDES_EXT             ${_IFA_OpKernelPrivateIncludesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_IFA_KernelTilingExtInclude}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_IFA_UTestCommonPrivateIncludeExt}
        UTEST_SOURCES_EXT                       ${_IFA_UTestSourcesExtForce}
        UTEST_SOURCES_EXT_FORCE                 ${UTEST_FORCE_FLAG}
)