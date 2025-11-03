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

set(_PFA_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp
        ${OPS_TRANSFORMER_DIR}/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_register.cpp
)

set(_PFA_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/attention/prompt_flash_attention/op_kernel/prompt_flash_attention_tiling_data.h
)

set(_PFA_UTestCommonPrivateIncludeExt
        ${OPBASE_INC_DIRS}
        ${OPS_TRANSFORMER_DIR}/common/include
)

set(_PFA_TargetPrivateIncludesExt)

set(_PFA_OpKernelPrivateIncludesExt
        ${OPS_TRANSFORMER_DIR}/common/include/kernel
)

aux_source_directory(${CMAKE_SOURCE_DIR}/attention/prompt_flash_attention/op_kernel _PFA_KernelSourcesExt)
set(_PFA_CompileDefintions
        PFA_UT
)

set(_PFA_UTestSourcesExtForce)
if (UT_TEST_ALL)
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cpp")
elseif(OP_HOST_UT)
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/op_host/*.cpp")
elseif(OP_KERNEL_UT)
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/op_kernel/*.cpp")
elseif(OP_API_UT)
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/op_api/*.cpp")        
endif()
list(APPEND _PFA_UTestSourcesExtForce ${_Src1} ${_Src2})

if (_PFA_UTestSourcesExtForce)
    set(UTEST_FORCE_FLAG ON)
else()
    set(UTEST_FORCE_FLAG OFF)
endif()

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   Pfa
        SNAKE                                   prompt_flash_attention
        KERNEL_TILING_DATA_DEF_H                ${_PFA_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_PFA_KernelSourcesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_PFA_CompileDefintions}
        TILING_SOURCES_EXT                      ${_PFA_TilingSourcesExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_PFA_TargetPrivateIncludesExt}
        KERNEL_PRIVATE_INCLUDES_EXT             ${_PFA_OpKernelPrivateIncludesExt}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_PFA_UTestCommonPrivateIncludeExt}
        UTEST_SOURCES_EXT                       ${_PFA_UTestSourcesExtForce}
        UTEST_SOURCES_EXT_FORCE                 ${UTEST_FORCE_FLAG}
)
