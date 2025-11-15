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

set(_NSA_SELECTED_ATTENTION_INFER_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/nsa_selected_attention_infer/op_host/nsa_selected_attention_infer_tiling.cpp
        ${OPS_TRANSFORMER_DIR}/attention/nsa_selected_attention_infer/op_host/nsa_selected_attention_infer_tiling_check.cpp
        ${OPS_TRANSFORMER_DIR}/attention/nsa_selected_attention_infer/op_host/nsa_selected_attention_infer_tiling_register.cpp
)

set(_NSA_SELECTED_ATTENTION_INFER_KernelTilingDataDefH
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention_infer/op_host/nsa_selected_attention_infer_tiling.h
)

set(_NCWC_OpKernelSourcesExt)

set(_NsaSelectedAttention_TilingPrivateIncludesExt
        ${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention_infer/op_host/
        ${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention_infer/op_host/op_api/
)

set(_NSA_SELECTED_ATTENTION_INFER_UTestCommonPrivateIncludeExt
        ${OPBASE_INC_DIRS}
        ${OPS_TRANSFORMER_DIR}/common/include
)

set(_NSA_SELECTED_ATTENTION_INFER_TargetPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling
        error_manager
)

set(_NCWC_UTestSourcesExtForce)

set(_NSA_SELECTED_ATTENTION_INFER_CompileDefintions
        ORIG_DTYPE_QUERY=DT_FLOAT16
)

set(_NCWC_UTestCompileDefinitions)

list(APPEND _NCWC_OpKernelSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/nsa_selected_attention_infer/op_kernel/nsa_selected_attention_infer.cpp
)

if (UT_TEST_ALL)
        message(STATUS "+++++ DEBUG: all ut")
        file(GLOB_RECURSE _Src1 "${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention_infer/tests/utest/*.cc")
        file(GLOB_RECURSE _Src2 "${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention_infer/tests/utest/*.cpp")
elseif(OP_HOST_UT)
        message(STATUS "+++++ DEBUG: only host :${CMAKE_SOURCE_DIR},${OPS_TRANSFORMER_DIR}")
        file(GLOB_RECURSE _Src1 "${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention_infer/tests/utest/*tiling.cc")
        file(GLOB_RECURSE _Src2 "${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention_infer/tests/utest/*tiling.cpp")
elseif(OP_KERNEL_UT)
        message(STATUS "+++++ DEBUG: only kernel")
        file(GLOB_RECURSE _Src1 "${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention_infer/tests/utest/*kernel.cc")
        file(GLOB_RECURSE _Src2 "${CMAKE_SOURCE_DIR}/attention/nsa_selected_attention_infer/tests/utest/*kernel.cpp")
endif()
list(APPEND _NCWC_UTestSourcesExtForce ${_Src1} ${_Src2})
message(STATUS "+++++ DEBUG: .cpp files: ${_NCWC_UTestSourcesExtForce}")

list(APPEND _NCWC_UTestCompileDefinitions)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   NsaSelectedAttentionInfer
        SNAKE                                   nsa_selected_attention_infer
        TILING_SOURCES_EXT                      ${_NSA_SELECTED_ATTENTION_INFER_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_NsaSelectedAttention_TilingPrivateIncludesExt}
        KERNEL_SOURCES_EXT                      ${_NCWC_OpKernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_NSA_SELECTED_ATTENTION_INFER_KernelTilingDataDefH}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_NSA_SELECTED_ATTENTION_INFER_TargetPrivateLinkLibrariesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_NSA_SELECTED_ATTENTION_INFER_CompileDefintions}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_NSA_SELECTED_ATTENTION_INFER_UTestCommonPrivateIncludeExt}
        UTEST_SOURCES_EXT                       ${_NCWC_UTestSourcesExtForce}
        UTEST_SOURCES_EXT_FORCE                   ON
        UTEST_COMMON_PRIVATE_COMPILE_DEFINITIONS  ${_NCWC_UTestCompileDefinitions}
        UTEST_PRIVATE_COMPILE_DEFINITIONS         ${_NCWC_UTestCompileDefinitions}
)
