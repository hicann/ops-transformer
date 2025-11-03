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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-deprecated-declarations")

set(_ncai_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/nsa_compress_attention_infer/op_host/nsa_compress_attention_infer_tiling.cpp
)

set(_ncai_KernelTilingDataDefH
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/attention/nsa_compress_attention_infer/op_host/nsa_compress_attention_infer_tiling.h
)

set(_ncai_CompileDefintions
        ORIG_DTYPE_QUERY=DT_FLOAT16
)

set(_ncai_KernelTilingExtInclude
        ${CMAKE_SOURCE_DIR}/attention/nsa_compress_attention_infer/op_host/
        ${CMAKE_SOURCE_DIR}/attention/nsa_compress_attention_infer/op_host/op_api/
)

set(_ncai_UTestCommonPrivateIncludeExt
        ${OPBASE_INC_DIRS}
        ${OPS_TRANSFORMER_DIR}/common/include
)

set(_ncai_TargetPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling
        error_manager
)

set(_ncai_OpKernelPrivateIncludesExt
        ${OPS_TRANSFORMER_DIR}/common/include/kernel
)

set(_NCAI_UTestSourcesExtForce)

if (UT_TEST_ALL)
        file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cc")
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cpp")
elseif(OP_HOST_UT)
        file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/op_host/*tiling.cc")
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/op_host/*tiling.cpp")
elseif(OP_KERNEL_UT)
        file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/op_kernel/*kernel.cc")
        file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/op_kernel/*kernel.cpp")
endif()

list(APPEND _NCAI_UTestSourcesExtForce ${_Src1} ${_Src2})

aux_source_directory(${CMAKE_SOURCE_DIR}/attention/nsa_compress_attention_infer/op_kernel _ncai_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   ncaInfer
        SNAKE                                   nsa_compress_attention_infer
        KERNEL_TILING_DATA_DEF_H                ${_ncai_KernelTilingDataDefH}
        KERNEL_PRIVATE_INCLUDES_EXT             ${_ncai_OpKernelPrivateIncludesExt}
        KERNEL_SOURCES_EXT                      ${_ncai_KernelSourcesExt}
        TILING_SOURCES_EXT                      ${_ncai_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_ncai_KernelTilingExtInclude}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_ncai_CompileDefintions}
        UTEST_SOURCES_EXT                       ${_NCAI_UTestSourcesExtForce}
        UTEST_SOURCES_EXT_FORCE                 ON
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_ncai_TargetPrivateLinkLibrariesExt}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_ncai_UTestCommonPrivateIncludeExt}
)
