# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

########################################################################################################################
# 调用编译方法, 生成对应编译目标
########################################################################################################################

# FA/FAG 算子的 Kernel 均包含较多的 TilingKey, 在编译态、执行态均区分是否只执行FA/FAG, 以便提高编译执行效率;
# Host 侧各 Target 暂不区分 FA/FAG 以便于用例实现.

set(_FA_OpApiSourcesExt
        # FAS
        ${OPS_TRANSFORMER_DIR}/attention/flash_attention_score/op_host/op_api/flash_attention_score.cpp
        ${OPS_TRANSFORMER_DIR}/attention/flash_attention_score/op_host/op_api/aclnn_flash_attention_score.cpp
        # FAG
        ${OPS_TRANSFORMER_DIR}/attention/flash_attention_score_grad/op_host/op_api/flash_attention_score_grad.cpp
        ${OPS_TRANSFORMER_DIR}/attention/flash_attention_score_grad/op_host/op_api/aclnn_flash_attention_score_grad.cpp
)

set(_FA_OpProtoSourcesExt
        # FAS
        ${OPS_TRANSFORMER_DIR}/attention/flash_attention_score/op_host/flash_attention_score_infershape.cpp
        # FAG
        ${OPS_TRANSFORMER_DIR}/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_infershape.cpp
)

set(_FA_OpTilingSourcesExt)
file(GLOB _Src1 "${OPS_TRANSFORMER_DIR}/attention/flash_attention_score/op_host/flash_attention_score_tiling*.cc")
file(GLOB _Src2 "${OPS_TRANSFORMER_DIR}/attention/flash_attention_score/op_host/flash_attention_score_tiling*.cpp")
file(GLOB _Src3 "${OPS_TRANSFORMER_DIR}/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling*.cc")
file(GLOB _Src4 "${OPS_TRANSFORMER_DIR}/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling*.cpp")
list(APPEND _FA_OpTilingSourcesExt ${_Src1} ${_Src2} ${_Src3} ${_Src4})

set(_FA_OpTilingPrivateIncludesExt
        ${OPS_TRANSFORMER_DIR}/attention/common/op_kernel/arch-310
)

file(GLOB _FAG_OpKernelTilingDataDefH_def "${OPS_TRANSFORMER_DIR}/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling*_def.h")
set(_FA_OpKernelTilingDataDefH
        # 公共
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        # FAS
        ${OPS_TRANSFORMER_DIR}/attention/flash_attention_score/op_kernel/flash_attention_score_tiling.h
        # FAG
        ${OPS_TRANSFORMER_DIR}/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_tiling.h
        ${_FAG_OpKernelTilingDataDefH_def}
)

set(_FA_OpKernelSourcesExt)

set(_FA_OpKernelPrivateIncludesExt
        ${OPS_TRANSFORMER_DIR}/common/include/kernel
)

set(_FA_OpKernelPrivateCompileDefinitionsExt
        KernelCtrlParam flash_attention_score_grad fp16 ORIG_DTYPE_QUERY=DT_FLOAT16 DTYPE_DQ=half       KFC_L1_RESERVER_SIZE=0
        KernelCtrlParam flash_attention_score_grad bf16 ORIG_DTYPE_QUERY=DT_BF16    DTYPE_DQ=bfloat16_t KFC_L1_RESERVER_SIZE=0
        KernelCtrlParam flash_attention_score_grad fp32 ORIG_DTYPE_QUERY=DT_FLOAT   DTYPE_DQ=float      KFC_L1_RESERVER_SIZE=0
)
set(_FA_UTestCommonPrivateIncludeExt
        ${OPBASE_INC_DIRS}
        ${OPS_TRANSFORMER_DIR}/common/include
        ${OPS_TRANSFORMER_DIR}/attention/flash_attention_score/op_host
        ${OPS_TRANSFORMER_DIR}/attention/flash_attention_score/op_host/op_api
        ${OPS_TRANSFORMER_DIR}/attention/flash_attention_score_grad/op_host
        ${OPS_TRANSFORMER_DIR}/attention/flash_attention_score_grad/op_host/op_api
)

set(_FA_UTestCommonPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling    # 用于 FAG 指定模板优先级相关实现
        error_manager                   # 用于 FAG 指定模板优先级相关实现
)

set(_FA_UTestSourcesExtForce)
file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/ts_fa/*.cc")
file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/ts_fa/*.cpp")
list(APPEND _FA_UTestSourcesExtForce ${_Src1} ${_Src2})

set(_FA_UTestAclnnSourcesExtForce)
file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/ts_fa/*.cc")
file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/ts_fa/*.cpp")
list(APPEND _FA_UTestAclnnSourcesExtForce ${_Src1} ${_Src2})

set(_FA_UTestCompileDefinitions)

list(APPEND _FA_OpKernelSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad.cpp
)

file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/ts_fag/*.cc")
file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/ts_fag/*.cpp")
list(APPEND _FA_UTestSourcesExtForce ${_Src1} ${_Src2})

file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/ts_fag/*.cc")
file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest_aclnn/ts_fag/*.cpp")
list(APPEND _FA_UTestAclnnSourcesExtForce ${_Src1} ${_Src2})

list(APPEND _FA_UTestCompileDefinitions TESTS_UT_OPS_TEST_FAG)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                                transformer
        BRIEF                                     Fa
        SNAKE                                     flash_attention
        OPAPI_SOURCES_EXT                         ${_FA_OpApiSourcesExt}
        PROTO_SOURCES_EXT                         ${_FA_OpProtoSourcesExt}
        TILING_SOURCES_EXT                        ${_FA_OpTilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT               ${_FA_OpTilingPrivateIncludesExt}
        KERNEL_SOURCES_EXT                        ${_FA_OpKernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                  ${_FA_OpKernelTilingDataDefH}
        KERNEL_PRIVATE_INCLUDES_EXT               ${_FA_OpKernelPrivateIncludesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT    ${_FA_OpKernelPrivateCompileDefinitionsExt}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT         ${_FA_UTestCommonPrivateIncludeExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT   ${_FA_UTestCommonPrivateLinkLibrariesExt}
        UTEST_COMMON_PRIVATE_COMPILE_DEFINITIONS  ${_FA_UTestCompileDefinitions}
        UTEST_SOURCES_EXT                         ${_FA_UTestSourcesExtForce}
        UTEST_SOURCES_EXT_FORCE                   ON
        UTEST_PRIVATE_COMPILE_DEFINITIONS         ${_FA_UTestCompileDefinitions}
        UTEST_ACLNN_SOURCES_EXT                   ${_FA_UTestAclnnSourcesExtForce}
        UTEST_ACLNN_SOURCES_EXT_FORCE             ON
        UTEST_ACLNN_PRIVATE_COMPILE_DEFINITIONS   ${_FA_UTestCompileDefinitions}
)
