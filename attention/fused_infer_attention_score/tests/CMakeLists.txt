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
)
