# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ########################################################################################################################
# # 调用编译方法, 生成对应编译目标
# ########################################################################################################################
# set(_FFN_TilingSourcesExt
#         ${OPS_TRANSFORMER_DIR}/ffn/ffn/op_host/ffn_tiling.cpp
# )

# set(_FFN_TilingPrivateIncludesExt
#         ${OPS_TRANSFORMER_DIR}/ffn/ffn/op_host
# )

# set(_FFN_KernelTilingDataDefH
#         ${OPS_TRANSFORMER_DIR}/ffn/ffn/op_host/ffn_tiling.h
# )

# set(_FFN_OpApiSourcesExt
#         ${OPS_TRANSFORMER_DIR}/ffn/ffn/op_host/op_api/ffn.cpp
#         ${OPS_TRANSFORMER_DIR}/ffn/ffn/op_host/op_api/aclnn_ffn.cpp
# )

# set(_FFN_OpProtoSourcesExt
#         ${OPS_TRANSFORMER_DIR}/ffn/ffn/op_host/ffn_infershape.cpp
# )

# aux_source_directory(${OPS_TRANSFORMER_DIR}/ffn/ffn/op_kernel _FFN_KernelSourcesExt)
# set(_FFN_KernelPrivateCompileDefinitionsExt
#         KernelCtrlParam ffn fp16 DTYPE_X=half DTYPE_WEIGHT1=half DTYPE_WEIGHT2=half DTYPE_BIAS1=half DTYPE_BIAS2=half DTYPE_Y=half
#         KernelCtrlParam ffn bf16 DTYPE_X=bfloat16_t DTYPE_WEIGHT1=bfloat16_t DTYPE_WEIGHT2=bfloat16_t DTYPE_BIAS1=float DTYPE_BIAS2=float DTYPE_Y=bfloat16_t
#         KernelCtrlParam ffn quant_fp16 DTYPE_X=int8_t DTYPE_WEIGHT1=int8_t DTYPE_WEIGHT2=int8_t DTYPE_BIAS1=int32_t DTYPE_BIAS2=int32_t DTYPE_Y=half
#         KernelCtrlParam ffn quant_bf16 DTYPE_X=int8_t DTYPE_WEIGHT1=int8_t DTYPE_WEIGHT2=int8_t DTYPE_BIAS1=int32_t DTYPE_BIAS2=int32_t DTYPE_Y=bfloat16_t
#         KernelCtrlParam ffn a16w8_fp16 DTYPE_X=half DTYPE_WEIGHT1=int8_t DTYPE_WEIGHT2=int8_t DTYPE_BIAS1=half DTYPE_BIAS2=half DTYPE_Y=half
#         KernelCtrlParam ffn a16w8_bf16 DTYPE_X=bfloat16_t DTYPE_WEIGHT1=int8_t DTYPE_WEIGHT2=int8_t DTYPE_BIAS1=float DTYPE_BIAS2=float DTYPE_Y=bfloat16_t
#         KernelCtrlParam ffn a16w4_fp16 DTYPE_X=half DTYPE_WEIGHT1=int4b_t DTYPE_WEIGHT2=int4b_t DTYPE_BIAS1=half DTYPE_BIAS2=half DTYPE_Y=half
#         KernelCtrlParam ffn a16w4_bf16 DTYPE_X=bfloat16_t DTYPE_WEIGHT1=int4b_t DTYPE_WEIGHT2=int4b_t DTYPE_BIAS1=float DTYPE_BIAS2=float DTYPE_Y=bfloat16_t
# )

# set(_FFN_TargetPrivateIncludeExt
#         ${OPS_TRANSFORMER_DIR}/ffn/ffn/op_host
#         ${CMAKE_SOURCE_DIR}/ffn/ffn/op_host/op_api
#         ${CMAKE_SOURCE_DIR}/common/include
#         ${OPBASE_INC_DIRS}
# )

# set(_FFN_TillingPrivateLinkLibrariesExt
# )

# OpsTest_Level2_AddOp(
#         SUB_SYSTEM                              transformer
#         BRIEF                                   FFN
#         SNAKE                                   ffn
#         OPAPI_SOURCES_EXT                       ${_FFN_OpApiSourcesExt}
#         PROTO_SOURCES_EXT                       ${_FFN_OpProtoSourcesExt}
#         TILING_SOURCES_EXT                      ${_FFN_TilingSourcesExt}
#         TILING_PRIVATE_INCLUDES_EXT             ${_FFN_TilingPrivateIncludesExt}
#         TILING_PRIVATE_LINK_LIBRARIES_EXT       ${_FFN_TillingPrivateLinkLibrariesExt}
#         KERNEL_SOURCES_EXT                      ${_FFN_KernelSourcesExt}
#         KERNEL_TILING_DATA_DEF_H                ${_FFN_KernelTilingDataDefH}
#         KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_FFN_KernelPrivateCompileDefinitionsExt}
#         UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_FFN_TargetPrivateIncludeExt}
#         UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_FFN_TargetPrivateLinkLibrariesExt}
# )
