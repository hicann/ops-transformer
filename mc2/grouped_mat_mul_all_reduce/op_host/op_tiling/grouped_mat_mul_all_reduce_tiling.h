/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_mat_mul_all_reduce_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_ALL_REDUCE_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_ALL_REDUCE_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr uint32_t MAX_TENSOR_CONT = 64;
constexpr uint32_t MC2_MSG_SIZE = 128;

BEGIN_TILING_DATA_DEF(GMMAllReduceBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, activeType);
TILING_DATA_FIELD_DEF(uint32_t, ubBaseM);
TILING_DATA_FIELD_DEF(uint32_t, ubBaseN);
TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
TILING_DATA_FIELD_DEF(uint32_t, ubRestSize);
TILING_DATA_FIELD_DEF(uint32_t, workspaceSize);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_TENSOR_CONT, mList);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_TENSOR_CONT, kList);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_TENSOR_CONT, nList);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMAllReduceBaseParamsOp, GMMAllReduceBaseParams)

BEGIN_TILING_DATA_DEF(GMMAllReduceCoreTiling)
TILING_DATA_FIELD_DEF_STRUCT(GMMAllReduceBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
TILING_DATA_FIELD_DEF(uint32_t, notifyOff);
TILING_DATA_FIELD_DEF(uint32_t, debugMode); // hccl debug mode
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMAllReduceCoreTilingOp, GMMAllReduceCoreTiling)

BEGIN_TILING_DATA_DEF(KFCGroupedTiling) // 同aicpu_hccl_def KFCGroupTilingData
TILING_DATA_FIELD_DEF_ARR(uint8_t, MAX_TENSOR_CONT* MC2_MSG_SIZE, msg);
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, groupTilingMagicNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(KFCGroupedTilingOp, KFCGroupedTiling)

BEGIN_TILING_DATA_DEF(GMMAllReduceTilingData)
TILING_DATA_FIELD_DEF_STRUCT(KFCGroupedTiling, aicpuTiling);
TILING_DATA_FIELD_DEF_STRUCT(GMMAllReduceCoreTiling, aicoreTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GroupedMatMulAllReduce, GMMAllReduceTilingData)

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_ALL_REDUCE_H
