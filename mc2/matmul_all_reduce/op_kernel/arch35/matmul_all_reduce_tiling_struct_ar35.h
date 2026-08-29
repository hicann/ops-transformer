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
 * \file matmul_all_reduce_tiling_struct
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_TILING_STRUCT_ARCH35_H
#define MATMUL_ALL_REDUCE_TILING_STRUCT_ARCH35_H

#include "kernel_tiling/kernel_tiling.h"
#include "../../../common/op_kernel/mc2_tiling_struct.h"
#include "../../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"
#include "../../../3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_tiling_data.h"
#include "../../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmv3_arch35_tiling_data.h"
#include "../unquant_matmul_all_reduce_tiling_data.h"
#include "../quant_matmul_all_reduce_tiling_data.h"
#include "../weight_quant_matmul_all_reduce_tiling_data.h"
namespace Mc2Tiling {

// 确保tilingData按8byte对齐
constexpr uint32_t STRUCT_ALIGNAS_EIGHT = 8;

#pragma pack(push, 8)
struct alignas(STRUCT_ALIGNAS_EIGHT) WeightQuantMatmulAllReduceA5TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    Mc2CcTiling mc2CcTilingComm;
    Mc2Tiling::RCSTiling param;
    Mc2WeightQuantBatchMatmulV2RegBaseTilingData tileRegBaseMmTiling;
    Mc2WeightQuantBatchMatmulV2RegBaseTilingData tailRegBaseMmTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(STRUCT_ALIGNAS_EIGHT) WeightQuantMatmulAllReduceA5Fp8TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    Mc2CcTiling mc2CcTilingComm;
    Mc2Tiling::RCSTiling param;
    Mc2WeightQuantBatchMatmulV2ASTilingData tileMmASTiling;
    Mc2WeightQuantBatchMatmulV2ASTilingData tailMmASTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(STRUCT_ALIGNAS_EIGHT) QuantMatmulAllReduceTilingDataA5 {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    Mc2CcTiling mc2CcTilingComm;
    Mc2Tiling::RCSTiling param;
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams tilematmulTiling;
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams tailmatmulTiling;
};
#pragma pack(pop)

// bias 移至 vec epilogue 累加的扩展 tiling 字段（统一收纳，便于后续扩展）
struct MatmulAllReduceExtTiling {
    uint32_t isVecBias; // bias 是否移至 vec epilogue（1=是，0=否）
    uint32_t biasUbCnt; // bias epilogue 的 N 方向 UB 切分粒度
};

#pragma pack(push, 8)
struct alignas(STRUCT_ALIGNAS_EIGHT) MatmulAllReduce910TilingDataA5 {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    Mc2CcTiling mc2CcTilingComm;
    Mc2Tiling::RCSTiling param;
    Mc2MatMulV3TilingData mC2Mmv3TileTilingData;
    Mc2MatMulV3TilingData mC2Mmv3TailTilingData;
    MatmulAllReduceExtTiling extTiling;
};
#pragma pack(pop)

} // namespace Mc2Tiling
#endif // MATMUL_ALL_REDUCE_TILING_STRUCT_ARCH35_H
