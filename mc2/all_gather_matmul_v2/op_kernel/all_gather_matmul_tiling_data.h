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
 * \file all_gather_matmul_tiling_data.h
 * \brief
 */
#ifndef __ALL_GATHER_MATMUL_TILING_DATA_H_
#define __ALL_GATHER_MATMUL_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"
#include "../../common/op_kernel/mc2_tiling_struct.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmv3_arch35_tiling_data.h"
#include "../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"

namespace Mc2Tiling {

struct AllGatherMatmulTilingDataV2 {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    RCSTiling param;
    uint32_t dataType;
    uint32_t debugMode;
    Mc2MatMulV3TilingData mc2MmV3LocalTilingData;
    Mc2MatMulV3TilingData mc2MmV3TileTilingData;
    Mc2MatMulV3TilingData mc2MmV3TailTilingData;
};

struct AllGatherMatmulTilingDataFp8 {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    RCSTiling param;
    uint32_t dataType;
    uint32_t debugMode;
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams quantBmmv3LocalTiling;
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams quantBmmv3TileTiling;
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams quantBmmv3TailTiling;
};

#if (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) || \
     (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
      ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))) || \
     ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1)))
struct MC2TileInfo {
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *mmTiling;
    uint64_t aOffset;
    uint64_t aAddrOffset;
    uint64_t cOffset;
    uint64_t cAddrOffset;
};
#endif
} // namespace Mc2Tiling
#endif
