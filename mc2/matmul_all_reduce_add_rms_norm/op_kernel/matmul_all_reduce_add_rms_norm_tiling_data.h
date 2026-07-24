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
 * \file matmul_all_reduce_add_rms_norm_tiling_data.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_DATA_H
#define MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#include "../../common/op_kernel/mc2_tiling_struct.h"
#include "../../matmul_all_reduce/op_kernel/quant_matmul_all_reduce_tiling_data.h"
#include "../../matmul_all_reduce/op_kernel/unquant_matmul_all_reduce_tiling_data.h"
#include "../../matmul_all_reduce/op_kernel/weight_quant_matmul_all_reduce_tiling_data.h"

namespace Mc2Tiling {
#pragma pack(push, 8) // 8 means 8 bytes aligned
struct AddRMSNormTilingData {
    uint32_t num_row;
    uint32_t num_col;
    uint32_t block_factor;
    uint32_t row_factor;
    uint32_t ub_factor;
    float epsilon;
    float avg_factor;
};

struct AddRMSNormTilingeKeyData {
    uint32_t ARNKeyTile;
    uint32_t ARNKeyTail;
    uint32_t ARNNumBlocksTile;
    uint32_t ARNNumBlocksTail;
};

struct MatmulAllReduceAddRmsNormTilingData {
    MatmulAllReduce910TilingData matmulAllReduceTilingData;
    AddRMSNormTilingData addRMSNormTileTilingData;
    AddRMSNormTilingData addRMSNormTailTilingData;
    AddRMSNormTilingeKeyData addRmsNormTilingeKeyData;
};

struct QuantMatmulAllReduceAddRmsNormTilingData {
    QuantMatmulAllReduceTilingData quantMatmulAllReduceTilingData;
    AddRMSNormTilingData addRMSNormTileTilingData;
    AddRMSNormTilingData addRMSNormTailTilingData;
    AddRMSNormTilingeKeyData addRmsNormTilingeKeyData;
};

struct WeightQuantMatmulAllReduceAddRmsNormTilingData {
    WeightQuantMatmulAllReduceTilingData weightQuantMatmulAllReduceTilingData;
    AddRMSNormTilingData addRMSNormTileTilingData;
    AddRMSNormTilingData addRMSNormTailTilingData;
    AddRMSNormTilingeKeyData addRmsNormTilingeKeyData;
};
#pragma pack(pop)
} // namespace Mc2Tiling
#endif
