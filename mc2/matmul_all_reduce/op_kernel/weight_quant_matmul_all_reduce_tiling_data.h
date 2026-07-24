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
 * \file unquant_matmul_all_reduce_tiling_data.h
 * \brief
 */

#ifndef WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_DATA_H
#define WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#ifdef __CCE_KT_TEST__
#include "../../common/op_kernel/mc2_tiling_struct.h"
#include "../../3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_tiling_data.h"
#else
#include "../../common/op_kernel/mc2_tiling_struct.h"
#include "../../3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_tiling_data.h"
#endif

namespace Mc2Tiling {

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) WeightQuantMatmulAllReduceTilingData {
    Mc2Tiling::Mc2Msg msg;
    Mc2Tiling::RCSTiling param;
    Mc2WeightQuantBatchMatmulV2TilingData tilematmulTiling;
    Mc2WeightQuantBatchMatmulV2TilingData tailmatmulTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) WeightQuantMatmulAllReduceNzTilingData {
    Mc2Tiling::Mc2Msg msg;
    Mc2Tiling::RCSTiling param;
    Mc2WeightQuantBatchMatmulV2NzTilingData tilematmulTiling;
    Mc2WeightQuantBatchMatmulV2NzTilingData tailmatmulTiling;
};
#pragma pack(pop)

} // namespace Mc2Tiling
#endif
