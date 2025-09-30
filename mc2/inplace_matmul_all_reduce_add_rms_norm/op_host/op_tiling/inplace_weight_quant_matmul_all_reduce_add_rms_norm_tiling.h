/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_weight_quant_matmul_all_reduce_add_rms_norm_tiling.h
 * \brief
 */
#ifndef _INPLACE_WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_
#define _INPLACE_WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_
#include "../../../matmul_all_reduce_add_rms_norm/op_host/op_tiling/weight_quant_matmul_all_reduce_add_rms_norm_tiling.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_310100, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_311100, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_310110, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_311110, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_310200, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_311200, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_310210, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_311210, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_310300, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_310310, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_311300, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceMatmulAllReduceAddRmsNorm_311310, WeightQuantMatmulAllReduceAddRmsNormTilingData);
} // namespace optiling

#endif // _INPLACE_WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_