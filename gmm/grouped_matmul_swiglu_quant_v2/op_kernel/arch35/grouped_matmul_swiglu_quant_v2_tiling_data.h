/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_v2_tiling_data.h
 * \brief Device-side tiling data for the default MX quantization path.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_TILING_DATA_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"
#include "grouped_matmul_swiglu_quant_swiglu_params.h"

#pragma pack(push, 8)
struct GMMSwigluQuantV2Params {
    uint32_t groupNum = 0;
    uint8_t groupListType = 0;
    uint8_t quantDtype = 0;
    uint8_t isMxWeightNzMultiTensor = 0;
    uint8_t dequantDtype = 0;
    uint32_t rowLen = 0;
    uint32_t ubAvail = 0;
};

struct GMMSwigluQuantV2TilingDataParams {
    GMMSwigluQuantV2Params gmmSwigluQuantParams;
    TCubeTiling mmTilingData;
    GMMSwigluQuantSwigluParams swigluParams;
};
#pragma pack(pop)

#endif // GROUPED_MATMUL_SWIGLU_QUANT_V2_TILING_DATA_H
