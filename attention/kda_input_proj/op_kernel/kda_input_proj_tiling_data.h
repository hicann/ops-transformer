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
 * \file kda_input_proj_tiling_data.h
 * \brief
 */

#ifndef KDA_INPUT_PROJ_TILING_DATA_H
#define KDA_INPUT_PROJ_TILING_DATA_H

#include <cstdint>

namespace optiling {

#pragma pack(push, 8)

struct alignas(8) KdaInputProjBaseParams {
    uint32_t tSize = 0;      // token 个数 T
    uint32_t hiddenSize = 0; // 隐藏层维度，如 7168
    uint32_t qkvSize = 0;    // qkv 投影输出维度，如 4608
    uint32_t betaSize = 0;   // beta 投影输出维度，如 12
    uint32_t gateSize = 0;   // gate 投影输出维度，如 1536
    uint32_t gSize = 0;      // g 投影输出维度，如 1536
};

struct alignas(8) KdaInputProjMmBggParams {
    uint32_t tTile{0};
    uint32_t betaTile{0};
    uint32_t gateTile{0};
    uint32_t gTile{0};
    uint32_t tDim{0};
    uint32_t betaDim{0};
    uint32_t gateDim{0};
    uint32_t gDim{0};
    uint32_t numBetaTile{0};
    uint32_t numGateTile{0};
    uint32_t numGTile{0};
    uint32_t reserved{0}; // pad to 8-byte size (11 -> 12 uint32)
};

// 占位字段保证非空且 8 字节对齐；后续填真实字段时可替换
struct alignas(8) KdaInputProjMxQuantParams {
    uint64_t reserved{0};
};

struct alignas(8) KdaInputProjQmmQkvParams {
    uint32_t kL1 = 0;
    uint32_t scaleKL1 = 0;
    uint16_t baseM = 0;
    uint16_t baseN = 0;
    uint16_t baseK = 0;
    uint16_t mTailTile = 0;
    uint16_t nTailTile = 0;
    uint16_t mBaseTailSplitCnt = 1;
    uint16_t nBaseTailSplitCnt = 1;
    uint16_t mTailMain = 0;
    uint16_t nTailMain = 0;
    uint8_t nBufferNum = 0;
    uint8_t dbL0C = 0;
    uint8_t reserved[4] = {0}; // pad to 8-byte size
};

struct alignas(8) KdaInputProjSigmoidParams {
    uint64_t reserved{0};
};

struct alignas(8) KdaInputProjTilingData {
    KdaInputProjBaseParams baseParams;
    KdaInputProjMmBggParams mmBggParams;
    KdaInputProjMxQuantParams mxQuantParams;
    KdaInputProjQmmQkvParams qmmQkvParams;
    KdaInputProjSigmoidParams sigmoidParams;
};

#pragma pack(pop)

} // namespace optiling

#endif // KDA_INPUT_PROJ_TILING_DATA_H
