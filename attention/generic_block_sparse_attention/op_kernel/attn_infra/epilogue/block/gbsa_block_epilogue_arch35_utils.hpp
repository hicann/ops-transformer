/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_ARCH35_UTILS_HPP
#define GBSA_BLOCK_EPILOGUE_ARCH35_UTILS_HPP

#include "../../../attn_infra/gbsa_base_defs.hpp"

namespace NpuArch::Epilogue::Block {

namespace Mxfp8UbLayout {
constexpr uint32_t UB_QUERY_ROW_ALIGNMENT = 16U;
constexpr uint32_t UB_STATS_ROWS = 64U;
constexpr uint32_t UB_COMPUTE_STAGES = 2U;
constexpr uint32_t UB_DELTA_STAGES = 3U;
constexpr uint32_t SCORE_ELEMENT_BYTES = sizeof(float);
constexpr uint32_t PROBABILITY_ELEMENT_BYTES = sizeof(uint8_t);

// D192 reserves D256 capacity; active tensor strides still use the original D.
__aicore__ inline uint64_t Align(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

__aicore__ inline uint64_t BufferDim(uint64_t dim)
{
    // D192 and D256 share the D256 capacity tier. This is a reservation width,
    // never the active tensor width or a GM/scale address stride.
    return Align(dim, 128U);
}

__aicore__ inline uint32_t VectorRows(uint64_t qTile)
{
    // Each AIV reserves half the Q tile, rounded up to a 16-row capacity.
    return static_cast<uint32_t>(Align(qTile / 2U, UB_QUERY_ROW_ALIGNMENT));
}

__aicore__ inline uint64_t ScoreUbSlotBytes(uint64_t qTile, uint64_t tile)
{
    // All four supported M/KV tiers reserve 32 KiB per FP32 S slot per AIV.
    return VectorRows(qTile) * tile * SCORE_ELEMENT_BYTES;
}

__aicore__ inline uint32_t PvUbStages(uint64_t qTile)
{
    return qTile <= 64U ? UB_COMPUTE_STAGES : 1U;
}

__aicore__ inline uint64_t OutputUbSlotBytes(uint64_t dim, uint64_t qTile)
{
    return VectorRows(qTile) * BufferDim(dim) * sizeof(float);
}

__aicore__ inline uint64_t ValueScaleUbBytes(uint64_t dim)
{
    return BufferDim(dim) * sizeof(uint16_t);
}
} // namespace Mxfp8UbLayout

struct UBufTileHelper {
    // Statistics retain their maximum row capacity independently of S/P/LO tiling.
    static constexpr uint32_t STATS_ROW_NUM = 64U;
    uint32_t qBaseTilePerSubCore;
    uint32_t kvBaseTilePerSubCore;
    uint32_t embedPerSubCore;
    // PV slot capacity can exceed the active row width (MXFP8 D192 in D256).
    uint32_t loUbColCapacity;
    uint32_t loUbBufNum;
    uint32_t loUbRowNum;
    uint32_t kvOffsetUbBufNum = 1U;
    bool loGmTransit;
    uint32_t sStartOffset;
    uint32_t pStartOffset;
    uint32_t loStartOffset;
    uint32_t goStartOffset;
    uint32_t lmStartOffset;
    uint32_t gmStartOffset;
    uint32_t dmStartOffset;
    uint32_t llStartOffset;
    uint32_t glStartOffset;
    uint32_t lseStartOffset;
    uint32_t maskStartOffset;
    // Hybrid MXFP8 PV uses a packed E8M0 pair for each output channel.
    uint32_t mxVScaleStartOffset = 0U;

    __aicore__ inline UBufTileHelper() {}

    __aicore__ inline UBufTileHelper(uint32_t qs, uint32_t kvs, uint32_t d, uint32_t s, uint32_t p, uint32_t lo,
                                     uint32_t go, uint32_t lm, uint32_t gm, uint32_t dm, uint32_t ll, uint32_t gl,
                                     uint32_t lse, uint32_t mask, uint32_t loBufNum = 2U)
        : qBaseTilePerSubCore(qs),
          kvBaseTilePerSubCore(kvs),
          embedPerSubCore(d),
          loUbColCapacity(d),
          loUbBufNum(loBufNum),
          loUbRowNum(qs),
          loGmTransit(false),
          sStartOffset(s),
          pStartOffset(p),
          loStartOffset(lo),
          goStartOffset(go),
          lmStartOffset(lm),
          gmStartOffset(gm),
          dmStartOffset(dm),
          llStartOffset(ll),
          glStartOffset(gl),
          lseStartOffset(lse),
          maskStartOffset(mask)
    {}
};

} // namespace NpuArch::Epilogue::Block

#endif // GBSA_BLOCK_EPILOGUE_ARCH35_UTILS_HPP
