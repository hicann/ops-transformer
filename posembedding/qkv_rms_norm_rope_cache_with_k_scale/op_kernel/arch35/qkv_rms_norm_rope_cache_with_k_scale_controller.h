/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTROLLER_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTROLLER_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_basic_block.h"

namespace QkvRmsNormRopeCacheWithKScale {

template <uint32_t HEAD_DIM, uint32_t QKV_LAYOUT, uint32_t Q_OUT_LAYOUT, uint32_t ROPE_MODE, uint32_t K_CACHE_DTYPE,
          uint32_t Q_QUANT_MODE, uint32_t K_QUANT_MODE>
class QkvRmsNormRopeCacheWithKScaleController {
    static constexpr uint32_t TILE_PARAM_BUFFER_NUM = 2U;
    static_assert(HEAD_DIM == QKV_K_SCALE_HEAD_DIM_D128, "RoPE/M-RoPE Controller only supports D=128");
    static_assert(K_QUANT_MODE == QKV_K_SCALE_K_QUANT_MODE_PER_TOKEN_PER_HEAD,
                  "RoPE/M-RoPE Controller only supports PerTokenPerHead K quantization");

public:
    __aicore__ inline void Process(const GlobalTensors &tensors,
                                   const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        uint32_t cubeIndex = GetBlockIdx();
        if ASCEND_IS_AIV {
            cubeIndex /= QKV_K_SCALE_MIX_AIV_PER_AIC;
        }
        TokenRange tokenRange = {0U, 0U};
        MakeCoreTokenRange(tokenRange, tilingData, cubeIndex);

        TileParam tileParam[TILE_PARAM_BUFFER_NUM];
        InitTileParamBuffer(tileParam);
        basicBlock_.Init(tensors, tilingData);
        basicBlock_.PrepareBeforeLoop(tokenRange.begin, tokenRange.end);
        const uint64_t tileCount = ForEachTile(tilingData, tokenRange, tileParam);
        basicBlock_.End(tileParam[GetPreviousTileParamBufferId(tileCount)], tileCount);
    }

private:
    using BasicBlock =
        QkvRmsNormRopeCacheWithKScaleBasicBlock<QKV_LAYOUT, Q_OUT_LAYOUT, ROPE_MODE, K_CACHE_DTYPE, Q_QUANT_MODE>;

    struct TokenRange {
        uint64_t begin;
        uint64_t end;
    };

    __aicore__ inline void MakeCoreTokenRange(TokenRange &range,
                                              const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData,
                                              uint32_t cubeIndex) const
    {
        range.begin = 0U;
        range.end = 0U;
        if (tilingData->coreGroupNum == 0U || tilingData->coreTokenTile == 0U ||
            cubeIndex >= tilingData->coreGroupNum || tilingData->totalTokens == 0U) {
            return;
        }

        range.begin = cubeIndex * tilingData->coreTokenTile;
        range.end = MinU64(tilingData->totalTokens, range.begin + tilingData->coreTokenTile);
    }

    __aicore__ inline void ResetTileParam(TileParam &tile) const
    {
        tile.tokenOffset = 0U;
        tile.tokenSize = 0U;
        tile.cubeTokenSize = 0U;
        tile.cubeHalfTokenSize = 0U;
        tile.aivTokenOffset = 0U;
        tile.aivTokenSize = 0U;
        tile.aivBlockTokenOffset = 0U;
        tile.vHeadSize = 0U;
    }

    __aicore__ inline void FillTileParam(TileParam &tile, const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData,
                                         uint64_t tokenOffset, uint64_t tokenSize) const
    {
        ResetTileParam(tile);
        tile.tokenOffset = tokenOffset;
        if (tile.tokenOffset >= tilingData->totalTokens || tokenSize == 0U) {
            return;
        }

        tile.tokenSize = MinU64(MinU64(tokenSize, tilingData->tokenTile), tilingData->totalTokens - tile.tokenOffset);
        tile.cubeTokenSize = AlignUp(tile.tokenSize, QKV_K_SCALE_MIX_AIV_PER_AIC);
        tile.cubeHalfTokenSize = tile.cubeTokenSize / QKV_K_SCALE_MIX_AIV_PER_AIC;
        tile.vHeadSize = tilingData->kvHeadNum;
        if ASCEND_IS_AIC {
            return;
        }

        FillAivTileParam(tile, GetSubBlockIdx());
    }

    __aicore__ inline void FillAivTileParam(TileParam &tile, uint32_t aivLocalId) const
    {
        tile.aivBlockTokenOffset = aivLocalId * tile.cubeHalfTokenSize;
        tile.aivTokenOffset = tile.tokenOffset + tile.aivBlockTokenOffset;
        tile.aivTokenSize = aivLocalId == 0U ? tile.cubeHalfTokenSize : tile.tokenSize - tile.cubeHalfTokenSize;
    }

    __aicore__ inline uint64_t ForEachTile(const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData,
                                           const TokenRange &range, TileParam tileParam[TILE_PARAM_BUFFER_NUM])
    {
        uint64_t tileCount = 0U;
        for (uint64_t tokenOffset = range.begin; tokenOffset < range.end;) {
            const uint64_t tokenSize = MinU64(tilingData->tokenTile, range.end - tokenOffset);
            if (tokenSize == 0U) {
                break;
            }

            const uint32_t currentBufferId = GetCurrentTileParamBufferId(tileCount);
            const uint32_t previousBufferId = GetPreviousTileParamBufferId(tileCount);
            FillTileParam(tileParam[currentBufferId], tilingData, tokenOffset, tokenSize);
            basicBlock_.ComputeTile(tileParam[currentBufferId], tileParam[previousBufferId], tileCount,
                                    tokenOffset + tokenSize >= range.end);
            tokenOffset += tokenSize;
            ++tileCount;
        }
        return tileCount;
    }

    __aicore__ inline void InitTileParamBuffer(TileParam tileParam[TILE_PARAM_BUFFER_NUM]) const
    {
        for (uint32_t bufferId = 0U; bufferId < TILE_PARAM_BUFFER_NUM; ++bufferId) {
            ResetTileParam(tileParam[bufferId]);
        }
    }

    __aicore__ inline uint32_t GetCurrentTileParamBufferId(uint64_t tileCount) const
    {
        return static_cast<uint32_t>(tileCount % TILE_PARAM_BUFFER_NUM);
    }

    __aicore__ inline uint32_t GetPreviousTileParamBufferId(uint64_t tileCount) const
    {
        return static_cast<uint32_t>((tileCount + TILE_PARAM_BUFFER_NUM - 1U) % TILE_PARAM_BUFFER_NUM);
    }

    BasicBlock basicBlock_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#include "qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_controller.h"

namespace QkvRmsNormRopeCacheWithKScale {

template <>
class QkvRmsNormRopeCacheWithKScaleController<QKV_K_SCALE_HEAD_DIM_D128, QKV_K_SCALE_LAYOUT_TND, QKV_K_SCALE_LAYOUT_TND,
                                              QKV_K_SCALE_ROPE_MODE_MROPE, QKV_K_SCALE_CACHE_DTYPE_FP8_E4M3FN,
                                              QKV_K_SCALE_Q_QUANT_MODE_MX, QKV_K_SCALE_K_QUANT_MODE_MX>
    : public QkvRmsNormRopeCacheWithKScaleMropeMxController {};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CONTROLLER_H_
