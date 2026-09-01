/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_CONTROLLER_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_CONTROLLER_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_struct.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_vec.h"

namespace QkvRmsNormRopeCacheWithKScale {

/*
 * Current tile schedule; time advances from left to right:
 *
 *   Controller : FillTile ------------------------------------------------ ReleaseInput
 *   MTE2/Scalar:        LoadTile(slot n) -> slot/input READY
 *   Vector     :                            ProcessV -> ProcessK -> ProcessQ
 *   MTE3       :                               V store -> K stores -> Q stores
 *
 * The current controller serializes V, K, and Q. Each stage owns a distinct
 * single-buffered output region and finishes its MTE3 write before Vector may
 * overwrite that stage's UB. Q is the final reader of qkv/rawCosSin, so the
 * input slot is returned only after ProcessQ. inputSlot alternates between two
 * physical UB slots, but this controller does not issue a next-tile preload;
 * the READY/FREE protocol is retained as the ownership contract for each slot.
 * Detailed producer/consumer event directions are documented beside the event
 * IDs in the Vector implementation.
 */
class QkvRmsNormRopeCacheWithKScaleMropeMxController {
public:
    __aicore__ inline void Process(const GlobalTensors &tensors,
                                   const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        TokenRange range{};
        MakeCoreTokenRange(range, tilingData, GetBlockIdx(), AscendC::GetBlockNum());
        QkvRmsNormRopeCacheWithKScaleMropeMxVec vec(tensors, tilingData, range.begin, range.end);
        vec.PrepareBeforeLoop();

        MropeMxTileDesc tile{};
        uint64_t ordinal = 0U;
        uint64_t tokenBegin = range.begin;
        while (FillTile(tile, range, tilingData, tokenBegin, ordinal)) {
            vec.LoadTile(tile);
            vec.WaitTileReady(tile);
            vec.ProcessV(tile);
            vec.ProcessK(tile);
            vec.ProcessQ(tile);
            vec.ReleaseInput(tile);
            tokenBegin += tile.tokenCount;
            ++ordinal;
        }
    }

private:
    struct TokenRange {
        uint64_t begin;
        uint64_t end;
    };

    __aicore__ inline void MakeCoreTokenRange(TokenRange &range,
                                              const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData,
                                              uint32_t aivIndex, uint32_t activeAiv) const
    {
        // Host tiling rejects T == 0 and aivNum == 0, then launches exactly
        // min(T, aivNum) AIV blocks. Therefore activeAiv is nonzero here.
        const uint64_t base = tilingData->totalTokens / activeAiv;
        const uint64_t remainder = tilingData->totalTokens % activeAiv;
        range.begin = static_cast<uint64_t>(aivIndex) * base + MinU64(aivIndex, remainder);
        range.end = range.begin + base + (aivIndex < remainder ? 1U : 0U);
    }

    __aicore__ inline bool FillTile(MropeMxTileDesc &tile, const TokenRange &range,
                                    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, uint64_t tokenBegin,
                                    uint64_t tileOrdinal) const
    {
        if (tokenBegin >= range.end) {
            return false;
        }
        const uint64_t relative = tokenBegin - range.begin;
        const uint64_t windowBegin =
            range.begin + relative / MROPE_MX_POSITION_WINDOW_TOKENS * MROPE_MX_POSITION_WINDOW_TOKENS;
        const uint64_t windowEnd = MinU64(windowBegin + MROPE_MX_POSITION_WINDOW_TOKENS, range.end);
        const uint64_t tileEnd = MinU64(MinU64(tokenBegin + tilingData->tokenTile, windowEnd), range.end);
        tile.tokenBegin = tokenBegin;
        tile.tokenCount = static_cast<uint32_t>(tileEnd - tokenBegin);
        tile.inputSlot = static_cast<uint32_t>(tileOrdinal & 1U);
        return tile.tokenCount != 0U;
    }
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_MROPE_MX_CONTROLLER_H_
