/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLASH_ATTENTION_SCORE_GRAD_REDUCE_QUANT_ARCH92_H_
#define FLASH_ATTENTION_SCORE_GRAD_REDUCE_QUANT_ARCH92_H_
#include "flash_attention_score_grad_quant_config_arch92.h"
namespace FagBaseApi {
namespace quant {
template <typename Tiling>
__aicore__ inline bool BalancedShape(Tiling data)
{
    const auto& p = data->s1s2BNGS1S2BaseParams;
    return FagBaseApi::quant::UseBalanced(static_cast<uint64_t>(p.b) * p.n2, p.s1, p.s2, p.coreNum);
}

// Each core owns a contiguous Q-block interval, intersecting at most two heads.
// Main writes two private head slots per core. This phase reduces only slots that
// actually exist, in ascending core order, so unused scratch needs no initialization.
// This reduction is required for both tiling keys, including NO_DETER.
template <typename Tiling>
__aicore__ inline void ReduceBalancedPartials(GM_ADDR workspace, Tiling data, TPipe* pipe)
{
    if ASCEND_IS_AIV {
        const auto& p = data->s1s2BNGS1S2BaseParams;
        uint64_t bn = static_cast<uint64_t>(p.b) * p.n2;
        constexpr uint64_t qRows = 512;
        uint64_t m = (static_cast<uint64_t>(p.s1) + qRows - 1) / qRows;
        constexpr uint32_t tileRows = 64;
        constexpr uint32_t elements = FagBaseApi::quant::BALANCED_REDUCE_ELEMS;
        uint64_t total = bn * m, rowsPerHead = (static_cast<uint64_t>(p.s2) + tileRows - 1) / tileRows;
        uint64_t partialBase = FagBaseApi::quant::PartialOffset(data->postTilingData.sfmgWorkSpaceOffset, bn, p.s1);
        TBuf<> accStorage, inStorage;
        pipe->InitBuffer(accStorage, elements * sizeof(float));
        pipe->InitBuffer(inStorage, elements * sizeof(float));
        auto acc = accStorage.template Get<float>();
        auto input = inStorage.template Get<float>();

        GlobalTensor<float> scratch[2], target[2];
        scratch[0].SetGlobalBuffer((__gm__ float*)(workspace + partialBase));
        scratch[1].SetGlobalBuffer((__gm__ float*)(workspace + partialBase) +
                                   static_cast<uint64_t>(p.coreNum) * 2 * p.s2 * 128);
        target[0].SetGlobalBuffer((__gm__ float*)(workspace + data->postTilingData.dkWorkSpaceOffset));
        target[1].SetGlobalBuffer((__gm__ float*)(workspace + data->postTilingData.dvWorkSpaceOffset));
        for (uint64_t tile = GetBlockIdx(); tile < bn * rowsPerHead; tile += p.coreNum) {
            uint64_t head = tile / rowsPerHead, row = tile % rowsPerHead * tileRows;
            uint32_t rows = Min<uint64_t>(tileRows, static_cast<uint64_t>(p.s2) - row);
            uint32_t count = rows * 128;
            uint64_t firstCore = 0, endCore = p.coreNum;

            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = count * sizeof(float);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            for (uint32_t kind = 0; kind < 2; ++kind) {
                Duplicate(acc, 0.0f, count);
                for (uint64_t c = firstCore; c < endCore; ++c) {
                    uint64_t begin = FagBaseApi::quant::TaskBegin(total, c, p.coreNum);
                    uint64_t end = FagBaseApi::quant::TaskBegin(total, c + 1, p.coreNum);
                    if (begin >= end || begin >= (head + 1) * m || end <= head * m)
                        continue;
                    uint64_t slot = c * 2 + head - begin / m;
                    uint64_t offset = (slot * p.s2 + row) * 128;
                    DataCopyPad(input, scratch[kind][offset], copyParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);
                    Add(acc, acc, input, count);
                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);
                }

                SetFlag<HardEvent::V_MTE3>(1);
                WaitFlag<HardEvent::V_MTE3>(1);
                uint64_t offset = ((head / p.n2 * p.s2 + row) * p.n2 + head % p.n2) * 128;
                DataCopyExtParams copy;
                copy.blockCount = static_cast<uint16_t>(rows);
                copy.blockLen = 128 * sizeof(float);
                copy.srcStride = 0;
                copy.dstStride = (p.n2 * 128 - 128) * sizeof(float);
                copy.rsv = 0;
                DataCopyPad(target[kind][offset], acc, copy);
                SetFlag<HardEvent::MTE3_V>(1);
                WaitFlag<HardEvent::MTE3_V>(1);
            }
        }
    }
    SyncAll<false>();
}
} // namespace quant
} // namespace FagBaseApi
#endif
