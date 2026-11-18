/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEGA_MOE_EPILOGUE_BF16_H
#define MEGA_MOE_EPILOGUE_BF16_H

#include "../arch35_92/common/mega_moe_arch_policy.h"
#ifdef MEGA_MOE_ARCH35_DEVICE
#include "../arch35_92/common/mega_moe_copy.h"
#include "../arch35_92/common/mega_moe_types.h"
#include "../arch35_92/blaze/epilogue/activation/activation_dispatch.h"

namespace MegaMoeImpl {

template <bool TopkWeightsPrefetch>
class Bf16ActivationEpilogue {
public:
    struct Params {
        GM_ADDR yGmAddr = nullptr;
        float clampLimit = 0.0f;
        uint8_t actMode = static_cast<uint8_t>(MegaMoeActMode::SWIGLU);
        uint8_t actSubMode = static_cast<uint8_t>(MegaMoeActSubMode::DEFAULT);
        float activationAlpha = 1.0f;
        float activationBeta = 1.0f;
    };

    __aicore__ inline void Init(const Params& params)
    {
        params_ = params;
    }

    __aicore__ inline void SetOutputRowOffset(uint64_t rowOffset)
    {
        outputRowOffset_ = rowOffset;
    }

    // GMM1 的 GM 结果按完整 gate/up 两半存放，激活按小行块消费，不分配量化缓冲区。
    __aicore__ inline void RunTile(GM_ADDR input, GM_ADDR metadata, uint32_t fullWidth,
                                    uint32_t row, uint32_t column, uint32_t rows, uint32_t columns)
    {
        constexpr uint32_t tileN = MegaMoeBf16GmmConfig::TILE_N;
        constexpr uint32_t chunkRows = MegaMoeBf16GmmConfig::ACTIVATION_TILE_ROWS;
        constexpr uint32_t slabBytes = chunkRows * tileN * sizeof(bfloat16_t);
        auto* gate = reinterpret_cast<__ubuf__ bfloat16_t*>(0U);
        auto* up = reinterpret_cast<__ubuf__ bfloat16_t*>(slabBytes);
        auto* output = reinterpret_cast<__ubuf__ bfloat16_t*>(2U * slabBytes);
        auto* weights = reinterpret_cast<__ubuf__ float*>(3U * slabBytes);
        const uint32_t divisor = GetActivationOutputDivisor(params_.actMode);
        const uint32_t outputWidth = fullWidth / divisor;
        for (uint32_t offset = 0U; offset < rows; offset += chunkRows) {
            const uint32_t count = rows - offset < chunkRows ? rows - offset : chunkRows;
            const uint64_t sourceOffset =
                (static_cast<uint64_t>(row + offset) * fullWidth + column) * sizeof(bfloat16_t);
            const CopyRows copyIn{static_cast<uint16_t>(count), columns * static_cast<uint32_t>(sizeof(bfloat16_t)),
                                  static_cast<uint64_t>(fullWidth) * sizeof(bfloat16_t), tileN * sizeof(bfloat16_t)};
            PipeFence<PIPE_MTE3, PIPE_MTE2, SYNC_EVENT_ID0>();
            CopyGmToUb(gate, input + sourceOffset, copyIn);
            if (divisor == ACTIVATION_N_HALF) {
                CopyGmToUb(up, input + sourceOffset + static_cast<uint64_t>(outputWidth) * sizeof(bfloat16_t), copyIn);
            }
            if constexpr (TopkWeightsPrefetch) {
                CopyGmToUb(weights, metadata + static_cast<uint64_t>(row + offset) * META_INFO_SIZE * sizeof(int32_t),
                           count * META_INFO_SIZE * static_cast<uint32_t>(sizeof(int32_t)));
            }
            PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID0>();
            if (divisor == 1U) {
                Activation::GeluActivationTileContext<bfloat16_t> context{
                    gate, output, weights, tileN, static_cast<uint16_t>(count), static_cast<uint16_t>(columns)};
                Activation::RunGeluActivation<bfloat16_t, TopkWeightsPrefetch>(context);
            } else {
                Activation::GatedActivationTileContext<bfloat16_t> context{};
                context.gate = gate;
                context.up = up;
                context.output = output;
                context.topkWeights = TopkWeightsPrefetch ? weights : nullptr;
                context.inputRowStrideElements = tileN;
                context.outputRowStrideElements = tileN;
                context.rowLoopCount = static_cast<uint16_t>(count);
                Activation::ConfigureGatedActivationTail(context, static_cast<uint16_t>(columns));
                Activation::RunGatedActivation<bfloat16_t, TopkWeightsPrefetch>(context, params_.actMode,
                    params_.actSubMode, params_.clampLimit, params_.activationAlpha, params_.activationBeta);
            }
            PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID0>();
            const uint64_t destinationOffset =
                ((outputRowOffset_ + row + offset) * outputWidth + column) * sizeof(bfloat16_t);
            CopyUbToGm(params_.yGmAddr + destinationOffset, output,
                CopyRows{static_cast<uint16_t>(count), columns * static_cast<uint32_t>(sizeof(bfloat16_t)),
                         static_cast<uint64_t>(outputWidth) * sizeof(bfloat16_t), tileN * sizeof(bfloat16_t)});
        }
    }

private:
    Params params_{};
    uint64_t outputRowOffset_ = 0U;
};

} // namespace MegaMoeImpl
#endif
#endif
