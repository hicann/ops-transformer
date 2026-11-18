/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEGA_MOE_BLOCK_EPILOGUE_ACTIVATION_BF16_H
#define MEGA_MOE_BLOCK_EPILOGUE_ACTIVATION_BF16_H

#include "../../../arch35_92/common/mega_moe_arch_policy.h"
#ifdef MEGA_MOE_ARCH35_DEVICE
#include "../../../arch35_92/common/mega_moe_copy.h"
#include "../../../arch35_92/common/mega_moe_types.h"
#include "../../../arch35_92/blaze/epilogue/activation/activation_assign.h"

namespace MegaMoeImpl {

class Bf16ActivationEpilogue {
public:
    struct Params {
        GM_ADDR yGmAddr = nullptr;
        float clampLimit = 0.0f;
        uint8_t actMode = static_cast<uint8_t>(MegaMoeActMode::SWIGLU);
        uint8_t actSubMode = static_cast<uint8_t>(MegaMoeActSubMode::DEFAULT);
        float activationAlpha = 1.0f;
        float activationBeta = 1.0f;
        bool topkWeightsPrefetch = false;
    };

    __aicore__ inline void Init(const Params& params)
    {
        params_ = params;
        constexpr uint32_t inputBytes =
            MegaMoeBf16GmmConfig::TILE_M * MegaMoeBf16GmmConfig::TILE_N * sizeof(bfloat16_t);
        const bool isGelu = params.actMode == static_cast<uint8_t>(MegaMoeActMode::GELU);
        input_ = reinterpret_cast<__ubuf__ bfloat16_t*>(0U);
        output_ = isGelu ? input_ : reinterpret_cast<__ubuf__ bfloat16_t*>(inputBytes);
        weights_ = reinterpret_cast<__ubuf__ float*>(isGelu ? inputBytes : inputBytes + inputBytes / ACTIVATION_N_HALF);
    }

    __aicore__ inline void SetOutputRowOffset(uint64_t rowOffset)
    {
        outputRowOffset_ = rowOffset;
    }

    __aicore__ inline void RunTile(GM_ADDR metadata, uint32_t outputWidth, uint32_t row, uint32_t column, uint32_t rows,
                                   uint32_t columns)
    {
        const uint32_t divisor = GetActivationOutputDivisor(params_.actMode);
        const uint32_t outputStride = MegaMoeBf16GmmConfig::TILE_N / divisor;
        if (params_.topkWeightsPrefetch) {
            CopyGmToUb(weights_, metadata + static_cast<uint64_t>(row) * META_INFO_SIZE * sizeof(int32_t),
                       rows * META_INFO_SIZE * static_cast<uint32_t>(sizeof(int32_t)));
            PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID0>();
            RunActivation<true>(rows, columns, outputStride);
        } else {
            RunActivation<false>(rows, columns, outputStride);
        }
        PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID0>();
        const uint64_t destinationOffset = ((outputRowOffset_ + row) * outputWidth + column) * sizeof(bfloat16_t);
        CopyUbToGm(params_.yGmAddr + destinationOffset, output_,
                   CopyRows{static_cast<uint16_t>(rows), columns * static_cast<uint32_t>(sizeof(bfloat16_t)),
                            static_cast<uint64_t>(outputWidth) * sizeof(bfloat16_t),
                            outputStride * static_cast<uint32_t>(sizeof(bfloat16_t))});
        // ACK 由 Vector 发出，等待搬出完成后再允许 AIC 复用输入槽。
        PipeFence<PIPE_MTE3, PIPE_V, SYNC_EVENT_ID0>();
        PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID0>();
    }

private:
    template <bool TopkWeightsPrefetch>
    __aicore__ inline void RunActivation(uint32_t rows, uint32_t columns, uint32_t outputStride)
    {
        constexpr uint32_t inputStride = MegaMoeBf16GmmConfig::TILE_N;
        if (params_.actMode == static_cast<uint8_t>(MegaMoeActMode::GELU)) {
            Activation::GeluActivationTileContext<bfloat16_t> context{input_,
                                                                      output_,
                                                                      TopkWeightsPrefetch ? weights_ : nullptr,
                                                                      inputStride,
                                                                      static_cast<uint16_t>(rows),
                                                                      static_cast<uint16_t>(columns)};
            Activation::RunGeluActivation<bfloat16_t, TopkWeightsPrefetch>(context);
        } else {
            Activation::GatedActivationTileContext<bfloat16_t> context{};
            context.gate = input_;
            context.up = input_ + inputStride / ACTIVATION_N_HALF;
            context.output = output_;
            context.topkWeights = TopkWeightsPrefetch ? weights_ : nullptr;
            context.inputRowStrideElements = inputStride;
            context.outputRowStrideElements = outputStride;
            context.rowLoopCount = static_cast<uint16_t>(rows);
            Activation::ConfigureGatedActivationTail(context, static_cast<uint16_t>(columns));
            Activation::RunGatedActivation<bfloat16_t, TopkWeightsPrefetch>(
                context, params_.actMode, params_.actSubMode, params_.clampLimit, params_.activationAlpha,
                params_.activationBeta);
        }
    }

    Params params_{};
    uint64_t outputRowOffset_ = 0U;
    __ubuf__ bfloat16_t* input_ = nullptr;
    __ubuf__ bfloat16_t* output_ = nullptr;
    __ubuf__ float* weights_ = nullptr;
};

} // namespace MegaMoeImpl
#endif
#endif
