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
 * \file flash_attention_score_grad_kernel_base_quant_arch92.h
 * \brief Dense BSND offset helpers for the arch92 quant pipeline.
 */
#ifndef FLASH_ATTENTION_SCORE_GRAD_KERNEL_BASE_QUANT_ARCH92_H_
#define FLASH_ATTENTION_SCORE_GRAD_KERNEL_BASE_QUANT_ARCH92_H_

#include "flash_attention_score_grad_common_quant_arch92.h"

namespace FagBaseApi {
namespace quant {
template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreGradKernelBase {
public:
    ARGS_TRAITS;

    __aicore__ inline void SyncALLCores()
    {
        SyncAll<false>();
    }

protected:
    // These are the BSNGD branches of the arch35 helpers. The quant tiler rejects other layouts.
    __aicore__ inline int64_t GetQueryOffset(FagRunInfo& runInfo)
    {
        const auto& info = constInfo.commonConstInfo;
        const auto& run = runInfo.commonRunInfo;
        return run.boIdx * info.n2GS1D + run.s1oIdx * static_cast<uint32_t>(s1TemplateType) * info.n2GD +
               run.n2oIdx * info.gD + run.goIdx * info.dSize;
    }

    __aicore__ inline int64_t GetKeyOffset(FagRunInfo& runInfo)
    {
        const auto& info = constInfo.commonConstInfo;
        const auto& run = runInfo.commonRunInfo;
        return run.boIdx * info.n2S2D + runInfo.s2CvBegin * info.n2D + run.n2oIdx * info.dSize;
    }

    uint32_t vBlockIdx = 0;
    uint32_t cBlockIdx = 0;
    uint32_t vSubBlockIdx = 0;
    FagConstInfo constInfo;
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};
} // namespace quant
} // namespace FagBaseApi
#endif
