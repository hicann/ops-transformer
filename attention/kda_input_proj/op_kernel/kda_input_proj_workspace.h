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
 * \file kda_input_proj_workspace.h
 * \brief User-workspace layout shared by host tiling and device orchestration.
 *
 * Stage1 DynamicMxQuant writes quantized x (fp8e4m3, [T, K]) and xScale (e8m0,
 * [T, CeilDiv(K, 64) * 2]). Stage2 QMM consumes the same slices. Offsets are
 * owned here so mx_quant and qmm_qkv do not slice workspace independently.
 *
 * 这些偏移量在 Host tiling 与 Device kernel 两侧共用，必须是 constexpr：
 * 写成不带 __aicore__ 的 static inline 时它们是 host-only 函数，Device 侧
 * KdaInputProjKernel 构造函数调用它们会让整条 op.Process() 调用链被裁掉
 * （编译无报错，但 mix_aic/mix_aiv 只剩入口壳）。
 */

#ifndef KDA_INPUT_PROJ_WORKSPACE_H
#define KDA_INPUT_PROJ_WORKSPACE_H

#include <cstdint>

namespace KdaInputProj {

struct KdaInputProjWorkspace {
    static constexpr uint64_t MXFP_GROUP_K = 64UL;
    static constexpr uint64_t MXFP_SCALE_MULTI = 2UL;
    static constexpr uint64_t ALIGN = 512UL;

    static constexpr uint64_t CeilDiv(uint64_t a, uint64_t b)
    {
        return b == 0UL ? 0UL : (a + b - 1UL) / b;
    }

    static constexpr uint64_t CeilAlign(uint64_t value, uint64_t align)
    {
        return align == 0UL ? value : CeilDiv(value, align) * align;
    }

    // Matches Blaze MX scaleKLen: CeilDiv(K, 64) * 2.
    static constexpr uint64_t ScaleKLen(uint32_t hiddenSize)
    {
        return CeilDiv(static_cast<uint64_t>(hiddenSize), MXFP_GROUP_K) * MXFP_SCALE_MULTI;
    }

    static constexpr uint64_t QuantXBytes(uint32_t tSize, uint32_t hiddenSize)
    {
        return static_cast<uint64_t>(tSize) * static_cast<uint64_t>(hiddenSize);
    }

    static constexpr uint64_t XScaleBytes(uint32_t tSize, uint32_t hiddenSize)
    {
        return static_cast<uint64_t>(tSize) * ScaleKLen(hiddenSize);
    }

    static constexpr uint64_t OffsetQuantX()
    {
        return 0UL;
    }

    static constexpr uint64_t OffsetScaleX(uint32_t tSize, uint32_t hiddenSize)
    {
        return CeilAlign(QuantXBytes(tSize, hiddenSize), ALIGN);
    }

    static constexpr uint64_t TotalBytes(uint32_t tSize, uint32_t hiddenSize)
    {
        return CeilAlign(OffsetScaleX(tSize, hiddenSize) + XScaleBytes(tSize, hiddenSize), ALIGN);
    }
};

} // namespace KdaInputProj

#endif // KDA_INPUT_PROJ_WORKSPACE_H
