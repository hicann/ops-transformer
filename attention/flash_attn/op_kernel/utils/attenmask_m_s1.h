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
 * \file attenmask_m_s1.h
 * \brief attention mask 不合轴（S1_ONLY）处理
 */

#ifndef ATTENMASK_M_S1_H
#define ATTENMASK_M_S1_H

#include "kernel_operator.h"
#include "flash_attn_common_def.h"

// 不合轴：每个任务只处理单个 G 头，M 轴为纯 S1，无合轴 gs1 坐标
struct MaskInfoUnmerged {
    uint32_t s1Size;
    uint32_t s2StartIdx;
    uint32_t s2dealNum;
    uint32_t s2Size;
    uint32_t nBaseSize;

    int64_t nextToken = 0;

    uint32_t attenMaskS1Stride;
    uint32_t attenMaskDstStride = 0;

    FAMaskMode maskMode;

    uint32_t s1StartIdx;
    uint32_t s1dealNum;
};

__aicore__ inline bool IsSkipAttentionmask(MaskInfoUnmerged& info)
{
    int32_t s1StartIdx = info.s1StartIdx;

    int64_t nextToken = 0;
    if (info.maskMode == FAMaskMode::CAUSAL) {
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算Token
    } else if (info.maskMode == FAMaskMode::BAND) {
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    }

    if (static_cast<int64_t>(info.s2StartIdx + info.s2dealNum) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
        return true;
    }
    return false;
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompressDn(MaskInfoUnmerged& info, uint32_t s1StartIdx)
{
    int64_t nextToken = 0;
    if (info.maskMode == FAMaskMode::CAUSAL) {
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算token
    } else if (info.maskMode == FAMaskMode::BAND) {
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    }
    uint64_t offset = 0;
    int64_t delta = nextToken + s1StartIdx - info.s2StartIdx;
    uint32_t attenMaskSizeAlign = info.s1dealNum;
    if (delta >= 0) {
        offset = (delta) < static_cast<int64_t>(info.nBaseSize) ? (delta) : info.nBaseSize; // min (-delta, s1Size)
    } else {
        offset = (-delta < static_cast<int64_t>(attenMaskSizeAlign) ? -delta : attenMaskSizeAlign) *
                 info.attenMaskS1Stride; // min(delta, s2inner)
    }
    return offset + 1;
}

template <typename T, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyInDn(LocalTensor<T>& attenMaskUb, GlobalTensor<T>& srcGmAddr,
                                             MaskInfoUnmerged& info)
{
    uint64_t maskOffset = ComputeAttenMaskOffsetCompressDn(info, info.s1StartIdx);
    if (info.s2Size % 32U == 0) { // 32： datablock size is 32B
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = info.s2dealNum;
        dataCopyParams.blockLen = (info.s1dealNum) >> 5;
        dataCopyParams.srcStride = (info.attenMaskS1Stride - info.s1dealNum) >> 5;
        dataCopyParams.dstStride = info.attenMaskDstStride;
        DataCopy(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams);
    } else {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = info.s2dealNum;
        dataCopyParams.blockLen = info.s1dealNum;
        dataCopyParams.srcStride = info.attenMaskS1Stride - info.s1dealNum;
        dataCopyParams.dstStride = info.attenMaskDstStride;
        AscendC::DataCopyPadExtParams<T> padParams;
        DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);
    }
}

#endif
