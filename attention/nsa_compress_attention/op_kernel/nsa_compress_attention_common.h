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
 * \file flash_attention_score_common.h
 * \brief
 */

#ifndef NSA_COMPRESS_ATTENTION_COMMON_H
#define NSA_COMPRESS_ATTENTION_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using AscendC::AdjustSoftMaxRes;
using AscendC::AIC;
using AscendC::BinaryRepeatParams;
using AscendC::Cast;
using AscendC::CopyRepeatParams;
using AscendC::Div;
using AscendC::Duplicate;
using AscendC::GetBlockIdx;
using AscendC::GetSubBlockIdx;
using AscendC::GetUserWorkspace;
using AscendC::Nd2NzParams;
using AscendC::RoundMode;
using AscendC::SelectWithBytesMask;
using AscendC::SelectWithBytesMaskShapeInfo;
using AscendC::SoftmaxFlashV2;
using AscendC::SoftMaxShapeInfo;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TSCM;

constexpr MatmulConfig CFG_EXCEED = GetNormalConfig(true);
constexpr MatmulConfig CFG_IBSHARE_EXCEED = GetIBShareNormConfig(true);
constexpr uint64_t BLOCK_BYTE = 32;
constexpr uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr uint32_t NEGATIVE_MIN_VAULE_FP16 = 0xFBFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP32 = 0x7F7FFFFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP16 = 0x7BFF;

template <typename T>
__aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar)
{
    if constexpr (IsSameType<T, float>::value) {
        uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
        uint32_t tmp2 = POSITIVE_MAX_VALUE_FP32;
        negativeScalar = *((float *)&tmp1);
        positiveScalar = *((float *)&tmp2);
    } else {
        uint16_t tmp1 = NEGATIVE_MIN_VAULE_FP16;
        uint16_t tmp2 = POSITIVE_MAX_VALUE_FP16;
        negativeScalar = *((half *)&tmp1);
        positiveScalar = *((half *)&tmp2);
    }
}

template <typename T>
__aicore__ inline void DataCopy2D(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGlobal, const uint32_t d0,
                                  const uint32_t d1, const uint32_t orgD1, uint64_t paddingValue = 0)
{
    if (d1 % (BLOCK_BYTE / sizeof(T)) == 0 && orgD1 % (BLOCK_BYTE / sizeof(T)) == 0) {
        auto d1Blocks = math::Ceil(d1 * sizeof(T), BLOCK_BYTE);
        auto orgD1Blocks = math::Ceil(orgD1 * sizeof(T), BLOCK_BYTE);
        DataCopyParams copyParams(d0, d1Blocks, orgD1Blocks - d1Blocks, 0);
        DataCopy(dstLocal, srcGlobal, copyParams);
    } else {
        auto d1Bytes = d1 * sizeof(T);
        auto d1Aligned = math::Align(static_cast<int64_t>(d1), static_cast<int64_t>(BLOCK_BYTE / sizeof(T)));
        DataCopyParams copyParams(static_cast<uint16_t>(d0), static_cast<uint16_t>(d1Bytes),
                                  orgD1 * sizeof(T) - d1Bytes, 0);
        DataCopyPadParams padParams(true, 0, static_cast<uint8_t>(d1Aligned - d1), paddingValue);
        DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
    }
}

#endif // NSA_COMPRESS_ATTENTION_COMMON_H
