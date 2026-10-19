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
 * \file mixed_quant_flash_attn_empty_kernel.h
 * \brief Empty-tensor kernel and sink handling for MixedQuantFlashAttn.
 */

#ifndef MIXED_QUANT_FLASH_ATTN_EMPTY_KERNEL_H_
#define MIXED_QUANT_FLASH_ATTN_EMPTY_KERNEL_H_

#include "kernel_operator.h"
#include "../mixed_quant_flash_attn_tiling_data.h"
#include "../utils/mixed_quant_flash_attn_common_def.h"
#include "mqfa_init_output.h"

using namespace AscendC;

namespace MixedQuantFlashAttnEmpty {

__aicore__ inline uint64_t GetSeqLen(GlobalTensor<int32_t>& seqLenGm, uint32_t size, uint32_t batchIdx,
                                     uint64_t defaultValue)
{
    if (size == 0) {
        return defaultValue;
    }
    return static_cast<uint64_t>(seqLenGm.GetValue(size == 1 ? 0 : batchIdx));
}

template <typename OUT_T, uint8_t IN_OUT_LAYOUT>
class MixedQuantFlashAttnEmptyKernel {
public:
    __aicore__ inline void Init(__gm__ uint8_t* attnOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* cuSeqLensQ,
                                __gm__ uint8_t* seqUsedQ, __gm__ uint8_t* seqUsedKv, __gm__ uint8_t* sinks,
                                __tiling_data_ptr__ optiling::MixedQuantFlashAttnTiling* tilingData)
    {
        attnOut_ = attnOut;
        softmaxLse_ = softmaxLse;
        cuSeqLensQ_ = cuSeqLensQ;
        seqUsedQ_ = seqUsedQ;
        seqUsedKv_ = seqUsedKv;
        sinks_ = sinks;
        tilingData_ = tilingData;
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const auto& baseParams = tilingData_->mixedQuantFlashAttnBaseParams;
            GlobalTensor<OUT_T> attentionOutGm;
            attentionOutGm.SetGlobalBuffer((__gm__ OUT_T*)attnOut_);
            constexpr uint32_t outputBufferElements = BUFFER_SIZE_BYTE_8K / sizeof(OUT_T);
            AttentionCommon::InitOutput<OUT_T, EVENT_ID0, 0, outputBufferElements>(
                attentionOutGm, baseParams.totalOutputSize, baseParams.coreNum, static_cast<OUT_T>(0));

            if (!baseParams.isSoftMaxLseEnable || baseParams.totalLseSize == 0) {
                return;
            }

            GlobalTensor<float> softmaxLseGm;
            softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse_);
            constexpr uint32_t lseBufferElements = BUFFER_SIZE_BYTE_8K / sizeof(float);
            AttentionCommon::InitOutput<float, EVENT_ID1, BUFFER_SIZE_BYTE_8K, lseBufferElements>(
                softmaxLseGm, baseParams.totalLseSize, baseParams.coreNum, static_cast<float>(3e+99));
            SyncAll();
        }
    }

private:
    __gm__ uint8_t* attnOut_ = nullptr;
    __gm__ uint8_t* softmaxLse_ = nullptr;
    __gm__ uint8_t* cuSeqLensQ_ = nullptr;
    __gm__ uint8_t* seqUsedQ_ = nullptr;
    __gm__ uint8_t* seqUsedKv_ = nullptr;
    __gm__ uint8_t* sinks_ = nullptr;
    __tiling_data_ptr__ optiling::MixedQuantFlashAttnTiling* tilingData_ = nullptr;
};

} // namespace MixedQuantFlashAttnEmpty

#endif // MIXED_QUANT_FLASH_ATTN_EMPTY_KERNEL_H_
