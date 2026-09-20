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
 * \file lightning_indexer_kernel_base_arch35.h
 * \brief Common kernel helpers shared by LightningIndexer and QuantLightningIndexer.
 */

#ifndef LIGHTNING_INDEXER_KERNEL_BASE_ARCH35_H
#define LIGHTNING_INDEXER_KERNEL_BASE_ARCH35_H

#include "../../../../lightning_indexer_v2/op_kernel/arch35/common/lightning_indexer_v2_kernel_base_arch35.h"
#include "../../lightning_indexer_common.h"

namespace LICommon {

template <typename ConstInfoT>
__aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                        ConstInfoT &constInfo, AscendC::GlobalTensor<uint32_t> &actualSeqLengthsGmQ,
                                        AscendC::GlobalTensor<uint32_t> &actualSeqLengthsGmK)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengthsK == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGmK.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsK, constInfo.actualLenDims);
    }
}

__aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                           AscendC::GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

} // namespace LICommon

#endif // LIGHTNING_INDEXER_KERNEL_BASE_ARCH35_H
