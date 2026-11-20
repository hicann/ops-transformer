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
 * \file init_output.h
 * \brief
 */

#ifndef INIT_OUTPUT_H
#define INIT_OUTPUT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace AttentionCommon {

template <typename T, uint32_t SYNC_ID, uint32_t POP_BUF_START_ADDR, uint32_t POP_BUF_ELE_SIZE,
          bool ENABLE_LOCK = false>
__aicore__ inline void InitOutput(GlobalTensor<T> outGm, uint64_t totalElementNum, uint32_t vecCoreNum, T initValue)
{
    if ASCEND_IS_AIV {
        uint64_t singleCoreMaxElementNum = (totalElementNum + vecCoreNum - 1U) / vecCoreNum;
        uint32_t tmpBlockIdx = AscendC::GetBlockIdx();
        uint64_t gmOffset = tmpBlockIdx * singleCoreMaxElementNum;
        if (gmOffset < totalElementNum) {
            uint64_t singleCoreActualElementNum = (gmOffset + singleCoreMaxElementNum > totalElementNum) ?
                                                      (totalElementNum - gmOffset) :
                                                      singleCoreMaxElementNum;
            LocalTensor<T> popBuffer =
                AscendC::LocalTensor<uint8_t>(TPosition::VECIN, POP_BUF_START_ADDR, POP_BUF_ELE_SIZE * sizeof(T))
                    .template ReinterpretCast<T>();

            if constexpr (ENABLE_LOCK) {
                Mutex::Lock<PIPE_V>(SYNC_ID);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_ID);

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_ID);
            }
            AscendC::Duplicate(popBuffer, initValue, POP_BUF_ELE_SIZE);
            if constexpr (ENABLE_LOCK) {
                Mutex::Unlock<PIPE_V>(SYNC_ID);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_ID);
            }

            if constexpr (ENABLE_LOCK) {
                Mutex::Lock<PIPE_MTE3>(SYNC_ID);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_ID);
            }
            uint64_t loopCnt = singleCoreActualElementNum / POP_BUF_ELE_SIZE;
            uint64_t tailSize = singleCoreActualElementNum - loopCnt * POP_BUF_ELE_SIZE;
            for (uint64_t loop = 0; loop < loopCnt; loop++) {
                AscendC::DataCopy(outGm[gmOffset], popBuffer, POP_BUF_ELE_SIZE);
                gmOffset += POP_BUF_ELE_SIZE;
            }
            if (tailSize > 0) {
                AscendC::DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = 1;
                dataCopyParams.blockLen = tailSize * sizeof(T);
                dataCopyParams.srcStride = 0;
                dataCopyParams.dstStride = 0;
                AscendC::DataCopyPad(outGm[gmOffset], popBuffer, dataCopyParams);
            }
            if constexpr (ENABLE_LOCK) {
                Mutex::Unlock<PIPE_MTE3>(SYNC_ID);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_ID);

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_ID);
            }
        }
    }
}

// Initialize contiguous head spans, reusing the same per-head values for each group.
template <typename T, uint32_t SYNC_ID, uint32_t POP_BUF_START_ADDR, uint32_t POP_BUF_ELE_SIZE>
__aicore__ inline void InitOutputPerHead(AscendC::GlobalTensor<T>& outGm, AscendC::GlobalTensor<T>& headValuesGm,
                                         uint64_t totalElementNum, uint64_t elementsPerHead, uint64_t headNum,
                                         uint32_t vecCoreNum)
{
    if ASCEND_IS_AIV {
        uint64_t coreSize = (totalElementNum + vecCoreNum - 1U) / vecCoreNum;
        uint64_t offset = AscendC::GetBlockIdx() * coreSize;
        uint64_t coreEnd = offset + coreSize;
        uint64_t end = coreEnd < totalElementNum ? coreEnd : totalElementNum;
        AscendC::LocalTensor<T> popBuffer =
            AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECIN, POP_BUF_START_ADDR, POP_BUF_ELE_SIZE * sizeof(T))
                .template ReinterpretCast<T>();

        while (offset < end) {
            uint64_t head = offset / elementsPerHead;
            uint64_t nextHeadOffset = (head + 1U) * elementsPerHead;
            uint64_t headEnd = nextHeadOffset < end ? nextHeadOffset : end;
            T value = headValuesGm.GetValue(head % headNum);
            AscendC::Mutex::Lock<PIPE_V>(SYNC_ID);
            AscendC::Duplicate(popBuffer, value, POP_BUF_ELE_SIZE);
            AscendC::Mutex::Unlock<PIPE_V>(SYNC_ID);
            AscendC::Mutex::Lock<PIPE_MTE3>(SYNC_ID);
            while (offset < headEnd) {
                uint64_t remaining = headEnd - offset;
                uint32_t count = remaining < POP_BUF_ELE_SIZE ? static_cast<uint32_t>(remaining) : POP_BUF_ELE_SIZE;
                AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPad(outGm[offset], popBuffer, copyParams);
                offset += count;
            }
            AscendC::Mutex::Unlock<PIPE_MTE3>(SYNC_ID);
        }
    }
}

} // namespace AttentionCommon

#endif // INIT_OUTPUT_H
