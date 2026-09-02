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
 * \file indexer_quant_cache_multi_row_mx_fp4.h
 * \brief MX-FP4 (quant_mode=3) multi-row kernel template.
 */

#ifndef INDEXER_QUANT_CACHE_MULTI_ROW_MX_FP4_H
#define INDEXER_QUANT_CACHE_MULTI_ROW_MX_FP4_H

#include "kernel_operator.h"
#include "indexer_quant_cache_base.h"
#include "indexer_quant_cache_mx_fp4_base.h"

namespace IndexerQuantCache {
using namespace AscendC;
template <typename T0, typename T1, typename T2>
class IndexerQuantCacheMultiRowMxFp4 {
public:
    __aicore__ inline IndexerQuantCacheMultiRowMxFp4() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR slotMapping, GM_ADDR cache, GM_ADDR cacheScale, GM_ADDR workspace,
                                const IndexerQuantCacheTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        cacheCol = (tilingData->d + 1) / 2;

        xGm.SetGlobalBuffer((__gm__ T0 *)x);
        slotMappingGm.SetGlobalBuffer((__gm__ int32_t *)slotMapping);
        cacheGm.SetGlobalBuffer((__gm__ int8_t *)cache); // fp4 cache 按字节寻址
        cacheScaleGm.SetGlobalBuffer((__gm__ T2 *)cacheScale);

        pipe->InitBuffer(cacheScaleQue, 2,
                         tilingData->rowFactor * ScaleRowAlignMxFp4(tilingData->scaleCol) * sizeof(T2));
        pipe->InitBuffer(xQue, 2, tilingData->rowFactor * RoundUp<T0>(tilingData->d) * sizeof(T0));
        pipe->InitBuffer(cacheQue, 2, tilingData->rowFactor * RoundUp<int8_t>(cacheCol) * sizeof(int8_t));
        pipe->InitBuffer(maxExpBuf, RoundUp<uint16_t>(tilingData->scaleCol) * sizeof(uint16_t));
        pipe->InitBuffer(halfScaleBuf, RoundUp<uint16_t>(tilingData->scaleCol) * sizeof(uint16_t));
        pipe->InitBuffer(indexBuf, RoundUp<int32_t>(tilingData->rowFactor) * sizeof(int32_t));
        indexLocal = indexBuf.Get<int32_t>();
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    }

    __aicore__ inline void Process()
    {
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop =
            (curBlockIdx == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        int64_t tailRowFactor = (curBlockIdx == GetBlockNum() - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                     tilingData->tailRowFactorOfFormerBlock;
        int64_t xGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->d;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            xLocal = xQue.template AllocTensor<T0>();
            validIdx = 0;
            for (int64_t rowInnerIdx = 0; rowInnerIdx < curRowFactor; rowInnerIdx++) {
                int64_t curSlotIdx =
                    curBlockIdx * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor + rowInnerIdx;
                int64_t slot = slotMappingGm.GetValue(curSlotIdx);
                if (slot == -1) {
                    continue;
                }
                CopyIn(xGm[xGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->d +
                           rowInnerIdx * tilingData->d],
                       xLocal[validIdx * RoundUp<T0>(tilingData->d)], 1, tilingData->d);
                indexLocal.SetValue(validIdx, slot);
                validIdx++;

                event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventId);
                WaitFlag<HardEvent::S_MTE3>(eventId);
            }
            xQue.template EnQue(xLocal);
            xLocal = xQue.template DeQue<T0>();

            cacheLocal = cacheQue.template AllocTensor<int8_t>();
            cacheScaleLocal = cacheScaleQue.AllocTensor<T2>();
            LocalTensor<uint16_t> maxExpLocal = maxExpBuf.Get<uint16_t>();
            LocalTensor<uint16_t> halfScaleLocal = halfScaleBuf.Get<uint16_t>();

            VFProcessDynamicMxFp4Quant<T1, T2, T0>(cacheLocal, cacheScaleLocal, xLocal, maxExpLocal, halfScaleLocal,
                                                   validIdx, tilingData->d);

            xQue.template FreeTensor(xLocal);
            cacheQue.template EnQue(cacheLocal);
            cacheScaleQue.template EnQue(cacheScaleLocal);

            cacheLocal = cacheQue.template DeQue<int8_t>();
            cacheScaleLocal = cacheScaleQue.template DeQue<T2>();

            for (int64_t curValidIdx = 0; curValidIdx < validIdx; curValidIdx++) {
                int64_t curSlotIdx = indexLocal.GetValue(curValidIdx);
                CopyOut(cacheLocal[curValidIdx * RoundUp<int8_t>(cacheCol)],
                        cacheGm[PagedSlotOffset(curSlotIdx, tilingData->blockSize, tilingData->cacheBlockStride,
                                                tilingData->cacheRowStride)],
                        1, cacheCol);
                CopyOut(cacheScaleLocal[curValidIdx * ScaleRowAlignMxFp4(tilingData->scaleCol)],
                        cacheScaleGm[PagedSlotOffset(curSlotIdx, tilingData->blockSize, tilingData->scaleBlockStride,
                                                     tilingData->scaleRowStride)],
                        1, tilingData->scaleCol);
            }
            cacheQue.template FreeTensor(cacheLocal);
            cacheScaleQue.template FreeTensor(cacheScaleLocal);
        }
    }

private:
    TPipe *pipe;
    const IndexerQuantCacheTilingData *tilingData;
    GlobalTensor<T0> xGm;
    GlobalTensor<int32_t> slotMappingGm;
    GlobalTensor<int8_t> cacheGm;
    GlobalTensor<T2> cacheScaleGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> cacheQue;
    TQue<QuePosition::VECOUT, 1> cacheScaleQue;
    TBuf<QuePosition::VECCALC> maxExpBuf;
    TBuf<QuePosition::VECCALC> halfScaleBuf;
    TBuf<QuePosition::VECCALC> indexBuf;

    LocalTensor<T0> xLocal;
    LocalTensor<int8_t> cacheLocal;
    LocalTensor<T2> cacheScaleLocal;
    LocalTensor<int32_t> indexLocal;
    int64_t validIdx = 0;
    int64_t cacheCol = 0;
};

} // namespace IndexerQuantCache

#endif
