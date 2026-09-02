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
 * \file indexer_quant_cache_single_row_mx_fp4.h
 * \brief MX-FP4 (quant_mode=3) single-row kernel template.
 */

#ifndef INDEXER_QUANT_CACHE_SINGLE_ROW_MX_FP4_H
#define INDEXER_QUANT_CACHE_SINGLE_ROW_MX_FP4_H

#include "kernel_operator.h"
#include "indexer_quant_cache_base.h"
#include "indexer_quant_cache_mx_fp4_base.h"

namespace IndexerQuantCache {
using namespace AscendC;
template <typename T0, typename T1, typename T2>
class IndexerQuantCacheSingleRowMxFp4 {
public:
    __aicore__ inline IndexerQuantCacheSingleRowMxFp4() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR slotMapping, GM_ADDR cache, GM_ADDR cacheScale, GM_ADDR workspace,
                                const IndexerQuantCacheTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        cacheCol = (tilingData->d + 1) / 2; // fp4 打包后每行字节数

        xGm.SetGlobalBuffer((__gm__ T0 *)x);
        slotMappingGm.SetGlobalBuffer((__gm__ int32_t *)slotMapping);
        cacheGm.SetGlobalBuffer((__gm__ int8_t *)cache); // fp4 cache 按字节寻址
        cacheScaleGm.SetGlobalBuffer((__gm__ T2 *)cacheScale);

        // scale Que 最先 InitBuffer 以获得 UB 64B 对齐基址(满足 DIST_PACK_B16 的 64B 对齐要求)。
        pipe->InitBuffer(cacheScaleQue, 2,
                         tilingData->rowFactor * ScaleRowAlignMxFp4(tilingData->scaleCol) * sizeof(T2));
        pipe->InitBuffer(xQue, 2, tilingData->rowFactor * RoundUp<T0>(tilingData->d) * sizeof(T0));
        pipe->InitBuffer(cacheQue, 2, tilingData->rowFactor * RoundUp<int8_t>(cacheCol) * sizeof(int8_t));
        pipe->InitBuffer(maxExpBuf, RoundUp<uint16_t>(tilingData->scaleCol) * sizeof(uint16_t));
        pipe->InitBuffer(halfScaleBuf, RoundUp<uint16_t>(tilingData->scaleCol) * sizeof(uint16_t));
        pipe->InitBuffer(indexBuf, RoundUp<int32_t>(tilingData->rowFactor) * sizeof(int32_t));
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    }

    __aicore__ inline void Process()
    {
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop =
            (curBlockIdx == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;

        int64_t xGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->d;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            xLocal = xQue.template AllocTensor<T0>();
            int64_t curSlotIdx = curBlockIdx * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor;
            int64_t slot = slotMappingGm.GetValue(curSlotIdx);
            if (slot == -1) {
                continue;
            }
            CopyIn(xGm[xGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->d], xLocal, 1, tilingData->d);

            xQue.template EnQue(xLocal);
            xLocal = xQue.template DeQue<T0>();

            cacheLocal = cacheQue.template AllocTensor<int8_t>();
            cacheScaleLocal = cacheScaleQue.AllocTensor<T2>();
            LocalTensor<uint16_t> maxExpLocal = maxExpBuf.Get<uint16_t>();
            LocalTensor<uint16_t> halfScaleLocal = halfScaleBuf.Get<uint16_t>();

            VFProcessDynamicMxFp4Quant<T1, T2, T0>(cacheLocal, cacheScaleLocal, xLocal, maxExpLocal, halfScaleLocal, 1,
                                                   tilingData->d);

            xQue.template FreeTensor(xLocal);
            cacheQue.template EnQue(cacheLocal);
            cacheScaleQue.template EnQue(cacheScaleLocal);

            cacheLocal = cacheQue.template DeQue<int8_t>();
            cacheScaleLocal = cacheScaleQue.template DeQue<T2>();

            CopyOut(cacheLocal,
                    cacheGm[PagedSlotOffset(slot, tilingData->blockSize, tilingData->cacheBlockStride,
                                            tilingData->cacheRowStride)],
                    1, cacheCol);
            CopyOut(cacheScaleLocal,
                    cacheScaleGm[PagedSlotOffset(slot, tilingData->blockSize, tilingData->scaleBlockStride,
                                                 tilingData->scaleRowStride)],
                    1, tilingData->scaleCol);

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
    int64_t cacheCol = 0;
};

} // namespace IndexerQuantCache

#endif
