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
 * \file indexer_quant_cache_multi_row_mx_fp8.h
 * \brief
 */

#ifndef INDEXER_QUANT_CACHE_MULTI_ROW_MX_FP8_H
#define INDEXER_QUANT_CACHE_MULTI_ROW_MX_FP8_H

#include "kernel_operator.h"
#include "indexer_quant_cache_base.h"

namespace IndexerQuantCache {
using namespace AscendC;
constexpr int32_t FOUR_UNFOLD = 4; // 拷出按 4 行展开
template <typename T0, typename T1, typename T2>
class IndexerQuantCacheMultiRowMxFp8 {
public:
    __aicore__ inline IndexerQuantCacheMultiRowMxFp8() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR slotMapping, GM_ADDR cache, GM_ADDR cacheScale, GM_ADDR workspace,
                                const IndexerQuantCacheTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T0 *)x);
        slotMappingGm.SetGlobalBuffer((__gm__ int32_t *)slotMapping);
        cacheGm.SetGlobalBuffer((__gm__ T1 *)cache);
        cacheScaleGm.SetGlobalBuffer((__gm__ T2 *)cacheScale);

        roundScale = (tilingData->roundScale == 1);
        // MX-FP8 宽化按 128 元素读 x, x 输入 buffer 必须按 128 对齐以保证 128 宽读不越界 (尾块)
        pipe->InitBuffer(xQue, 2, tilingData->rowFactor * RoundUp<T0>(tilingData->d, 128) * sizeof(T0));
        pipe->InitBuffer(cacheQue, 2, tilingData->rowFactor * RoundUp<T1>(tilingData->d) * sizeof(T1));
        pipe->InitBuffer(cacheScaleQue, 2, tilingData->rowFactor * RoundUp<T2>(tilingData->scaleCol) * sizeof(T2));
        // MX-FP8 两函数式: fp32 中间 buffer (InvScale 写, Quant 读) + invScale 广播 buffer
        pipe->InitBuffer(yFp32Buf, tilingData->rowFactor * RoundUp<float>(tilingData->d) * sizeof(float));
        pipe->InitBuffer(invScaleBuf, tilingData->rowFactor * CeilDiv(tilingData->d, 128) * 16 * sizeof(float));
        yFp32Local = yFp32Buf.Get<float>();
        mxInvScaleLocal = invScaleBuf.Get<float>();
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    }

    __aicore__ inline void Process()
    {
        SetMaxValue();
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop =
            (curBlockIdx == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        int64_t tailRowFactor = (curBlockIdx == GetBlockNum() - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                     tilingData->tailRowFactorOfFormerBlock;
        int64_t xGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->d;
        int64_t xRowAlign = RoundUp<T0>(tilingData->d, 128); // MX-FP8 x 行按 128 对齐 (与量化读取步长一致)
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            int64_t xRowBase = xGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->d;
            int64_t slotGmBase = curBlockIdx * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor;
            xLocal = xQue.template AllocTensor<T0>();
            if (xRowAlign == tilingData->d) {
                CopyIn(xGm[xRowBase], xLocal, curRowFactor, tilingData->d);
            } else {
                for (int64_t r = 0; r < curRowFactor; r++) {
                    CopyIn(xGm[xRowBase + r * tilingData->d], xLocal[r * xRowAlign], 1, tilingData->d);
                }
            }
            xQue.template EnQue(xLocal);
            xLocal = xQue.template DeQue<T0>();

            cacheLocal = cacheQue.template AllocTensor<T1>();
            cacheScaleLocal = cacheScaleQue.AllocTensor<T2>();
            VFProcessMxFp8InvScale<T0, T2>(yFp32Local, cacheScaleLocal, mxInvScaleLocal, xLocal, maxValue, curRowFactor,
                                           tilingData->d);
            VFProcessMxFp8Quant<T1>(cacheLocal, yFp32Local, mxInvScaleLocal, curRowFactor, tilingData->d);
            xQue.template FreeTensor(xLocal);
            cacheQue.template EnQue(cacheLocal);
            cacheScaleQue.template EnQue(cacheScaleLocal);

            cacheLocal = cacheQue.template DeQue<T1>();
            cacheScaleLocal = cacheScaleQue.template DeQue<T2>();

            // ---- 优化1+3: 拷出 4 展开 + 直接从 GM 读 slot (无 UB slot 缓冲); slot==-1 丢弃; 保留分页 PagedSlotOffset
            // ----
            int64_t r = 0;
            for (; r + FOUR_UNFOLD <= curRowFactor; r += FOUR_UNFOLD) {
                for (int32_t k = 0; k < FOUR_UNFOLD; k++) {
                    int64_t curSlotIdx = slotMappingGm.GetValue(slotGmBase + r + k);
                    if (curSlotIdx == -1) {
                        continue;
                    }
                    CopyOut(cacheLocal[(r + k) * RoundUp<T1>(tilingData->d)],
                            cacheGm[PagedSlotOffset(curSlotIdx, tilingData->blockSize, tilingData->cacheBlockStride,
                                                    tilingData->cacheRowStride)],
                            1, tilingData->d);
                    CopyOut(cacheScaleLocal[(r + k) * RoundUp<T2>(tilingData->scaleCol)],
                            cacheScaleGm[PagedSlotOffset(curSlotIdx, tilingData->blockSize,
                                                         tilingData->scaleBlockStride, tilingData->scaleRowStride)],
                            1, tilingData->scaleCol);
                }
            }
            for (; r < curRowFactor; r++) {
                int64_t curSlotIdx = slotMappingGm.GetValue(slotGmBase + r);
                if (curSlotIdx == -1) {
                    continue;
                }
                CopyOut(cacheLocal[r * RoundUp<T1>(tilingData->d)],
                        cacheGm[PagedSlotOffset(curSlotIdx, tilingData->blockSize, tilingData->cacheBlockStride,
                                                tilingData->cacheRowStride)],
                        1, tilingData->d);
                CopyOut(cacheScaleLocal[r * RoundUp<T2>(tilingData->scaleCol)],
                        cacheScaleGm[PagedSlotOffset(curSlotIdx, tilingData->blockSize, tilingData->scaleBlockStride,
                                                     tilingData->scaleRowStride)],
                        1, tilingData->scaleCol);
            }
            cacheQue.template FreeTensor(cacheLocal);
            cacheScaleQue.template FreeTensor(cacheScaleLocal);
        }
    }

    __aicore__ inline void SetMaxValue()
    {
        if constexpr (IsSameType<T1, fp8_e5m2_t>::value) {
            maxValue = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE;
            fp8Max = FP8_E5M2_MAX_VALUE;
            fp8Min = FP8_E5M2_MIN_VALUE;
        } else if constexpr (IsSameType<T1, fp8_e4m3fn_t>::value) {
            maxValue = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
            fp8Max = FP8_E4M3FN_MAX_VALUE;
            fp8Min = FP8_E4M3FN_MIN_VALUE;
        }
    }

private:
    TPipe *pipe;
    const IndexerQuantCacheTilingData *tilingData;
    GlobalTensor<T0> xGm;
    GlobalTensor<int32_t> slotMappingGm;
    GlobalTensor<T1> cacheGm;
    GlobalTensor<T2> cacheScaleGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> cacheQue;
    TQue<QuePosition::VECOUT, 1> cacheScaleQue;
    TBuf<QuePosition::VECCALC> yFp32Buf;
    TBuf<QuePosition::VECCALC> invScaleBuf;

    LocalTensor<T0> xLocal;
    LocalTensor<T1> cacheLocal;
    LocalTensor<T2> cacheScaleLocal;
    LocalTensor<float> yFp32Local;
    LocalTensor<float> mxInvScaleLocal;
    int64_t validIdx = 0;
    float maxValue = 0.0f;
    float fp8Min = 0.0f;
    float fp8Max = 0.0f;
    bool roundScale = true;
};

} // namespace IndexerQuantCache

#endif
