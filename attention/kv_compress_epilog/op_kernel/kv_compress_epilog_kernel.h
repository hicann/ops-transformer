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
 * \file kv_compress_epilog_kernel.h
 * \brief KV compress epilog kernel implementation
 */

#ifndef KV_COMPRESS_EPILOG_KERNEL_H
#define KV_COMPRESS_EPILOG_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kv_compress_epilog_common.h"

using namespace AscendC;

namespace KvCompressEpilogOps {

constexpr int32_t FOUR_UNFOLD = 4; // 拷出按 4 行展开

template <typename T0, typename U, typename T1>
class KvCompressEpilog {
public:
    __aicore__ inline KvCompressEpilog(TPipe *pipe)
        : pipe_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR slotMapping, GM_ADDR kvCache,
                                const KvCompressEpilogTilingData *tilingData);

    __aicore__ inline void Process();

    __aicore__ inline void SetMaxValue();

    __aicore__ inline void Scatter(const LocalTensor<T1> &kvLocal, int64_t slotBase, int64_t rf, int64_t blkSize,
                                   int64_t blkStride, int64_t rowStride, uint32_t cacheCol, int64_t cacheColAlign);

private:
    TPipe *pipe_ = nullptr;

    GlobalTensor<T0> xGm;
    GlobalTensor<U> slotMappingGm;
    GlobalTensor<T1> kvCacheGm;

    TQue<QuePosition::VECIN, 2> xQue;
    TQue<QuePosition::VECOUT, 1> kvCacheQue;
    TBuf<QuePosition::VECCALC> kvCacheScaleBuf;
    TBuf<TPosition::VECCALC> scratchBuf;

    LocalTensor<T0> xLocal;
    LocalTensor<T1> kvCacheLocal;
    int64_t validIdx = 0;
    float maxValue = 0.0f;
    float fp8Min = 0.0f;
    float fp8Max = 0.0f;
    float scalesAttr = 1.0f;

    // Tiling data
    const KvCompressEpilogTilingData *tilingData = nullptr;
};

// Template implementations

template <typename T0, typename U, typename T1>
__aicore__ inline void KvCompressEpilog<T0, U, T1>::Init(GM_ADDR x, GM_ADDR slotMapping, GM_ADDR kvCache,
                                                         const KvCompressEpilogTilingData *tilingDataPtr)
{
    tilingData = tilingDataPtr;
    scalesAttr = tilingData->scalesAttr;

    int64_t xGmBaseOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->d;

    xGm.SetGlobalBuffer((__gm__ T0 *)x + xGmBaseOffset);
    kvCacheGm.SetGlobalBuffer((__gm__ T1 *)kvCache);
    slotMappingGm.SetGlobalBuffer((__gm__ U *)slotMapping);

    pipe_->InitBuffer(xQue, 2, tilingData->rowFactor * RoundUp<T0>(tilingData->d) * sizeof(T0));
    pipe_->InitBuffer(kvCacheQue, 2, tilingData->rowFactor * RoundUp<T1>(tilingData->kvCacheCol) * sizeof(T1));

    if (tilingData->quantMode == QUANT_MODE_HIF8_FP4) {
        uint32_t nopeElems = static_cast<uint32_t>(tilingData->d) - ROPE_HIF8_COLS;
        uint32_t numChunks = nopeElems / FP4_CHUNK_ELEMS;
        uint32_t scratchRowB16 = numChunks * SCRATCH_BLK_PER_CHUNK;
        pipe_->InitBuffer(scratchBuf, tilingData->rowFactor * scratchRowB16 * sizeof(T0));
    }

    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
}

template <typename T0, typename U, typename T1>
__aicore__ inline void KvCompressEpilog<T0, U, T1>::Process()
{
    SetMaxValue();
    int64_t rowOuterLoop =
        (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
    int64_t tailRowFactor = (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                   tilingData->tailRowFactorOfFormerBlock;
    if (rowOuterLoop <= 0) {
        return;
    }
    const int64_t dCol = tilingData->d;
    const int64_t rowFactorC = tilingData->rowFactor;
    const int64_t slotBlkBase = GetBlockIdx() * tilingData->rowOfFormerBlock;
    const int64_t blkSize = tilingData->kvCacheBlockSize;
    const int64_t blkStride = tilingData->kvCacheBlockStride;
    const int64_t rowStride = tilingData->kvCacheRowStride;
    const uint32_t cacheCol = tilingData->kvCacheCol;
    const int64_t cacheColAlign = RoundUp<T1>(tilingData->kvCacheCol);
    const uint32_t quantMode = tilingData->quantMode;
    const bool roundScale = (tilingData->roundScale == 1);
    const uint32_t concatCol = tilingData->concatCol;
    const uint32_t padCol = tilingData->padCol;
    // mode2(QUANT_MODE_HIF8_FP4) 专用参数: nope per-group 量化的组数/组大小, 及目标行字节步长。
    const uint32_t nGroup = tilingData->scaleCol;
    const uint32_t groupSize = tilingData->perGroupSize;
    const uint32_t dstRowStride = static_cast<uint32_t>(cacheColAlign);
    LocalTensor<T0> scratchLocal;
    if (quantMode == QUANT_MODE_HIF8_FP4) {
        scratchLocal = scratchBuf.template Get<T0>();
    }

    int64_t rf0 = (rowOuterLoop == 1) ? tailRowFactor : rowFactorC;
    LocalTensor<T0> x0 = xQue.template AllocTensor<T0>();
    CopyIn(xGm[0], x0, rf0, dCol);
    xQue.template EnQue(x0);

    LocalTensor<T1> prevKv;
    int64_t prevSlotBase = 0;
    int64_t prevRF = 0;
    bool havePrev = false;

    for (int64_t i = 0; i < rowOuterLoop; i++) {
        int64_t curRowFactor = (i == rowOuterLoop - 1) ? tailRowFactor : rowFactorC;
        // 预取下一块搬入(其 MTE2 与本块量化、上一块散写并行)
        if (i + 1 < rowOuterLoop) {
            int64_t nextRF = (i + 1 == rowOuterLoop - 1) ? tailRowFactor : rowFactorC;
            int64_t nextXBase = (i + 1) * rowFactorC * dCol;
            LocalTensor<T0> xn = xQue.template AllocTensor<T0>();
            CopyIn(xGm[nextXBase], xn, nextRF, dCol);
            xQue.template EnQue(xn);
        }
        // 量化本块(VEC) —— 取本块搬入结果
        LocalTensor<T0> xl = xQue.template DeQue<T0>();
        LocalTensor<T1> curKv = kvCacheQue.template AllocTensor<T1>();
        if (quantMode == QUANT_MODE_HIF8_FP4) {
            VFProcessHif8Fp4<T0, T1>(curKv, xl, scratchLocal, scalesAttr, static_cast<uint16_t>(curRowFactor),
                                     static_cast<uint32_t>(dCol), nGroup, groupSize, dstRowStride, concatCol, padCol);
        } else if (quantMode == QUANT_MODE_GROUP_QUANT_BF16) {
            if (roundScale) {
                VFProcessFP8PerGroupQuant<T0, T1, true, true>(curKv, xl, maxValue, fp8Min, fp8Max, curRowFactor, dCol,
                                                              concatCol, padCol);
            } else {
                VFProcessFP8PerGroupQuant<T0, T1, false, true>(curKv, xl, maxValue, fp8Min, fp8Max, curRowFactor, dCol,
                                                               concatCol, padCol);
            }
        } else if (quantMode == QUANT_MODE_GROUP_QUANT_E8M0) {
            if (roundScale) {
                VFProcessFP8PerGroupQuant<T0, T1, true, false>(curKv, xl, maxValue, fp8Min, fp8Max, curRowFactor, dCol,
                                                               concatCol, padCol);
            } else {
                VFProcessFP8PerGroupQuant<T0, T1, false, false>(curKv, xl, maxValue, fp8Min, fp8Max, curRowFactor, dCol,
                                                                concatCol, padCol);
            }
        }
        xQue.template FreeTensor(xl);
        kvCacheQue.template EnQue(curKv);

        if (havePrev) {
            Scatter(prevKv, prevSlotBase, prevRF, blkSize, blkStride, rowStride, cacheCol, cacheColAlign);
            kvCacheQue.template FreeTensor(prevKv);
        }
        // 取本块量化结果做下一轮的"上一块"(DeQue 等待本块 VEC, 已被上一块散写掩盖)
        prevKv = kvCacheQue.template DeQue<T1>();
        prevSlotBase = slotBlkBase + i * rowFactorC;
        prevRF = curRowFactor;
        havePrev = true;
    }
    // 收尾: 散写最后一块
    if (havePrev) {
        Scatter(prevKv, prevSlotBase, prevRF, blkSize, blkStride, rowStride, cacheCol, cacheColAlign);
        kvCacheQue.template FreeTensor(prevKv);
    }
}

template <typename T0, typename U, typename T1>
__aicore__ inline void KvCompressEpilog<T0, U, T1>::Scatter(const LocalTensor<T1> &kvLocal, int64_t slotBase,
                                                            int64_t rf, int64_t blkSize, int64_t blkStride,
                                                            int64_t rowStride, uint32_t cacheCol, int64_t cacheColAlign)
{
    // 拷出 4 展开 + 直接从 GM 读 slot (无 UB slot 缓冲); slot==-1 丢弃; 分页偏移
    int64_t r = 0;
    for (; r + FOUR_UNFOLD <= rf; r += FOUR_UNFOLD) {
        for (int32_t k = 0; k < FOUR_UNFOLD; k++) {
            int64_t slot = static_cast<int64_t>(slotMappingGm.GetValue(slotBase + r + k));
            if (slot == -1) {
                continue;
            }
            int64_t block = slot / blkSize;
            int64_t posInBlock = slot % blkSize;
            int64_t gmOffset = block * blkStride + posInBlock * rowStride;
            CopyOut(kvLocal[(r + k) * cacheColAlign], kvCacheGm[gmOffset], 1, cacheCol);
        }
    }
    for (; r < rf; r++) {
        int64_t slot = static_cast<int64_t>(slotMappingGm.GetValue(slotBase + r));
        if (slot == -1) {
            continue;
        }
        int64_t block = slot / blkSize;
        int64_t posInBlock = slot % blkSize;
        int64_t gmOffset = block * blkStride + posInBlock * rowStride;
        CopyOut(kvLocal[r * cacheColAlign], kvCacheGm[gmOffset], 1, cacheCol);
    }
}

template <typename T0, typename U, typename T1>
__aicore__ inline void KvCompressEpilog<T0, U, T1>::SetMaxValue()
{
    maxValue = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
    fp8Max = FP8_E4M3FN_MAX_VALUE;
    fp8Min = FP8_E4M3FN_MIN_VALUE;
}

} // namespace KvCompressEpilogOps
#endif // KV_COMPRESS_EPILOG_KERNEL_H
