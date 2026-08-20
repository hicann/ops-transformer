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
 * \file stem_indexer_topk.h
 * \brief
 */
#ifndef stem_indexer_TOPK_H
#define stem_indexer_TOPK_H

#include "kernel_operator.h"
#include "vf_topk_gather.h"
#include "vf_topk_16_gather.h"

namespace SIKernel {
template <typename SCORE_T>
class StemIndexerTopk {
public:
    static constexpr uint32_t MAX_LOOP_M = 4;

    // reuseMm1ResLocal 内的复用偏移，以下大小均以 uint32_t 元素为单位。
    static constexpr uint64_t TOPK_ALIGN_SIZE = 256ULL;
    static constexpr uint64_t NK_VALUE_PER_ROW = 64ULL;
    // uint16只需要idxHigh和idxLow两组；uint32需要idx0到idx3四组。
    static constexpr uint64_t IDX_BUFFER_NUM = std::is_same_v<SCORE_T, uint16_t> ? 2ULL : 4ULL;
    static constexpr uint64_t IDX_BUFFER_SIZE_U32 = MAX_LOOP_M * TOPK_ALIGN_SIZE;

    static constexpr uint64_t IDX_WORKSPACE_SIZE_U32 = IDX_BUFFER_NUM * IDX_BUFFER_SIZE_U32;
    static constexpr uint64_t HISTOGRAM_SIZE_U32 = MAX_LOOP_M * TOPK_ALIGN_SIZE;
    static constexpr uint64_t NK_VALUE_SIZE_U32 = MAX_LOOP_M * NK_VALUE_PER_ROW;
    // uint16 路径每行使用 512 个 uint16_t，折算后同样占 256 个 uint32_t。
    static constexpr uint64_t TMP_INDEX_SIZE_U32 = MAX_LOOP_M * TOPK_ALIGN_SIZE;
    // indicesOut在mm1Res复用区内使用独立子区，避免覆盖仍需读取的历史索引。
    static constexpr uint64_t INDICES_OUT_SIZE_U32 = MAX_LOOP_M * TOPK_ALIGN_SIZE;

    static constexpr uint64_t REUSE_INDICES_OUT_OFFSET =
        IDX_WORKSPACE_SIZE_U32 + NK_VALUE_SIZE_U32 + TMP_INDEX_SIZE_U32;
    static constexpr uint64_t REUSE_SCORE_OUT_OFFSET_U32 = REUSE_INDICES_OUT_OFFSET + INDICES_OUT_SIZE_U32;
    static constexpr uint64_t REUSE_SCORE_OUT_OFFSET = REUSE_SCORE_OUT_OFFSET_U32 * sizeof(uint32_t) / sizeof(SCORE_T);

    __aicore__ inline void InitBuffers(const LocalTensor<uint32_t> &reuseMm1ResLocal,
                                       const LocalTensor<uint32_t> &histogramLocal,
                                       const LocalTensor<uint32_t> &reuseGlobalIndexLocal)
    {
        // reuseMm1ResLocal按idx -> nk -> tmpIndex -> indicesOut -> scoreOut排布；histogram独立申请。
        idx0Local = reuseMm1ResLocal;
        idx1Local = idx0Local[IDX_BUFFER_SIZE_U32];
        if constexpr (!std::is_same_v<SCORE_T, uint16_t>) {
            idx2Local = idx1Local[IDX_BUFFER_SIZE_U32];
            idx3Local = idx2Local[IDX_BUFFER_SIZE_U32];
        }
        histogramsLocal = histogramLocal;
        nkValueLocal = reuseMm1ResLocal[IDX_WORKSPACE_SIZE_U32];
        tmpIndexLocal = nkValueLocal[NK_VALUE_SIZE_U32];
        indicesOutLocal = reuseMm1ResLocal[REUSE_INDICES_OUT_OFFSET];
        hisValueLocal = reuseMm1ResLocal[REUSE_SCORE_OUT_OFFSET_U32].ReinterpretCast<SCORE_T>();

        // globalIndexLocal保存跨S2轮次的索引；indicesOutLocal作为本轮gather输出，避免原地覆盖。
        hisIndexLocal[0] = reuseGlobalIndexLocal;
        hisIndexLocal[1] = indicesOutLocal;
    }

    __aicore__ inline void Batch4Rows(const LocalTensor<SCORE_T> &mrgValueLocal,
                                      const LocalTensor<uint32_t> &reuseMm1ResLocal,
                                      const LocalTensor<uint32_t> &histogramLocal,
                                      const LocalTensor<uint32_t> &reuseGlobalIndexLocal,
                                      const SICommon::RowIdx4 &rowIdx4, const SICommon::TopkNum4 &topkNum4,
                                      uint32_t batchRowNum, uint32_t mrgRowStride, uint32_t inputOffset,
                                      uint32_t validLen, uint32_t loopIdx, uint32_t s2LoopNum)
    {
        const uint32_t offset = mrgRowStride;
        const uint32_t topkAlign = 256U;
        const uint32_t tmpIdxStride16 = topkAlign + SICommon::TRUNK_LEN_256;
        InitBuffers(reuseMm1ResLocal, histogramLocal, reuseGlobalIndexLocal);

        LocalTensor<SCORE_T> inputValueLocal = mrgValueLocal[inputOffset];

        if (s2LoopNum == 1) {
            if constexpr (std::is_same_v<SCORE_T, uint16_t>) {
                LocalTensor<uint16_t> tmpIdxLocal16 = tmpIndexLocal.ReinterpretCast<uint16_t>();
                SITopkb16gather::SiTopKVF<false>(tmpIdxLocal16, hisValueLocal, inputValueLocal, histogramsLocal,
                                                 idx0Local, idx1Local, nkValueLocal, validLen, MAX_LOOP_M, offset,
                                                 rowIdx4.v0, rowIdx4.v1, rowIdx4.v2, rowIdx4.v3, topkNum4.v0,
                                                 topkNum4.v1, topkNum4.v2, topkNum4.v3, tmpIdxStride16, topkAlign);
            } else {
                SITopkb32gather::SiTopKVF<false>(tmpIndexLocal, hisValueLocal, inputValueLocal, histogramsLocal,
                                                 idx0Local, idx1Local, idx2Local, idx3Local, nkValueLocal, validLen,
                                                 MAX_LOOP_M, offset, rowIdx4.v0, rowIdx4.v1, rowIdx4.v2, rowIdx4.v3,
                                                 topkNum4.v0, topkNum4.v1, topkNum4.v2, topkNum4.v3);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t m = 0; m < batchRowNum; ++m) {
                uint32_t mInnerIdx = SICommon::GetLane(rowIdx4, m);
                if constexpr (std::is_same_v<SCORE_T, uint16_t>) {
                    LocalTensor<uint16_t> tmpIdxLocal16 = tmpIndexLocal.ReinterpretCast<uint16_t>();
                    AscendC::Cast(hisIndexLocal[0][mInnerIdx * topkAlign], tmpIdxLocal16[m * tmpIdxStride16],
                                  RoundMode::CAST_NONE, topkAlign);
                } else {
                    AscendC::DataCopy(hisIndexLocal[0][mInnerIdx * topkAlign], tmpIndexLocal[m * topkAlign], topkAlign);
                }
            }
        } else if (loopIdx == 0) {
            if constexpr (std::is_same_v<SCORE_T, uint16_t>) {
                LocalTensor<uint16_t> tmpIdxLocal16 = tmpIndexLocal.ReinterpretCast<uint16_t>();
                SITopkb16gather::SiTopKVF<true>(tmpIdxLocal16, hisValueLocal, inputValueLocal, histogramsLocal,
                                                idx0Local, idx1Local, nkValueLocal, validLen, MAX_LOOP_M, offset,
                                                rowIdx4.v0, rowIdx4.v1, rowIdx4.v2, rowIdx4.v3, topkNum4.v0,
                                                topkNum4.v1, topkNum4.v2, topkNum4.v3, tmpIdxStride16, topkAlign);
            } else {
                SITopkb32gather::SiTopKVF<true>(tmpIndexLocal, hisValueLocal, inputValueLocal, histogramsLocal,
                                                idx0Local, idx1Local, idx2Local, idx3Local, nkValueLocal, validLen,
                                                MAX_LOOP_M, offset, rowIdx4.v0, rowIdx4.v1, rowIdx4.v2, rowIdx4.v3,
                                                topkNum4.v0, topkNum4.v1, topkNum4.v2, topkNum4.v3);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t m = 0; m < batchRowNum; ++m) {
                uint32_t mInnerIdx = SICommon::GetLane(rowIdx4, m);
                // 输出到 globalIndexLocal(hisIndexLocal[0])，给下一轮 gather 和最终打包搬出使用
                if constexpr (std::is_same_v<SCORE_T, uint16_t>) {
                    LocalTensor<uint16_t> tmpIdxLocal16 = tmpIndexLocal.ReinterpretCast<uint16_t>();
                    AscendC::Cast(hisIndexLocal[0][mInnerIdx * topkAlign], tmpIdxLocal16[m * tmpIdxStride16],
                                  RoundMode::CAST_NONE, topkAlign);
                } else {
                    AscendC::DataCopy(hisIndexLocal[0][mInnerIdx * topkAlign], tmpIndexLocal[m * topkAlign], topkAlign);
                }
                AscendC::DataCopy(mrgValueLocal[mInnerIdx * mrgRowStride], hisValueLocal[m * topkAlign], topkAlign);
            }
        } else {
            if constexpr (std::is_same_v<SCORE_T, uint16_t>) {
                LocalTensor<uint16_t> tmpIdxLocal16 = tmpIndexLocal.ReinterpretCast<uint16_t>();
                SITopkb16gather::SiTopKVF<true>(tmpIdxLocal16, hisValueLocal, inputValueLocal, histogramsLocal,
                                                idx0Local, idx1Local, nkValueLocal, validLen, MAX_LOOP_M, offset,
                                                rowIdx4.v0, rowIdx4.v1, rowIdx4.v2, rowIdx4.v3, topkNum4.v0,
                                                topkNum4.v1, topkNum4.v2, topkNum4.v3, tmpIdxStride16, topkAlign);
            } else {
                SITopkb32gather::SiTopKVF<true>(tmpIndexLocal, hisValueLocal, inputValueLocal, histogramsLocal,
                                                idx0Local, idx1Local, idx2Local, idx3Local, nkValueLocal, validLen,
                                                MAX_LOOP_M, offset, rowIdx4.v0, rowIdx4.v1, rowIdx4.v2, rowIdx4.v3,
                                                topkNum4.v0, topkNum4.v1, topkNum4.v2, topkNum4.v3);
            }
            PipeBarrier<PIPE_V>();
            // gather: 一次VF处理4个compact行，从globalIndexLocal读取上轮结果，输出到indicesOutLocal。
            if constexpr (std::is_same_v<SCORE_T, uint16_t>) {
                LocalTensor<uint16_t> tmpIdxLocal16 = tmpIndexLocal.ReinterpretCast<uint16_t>();
                SITopkb16gather::SiTopKGatherVF(hisIndexLocal[1], tmpIdxLocal16, hisIndexLocal[0], topkAlign,
                                                tmpIdxStride16, topkAlign, rowIdx4.v0, rowIdx4.v1, rowIdx4.v2,
                                                rowIdx4.v3, topkNum4.v0, topkNum4.v1, topkNum4.v2, topkNum4.v3,
                                                loopIdx * SICommon::TRUNK_LEN_256 - topkAlign);
            } else {
                SITopkb32gather::SiTopKGatherVF(hisIndexLocal[1], tmpIndexLocal, hisIndexLocal[0], topkAlign, topkAlign,
                                                topkAlign, rowIdx4.v0, rowIdx4.v1, rowIdx4.v2, rowIdx4.v3, topkNum4.v0,
                                                topkNum4.v1, topkNum4.v2, topkNum4.v3,
                                                loopIdx * SICommon::TRUNK_LEN_256 - topkAlign);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t m = 0; m < batchRowNum; ++m) {
                uint32_t mInnerIdx = SICommon::GetLane(rowIdx4, m);
                // 拷回 globalIndexLocal(hisIndexLocal[0])，给下一轮和最终打包搬出使用
                AscendC::DataCopy(hisIndexLocal[0][mInnerIdx * topkAlign], hisIndexLocal[1][m * topkAlign], topkAlign);
                AscendC::DataCopy(mrgValueLocal[mInnerIdx * mrgRowStride], hisValueLocal[m * topkAlign], topkAlign);
            }
        }
    }

private:
    LocalTensor<uint32_t> hisIndexLocal[2]; // 每trunkLen长度的s2选出的topK个索引
    LocalTensor<uint32_t> histogramsLocal;  // 直方图的临时Buf MAX_LOOP_M * 256 * 4B
    LocalTensor<uint32_t> idx0Local;        // 输入数据第1个8位Buf MAX_LOOP_M * 256 * 4B
    LocalTensor<uint32_t> idx1Local;        // 输入数据第2个8位Buf MAX_LOOP_M * 256 * 4B
    LocalTensor<uint32_t> idx2Local;        // 输入数据第3个8位Buf MAX_LOOP_M * 256 * 4B
    LocalTensor<uint32_t> idx3Local;        // 输入数据第4个8位Buf MAX_LOOP_M * 256 * 4B
    LocalTensor<uint32_t> nkValueLocal;     // next_k 暂存Buf MAX_LOOP_M * 64 * 4B
    LocalTensor<uint32_t> tmpIndexLocal;    // 每行tmpIdx MAX_LOOP_M * Align(topK,256) * 4B
    LocalTensor<uint32_t> indicesOutLocal;  // 本轮gather产生的TopK索引 MAX_LOOP_M * 256 * 4B
    LocalTensor<SCORE_T> hisValueLocal;     // 本轮TopK value MAX_LOOP_M * 256 * sizeof(SCORE_T)
};
} // namespace SIKernel
#endif
