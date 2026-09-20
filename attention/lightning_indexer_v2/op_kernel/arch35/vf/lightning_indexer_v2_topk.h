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
 * \file lightning_indexer_v2_topk.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_V2_TOPK_H
#define LIGHTNING_INDEXER_V2_TOPK_H

#include "kernel_operator.h"
#include "vf_topk_gather_v2.h"

#include "../common/vf/lightning_indexer_v2_topk_base.h"

namespace liV2Topk {
template <typename T>
class LIV2Topk {
public:
    __aicore__ inline void operator()(LocalTensor<uint32_t> &outputIdxLocal, LocalTensor<T> &inputLocal,
                                      uint32_t s2SeqLen)
    {}
};

template <>
class LIV2Topk<uint32_t> {
public:
    __aicore__ inline uint32_t GetSharedTmpBufferSize()
    {
        return liV2TopkCommon::GetGatherTmpBufferSize<uint32_t, liV2TopkCommon::B32_RADIX_BUFFER_NUM>(topK, trunkLen);
    }

    __aicore__ inline void Init(uint32_t topK, uint32_t trunkLen)
    {
        this->topK = topK;
        this->trunkLen = trunkLen;
    }

    __aicore__ inline void InitBuffers(LocalTensor<uint32_t> &sharedTmpBuffer, LocalTensor<uint32_t> &indicesOutLocal)
    {
        LocalTensor<uint32_t> hisIndexLocal1 = indicesOutLocal;
        LocalTensor<uint32_t> hisIndexLocal2 = sharedTmpBuffer[0];
        hisIndexLocal[0] = hisIndexLocal1;
        hisIndexLocal[1] = hisIndexLocal2;
        histogramsLocal = hisIndexLocal2[LIV2Common::Align(topK, (uint32_t)256)]; // 256：本地内存对齐基线
        idx0Local = histogramsLocal[256];                                         // 256：同上
        idx1Local = idx0Local[256];                                               // 256：同上
        idx2Local = idx1Local[256];                                               // 256：同上
        idx3Local = idx2Local[256];                                               // 256：同上
        nkValueLocal = idx3Local[256];                                            // 256：同上
        tmpIndexLocal = nkValueLocal[64]; // 64：单核/单线程输出元素容量（G维度切分阈值）
    }

    __aicore__ inline void operator()(LocalTensor<uint32_t> &mrgValueLocal, LocalTensor<uint32_t> &indicesOutLocal,
                                      LocalTensor<uint32_t> &hisValueLocal, uint32_t s2SeqLen, uint32_t loopIdx,
                                      uint32_t s2LoopNum, bool isNeedLD, bool returnValueFlag, uint32_t outputIdxOffset)
    {
        if (s2LoopNum == 1) {
            if (isNeedLD || returnValueFlag) {
                liV2Topkb32gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
                                                  idx0Local, idx1Local, idx2Local, idx3Local, nkValueLocal, topK,
                                                  s2SeqLen);
            } else {
                liV2Topkb32gather::LiTopKVF<false>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
                                                   idx0Local, idx1Local, idx2Local, idx3Local, nkValueLocal, topK,
                                                   s2SeqLen);
            }
            PipeBarrier<PIPE_V>();
            AscendC::DataCopy(indicesOutLocal, tmpIndexLocal,
                              LIV2Common::Align(topK, (uint32_t)256)); // 256：拷贝长度对齐大小
        } else {
            if (loopIdx == 0) {
                liV2Topkb32gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
                                                  idx0Local, idx1Local, idx2Local, idx3Local, nkValueLocal, topK,
                                                  s2SeqLen);
                PipeBarrier<PIPE_V>();
                AscendC::DataCopy(
                    hisIndexLocal[(loopIdx + 1) % 2], tmpIndexLocal, // 2：使用两个本地存储单元进行循环交替存储
                    LIV2Common::Align(topK, (uint32_t)256)); // 对复制的数据量进行对齐处理，确保其为256的倍数
            } else if (loopIdx != 0) {
                liV2Topkb32gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
                                                  idx0Local, idx1Local, idx2Local, idx3Local, nkValueLocal, topK,
                                                  s2SeqLen);
                PipeBarrier<PIPE_V>();
                uint32_t loopBasicIdx = liV2TopkCommon::GetGatherLoopOffset(topK, trunkLen, loopIdx);
                // 2：使用两个本地存储单元进行循环交替存储
                liV2Topkb32gather::LiTopKGatherVF(hisIndexLocal[(loopIdx + 1) % 2], hisValueLocal, mrgValueLocal,
                                                  // 2：同上
                                                  tmpIndexLocal, hisIndexLocal[loopIdx % 2], topK, loopBasicIdx,
                                                  s2SeqLen);
                if (loopIdx == s2LoopNum - 1) {
                    PipeBarrier<PIPE_V>();
                    if ((loopIdx + 1) % 2 == 1) { // 2：同上
                        AscendC::DataCopy(
                            indicesOutLocal,
                            hisIndexLocal[(loopIdx + 1) % 2], // 2：同上
                            LIV2Common::Align(topK, (uint32_t)256)); // 对复制的数据量进行对齐处理，确保其为256的倍数
                    }
                }
            }
        }
        // indices加偏移
        if (outputIdxOffset != 0) {
            liV2Topkb32gather::IndicesAddOffset(indicesOutLocal, outputIdxOffset, topK);
        }
    }

    __aicore__ inline void LdTopK(LocalTensor<uint32_t> &mrgValueLocal, LocalTensor<uint32_t> indexLocal,
                                  LocalTensor<uint32_t> &indicesOutLocal, LocalTensor<uint32_t> &hisValueLocal,
                                  uint32_t s2SeqLen, uint32_t loopIdx, uint32_t s2LoopNum)
    {
        liV2Topkb32gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idx0Local,
                                          idx1Local, idx2Local, idx3Local, nkValueLocal, topK, s2SeqLen);
        PipeBarrier<PIPE_V>();
        liV2Topkb32gather::LiTopKLDGatherVF(indicesOutLocal, tmpIndexLocal, indexLocal, topK);
    }

private:
    LocalTensor<uint32_t> hisIndexLocal[2]; // 每trunkLen长度的s2选出的topK个索引
    LocalTensor<uint32_t> histogramsLocal;  // 直方图的临时Buf 256 * 4B
    LocalTensor<uint32_t> idx0Local;        // 输入数据第1个8位Buf 256 * 4B
    LocalTensor<uint32_t> idx1Local;        // 输入数据第2个8位Buf 256 * 4B
    LocalTensor<uint32_t> idx2Local;        // 输入数据第3个8位Buf 256 * 4B
    LocalTensor<uint32_t> idx3Local;        // 输入数据第4个8位Buf 256 * 4B
    LocalTensor<uint32_t> nkValueLocal;     // next_k 暂存Buf 64 * 4B
    LocalTensor<uint32_t> tmpIndexLocal;    // 每trunkLen + topK的临时index
    uint32_t topK = 512;
    uint32_t trunkLen = 8192;
};
} // namespace liV2Topk
#endif
