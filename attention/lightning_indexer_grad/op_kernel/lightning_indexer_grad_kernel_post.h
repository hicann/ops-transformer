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
 * \file lightning_indexer_grad_service_vector_post.h
 * \brief common post process
 */

#ifndef LIGHTNING_INDEXER_GRAD_SERVICE_VECTOR_POST_H
#define LIGHTNING_INDEXER_GRAD_SERVICE_VECTOR_POST_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lightning_indexer_grad_common.h"

namespace LigKernel {
using namespace LIGCommon;
using namespace AscendC;

template <typename LIGT>
class LIGVectorPost {
public:
    __aicore__ inline LIGVectorPost()
        : tilingData(nullptr),
          pipe(nullptr){};
    __aicore__ inline void Init(TPipe *pipe_in, __gm__ uint8_t *dk, __gm__ uint8_t *workspace,
                                const LIGTilingData *__restrict ordTilingData);
    __aicore__ inline void Process();

    using dataType = typename LIGT::dataType;
    constexpr static uint32_t POST_BUFFER_NUM = 1;

    TPipe *pipe;
    TQue<QuePosition::VECIN, POST_BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, POST_BUFFER_NUM> outQueue;

    // input
    AscendC::GlobalTensor<float> dkWorkSpaceGm;

    // output
    AscendC::GlobalTensor<dataType> dkGm;

    const LIGTilingData *__restrict tilingData;

    // key
    int64_t cBlockIdx;
    int64_t ubBaseSize;
    int64_t kPostBlockFactor;
    uint64_t kPostBlockTotal;
    int64_t kPostBaseNum;
    int64_t kPostTailNum;
};

template <typename LIGT>
__aicore__ inline void LIGVectorPost<LIGT>::Init(TPipe *pipe_in, __gm__ uint8_t *dk, __gm__ uint8_t *workspace,
                                                 const LIGTilingData *__restrict ordTilingData)
{
    tilingData = nullptr;
    pipe = nullptr;

    if (ordTilingData == nullptr || pipe_in == nullptr || dk == nullptr || workspace == nullptr) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "LIGVectorPost Init input pointer is null."); });
        return;
    }

    cBlockIdx = GetBlockIdx();

    tilingData = ordTilingData;
    pipe = pipe_in;

    dkGm.SetGlobalBuffer((__gm__ dataType *)dk);

    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dkWorkSpaceOffset / sizeof(float));

    uint32_t coreNum = tilingData->usedCoreNum;

    // compute tiling
    constexpr static uint32_t POST_COEX_NODE = 3;
    constexpr static uint32_t WORKSPACE_NUM_ALIGN = 256;
    uint32_t curPostCoexNode = POST_COEX_NODE;
    uint32_t ubSize = 192 * 1024;
    ubBaseSize = ubSize / curPostCoexNode / POST_BUFFER_NUM;
    ubBaseSize = ubBaseSize / WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN; // align

    // dk
    kPostBaseNum = ubBaseSize / sizeof(float);
    kPostBlockTotal = tilingData->dkSize;

    int64_t kPostTailNumTmp = kPostBlockTotal % kPostBaseNum;
    int64_t kPostBlockOuterTotal = (kPostBlockTotal + kPostBaseNum - 1) / kPostBaseNum;

    kPostTailNum = kPostTailNumTmp == 0 ? kPostBaseNum : kPostTailNumTmp;
    kPostBlockFactor = (kPostBlockOuterTotal + coreNum - 1) / coreNum;

    pipe->InitBuffer(inQueue, 1, ubBaseSize * 2);
    pipe->InitBuffer(outQueue, 1, ubBaseSize);
}

template <typename LIGT>
__aicore__ inline void LIGVectorPost<LIGT>::Process()
{
    if (tilingData == nullptr || pipe == nullptr) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "LIGVectorPost Process called before successful Init."); });
        return;
    }
    // init k
    if (g_coreType == AIV) {
        uint64_t kvBegin = cBlockIdx * kPostBlockFactor * kPostBaseNum;
        uint64_t kvEnd = (cBlockIdx + 1) * kPostBlockFactor * kPostBaseNum;
        if (((cBlockIdx + 1) * kPostBlockFactor * kPostBaseNum) > kPostBlockTotal) {
            kvEnd = kPostBlockTotal;
        }

        for (uint64_t i = kvBegin; i < kvEnd; i = i + kPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<dataType> vecOut = outQueue.template AllocTensor<dataType>();
            uint64_t dataSize = i + kPostBaseNum < kPostBlockTotal ? kPostBaseNum : kPostTailNum;
            DataCopy(vecIn, dkWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            inQueue.EnQue(vecIn);
            inQueue.template DeQue<float>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<dataType>();
            DataCopy(dkGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
} // namespace LigKernel
#endif // _FLASH_ATTENTION_SCORE_GRAD_POST_H_
