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
 * \file lightning_indexer_grad_service_vector_pre.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_GRAD_SERVICE_VECTOR_PRE_H
#define LIGHTNING_INDEXER_GRAD_SERVICE_VECTOR_PRE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_grad_common.h"
#include "lightning_indexer_grad_tiling.h"

namespace LigKernel {
using namespace LIGCommon;
using namespace AscendC;
using namespace optiling;

template <typename LIGT>
class LIGVectorPre {
public:
    using dataType = typename LIGT::dataType;

    __aicore__ inline LIGVectorPre(){};
    __aicore__ inline void Init(TPipe *pipe_in, __gm__ uint8_t *dq, __gm__ uint8_t *dweights, __gm__ uint8_t *workspace,
                                const LIGTilingData *__restrict orgTilingData, __gm__ uint8_t *actualSeqLengthsQ);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void DoSplitOnGM(uint32_t coreNum, uint32_t gmSize, uint32_t &PreBlockTotal, int64_t &initSize,
                                       int64_t &Offset);

    using D_T = typename LIGT::dataType;

protected:
    TPipe *pipe;
    GlobalTensor<float> dkWorkSpaceGm;
    GlobalTensor<float> dkCoreWorkspaceGM;

    const LIGTilingData *__restrict tilingData;

    uint32_t cBlockIdx;
    uint32_t kPreBlockTotal;
    int64_t initdkSize;
    int64_t dkOffset;

    uint32_t qPreBlockTotal = 0;
    int64_t initdqSize = 0;
    int64_t dqOffset = 0;

    uint32_t wPreBlockTotal = 0;
    int64_t initdwSize = 0;
    int64_t dwOffset = 0;

    // Output GlobalTensor
    GlobalTensor<D_T> dqGm;
    GlobalTensor<D_T> dweightsGm;
    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
};

template <typename LIGT>
__aicore__ inline void LIGVectorPre<LIGT>::DoSplitOnGM(uint32_t coreNum, uint32_t gmSize, uint32_t &PreBlockTotal,
                                                       int64_t &initSize, int64_t &Offset)
{
    uint32_t PreBlockFactor = CeilDiv(gmSize, coreNum);
    PreBlockTotal = CeilDiv(gmSize, PreBlockFactor);
    int64_t PreTailNumTmp = gmSize % PreBlockFactor;
    uint32_t PreBlockTail = PreTailNumTmp == 0 ? PreBlockFactor : PreTailNumTmp;
    initSize = cBlockIdx == PreBlockTotal - 1 ? PreBlockTail : PreBlockFactor;
    Offset = ((int64_t)cBlockIdx) * PreBlockFactor;
}

template <typename LIGT>
__aicore__ inline void LIGVectorPre<LIGT>::Init(TPipe *pipe_in, __gm__ uint8_t *dq, __gm__ uint8_t *dweights,
                                                __gm__ uint8_t *workspace,
                                                const LIGTilingData *__restrict orgTilingData,
                                                __gm__ uint8_t *actualSeqLengthsQ)
{
    cBlockIdx = GetBlockIdx();
    pipe = pipe_in;
    tilingData = orgTilingData;

    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dkWorkSpaceOffset / sizeof(float));

    // Init workspace to store per-core dKey partial results.
    dkCoreWorkspaceGM.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dkCoreWorkspaceOffset / sizeof(float));

    DoSplitOnGM(tilingData->usedCoreNum, tilingData->dkSize, kPreBlockTotal, initdkSize, dkOffset);

    if (LIGT::layout == LIG_LAYOUT::TND) {
        dqGm.SetGlobalBuffer((__gm__ D_T *)dq);
        dweightsGm.SetGlobalBuffer((__gm__ D_T *)dweights);
        if (actualSeqLengthsQ != nullptr) {
            actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsQ, tilingData->batch);
        }
        uint32_t actualSeqTotalLenQ = 0;
        if (tilingData->batch > 0) {
            actualSeqTotalLenQ = actualSeqLengthsGmQ.GetValue(tilingData->batch - 1);
        }
        if (tilingData->seqlenQ > actualSeqTotalLenQ) {
            uint32_t dqPaddingSize =
                (tilingData->seqlenQ - actualSeqTotalLenQ) * tilingData->headNumQ * tilingData->headDim;
            uint32_t dwPaddingSize = (tilingData->seqlenQ - actualSeqTotalLenQ) * tilingData->headNumQ;
            DoSplitOnGM(tilingData->usedCoreNum, dqPaddingSize, qPreBlockTotal, initdqSize, dqOffset);
            dqOffset += actualSeqTotalLenQ * tilingData->headNumQ * tilingData->headDim;
            DoSplitOnGM(tilingData->usedCoreNum, dwPaddingSize, wPreBlockTotal, initdwSize, dwOffset);
            dwOffset += actualSeqTotalLenQ * tilingData->headNumQ;
        }
    }
}

template <typename LIGT>
__aicore__ inline void LIGVectorPre<LIGT>::Process()
{
    // process clear dk workspace
    if (g_coreType == AIV && cBlockIdx < kPreBlockTotal) {
        InitOutput<float>(dkWorkSpaceGm[dkOffset], initdkSize, 0);
    }
    // TND layout: process clear dq, dk, dweights's padding

    if (g_coreType == AIV && LIGT::layout == LIG_LAYOUT::TND) {
        if (cBlockIdx < qPreBlockTotal) {
            InitOutput<D_T>(dqGm[dqOffset], initdqSize, 0);
        }
        if (cBlockIdx < wPreBlockTotal) {
            InitOutput<D_T>(dweightsGm[dwOffset], initdwSize, 0);
        }
    }
}

template <typename LIGT>
__aicore__ inline void LIGVectorPre<LIGT>::SyncALLCores()
{
    SyncAll();
}
} // namespace LigKernel
#endif
