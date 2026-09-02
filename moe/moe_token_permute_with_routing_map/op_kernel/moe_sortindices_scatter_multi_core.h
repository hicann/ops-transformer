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
 * \file moe_sortindices_scatter_multi_core.h
 * \brief 非对齐路径：value 流 scatter 到 sorted_indices（参考 moe_v2_src_to_dst_op，禁止 GM SetValue）。
 */
#ifndef MOE_SORTINDICES_SCATTER_MULTI_CORE_H
#define MOE_SORTINDICES_SCATTER_MULTI_CORE_H

#include "kernel_tiling/kernel_tiling.h"
#include "moe_token_permute_with_routing_map_common.h"

namespace MoeTokenPermute {
using namespace AscendC;

class MoeSortIndicesScatterMultiCore {
public:
    __aicore__ inline MoeSortIndicesScatterMultiCore() {}

    __aicore__ inline void Init(GM_ADDR sortedIndicesOut, GM_ADDR valueStreamGm, int32_t actualOutTokensIn,
                                const MoeTokenPermuteWithRoutingMapTilingData *tilingData, GM_ADDR workspace,
                                TPipe *tPipeIn);

    __aicore__ inline void Process();

private:
    __aicore__ inline void AssistInit();
    __aicore__ inline void InitMinusOne();
    __aicore__ inline void CopyIn(int64_t progress);
    __aicore__ inline void Compute(int64_t progress);
    __aicore__ inline void CopyOut();

private:
    static constexpr int32_t SCATTER_MINUS_ONE = -1;

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> copyInQueue;
    TQue<QuePosition::VECOUT, 1> copyOutQueue;
    TBuf<TPosition::VECCALC> assistBuffer;

    GlobalTensor<int32_t> outGm;
    GlobalTensor<int32_t> valueGm;
    GlobalTensor<int32_t> assistGm;

    int64_t blockIdx;
    int32_t actualOutTokens;
    int64_t totalLength;
    int64_t scatterNeedCoreNum;
    int64_t scatterPerCore;
    int64_t valueStart;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
    int64_t currentLoopRows;
    int64_t initNeedCoreNum;
    int64_t initPerCore;
    int64_t initSliceStart;
    int64_t initSliceCount;
};

__aicore__ inline void MoeSortIndicesScatterMultiCore::Init(GM_ADDR sortedIndicesOut, GM_ADDR valueStreamGm,
                                                            int32_t actualOutTokensIn,
                                                            const MoeTokenPermuteWithRoutingMapTilingData *tilingData,
                                                            GM_ADDR workspace, TPipe *tPipeIn)
{
    (void)workspace;
    pipe = tPipeIn;
    blockIdx = GetBlockIdx();
    actualOutTokens = actualOutTokensIn;
    totalLength = tilingData->n * tilingData->topK;

    outGm.SetGlobalBuffer((__gm__ int32_t *)sortedIndicesOut, Align(totalLength, sizeof(int32_t)));
    valueGm.SetGlobalBuffer((__gm__ int32_t *)valueStreamGm, actualOutTokens);

    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    initPerCore = Ceil(Ceil(totalLength, blockNum), ASSIST_NUM) * ASSIST_NUM;
    initNeedCoreNum = Ceil(totalLength, initPerCore);
    initSliceStart = blockIdx * initPerCore;
    initSliceCount = initPerCore;
    if (initSliceStart + initSliceCount > totalLength) {
        initSliceCount = totalLength - initSliceStart;
    }
    if (initSliceStart >= totalLength) {
        initSliceCount = 0;
    }

    valueStart = 0;
    coreRows = 0;
    scatterNeedCoreNum = 0;
    scatterPerCore = 0;
    if (actualOutTokens > 0) {
        scatterPerCore = Ceil(Ceil(actualOutTokens, blockNum), ASSIST_NUM) * ASSIST_NUM;
        scatterNeedCoreNum = Ceil(actualOutTokens, scatterPerCore);
        valueStart = blockIdx * scatterPerCore;
        coreRows = scatterPerCore;
        if (valueStart + coreRows > actualOutTokens) {
            coreRows = actualOutTokens - valueStart;
        }
        if (valueStart >= actualOutTokens) {
            coreRows = 0;
        }
    }

    perLoopRows = Min(coreRows, CalcScatterPerLoopMaxRows(blockNum));
    if (perLoopRows <= 0) {
        perLoopRows = 1;
    }
    int64_t loops = coreRows > 0 ? (coreRows + perLoopRows - 1) / perLoopRows : 0;
    lastLoopRows = coreRows - (loops - 1) * perLoopRows;
    if (lastLoopRows <= 0) {
        lastLoopRows = perLoopRows;
    }

    pipe->InitBuffer(copyInQueue, 1, perLoopRows * BLOCK_BYTES);
    pipe->InitBuffer(copyOutQueue, 1, Ceil(perLoopRows, ASSIST_INDEX_NUM) * ASSIST_INDEX_NUM * BLOCK_BYTES);
    pipe->InitBuffer(assistBuffer, ASSIST_NUM * sizeof(int32_t));
    assistGm.SetGlobalBuffer((__gm__ int32_t *)scatterAssist, ASSIST_NUM);
}

__aicore__ inline void MoeSortIndicesScatterMultiCore::AssistInit()
{
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    OOMCheckAddrRange(assistGm.GetPhyAddr(), ASSIST_NUM * sizeof(int32_t));
#endif
    LocalTensor<int32_t> assistTensor = assistBuffer.Get<int32_t>(ASSIST_NUM);
    DataCopy(assistTensor, assistGm, ASSIST_NUM);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Adds(assistTensor, assistTensor, static_cast<int32_t>(valueStart), ASSIST_NUM);
}

__aicore__ inline void MoeSortIndicesScatterMultiCore::InitMinusOne()
{
    if (initSliceCount <= 0) {
        return;
    }
    GlobalTensor<int32_t> initOutGm = outGm[initSliceStart];
    InitGlobalMemory(initOutGm, initSliceCount, SCATTER_MINUS_ONE);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

__aicore__ inline void MoeSortIndicesScatterMultiCore::CopyIn(int64_t progress)
{
    LocalTensor<int32_t> inLocal = copyInQueue.AllocTensor<int32_t>();
    DataCopy(inLocal, valueGm[valueStart + progress * perLoopRows], Align(currentLoopRows, sizeof(int32_t)));
    copyInQueue.EnQue<int32_t>(inLocal);
}

__aicore__ inline void MoeSortIndicesScatterMultiCore::Compute(int64_t progress)
{
    LocalTensor<int32_t> outLocal = copyOutQueue.AllocTensor<int32_t>();
    LocalTensor<int32_t> assistTensor = assistBuffer.Get<int32_t>(ASSIST_NUM);

    PipeBarrier<PIPE_V>();
    int64_t loops = Ceil(currentLoopRows, ASSIST_INDEX_NUM);
    for (int64_t i = 0; i < loops; i++) {
        Adds(outLocal[i * ASSIST_NUM], assistTensor,
             static_cast<int32_t>(perLoopRows * progress + i * ASSIST_INDEX_NUM), ASSIST_NUM);
    }
    PipeBarrier<PIPE_V>();
    copyOutQueue.EnQue<int32_t>(outLocal);
}

__aicore__ inline void MoeSortIndicesScatterMultiCore::CopyOut()
{
    LocalTensor<int32_t> inLocal = copyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> outLocal = copyOutQueue.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = sizeof(int32_t);
    for (int64_t idx = 0; idx < currentLoopRows; idx++) {
        uint32_t outOffset = static_cast<uint32_t>(inLocal.GetValue(idx));
        if (outOffset < static_cast<uint32_t>(totalLength)) {
            DataCopyPad(outGm[outOffset], outLocal[idx * INT32_ONE_BLOCK_NUM], intriParams);
        }
    }

    copyInQueue.FreeTensor(inLocal);
    copyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortIndicesScatterMultiCore::Process()
{
    AscendC::SyncAll();
    if (blockIdx < initNeedCoreNum) {
        InitMinusOne();
    }
    AscendC::SyncAll();

    if (blockIdx < scatterNeedCoreNum && coreRows > 0) {
        int64_t loops = (coreRows + perLoopRows - 1) / perLoopRows;
        currentLoopRows = perLoopRows;
        AssistInit();
        for (int64_t loop = 0; loop < loops - 1; loop++) {
            CopyIn(loop);
            Compute(loop);
            CopyOut();
        }
        currentLoopRows = lastLoopRows;
        CopyIn(loops - 1);
        Compute(loops - 1);
        CopyOut();
    }
    AscendC::SyncAll();
}

} // namespace MoeTokenPermute

#endif // MOE_SORTINDICES_SCATTER_MULTI_CORE_H
