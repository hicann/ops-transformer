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
 * \file grouped_matmul_antiquant_a8w4_msd_pre_nz.h
 * \brief
 */

#ifndef GROUPED_MATMUL_ANTIQUANT_A8W4_MSD_PRE_NZ_H
#define GROUPED_MATMUL_ANTIQUANT_A8W4_MSD_PRE_NZ_H

#include "kernel_operator.h"
#include "grouped_matmul_antiquant_a8w4_msd_pre.h"

#ifdef GMM_ANTI_QUANT_A8W4_MSD
namespace GROUPED_MATMUL {
using namespace AscendC;
constexpr uint32_t QUE_DEPTH = 1U;
constexpr uint32_t BUFFER_NUM_ONE = 1U;
constexpr uint32_t BUFFER_NUM_TWO = 2U;
constexpr uint32_t MASK_LEN = 256U;
constexpr int BUFFER_SIZE_256B = 256;
constexpr uint32_t DATA_BLOCK_SIZE_32_NZ = 32U;
constexpr uint32_t WITH_OFFSET_NZ = 1U;
constexpr uint32_t A8W4PRE_VECTOR_BASE_M = 16;
constexpr uint32_t A8W4PRE_VECTOR_BASE_K = 512;
constexpr int32_t MASK = 128;
constexpr half ONE_SIXTEEN = 0.0625;
constexpr uint32_t ALIGN_MTE = 32;
constexpr uint32_t kNZInt4BlockSize = 64;

class GMMA8W4PreProcessNZ {
public:
    __aicore__ inline GMMA8W4PreProcessNZ(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR x_out, GM_ADDR groupList, GM_ADDR workspace,
                                const GMMBaseParams &tilingData, TPipe *pipe);
    __aicore__ inline void Process();
    __aicore__ inline void copyIn(uint32_t mIdx, uint32_t kIdx, uint32_t curVecBaseM, uint32_t curVecBaseK);
    __aicore__ inline void compute(uint32_t mIdx, uint32_t kIdx, uint32_t curVecBaseM, uint32_t curVecBaseK);
    __aicore__ inline void copyOut(uint32_t mIdx, uint32_t kIdx, uint32_t curVecBaseM, uint32_t curVecBaseK);

private:
    TQue<QuePosition::VECIN, QUE_DEPTH> vecInQueueX;
    TQue<QuePosition::VECOUT, QUE_DEPTH> vecOutQueue;

    TQue<QuePosition::VECOUT, QUE_DEPTH> vecOutQueueRowSum;

    TBuf<TPosition::VECCALC> tempBuff;
    TBuf<TPosition::VECCALC> xHighHalfBuff;
    TBuf<TPosition::VECCALC> xLowHalfBuff;

    TBuf<TPosition::VECIN> groupListBuff;
    TBuf<TPosition::VECCALC> groupListFloatBuff;
    TBuf<TPosition::VECCALC> reduceSumWorkSpace;
    TBuf<TPosition::VECCALC> maskBuff;
    TBuf<TPosition::VECCALC> fp16Buff;

    LocalTensor<int8_t> xTensor;
    LocalTensor<half> xHighHalfTensor;
    LocalTensor<int4b_t> xHighI4Tensor;
    LocalTensor<half> xLowHalfTensor;
    LocalTensor<int16_t> xLowI16Tensor;
    LocalTensor<int4b_t> xLowI4Tensor;
    LocalTensor<int16_t> maskTensor;

    LocalTensor<int64_t> groupListTensor;
    LocalTensor<float> groupListFTensor;
    LocalTensor<float> workTensor;

    GlobalTensor<int8_t> xGm;
    GlobalTensor<int4b_t> xOutInt4;
    GlobalTensor<int64_t> groupListGm;
    GlobalTensor<float> xRowSumGm;

    uint32_t vectorBaseSize{A8W4PRE_VECTOR_BASE_M * A8W4PRE_VECTOR_BASE_K};
    uint32_t vectorBlockSizeAlign{DATA_BLOCK_SIZE_32_NZ};
    uint32_t totalGroup{0};
    uint32_t blockDim{0};
    uint32_t coreId{0};
    uint32_t currentNum{0};
    uint32_t startNum{0};
    uint32_t groupNum{0};
    uint32_t xRowSumRepeatNum{0};
    uint32_t withOffset{0};
    uint32_t m{0};
    uint32_t k{0};
};

__aicore__ inline void GMMA8W4PreProcessNZ::Init(GM_ADDR x, GM_ADDR x_out, GM_ADDR groupList, GM_ADDR workspace,
                                                 const GMMBaseParams &tilingData, TPipe *pipe)
{
    groupListGm.SetGlobalBuffer((__gm__ int64_t *)groupList);
    xGm.SetGlobalBuffer(GetTensorAddr<int8_t>(0, x));
    xOutInt4.SetGlobalBuffer((__gm__ int4b_t *)((__gm__ int8_t *)x_out));
    groupNum = static_cast<uint32_t>(tilingData.groupNum);
    withOffset = tilingData.withOffset;
    m = tilingData.m;
    k = tilingData.k;

    pipe->InitBuffer(vecInQueueX, BUFFER_NUM_TWO, vectorBaseSize * sizeof(int8_t));
    pipe->InitBuffer(vecOutQueue, BUFFER_NUM_TWO, vectorBaseSize * sizeof(int4b_t) * 2);

    pipe->InitBuffer(groupListBuff, Ceil(static_cast<uint32_t>(groupNum * sizeof(int64_t)), DATA_BLOCK_SIZE_32_NZ) *
                                        DATA_BLOCK_SIZE_32_NZ);
    pipe->InitBuffer(groupListFloatBuff, Ceil(static_cast<uint32_t>(groupNum * sizeof(float)), DATA_BLOCK_SIZE_32_NZ) *
                                             DATA_BLOCK_SIZE_32_NZ);
    pipe->InitBuffer(reduceSumWorkSpace, BUFFER_SIZE_256B * sizeof(float));

    pipe->InitBuffer(xHighHalfBuff, A8W4PRE_VECTOR_BASE_M * A8W4PRE_VECTOR_BASE_K * sizeof(half));
    xHighHalfTensor = xHighHalfBuff.Get<half>();
    xHighI4Tensor = xHighHalfBuff.Get<int4b_t>();

    pipe->InitBuffer(xLowHalfBuff, A8W4PRE_VECTOR_BASE_M * A8W4PRE_VECTOR_BASE_K * sizeof(half));
    xLowHalfTensor = xLowHalfBuff.Get<half>();
    xLowI16Tensor = xLowHalfBuff.GetWithOffset<int16_t>(A8W4PRE_VECTOR_BASE_M * A8W4PRE_VECTOR_BASE_K / 2,
                                                        A8W4PRE_VECTOR_BASE_M * A8W4PRE_VECTOR_BASE_K);
    xLowI4Tensor = xLowHalfBuff.Get<int4b_t>();

    pipe->InitBuffer(maskBuff, MASK_LEN);
    maskTensor = maskBuff.Get<int16_t>();
    Duplicate(maskTensor, static_cast<int16_t>(0x0F0F), MASK);

    startNum = GetBlockIdx();
    blockDim = GetBlockNum() * GetTaskRation();
}

__aicore__ inline void GMMA8W4PreProcessNZ::copyIn(uint32_t mIdx, uint32_t kIdx, uint32_t curVecBaseM,
                                                   uint32_t curVecBaseK)
{
    DataCopyParams dataCopyParams{static_cast<uint16_t>(curVecBaseM),
                                  static_cast<uint16_t>(curVecBaseK * sizeof(int8_t)),
                                  static_cast<uint16_t>(k - curVecBaseK * sizeof(int8_t)), 0};
    DataCopyPadParams padParams{false, 0, 0, 0};
    auto xTensor = vecInQueueX.AllocTensor<int8_t>();
    DataCopyPad(xTensor, xGm[mIdx * A8W4PRE_VECTOR_BASE_M * k + kIdx * A8W4PRE_VECTOR_BASE_K], dataCopyParams,
                padParams);
    vecInQueueX.EnQue(xTensor);
}
__aicore__ inline void GMMA8W4PreProcessNZ::compute(uint32_t mIdx, uint32_t kIdx, uint32_t curVecBaseM,
                                                    uint32_t curVecBaseK)
{
    auto xTensor = vecInQueueX.DeQue<int8_t>();
    auto totalLen = curVecBaseM * curVecBaseK;
    const size_t LEN_VK = (totalLen / 2) / LEN_128;
    const size_t LAST_LEN_VK = (totalLen % MASK_LEN) / 2;

    Cast(xHighHalfTensor, xTensor, AscendC::RoundMode::CAST_NONE, totalLen);
    PipeBarrier<PIPE_V>();
    Muls(xHighHalfTensor, xHighHalfTensor, ONE_SIXTEEN, totalLen);
    PipeBarrier<PIPE_V>();
    Cast(xHighI4Tensor, xHighHalfTensor, AscendC::RoundMode::CAST_FLOOR, totalLen);

    And(xLowI16Tensor, xTensor.ReinterpretCast<int16_t>(), maskTensor, LEN_128, LEN_VK, {1, 1, 1, 8, 8, 0});
    if (LAST_LEN_VK > 0) {
        And(xLowI16Tensor[LEN_VK * LEN_128], xTensor[LEN_VK * LEN_128 * TWO].ReinterpretCast<int16_t>(), maskTensor,
            LAST_LEN_VK, 1, {1, 1, 1, 8, 8, 0});
    }
    PipeBarrier<PIPE_V>();

    vecInQueueX.FreeTensor(xTensor);
    Cast(xLowHalfTensor, xLowI16Tensor.ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE, totalLen);

    PipeBarrier<PIPE_V>();
    const half MINUS_EIGHT = static_cast<half>(-8);
    Adds(xLowHalfTensor, xLowHalfTensor, MINUS_EIGHT, totalLen);

    PipeBarrier<PIPE_V>();
    Cast(xLowI4Tensor, xLowHalfTensor, AscendC::RoundMode::CAST_NONE, totalLen);
    PipeBarrier<PIPE_V>();
    auto tensorOut = vecOutQueue.AllocTensor<int4b_t>();

    DataCopyParams dataCopyParams{static_cast<uint16_t>(curVecBaseM), 1,
                                  static_cast<uint16_t>(curVecBaseK / kNZInt4BlockSize - 1), 1};
    auto loopCnt = curVecBaseK / kNZInt4BlockSize;
    for (int i = 0; i < loopCnt; i++) {
        DataCopy(tensorOut[i * curVecBaseM * 2 * kNZInt4BlockSize], xHighI4Tensor[i * kNZInt4BlockSize],
                 dataCopyParams);
        DataCopy(tensorOut[kNZInt4BlockSize + i * curVecBaseM * 2 * kNZInt4BlockSize],
                 xLowI4Tensor[i * kNZInt4BlockSize], dataCopyParams);
    }

    vecOutQueue.EnQue(tensorOut);
}
__aicore__ inline void GMMA8W4PreProcessNZ::copyOut(uint32_t mIdx, uint32_t kIdx, uint32_t curVecBaseM,
                                                    uint32_t curVecBaseK)
{
    auto m_Align_8 = AlignUp<8>(m);
    auto tensorOut = vecOutQueue.DeQue<int4b_t>();
    auto baseOffset =
        mIdx * A8W4PRE_VECTOR_BASE_M * kNZInt4BlockSize * 2 + kIdx * A8W4PRE_VECTOR_BASE_K * m_Align_8 * 2;

    baseOffset = (baseOffset + ALIGN_MTE - 1) / ALIGN_MTE * ALIGN_MTE;
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(curVecBaseK / kNZInt4BlockSize),
                                     curVecBaseM * kNZInt4BlockSize, 0, (m_Align_8 - curVecBaseM) * 2 * 32, 0};

    DataCopyPadExtParams<int4b_t> padParams{false, 0, 0, 0};

    DataCopyPad(xOutInt4[baseOffset], tensorOut, dataCopyParams);
    vecOutQueue.FreeTensor(tensorOut);
}

__aicore__ inline void GMMA8W4PreProcessNZ::Process()
{
    groupListTensor = groupListBuff.Get<int64_t>();
    groupListFTensor = groupListFloatBuff.Get<float>();
    workTensor = reduceSumWorkSpace.Get<float>();
    DataCopyParams dataCopyParams{1, static_cast<uint16_t>(groupNum * sizeof(int64_t)), 0, 0};
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyPad(groupListTensor, groupListGm, dataCopyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(groupListFTensor, groupListTensor, AscendC::RoundMode::CAST_ROUND, groupNum);
    PipeBarrier<PIPE_V>();
    ReduceSum(groupListFTensor, groupListFTensor, workTensor, groupNum);
    PipeBarrier<PIPE_V>();
    Cast(groupListTensor, groupListFTensor, AscendC::RoundMode::CAST_ROUND, 1);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    totalGroup = groupListTensor.GetValue(0);
    uint32_t m_Align_8 = AlignUp<8>(m);
    uint32_t mDim = Ceil(totalGroup, A8W4PRE_VECTOR_BASE_M);
    uint32_t kDim = Ceil(k, A8W4PRE_VECTOR_BASE_K);
    uint32_t mIdx = 0;
    uint32_t kIdx = 0;
    uint32_t totalTaskNum = mDim * kDim;
    for (uint32_t taskIdx = startNum; taskIdx < totalTaskNum; taskIdx += blockDim) {
        uint32_t curVecBaseM = A8W4PRE_VECTOR_BASE_M;
        uint32_t curVecBaseK = A8W4PRE_VECTOR_BASE_K;
        mIdx = taskIdx / kDim;
        kIdx = taskIdx % kDim;
        if (mIdx >= mDim - 1) {
            curVecBaseM = totalGroup - mIdx * A8W4PRE_VECTOR_BASE_M;
        }
        if (kIdx >= kDim - 1) {
            curVecBaseK = k - kIdx * A8W4PRE_VECTOR_BASE_K;
        }
        copyIn(mIdx, kIdx, curVecBaseM, curVecBaseK);
        compute(mIdx, kIdx, curVecBaseM, curVecBaseK);
        copyOut(mIdx, kIdx, curVecBaseM, curVecBaseK);
    }
    SyncAll<false>();
}

} // namespace GROUPED_MATMUL
#endif
#endif
