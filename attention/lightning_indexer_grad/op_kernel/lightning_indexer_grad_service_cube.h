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
 * \file lightning_indexer_grad_service_cube.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef LIGHTNING_INDEXER_GRAD_SERVICE_CUBE_H
#define LIGHTNING_INDEXER_GRAD_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_grad_common.h"

namespace LigKernel {
using namespace LIGCommon;
using namespace AscendC;

template <typename LIGT>
class LIGMatmul {
public:
    using dataType = typename LIGT::dataType;

    __aicore__ inline LIGMatmul(){};
    __aicore__ inline void Init(TPipe *pipe);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void Cube1(GlobalTensor<dataType> leftMatrixGmTensor, GlobalTensor<dataType> rightMatrixGmTensor,
                                 GlobalTensor<float> outputMatrixGmTensor, ConstInfo constInfo, RunInfo runInfo);
    __aicore__ inline void Cube2(GlobalTensor<dataType> leftMatrixGmTensor, GlobalTensor<dataType> rightMatrixGmTensor,
                                 GlobalTensor<dataType> outputMatrixGmTensor, ConstInfo constInfo, RunInfo runInfo);
    __aicore__ inline void Cube3(GlobalTensor<dataType> leftMatrixGmTensor, GlobalTensor<dataType> rightMatrixGmTensor,
                                 GlobalTensor<dataType> outputMatrixGmTensor, ConstInfo constInfo, RunInfo runInfo);
    __aicore__ inline void Cube4(GlobalTensor<dataType> leftMatrixGmTensor, GlobalTensor<dataType> rightMatrixGmTensor,
                                 GlobalTensor<float> outputMatrixGmTensor, ConstInfo constInfo, RunInfo runInfo);

    static constexpr uint32_t BASIC_BLOCK_LENGTH = 128;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t C0_SIZE = 16;

    // flag shift detail
    static constexpr uint32_t A_FLAG_SHIFT = 0;
    static constexpr uint32_t A_CONFLICT_FLAG_SHIFT = 3;
    static constexpr uint32_t B_FLAG_SHIFT = 2;
    static constexpr uint32_t B_CONFLICT_FLAG_SHIFT = 5;

protected:
    TPipe *pipe;

    TBuf<TPosition::A1> leftMatrixL1Buf;
    LocalTensor<dataType> leftMatrixL1PingTensor;
    LocalTensor<dataType> leftMatrixL1PongTensor;

    TBuf<TPosition::B1> rightMatrixL1Buf;
    LocalTensor<dataType> rightMatrixL1PingTensor;
    LocalTensor<dataType> rightMatrixL1PongTensor;

    TBuf<TPosition::A2> leftMatrixL0ABuf;
    LocalTensor<dataType> leftMatrixL0APingTensor;
    LocalTensor<dataType> leftMatrixL0APongTensor;

    TBuf<TPosition::B2> rightMatrixL0BBuf;
    LocalTensor<dataType> rightMatrixL0BPingTensor;
    LocalTensor<dataType> rightMatrixL0BPongTensor;

    TBuf<TPosition::CO1> outputMatrixL0CBuf;
    LocalTensor<float> outputMatrixL0CPingTensor;
    LocalTensor<float> outputMatrixL0CPongTensor;

    AscendC::Nd2NzParams commonNd2NzParams{
        1, BASIC_BLOCK_LENGTH, BASIC_BLOCK_LENGTH, 0, BASIC_BLOCK_LENGTH, BASIC_BLOCK_LENGTH, 1, 0};

    AscendC::LoadData2dParams commonLoadData2dParamsTranspose{0, BASIC_BLOCK_LENGTH, BASIC_BLOCK_LENGTH, 0, 0, true, 0};

    AscendC::LoadData2dParams commonLoadData2dParamsNoTranspose{0, BASIC_BLOCK_LENGTH, BASIC_BLOCK_LENGTH, 0, 0, false,
                                                                0};

    AscendC::MmadParams commonMadParams{BASIC_BLOCK_LENGTH, BASIC_BLOCK_LENGTH, BASIC_BLOCK_LENGTH, 3, false, true};

    AscendC::FixpipeParamsV220 commonFixpipeParamsV220{BASIC_BLOCK_LENGTH, BASIC_BLOCK_LENGTH, BASIC_BLOCK_LENGTH,
                                                       BASIC_BLOCK_LENGTH, false};
};

template <typename LIT>
__aicore__ inline void LIGMatmul<LIT>::Init(TPipe *pipeIn)
{
    pipe = pipeIn;
#if __CCE_AICORE__ == 310
#else
    AscendC::SetLoadDataPaddingValue<uint64_t>(0);
    AscendC::SetNdParaImpl(0x1);
#endif
    InitBuffers();

    commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
    commonFixpipeParamsV220.unitFlag = 3;
}

template <typename LIT>
__aicore__ inline void LIGMatmul<LIT>::InitBuffers()
{
    pipe->InitBuffer(leftMatrixL1Buf, DB * BASIC_BLOCK_LENGTH * BASIC_BLOCK_LENGTH * sizeof(dataType));
    leftMatrixL1PingTensor = leftMatrixL1Buf.Get<dataType>();
    leftMatrixL1PongTensor = leftMatrixL1PingTensor[BASIC_BLOCK_LENGTH * BASIC_BLOCK_LENGTH];

    pipe->InitBuffer(rightMatrixL1Buf, DB * BASIC_BLOCK_LENGTH * BASIC_BLOCK_LENGTH * sizeof(dataType));
    rightMatrixL1PingTensor = rightMatrixL1Buf.Get<dataType>();
    rightMatrixL1PongTensor = rightMatrixL1PingTensor[BASIC_BLOCK_LENGTH * BASIC_BLOCK_LENGTH];

    pipe->InitBuffer(leftMatrixL0ABuf, DB * BASIC_BLOCK_LENGTH * BASIC_BLOCK_LENGTH * sizeof(dataType));
    leftMatrixL0APingTensor = leftMatrixL0ABuf.Get<dataType>();
    leftMatrixL0APongTensor = leftMatrixL0APingTensor[BASIC_BLOCK_LENGTH * BASIC_BLOCK_LENGTH];

    pipe->InitBuffer(rightMatrixL0BBuf, DB * BASIC_BLOCK_LENGTH * BASIC_BLOCK_LENGTH * sizeof(dataType));
    rightMatrixL0BPingTensor = rightMatrixL0BBuf.Get<dataType>();
    rightMatrixL0BPongTensor = rightMatrixL0BPingTensor[BASIC_BLOCK_LENGTH * BASIC_BLOCK_LENGTH];

    pipe->InitBuffer(outputMatrixL0CBuf, DB * BASIC_BLOCK_LENGTH * BASIC_BLOCK_LENGTH * sizeof(float));
    outputMatrixL0CPingTensor = outputMatrixL0CBuf.Get<float>();
    outputMatrixL0CPongTensor = outputMatrixL0CPingTensor[BASIC_BLOCK_LENGTH * BASIC_BLOCK_LENGTH];
}

template <typename LIGT>
__aicore__ inline void LIGMatmul<LIGT>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);

    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID6);
}

template <typename LIGT>
__aicore__ inline void LIGMatmul<LIGT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);

    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID6);
}

// (G, 128) @ (2048, 128)^T -> (G, 2048)
// baseM = G, baseN = 128, baseK = 128
template <typename LIGT>
__aicore__ inline void LIGMatmul<LIGT>::Cube1(GlobalTensor<dataType> leftMatrixGmTensor,
                                              GlobalTensor<dataType> rightMatrixGmTensor,
                                              GlobalTensor<float> outputMatrixGmTensor, ConstInfo constInfo,
                                              RunInfo runInfo)
{
    uint32_t singleM = constInfo.groupNum;
    uint32_t singleK = constInfo.headDim;
    uint32_t singleN = runInfo.realTopk;
    uint32_t baseM = constInfo.groupNum;
    uint32_t baseN = BASIC_BLOCK_LENGTH;
    uint32_t baseK = constInfo.headDim;
    uint32_t rightMatrixPingPong = 0;

    // [B, S1, N1, D]
    uint64_t leftMatrixGmOffset = 0;
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        leftMatrixGmOffset = runInfo.bIdx * constInfo.seqlenQ * constInfo.headNumQ * constInfo.headDim +
                             runInfo.s1Idx * constInfo.headNumQ * constInfo.headDim +
                             runInfo.n2Idx * constInfo.groupNum * constInfo.headDim;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        leftMatrixGmOffset = (runInfo.prefixSumS1 + runInfo.s1Idx) * constInfo.headNumQ * constInfo.headDim +
                             runInfo.n2Idx * constInfo.groupNum * constInfo.headDim;
    }

    // load A matrix(query) from gm to L1
    WaitFlag<HardEvent::MTE1_MTE2>(A_FLAG_SHIFT);
    commonNd2NzParams.nValue = baseM;
    commonNd2NzParams.dValue = baseK;
    commonNd2NzParams.srcDValue = constInfo.headDim;
    commonNd2NzParams.dstNzC0Stride = RoundUp(baseM, C0_SIZE);
    AscendC::DataCopy(leftMatrixL1PingTensor, leftMatrixGmTensor[leftMatrixGmOffset], commonNd2NzParams);
    SetFlag<HardEvent::MTE2_MTE1>(A_FLAG_SHIFT);

    // load A matrix from L1 -> L0A
    WaitFlag<HardEvent::MTE2_MTE1>(A_FLAG_SHIFT);
    WaitFlag<HardEvent::M_MTE1>(A_CONFLICT_FLAG_SHIFT);
#if __CCE_AICORE__ == 310
    AscendC::LoadData2DParamsV2 loadParamsA1;
    loadParamsA1.mStartPosition = 0;
    loadParamsA1.kStartPosition = 0;
    loadParamsA1.mStep = RoundUp(baseM, C0_SIZE) / C0_SIZE;
    loadParamsA1.kStep = RoundUp(baseK, C0_SIZE) / C0_SIZE;
    loadParamsA1.srcStride = RoundUp(baseM, C0_SIZE) / C0_SIZE;
    loadParamsA1.dstStride = RoundUp(baseM, C0_SIZE) / C0_SIZE;
    loadParamsA1.ifTranspose = false;
    AscendC::LoadData(leftMatrixL0APingTensor, leftMatrixL1PingTensor, loadParamsA1);
#else
    commonLoadData2dParamsNoTranspose.repeatTimes = baseK / C0_SIZE;
    commonLoadData2dParamsNoTranspose.srcStride = RoundUp(baseM, C0_SIZE) / C0_SIZE;
    for (int32_t i = 0; i < RoundUp(baseM, C0_SIZE) / C0_SIZE; i++) {
        AscendC::LoadData(leftMatrixL0APingTensor[i * C0_SIZE * baseK], leftMatrixL1PingTensor[i * C0_SIZE * C0_SIZE],
                          commonLoadData2dParamsNoTranspose);
    }
#endif
    SetFlag<HardEvent::MTE1_MTE2>(A_FLAG_SHIFT);
    SetFlag<HardEvent::MTE1_M>(A_FLAG_SHIFT);

    uint64_t loopN = LIGCommon::Align(singleN, baseN) / baseN;
    uint32_t baseNTail = (singleN % baseN == 0) ? baseN : singleN % baseN;
    for (uint32_t i = 0; i < loopN; i++) {
        LocalTensor<dataType> rightMatrixL1Tensor =
            (rightMatrixPingPong & 1) ? rightMatrixL1PingTensor : rightMatrixL1PongTensor;
        LocalTensor<dataType> rightMatrixL0BTensor =
            (rightMatrixPingPong & 1) ? rightMatrixL0BPingTensor : rightMatrixL0BPongTensor;
        LocalTensor<float> outputMatrixL0CTensor =
            (rightMatrixPingPong & 1) ? outputMatrixL0CPingTensor : outputMatrixL0CPongTensor;

        uint32_t realN = (i == loopN - 1) ? baseNTail : baseN;
        // load B matrix(keyGather) from GM to L1
        WaitFlag<HardEvent::MTE1_MTE2>(B_FLAG_SHIFT + rightMatrixPingPong);
        commonNd2NzParams.nValue = realN;
        commonNd2NzParams.dValue = baseK;
        commonNd2NzParams.srcDValue = constInfo.headDim;
        commonNd2NzParams.dstNzC0Stride = RoundUp(realN, C0_SIZE);
        AscendC::DataCopy(rightMatrixL1Tensor, rightMatrixGmTensor[i * baseN * constInfo.headDim], commonNd2NzParams);
        SetFlag<HardEvent::MTE2_MTE1>(B_FLAG_SHIFT + rightMatrixPingPong);

        // load B matrix from L1 -> l0B
        WaitFlag<HardEvent::MTE2_MTE1>(B_FLAG_SHIFT + rightMatrixPingPong);
        WaitFlag<HardEvent::M_MTE1>(B_CONFLICT_FLAG_SHIFT + rightMatrixPingPong);
        commonLoadData2dParamsNoTranspose.repeatTimes = RoundUp(realN, C0_SIZE) / C0_SIZE;
        commonLoadData2dParamsNoTranspose.srcStride = 1;
        for (int i = 0; i < baseK / C0_SIZE; i++) {
            AscendC::LoadData(rightMatrixL0BTensor[i * RoundUp(realN, C0_SIZE) * C0_SIZE],
                              rightMatrixL1Tensor[i * RoundUp(realN, C0_SIZE) * C0_SIZE],
                              commonLoadData2dParamsNoTranspose);
        }
        SetFlag<HardEvent::MTE1_MTE2>(B_FLAG_SHIFT + rightMatrixPingPong);
        SetFlag<HardEvent::MTE1_M>(B_FLAG_SHIFT + rightMatrixPingPong);

        // mad (M, N, K)
        if (i == 0) {
            WaitFlag<HardEvent::MTE1_M>(A_FLAG_SHIFT);
        }
        WaitFlag<HardEvent::MTE1_M>(B_FLAG_SHIFT + rightMatrixPingPong);
        uint32_t realM = (baseM == 1) ? 2 : baseM;
        commonMadParams.m = realM;
        commonMadParams.n = realN;
        commonMadParams.k = baseK;
        commonMadParams.unitFlag = 3;
        commonMadParams.cmatrixInitVal = true;
        AscendC::Mmad(outputMatrixL0CTensor, leftMatrixL0APingTensor, rightMatrixL0BTensor, commonMadParams);

        SetFlag<HardEvent::M_MTE1>(B_CONFLICT_FLAG_SHIFT + rightMatrixPingPong);
        if (i == (loopN - 1)) {
            SetFlag<HardEvent::M_MTE1>(A_CONFLICT_FLAG_SHIFT);
        }

        // FixPipe  L0C -> GM
        commonFixpipeParamsV220.mSize = baseM;
        commonFixpipeParamsV220.nSize = realN;
        commonFixpipeParamsV220.srcStride = RoundUp(baseM, C0_SIZE);
        commonFixpipeParamsV220.dstStride = runInfo.realTopk;
        commonFixpipeParamsV220.reluEn = true;
        commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
        commonFixpipeParamsV220.unitFlag = 3;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(outputMatrixGmTensor[i * baseN], outputMatrixL0CTensor,
                                                               commonFixpipeParamsV220);

        rightMatrixPingPong = 1 - rightMatrixPingPong;
    }
}

// (G, 1) @ (1, K) -> (G, K)
template <typename LIGT>
__aicore__ inline void LIGMatmul<LIGT>::Cube2(GlobalTensor<dataType> leftMatrixGmTensor,
                                              GlobalTensor<dataType> rightMatrixGmTensor,
                                              GlobalTensor<dataType> outputMatrixGmTensor, ConstInfo constInfo,
                                              RunInfo runInfo)
{
    uint32_t singleM = constInfo.groupNum;
    uint32_t singleK = 1;
    uint32_t singleN = runInfo.realTopk;
    uint32_t baseM = constInfo.groupNum;
    uint32_t baseN = BASIC_BLOCK_LENGTH;
    uint32_t baseK = 16;
    uint32_t rightMatrixPingPong = 0;

    // [B, S1, N1]
    uint64_t leftMatrixGmOffset = 0;
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        leftMatrixGmOffset = runInfo.bIdx * constInfo.seqlenQ * constInfo.headNumQ +
                             runInfo.s1Idx * constInfo.headNumQ + runInfo.n2Idx * constInfo.groupNum;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        leftMatrixGmOffset =
            (runInfo.prefixSumS1 + runInfo.s1Idx) * constInfo.headNumQ + runInfo.n2Idx * constInfo.groupNum;
    }

    // [B, S1, K]
    uint64_t rightMatrixGmOffset = 0;
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        rightMatrixGmOffset = runInfo.bIdx * constInfo.seqlenQ * constInfo.topK + runInfo.s1Idx * constInfo.topK;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        rightMatrixGmOffset = (runInfo.prefixSumS1 + runInfo.s1Idx) * constInfo.topK;
    }

    // load A matrix(weights) from gm to L1
    WaitFlag<HardEvent::MTE1_MTE2>(A_FLAG_SHIFT);
    commonNd2NzParams.nValue = baseM;
    commonNd2NzParams.dValue = 1;
    commonNd2NzParams.srcDValue = 1;
    commonNd2NzParams.dstNzC0Stride = RoundUp(baseM, C0_SIZE);
    AscendC::DataCopy(leftMatrixL1PingTensor, leftMatrixGmTensor[leftMatrixGmOffset], commonNd2NzParams);
    SetFlag<HardEvent::MTE2_MTE1>(A_FLAG_SHIFT);

    // load A matrix from L1 -> L0A
    WaitFlag<HardEvent::MTE2_MTE1>(A_FLAG_SHIFT);
    WaitFlag<HardEvent::M_MTE1>(A_CONFLICT_FLAG_SHIFT);
#if __CCE_AICORE__ == 310
    AscendC::LoadData2DParamsV2 loadParamsA2;
    loadParamsA2.mStartPosition = 0;
    loadParamsA2.kStartPosition = 0;
    loadParamsA2.mStep = RoundUp(baseM, C0_SIZE) / C0_SIZE;
    loadParamsA2.kStep = RoundUp(baseK, C0_SIZE) / C0_SIZE;
    loadParamsA2.srcStride = RoundUp(baseM, C0_SIZE) / C0_SIZE;
    loadParamsA2.dstStride = RoundUp(baseM, C0_SIZE) / C0_SIZE;
    loadParamsA2.ifTranspose = false;
    AscendC::LoadData(leftMatrixL0APingTensor, leftMatrixL1PingTensor, loadParamsA2);
#else
    commonLoadData2dParamsNoTranspose.repeatTimes = baseK / C0_SIZE;
    commonLoadData2dParamsNoTranspose.srcStride = RoundUp(baseM, C0_SIZE) / C0_SIZE;
    for (int32_t i = 0; i < RoundUp(baseM, C0_SIZE) / C0_SIZE; i++) {
        AscendC::LoadData(leftMatrixL0APingTensor[i * C0_SIZE * baseK], leftMatrixL1PingTensor[i * C0_SIZE * C0_SIZE],
                          commonLoadData2dParamsNoTranspose);
    }
#endif
    SetFlag<HardEvent::MTE1_MTE2>(A_FLAG_SHIFT);
    SetFlag<HardEvent::MTE1_M>(A_FLAG_SHIFT);

    uint64_t loopN = LIGCommon::Align(singleN, baseN) / baseN;
    uint32_t baseNTail = (singleN % baseN == 0) ? baseN : singleN % baseN;
    for (uint32_t i = 0; i < loopN; i++) {
        LocalTensor<dataType> rightMatrixL1Tensor =
            (rightMatrixPingPong & 1) ? rightMatrixL1PingTensor : rightMatrixL1PongTensor;
        LocalTensor<dataType> rightMatrixL0BTensor =
            (rightMatrixPingPong & 1) ? rightMatrixL0BPingTensor : rightMatrixL0BPongTensor;
        LocalTensor<float> outputMatrixL0CTensor =
            (rightMatrixPingPong & 1) ? outputMatrixL0CPingTensor : outputMatrixL0CPongTensor;

        uint32_t realN = (i == loopN - 1) ? baseNTail : baseN;
        // load B matrix(dOut) from GM to L1
        WaitFlag<HardEvent::MTE1_MTE2>(B_FLAG_SHIFT + rightMatrixPingPong);
        commonNd2NzParams.nValue = realN;
        commonNd2NzParams.dValue = 1;
        commonNd2NzParams.srcDValue = 1;
        commonNd2NzParams.dstNzC0Stride = RoundUp(realN, C0_SIZE);
        AscendC::DataCopy(rightMatrixL1Tensor, rightMatrixGmTensor[rightMatrixGmOffset + i * baseN * 1],
                          commonNd2NzParams);
        SetFlag<HardEvent::MTE2_MTE1>(B_FLAG_SHIFT + rightMatrixPingPong);

        // load B matrix from L1 -> l0B
        WaitFlag<HardEvent::MTE2_MTE1>(B_FLAG_SHIFT + rightMatrixPingPong);
        WaitFlag<HardEvent::M_MTE1>(B_CONFLICT_FLAG_SHIFT + rightMatrixPingPong);
        commonLoadData2dParamsNoTranspose.repeatTimes = RoundUp(realN, C0_SIZE) / C0_SIZE;
        commonLoadData2dParamsNoTranspose.srcStride = 1;
        for (int i = 0; i < baseK / C0_SIZE; i++) {
            AscendC::LoadData(rightMatrixL0BTensor[i * RoundUp(realN, C0_SIZE) * C0_SIZE],
                              rightMatrixL1Tensor[i * RoundUp(realN, C0_SIZE) * C0_SIZE],
                              commonLoadData2dParamsNoTranspose);
        }
        SetFlag<HardEvent::MTE1_MTE2>(B_FLAG_SHIFT + rightMatrixPingPong);
        SetFlag<HardEvent::MTE1_M>(B_FLAG_SHIFT + rightMatrixPingPong);

        // mad (M, N, K)
        if (i == 0) {
            WaitFlag<HardEvent::MTE1_M>(A_FLAG_SHIFT);
        }
        WaitFlag<HardEvent::MTE1_M>(B_FLAG_SHIFT + rightMatrixPingPong);
        uint32_t realM = (baseM == 1) ? 2 : baseM;
        commonMadParams.m = realM;
        commonMadParams.n = realN;
        commonMadParams.k = 1;
        commonMadParams.unitFlag = 3;
        commonMadParams.cmatrixInitVal = true;
        AscendC::Mmad(outputMatrixL0CTensor, leftMatrixL0APingTensor, rightMatrixL0BTensor, commonMadParams);

        SetFlag<HardEvent::M_MTE1>(B_CONFLICT_FLAG_SHIFT + rightMatrixPingPong);
        if (i == loopN - 1) {
            SetFlag<HardEvent::M_MTE1>(A_CONFLICT_FLAG_SHIFT);
        }

        // FixPipe  L0C -> GM
        commonFixpipeParamsV220.mSize = baseM;
        commonFixpipeParamsV220.nSize = realN;
        commonFixpipeParamsV220.srcStride = RoundUp(baseM, C0_SIZE);
        commonFixpipeParamsV220.dstStride = runInfo.realTopk;
        commonFixpipeParamsV220.reluEn = false;
        if constexpr (std::is_same_v<dataType, half>) {
            commonFixpipeParamsV220.quantPre = QuantMode_t::F322F16;
        } else if constexpr (std::is_same_v<dataType, bfloat16_t>) {
            commonFixpipeParamsV220.quantPre = QuantMode_t::F322BF16;
        }
        commonFixpipeParamsV220.unitFlag = 3;
        AscendC::Fixpipe<dataType, float, AscendC::CFG_ROW_MAJOR>(outputMatrixGmTensor[i * baseN],
                                                                  outputMatrixL0CTensor, commonFixpipeParamsV220);
        rightMatrixPingPong = 1 - rightMatrixPingPong;
    }
}

// (G, K) @ (K, 128) -> (G, 128)
// L0C one buffer to add
template <typename LIGT>
__aicore__ inline void LIGMatmul<LIGT>::Cube3(GlobalTensor<dataType> leftMatrixGmTensor,
                                              GlobalTensor<dataType> rightMatrixGmTensor,
                                              GlobalTensor<dataType> outputMatrixGmTensor, ConstInfo constInfo,
                                              RunInfo runInfo)
{
    uint32_t singleM = constInfo.groupNum;
    uint32_t singleK = runInfo.realTopk;
    uint32_t singleN = constInfo.headDim;
    uint32_t baseM = constInfo.groupNum;
    uint32_t baseN = constInfo.headDim;
    uint32_t baseK = BASIC_BLOCK_LENGTH;
    uint32_t leftMatrixPingPong = 0;
    uint32_t rightMatrixPingPong = 0;

    // [B, S1, N1, D]
    uint64_t outputMatrixGmOffset = 0;
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        outputMatrixGmOffset = runInfo.bIdx * constInfo.seqlenQ * constInfo.headNumQ * constInfo.headDim +
                               runInfo.s1Idx * constInfo.headNumQ * constInfo.headDim +
                               runInfo.n2Idx * constInfo.groupNum * constInfo.headDim;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        outputMatrixGmOffset = (runInfo.prefixSumS1 + runInfo.s1Idx) * constInfo.headNumQ * constInfo.headDim +
                               runInfo.n2Idx * constInfo.groupNum * constInfo.headDim;
    }

    uint64_t loopK = LIGCommon::Align(singleK, baseK) / baseK;
    uint32_t baseKTail = (singleK % baseK == 0) ? baseK : singleK % baseK;
    for (uint32_t i = 0; i < loopK; i++) {
        LocalTensor<dataType> leftMatrixL1Tensor =
            (leftMatrixPingPong & 1) ? leftMatrixL1PingTensor : leftMatrixL1PongTensor;
        LocalTensor<dataType> leftMatrixL0ATensor =
            (leftMatrixPingPong & 1) ? leftMatrixL0APingTensor : leftMatrixL0APongTensor;
        LocalTensor<dataType> rightMatrixL1Tensor =
            (rightMatrixPingPong & 1) ? rightMatrixL1PingTensor : rightMatrixL1PongTensor;
        LocalTensor<dataType> rightMatrixL0BTensor =
            (rightMatrixPingPong & 1) ? rightMatrixL0BPingTensor : rightMatrixL0BPongTensor;
        LocalTensor<float> outputMatrixL0CTensor = outputMatrixL0CPingTensor;

        uint32_t realK = (i == loopK - 1) ? baseKTail : baseK;
        // load A matrix from gm to L1
        WaitFlag<HardEvent::MTE1_MTE2>(A_FLAG_SHIFT + leftMatrixPingPong);
        commonNd2NzParams.nValue = baseM;
        commonNd2NzParams.dValue = realK;
        commonNd2NzParams.srcDValue = runInfo.realTopk;
        commonNd2NzParams.dstNzC0Stride = RoundUp(baseM, C0_SIZE);
        AscendC::DataCopy(leftMatrixL1Tensor, leftMatrixGmTensor[i * baseK], commonNd2NzParams);
        SetFlag<HardEvent::MTE2_MTE1>(A_FLAG_SHIFT + leftMatrixPingPong);

        // load A matrix from L1 -> L0A
        WaitFlag<HardEvent::MTE2_MTE1>(A_FLAG_SHIFT + leftMatrixPingPong);
        WaitFlag<HardEvent::M_MTE1>(A_CONFLICT_FLAG_SHIFT + leftMatrixPingPong);
#if __CCE_AICORE__ == 310
        AscendC::LoadData2DParamsV2 loadParamsA;
        loadParamsA.mStartPosition = 0;
        loadParamsA.kStartPosition = 0;
        loadParamsA.mStep = RoundUp(baseM, C0_SIZE) / C0_SIZE;
        loadParamsA.kStep = RoundUp(realK, C0_SIZE) / C0_SIZE;
        loadParamsA.srcStride = RoundUp(baseM, C0_SIZE) / C0_SIZE;
        loadParamsA.dstStride = RoundUp(baseM, C0_SIZE) / C0_SIZE;
        loadParamsA.ifTranspose = false;
        AscendC::LoadData(leftMatrixL0ATensor, leftMatrixL1Tensor, loadParamsA);
#else
        commonLoadData2dParamsNoTranspose.repeatTimes = RoundUp(realK, C0_SIZE) / C0_SIZE;
        commonLoadData2dParamsNoTranspose.srcStride = RoundUp(baseM, C0_SIZE) / C0_SIZE;
        for (int32_t i = 0; i < RoundUp(baseM, C0_SIZE) / C0_SIZE; i++) {
            AscendC::LoadData(leftMatrixL0ATensor[i * C0_SIZE * RoundUp(realK, C0_SIZE)],
                              leftMatrixL1Tensor[i * C0_SIZE * C0_SIZE], commonLoadData2dParamsNoTranspose);
        }
#endif
        SetFlag<HardEvent::MTE1_MTE2>(A_FLAG_SHIFT + leftMatrixPingPong);
        SetFlag<HardEvent::MTE1_M>(A_FLAG_SHIFT + leftMatrixPingPong);

        // load B matrix from GM to L1
        WaitFlag<HardEvent::MTE1_MTE2>(B_FLAG_SHIFT + rightMatrixPingPong);
        commonNd2NzParams.nValue = realK;
        commonNd2NzParams.dValue = constInfo.headDim;
        commonNd2NzParams.srcDValue = constInfo.headDim;
        commonNd2NzParams.dstNzC0Stride = RoundUp(realK, C0_SIZE);
        AscendC::DataCopy(rightMatrixL1Tensor, rightMatrixGmTensor[i * BASIC_BLOCK_LENGTH * constInfo.headDim],
                          commonNd2NzParams);
        SetFlag<HardEvent::MTE2_MTE1>(B_FLAG_SHIFT + rightMatrixPingPong);

        // load B matrix from L1 -> l0B
        WaitFlag<HardEvent::MTE2_MTE1>(B_FLAG_SHIFT + rightMatrixPingPong);
        WaitFlag<HardEvent::M_MTE1>(B_CONFLICT_FLAG_SHIFT + rightMatrixPingPong);
        commonLoadData2dParamsTranspose.repeatTimes = constInfo.headDim / C0_SIZE;
        commonLoadData2dParamsTranspose.srcStride = RoundUp(realK, C0_SIZE) / C0_SIZE;
        for (int i = 0; i < RoundUp(realK, C0_SIZE) / C0_SIZE; i++) {
            AscendC::LoadData(rightMatrixL0BTensor[i * RoundUp(constInfo.headDim, C0_SIZE) * C0_SIZE],
                              rightMatrixL1Tensor[i * C0_SIZE * C0_SIZE], commonLoadData2dParamsTranspose);
        }
        SetFlag<HardEvent::MTE1_MTE2>(B_FLAG_SHIFT + rightMatrixPingPong);
        SetFlag<HardEvent::MTE1_M>(B_FLAG_SHIFT + rightMatrixPingPong);

        // mad (M, N, K)
        WaitFlag<HardEvent::MTE1_M>(A_FLAG_SHIFT + leftMatrixPingPong);
        WaitFlag<HardEvent::MTE1_M>(B_FLAG_SHIFT + rightMatrixPingPong);
        uint32_t realM = (baseM == 1) ? 2 : baseM;
        commonMadParams.m = realM;
        commonMadParams.n = baseN;
        commonMadParams.k = realK;
        commonMadParams.unitFlag = (i == loopK - 1) ? 3 : 2;
        commonMadParams.cmatrixInitVal = (i == 0) ? true : false;
        AscendC::Mmad(outputMatrixL0CTensor, leftMatrixL0ATensor, rightMatrixL0BTensor, commonMadParams);
        SetFlag<HardEvent::M_MTE1>(B_CONFLICT_FLAG_SHIFT + rightMatrixPingPong);
        SetFlag<HardEvent::M_MTE1>(A_CONFLICT_FLAG_SHIFT + leftMatrixPingPong);
        if (i == loopK - 1) {
            // FixPipe  L0C -> GM
            commonFixpipeParamsV220.mSize = baseM;
            commonFixpipeParamsV220.nSize = baseN;
            commonFixpipeParamsV220.srcStride = RoundUp(baseM, C0_SIZE);
            commonFixpipeParamsV220.dstStride = constInfo.headDim;
            commonFixpipeParamsV220.reluEn = false;
            if constexpr (std::is_same_v<dataType, half>) {
                commonFixpipeParamsV220.quantPre = QuantMode_t::F322F16;
            } else if constexpr (std::is_same_v<dataType, bfloat16_t>) {
                commonFixpipeParamsV220.quantPre = QuantMode_t::F322BF16;
            }
            commonFixpipeParamsV220.unitFlag = 3;
            AscendC::Fixpipe<dataType, float, AscendC::CFG_ROW_MAJOR>(outputMatrixGmTensor[outputMatrixGmOffset],
                                                                      outputMatrixL0CTensor, commonFixpipeParamsV220);
        } else {
            // Avoid L0C read/write conflicts
            AscendC::PipeBarrier<PIPE_M>();
        }

        leftMatrixPingPong = 1 - leftMatrixPingPong;
        rightMatrixPingPong = 1 - rightMatrixPingPong;
    }
}

// (G, K)^T @ (G, 128) -> (K, 128)
// L0B will always stay in L0B buffer
template <typename LIGT>
__aicore__ inline void LIGMatmul<LIGT>::Cube4(GlobalTensor<dataType> leftMatrixGmTensor,
                                              GlobalTensor<dataType> rightMatrixGmTensor,
                                              GlobalTensor<float> outputMatrixGmTensor, ConstInfo constInfo,
                                              RunInfo runInfo)
{
    uint32_t singleM = runInfo.realTopk;
    uint32_t singleK = constInfo.groupNum;
    uint32_t singleN = constInfo.headDim;
    uint32_t baseM = BASIC_BLOCK_LENGTH;
    uint32_t baseN = constInfo.headDim;
    uint32_t baseK = constInfo.groupNum;
    uint32_t leftMatrixPingPong = 0;

    // [B, S1, N1, D]
    uint64_t rightMatrixGmOffset = 0;
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        rightMatrixGmOffset = runInfo.bIdx * constInfo.seqlenQ * constInfo.headNumQ * constInfo.headDim +
                              runInfo.s1Idx * constInfo.headNumQ * constInfo.headDim +
                              runInfo.n2Idx * constInfo.groupNum * constInfo.headDim;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        rightMatrixGmOffset = (runInfo.prefixSumS1 + runInfo.s1Idx) * constInfo.headNumQ * constInfo.headDim +
                              runInfo.n2Idx * constInfo.groupNum * constInfo.headDim;
    }

    // load B matrix(query) from GM to L1
    WaitFlag<HardEvent::MTE1_MTE2>(B_FLAG_SHIFT);
    commonNd2NzParams.nValue = baseK;
    commonNd2NzParams.dValue = baseN;
    commonNd2NzParams.srcDValue = constInfo.headDim;
    commonNd2NzParams.dstNzC0Stride = RoundUp(baseK, C0_SIZE);
    AscendC::DataCopy(rightMatrixL1PingTensor, rightMatrixGmTensor[rightMatrixGmOffset], commonNd2NzParams);
    SetFlag<HardEvent::MTE2_MTE1>(B_FLAG_SHIFT);

    // load B matrix from L1 -> l0B
    WaitFlag<HardEvent::MTE2_MTE1>(B_FLAG_SHIFT);
    WaitFlag<HardEvent::M_MTE1>(B_CONFLICT_FLAG_SHIFT);
    commonLoadData2dParamsTranspose.repeatTimes = constInfo.headDim / C0_SIZE;
    commonLoadData2dParamsTranspose.srcStride = RoundUp(baseK, C0_SIZE) / C0_SIZE;
    for (int i = 0; i < RoundUp(baseK, C0_SIZE) / C0_SIZE; i++) {
        AscendC::LoadData(rightMatrixL0BPingTensor[i * constInfo.headDim * C0_SIZE],
                          rightMatrixL1PingTensor[i * C0_SIZE * C0_SIZE], commonLoadData2dParamsTranspose);
    }
    SetFlag<HardEvent::MTE1_MTE2>(B_FLAG_SHIFT);
    SetFlag<HardEvent::MTE1_M>(B_FLAG_SHIFT);

    uint64_t loopM = LIGCommon::Align(singleM, baseM) / baseM;
    uint32_t baseMTail = (singleM % baseM == 0) ? baseM : singleM % baseM;
    for (uint32_t i = 0; i < loopM; i++) {
        LocalTensor<dataType> leftMatrixL1Tensor =
            (leftMatrixPingPong & 1) ? leftMatrixL1PingTensor : leftMatrixL1PongTensor;
        LocalTensor<dataType> leftMatrixL0ATensor =
            (leftMatrixPingPong & 1) ? leftMatrixL0APingTensor : leftMatrixL0APongTensor;
        LocalTensor<float> outputMatrixL0CTensor =
            (leftMatrixPingPong & 1) ? outputMatrixL0CPingTensor : outputMatrixL0CPongTensor;

        uint32_t realM = (i == loopM - 1) ? baseMTail : baseM;
        // load A matrix from gm to L1
        WaitFlag<HardEvent::MTE1_MTE2>(A_FLAG_SHIFT + leftMatrixPingPong);
        commonNd2NzParams.nValue = baseK;
        commonNd2NzParams.dValue = realM;
        commonNd2NzParams.srcDValue = runInfo.realTopk;
        commonNd2NzParams.dstNzC0Stride = RoundUp(baseK, C0_SIZE);
        AscendC::DataCopy(leftMatrixL1Tensor, leftMatrixGmTensor[i * BASIC_BLOCK_LENGTH], commonNd2NzParams);
        SetFlag<HardEvent::MTE2_MTE1>(A_FLAG_SHIFT + leftMatrixPingPong);

        // load A matrix from L1 -> L0A
        WaitFlag<HardEvent::MTE2_MTE1>(A_FLAG_SHIFT + leftMatrixPingPong);
        WaitFlag<HardEvent::M_MTE1>(A_CONFLICT_FLAG_SHIFT + leftMatrixPingPong);
#if __CCE_AICORE__ == 310
        /*
         * Load A matrix with transpose: src in L1 is [baseK x realM] = [G x realM],
         * dst in L0A should be [realM x G] after transpose.
         * mSizeAlign = Align(G, 16), kSizeAlign = Align(realM, 16)
         * Reference ZN pattern: kStep = kSizeAlign/16, srcStride = mSizeAlign/16
         */
        AscendC::LoadData2DParamsV2 loadParams;
        loadParams.kStartPosition = 0;
        loadParams.mStep = 1;
        loadParams.kStep = LIGCommon::Align(realM, C0_SIZE) / C0_SIZE;
        loadParams.srcStride = LIGCommon::Align(baseK, C0_SIZE) / C0_SIZE;
        loadParams.dstStride = 1;
        loadParams.ifTranspose = true;
        for (int i = 0; i < LIGCommon::Align(baseK, C0_SIZE) / C0_SIZE; i++) {
            loadParams.mStartPosition = i;
            AscendC::LoadData(leftMatrixL0ATensor[i * LIGCommon::Align(realM, C0_SIZE) * C0_SIZE], leftMatrixL1Tensor,
                              loadParams);
        }
#else
        commonLoadData2dParamsTranspose.repeatTimes =
            (LIGCommon::Align(realM, C0_SIZE) / C0_SIZE) * (LIGCommon::Align(baseK, C0_SIZE) / C0_SIZE);
        commonLoadData2dParamsTranspose.srcStride = 1;
        AscendC::LoadData(leftMatrixL0ATensor, leftMatrixL1Tensor, commonLoadData2dParamsTranspose);
#endif
        SetFlag<HardEvent::MTE1_MTE2>(A_FLAG_SHIFT + leftMatrixPingPong);
        SetFlag<HardEvent::MTE1_M>(A_FLAG_SHIFT + leftMatrixPingPong);

        // mad (M, N, K)
        if (i == 0) {
            WaitFlag<HardEvent::MTE1_M>(B_FLAG_SHIFT);
        }
        WaitFlag<HardEvent::MTE1_M>(A_FLAG_SHIFT + leftMatrixPingPong);
        commonMadParams.m = (realM == 1) ? 2 : realM;
        commonMadParams.n = baseN;
        commonMadParams.k = baseK;
        commonMadParams.unitFlag = 3;
        commonMadParams.cmatrixInitVal = true;
        AscendC::Mmad(outputMatrixL0CTensor, leftMatrixL0ATensor, rightMatrixL0BPingTensor, commonMadParams);

        SetFlag<HardEvent::M_MTE1>(A_CONFLICT_FLAG_SHIFT + leftMatrixPingPong);
        if (i == loopM - 1) {
            SetFlag<HardEvent::M_MTE1>(B_CONFLICT_FLAG_SHIFT);
        }

        // FixPipe  L0C -> GM
        commonFixpipeParamsV220.mSize = realM;
        commonFixpipeParamsV220.nSize = baseN;
        commonFixpipeParamsV220.srcStride = RoundUp(realM, C0_SIZE);
        commonFixpipeParamsV220.dstStride = constInfo.headDim;
        commonFixpipeParamsV220.reluEn = false;
        commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
        commonFixpipeParamsV220.unitFlag = 3;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(outputMatrixGmTensor[i * baseM * constInfo.headDim],
                                                               outputMatrixL0CTensor, commonFixpipeParamsV220);
        leftMatrixPingPong = 1 - leftMatrixPingPong;
    }
}
} // namespace LigKernel
#endif
