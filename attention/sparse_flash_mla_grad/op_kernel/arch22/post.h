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
 * \file post.h
 * \brief common post process
 */

#pragma once
#include "kernel_operator.h"
#include "basic_modules/sparse_flash_mla_grad_common_header.h"
using namespace AscendC;

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t MODE = SMLAG_SCFA_MODE>
class SparseFlashMlaGradPost {
public:
    __aicore__ inline SparseFlashMlaGradPost(){};
    __aicore__ inline void Init(__gm__ uint8_t* dq, __gm__ uint8_t* d_ori_kv, __gm__ uint8_t* d_cmp_kv,
                                __gm__ uint8_t* dsinks, __gm__ uint8_t* workspace,
                                const TILING_TYPE* __restrict ordTilingData, TPipe* pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessSink();

    constexpr static uint32_t BUFFER_NUM = 1;
    TPipe* pipe;
    TBuf<> tmpBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    AscendC::GlobalTensor<OUT_TYPE> dqGm;
    AscendC::GlobalTensor<OUT_TYPE> dOriKvGm;
    AscendC::GlobalTensor<OUT_TYPE> dCmpKvGm;
    AscendC::GlobalTensor<float> dsinkGm;
    AscendC::GlobalTensor<float> dsinkWorkspaceGm;
    // input
    AscendC::GlobalTensor<float> dqWorkSpaceGm;
    AscendC::GlobalTensor<float> dkWorkSpaceGm;

    const TILING_TYPE* __restrict tilingData;
    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint64_t C0_SIZE = 16;
    constexpr static uint64_t VEC_REPEAT = 8;
    constexpr static uint32_t cal_block_num = 32 / sizeof(float);

    int64_t usedCoreNum;
    int64_t cBlockIdx;
    int64_t aivNum{0};
    int64_t dimG{0};
    uint32_t isDeterministic{0};
    // query
    int64_t ubBaseSize;
    int64_t qPostBlockFactor;
    uint64_t qPostBlockTotal;
    int64_t qPostBaseNum;
    int64_t qPostTailNum;
    // oriKv
    int64_t oriKvPostBlockFactor;
    uint64_t oriKvPostBlockTotal;
    int64_t oriKvPostBaseNum;
    int64_t oriKvPostTailNum;
    // cmpKv
    int64_t cmpKvPostBlockFactor;
    uint64_t cmpKvPostBlockTotal;
    int64_t cmpKvPostBaseNum;
    int64_t cmpKvPostTailNum;

    // org shape info
    int64_t dimDqk;
    int64_t dOriKvSize;
    int64_t dCmpKvSize;

    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t TND = 3;

    constexpr static uint32_t ND = 0;
    constexpr static uint32_t NZ = 1;

    event_t mte2WaitV;
    event_t mte3WaitV;
    event_t vWaitMte2;
    event_t vWaitMte3;
};

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t MODE>
__aicore__ inline void SparseFlashMlaGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, MODE>::Init(
    __gm__ uint8_t* dq, __gm__ uint8_t* d_ori_kv, __gm__ uint8_t* d_cmp_kv, __gm__ uint8_t* dsinks,
    __gm__ uint8_t* workspace, const TILING_TYPE* __restrict ordTilingData, TPipe* pipe_in)
{
    cBlockIdx = GetBlockIdx();

    tilingData = ordTilingData;
    pipe = pipe_in;

    dqGm.SetGlobalBuffer((__gm__ OUT_TYPE*)dq);
    dOriKvGm.SetGlobalBuffer((__gm__ OUT_TYPE*)d_ori_kv);
    dCmpKvGm.SetGlobalBuffer((__gm__ OUT_TYPE*)d_cmp_kv);
    dsinkGm.SetGlobalBuffer((__gm__ float*)dsinks);

    // tiling_data
    usedCoreNum = tilingData->postTilingData.coreNum;
    ubBaseSize = tilingData->postTilingData.postUbBaseSize;
    qPostBlockFactor = tilingData->postTilingData.qPostBlockFactor;
    qPostBlockTotal = tilingData->postTilingData.qPostBlockTotal;
    qPostBaseNum = tilingData->postTilingData.qPostBaseNum;
    qPostTailNum = tilingData->postTilingData.qPostTailNum;

    oriKvPostBlockFactor = tilingData->postTilingData.oriKvPostBlockFactor;
    oriKvPostBlockTotal = tilingData->postTilingData.oriKvPostBlockTotal;
    oriKvPostBaseNum = tilingData->postTilingData.oriKvPostBaseNum;
    oriKvPostTailNum = tilingData->postTilingData.oriKvPostTailNum;

    cmpKvPostBlockFactor = tilingData->postTilingData.cmpKvPostBlockFactor;
    cmpKvPostBlockTotal = tilingData->postTilingData.cmpKvPostBlockTotal;
    cmpKvPostBaseNum = tilingData->postTilingData.cmpKvPostBaseNum;
    cmpKvPostTailNum = tilingData->postTilingData.cmpKvPostTailNum;

    dimDqk = tilingData->opInfo.D;
    dimG = tilingData->opInfo.G;
    isDeterministic = tilingData->opInfo.isDeterministic;
    aivNum = tilingData->opInfo.aivNum;
    dOriKvSize = LAYOUT == 3 ? tilingData->opInfo.S2 * tilingData->opInfo.N2 * dimDqk :
                               tilingData->opInfo.B * tilingData->opInfo.S2 * tilingData->opInfo.N2 * dimDqk;
    dCmpKvSize = LAYOUT == 3 ? tilingData->opInfo.S3 * tilingData->opInfo.N2 * dimDqk :
                               tilingData->opInfo.B * tilingData->opInfo.S3 * tilingData->opInfo.N2 * dimDqk;

    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                  tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                  tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
    if (isDeterministic) {
        dsinkWorkspaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                         tilingData->postTilingData.dSinkWorkSpaceOffset / sizeof(float));
    }

    pipe->InitBuffer(inQueue, 1, ubBaseSize * 2);
    pipe->InitBuffer(outQueue, 1, ubBaseSize);

    pipe->InitBuffer(tmpBuf, ubBaseSize * 4);
    mte2WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    mte3WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    vWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    vWaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
}

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t MODE>
__aicore__ inline void SparseFlashMlaGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, MODE>::Process()
{
    // init q
    uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
    uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;

    if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
        qEnd = qPostBlockTotal;
    }
    uint64_t dqOutGmOffset = cBlockIdx * qPostBlockFactor * (qPostBaseNum / dimDqk) * dimDqk;

    SetFlag<HardEvent::V_MTE2>(0);
    for (uint64_t i = qBegin; i < qEnd; i = i + qPostBaseNum) {
        AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
        AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
        uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
        WaitFlag<HardEvent::V_MTE2>(0);
        DataCopy(vecIn, dqWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
        inQueue.EnQue(vecIn);
        inQueue.template DeQue<float>();

        Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
        PIPE_BARRIER(PIPE_V);

        Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
        outQueue.EnQue(vecOut);
        outQueue.template DeQue<OUT_TYPE>();

        DataCopyParams repeatParams;
        repeatParams.blockCount = dataSize / dimDqk;
        repeatParams.blockLen = dimDqk * sizeof(OUT_TYPE) / 32;
        repeatParams.srcStride = 0;
        repeatParams.dstStride = 0;

        DataCopy(dqGm[dqOutGmOffset], vecOut, repeatParams);
        dqOutGmOffset += repeatParams.blockCount * dimDqk;

        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
        SetFlag<HardEvent::V_MTE2>(0);
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    PIPE_BARRIER(PIPE_ALL);

    // muls -> atomicAdd (非确定性路径；确定性路径已在 AccumulateKvDeter 中完成 scale+dv)
    if (!isDeterministic) {
        uint64_t oriKvBegin = cBlockIdx * oriKvPostBlockFactor * oriKvPostBaseNum;
        uint64_t oriKvEnd = (cBlockIdx + 1) * oriKvPostBlockFactor * oriKvPostBaseNum;
        if (((cBlockIdx + 1) * oriKvPostBlockFactor * oriKvPostBaseNum) > oriKvPostBlockTotal) {
            oriKvEnd = oriKvPostBlockTotal;
        }
        SetFlag<HardEvent::V_MTE2>(mte2WaitV);
        SetFlag<HardEvent::MTE3_V>(vWaitMte3);
        for (uint64_t i = oriKvBegin; i < oriKvEnd; i = i + oriKvPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = tmpBuf.GetWithOffset<float>(oriKvPostBaseNum, 0);
            AscendC::LocalTensor<float> vecOut =
                tmpBuf.GetWithOffset<float>(oriKvPostBaseNum, oriKvPostBaseNum * sizeof(float));
            uint64_t dataSize = i + oriKvPostBaseNum < oriKvPostBlockTotal ? oriKvPostBaseNum : oriKvPostTailNum;
            WaitFlag<HardEvent::V_MTE2>(mte2WaitV);
            DataCopy(vecIn, dkWorkSpaceGm[i + dOriKvSize + dCmpKvSize],
                     (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            SetFlag<HardEvent::MTE2_V>(vWaitMte2);

            WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
            WaitFlag<HardEvent::MTE3_V>(vWaitMte3);
            Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
            SetFlag<HardEvent::V_MTE2>(mte2WaitV);

            DataCopyParams repeatParams;
            repeatParams.blockCount = dataSize / dimDqk;
            repeatParams.blockLen = dimDqk * sizeof(float) / 32;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;

            SetFlag<HardEvent::V_MTE3>(mte3WaitV);
            WaitFlag<HardEvent::V_MTE3>(mte3WaitV);
            SetAtomicAdd<float>();
            DataCopy(dkWorkSpaceGm[i], vecOut, repeatParams);
            SetAtomicNone();
            SetFlag<HardEvent::MTE3_V>(vWaitMte3);
        }
        WaitFlag<HardEvent::V_MTE2>(mte2WaitV);
        WaitFlag<HardEvent::MTE3_V>(vWaitMte3);
        PIPE_BARRIER(PIPE_ALL);

        if constexpr (MODE == SMLAG_CFA_MODE) {
            uint64_t cmpKvBegin = cBlockIdx * cmpKvPostBlockFactor * cmpKvPostBaseNum;
            uint64_t cmpKvEnd = (cBlockIdx + 1) * cmpKvPostBlockFactor * cmpKvPostBaseNum;
            if (((cBlockIdx + 1) * cmpKvPostBlockFactor * cmpKvPostBaseNum) > cmpKvPostBlockTotal) {
                cmpKvEnd = cmpKvPostBlockTotal;
            }
            SetFlag<HardEvent::V_MTE2>(mte2WaitV);
            SetFlag<HardEvent::MTE3_V>(vWaitMte3);
            for (uint64_t i = cmpKvBegin; i < cmpKvEnd; i = i + cmpKvPostBaseNum) {
                AscendC::LocalTensor<float> vecIn = tmpBuf.GetWithOffset<float>(cmpKvPostBaseNum, 0);
                AscendC::LocalTensor<float> vecOut =
                    tmpBuf.GetWithOffset<float>(cmpKvPostBaseNum, cmpKvPostBaseNum * sizeof(float));
                uint64_t dataSize = i + cmpKvPostBaseNum < cmpKvPostBlockTotal ? cmpKvPostBaseNum : cmpKvPostTailNum;
                WaitFlag<HardEvent::V_MTE2>(mte2WaitV);
                DataCopy(vecIn, dkWorkSpaceGm[i + dOriKvSize * 2 + dCmpKvSize], (dataSize + 7) / 8 * 8);
                SetFlag<HardEvent::MTE2_V>(vWaitMte2);

                WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
                WaitFlag<HardEvent::MTE3_V>(vWaitMte3);
                Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                SetFlag<HardEvent::V_MTE2>(mte2WaitV);

                DataCopyParams repeatParams;
                repeatParams.blockCount = dataSize / dimDqk;
                repeatParams.blockLen = dimDqk * sizeof(float) / 32;
                repeatParams.srcStride = 0;
                repeatParams.dstStride = 0;

                SetFlag<HardEvent::V_MTE3>(mte3WaitV);
                WaitFlag<HardEvent::V_MTE3>(mte3WaitV);
                SetAtomicAdd<float>();
                DataCopy(dkWorkSpaceGm[dOriKvSize + i], vecOut, repeatParams);
                SetAtomicNone();
                SetFlag<HardEvent::MTE3_V>(vWaitMte3);
            }
            WaitFlag<HardEvent::V_MTE2>(mte2WaitV);
            WaitFlag<HardEvent::MTE3_V>(vWaitMte3);
            PIPE_BARRIER(PIPE_ALL);
        }
    } // !isDeterministic

    // init oriKv
    uint64_t oriKvBegin = cBlockIdx * oriKvPostBlockFactor * oriKvPostBaseNum;
    uint64_t oriKvEnd = (cBlockIdx + 1) * oriKvPostBlockFactor * oriKvPostBaseNum;
    if (((cBlockIdx + 1) * oriKvPostBlockFactor * oriKvPostBaseNum) > oriKvPostBlockTotal) {
        oriKvEnd = oriKvPostBlockTotal;
    }
    uint64_t dOriKvOutGmOffset = cBlockIdx * oriKvPostBlockFactor * (oriKvPostBaseNum / dimDqk) * dimDqk;

    SetFlag<HardEvent::V_MTE2>(0);
    for (uint64_t i = oriKvBegin; i < oriKvEnd; i = i + oriKvPostBaseNum) {
        AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
        AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
        uint64_t dataSize = i + oriKvPostBaseNum < oriKvPostBlockTotal ? oriKvPostBaseNum : oriKvPostTailNum;
        WaitFlag<HardEvent::V_MTE2>(0);
        DataCopy(vecIn, dkWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
        inQueue.EnQue(vecIn);
        inQueue.template DeQue<float>();

        Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
        outQueue.EnQue(vecOut);
        outQueue.template DeQue<OUT_TYPE>();

        DataCopyParams repeatParams;
        repeatParams.blockCount = dataSize / dimDqk;
        repeatParams.blockLen = dimDqk * sizeof(OUT_TYPE) / 32;
        repeatParams.srcStride = 0;
        repeatParams.dstStride = 0;

        DataCopy(dOriKvGm[dOriKvOutGmOffset], vecOut, repeatParams);
        dOriKvOutGmOffset += repeatParams.blockCount * dimDqk;

        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
        SetFlag<HardEvent::V_MTE2>(0);
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    PIPE_BARRIER(PIPE_ALL);

    if constexpr (MODE != SMLAG_SWA_MODE) {
        // init cmpKv
        uint64_t cmpKvBegin = cBlockIdx * cmpKvPostBlockFactor * cmpKvPostBaseNum;
        uint64_t cmpKvEnd = (cBlockIdx + 1) * cmpKvPostBlockFactor * cmpKvPostBaseNum;
        if (((cBlockIdx + 1) * cmpKvPostBlockFactor * cmpKvPostBaseNum) > cmpKvPostBlockTotal) {
            cmpKvEnd = cmpKvPostBlockTotal;
        }
        uint64_t dCmpKvOutGmOffset = cBlockIdx * cmpKvPostBlockFactor * (cmpKvPostBaseNum / dimDqk) * dimDqk;

        SetFlag<HardEvent::V_MTE2>(0);
        for (uint64_t i = cmpKvBegin; i < cmpKvEnd; i = i + cmpKvPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
            uint64_t dataSize = i + cmpKvPostBaseNum < cmpKvPostBlockTotal ? cmpKvPostBaseNum : cmpKvPostTailNum;
            WaitFlag<HardEvent::V_MTE2>(0);
            DataCopy(vecIn, dkWorkSpaceGm[i + dOriKvSize], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            inQueue.EnQue(vecIn);
            inQueue.template DeQue<float>();

            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();

            DataCopyParams repeatParams;
            repeatParams.blockCount = dataSize / dimDqk;
            repeatParams.blockLen = dimDqk * sizeof(OUT_TYPE) / 32;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;

            DataCopy(dCmpKvGm[dCmpKvOutGmOffset], vecOut, repeatParams);
            dCmpKvOutGmOffset += repeatParams.blockCount * dimDqk;

            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
            SetFlag<HardEvent::V_MTE2>(0);
        }
        WaitFlag<HardEvent::V_MTE2>(0);
        PIPE_BARRIER(PIPE_ALL);
    }

    if (isDeterministic) {
        ProcessSink();
    }
}

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t MODE>
__aicore__ inline void SparseFlashMlaGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, MODE>::ProcessSink()
{
    // CopyOutDsinks 只写入 mix 实际 AIV：vecBlockIdx ∈ [0, usedCoreNum*2)
    int64_t sinkAivNum = static_cast<int64_t>(tilingData->opInfo.usedCoreNum) * 2;
    if (sinkAivNum <= 0 || dimG == 0) {
        return;
    }
    // 按实际写 workspace 的 AIV 数均分 G，避免用平台 aivNum 导致尾部 G 无人写
    int64_t gPerCore = (dimG + sinkAivNum - 1) / sinkAivNum;
    int64_t gStart = cBlockIdx * gPerCore;
    int64_t gEnd = gStart + gPerCore;
    if (gEnd > dimG) {
        gEnd = dimG;
    }
    if (gStart >= dimG) {
        return;
    }
    int64_t gLen = gEnd - gStart;
    // VEC/MTE 要求 UB 基址与长度按 32B（8 个 float）对齐
    constexpr int64_t FP32_ALIGN = 8;
    int64_t gLenAlign = (gLen + FP32_ALIGN - 1) / FP32_ALIGN * FP32_ALIGN;
    AscendC::LocalTensor<float> accTensor = tmpBuf.GetWithOffset<float>(gLenAlign, 0);
    AscendC::LocalTensor<float> tmpTensor = tmpBuf.GetWithOffset<float>(gLenAlign, gLenAlign * sizeof(float));

    Duplicate(accTensor, (float)0.0, gLenAlign);
    PIPE_BARRIER(PIPE_V);
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
    for (int64_t srcAiv = 0; srcAiv < sinkAivNum; srcAiv++) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        DataCopyPad(tmpTensor, dsinkWorkspaceGm[srcAiv * dimG + gStart], {1, (uint32_t)(gLen * sizeof(float)), 0, 0, 0},
                    {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
        Add(accTensor, accTensor, tmpTensor, gLenAlign);
        SetFlag<HardEvent::V_MTE3>(mte3WaitV);
        WaitFlag<HardEvent::V_MTE3>(mte3WaitV);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    DataCopyPad(dsinkGm[gStart], accTensor, {1, (uint32_t)(gLen * sizeof(float)), 0, 0, 0});
}
