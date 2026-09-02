/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _fused_floyd_attention_grad_POST_H_
#define _fused_floyd_attention_grad_POST_H_
#include "kernel_operator.h"

using namespace AscendC;
using AscendC::CopyRepeatParams;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT>
class FusedFloydAttentionGradPost {
public:
    __aicore__ inline FusedFloydAttentionGradPost(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dk_1,
                                __gm__ uint8_t *dv_1, __gm__ uint8_t *workspace,
                                const TILING_TYPE *__restrict ordTilingData, TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void InitIndex(uint64_t startIdx, int64_t curG, int64_t &curS, GM_ADDR seqS);
    __aicore__ inline void NZ2ND(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, uint64_t sLen,
                                 uint64_t ubOffset, uint64_t srcUbOffset);
    __aicore__ inline void NZVecClc(GlobalTensor<float> srcGm, GlobalTensor<OUT_TYPE> dstGm, uint64_t dataSize,
                                    GM_ADDR seqS, int64_t curG, int64_t &curS, bool needMuls, int64_t flag);
    __aicore__ inline void NZProcess();
    __aicore__ inline void ComputeDataCopyOffset(int64_t curG, int64_t &curS);

    constexpr static uint32_t BUFFER_NUM = 1;
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    // NZ buffer
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePing;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePong;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuePing;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuePong;
    TBuf<> tmpBufPing;
    TBuf<> tmpBufPong;

    AscendC::GlobalTensor<OUT_TYPE> dqGm, dkGm, dvGm, dk1Gm, dv1Gm;
    // input
    AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, dk1WorkSpaceGm, dv1WorkSpaceGm;

    const TILING_TYPE *__restrict tilingData;
    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint64_t C0_SIZE = 16;
    constexpr static uint64_t VEC_REPEAT = 8;
    constexpr static uint32_t cal_block_num = 32 / sizeof(float);

    int64_t usedCoreNum;
    int64_t cBlockIdx;
    // query
    int64_t ubBaseSize;
    uint32_t nzReservedSize;
    int64_t qPostBlockFactor;
    uint64_t qPostBlockTotal;
    int64_t qPostBaseNum;
    int64_t qPostTailNum;
    uint64_t qSizeAlign;
    int64_t kvPostBlockFactor;
    uint64_t kvPostBlockTotal;
    int64_t kvPostBaseNum;
    int64_t kvPostTailNum;
    uint64_t kvSizeAlign;
    int64_t k1v1PostBlockFactor;
    uint64_t k1v1PostBlockTotal;
    int64_t k1v1PostBaseNum;
    int64_t k1v1PostTailNum;
    uint64_t k1v1SizeAlign;

    // org shape info
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dAlign;

    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t TND = 3;

    constexpr static uint32_t ND = 0;
    constexpr static uint32_t NZ = 1;

    GM_ADDR actual_seq_qlen_addr;
    GM_ADDR actual_seq_kvlen_addr;

    uint64_t bIdx;
    uint64_t nIdx;
    uint64_t sIdx;

    uint64_t scrOffsetBase = 0;
    uint64_t dstOffsetBase = 0;
    uint64_t copyInSrcOffset = 0;
    uint64_t copyOutDstOffset = 0;
    uint64_t copyOutDstStride = 0;
};

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT>
__aicore__ inline void FusedFloydAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dk_1, __gm__ uint8_t *dv_1,
    __gm__ uint8_t *workspace, const TILING_TYPE *__restrict ordTilingData, TPipe *pipe_in)
{
    cBlockIdx = GetBlockIdx();

    tilingData = ordTilingData;
    pipe = pipe_in;

    dqGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dq);
    dkGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dk);
    dvGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dv);
    dk1Gm.SetGlobalBuffer((__gm__ OUT_TYPE *)dk_1);
    dv1Gm.SetGlobalBuffer((__gm__ OUT_TYPE *)dv_1);

    // tiling_data
    usedCoreNum = tilingData->postTilingData.coreNum;
    ubBaseSize = tilingData->postTilingData.postUbBaseSize;
    nzReservedSize = tilingData->postTilingData.nzReservedSize;
    qPostBlockFactor = tilingData->postTilingData.qPostBlockFactor;
    qPostBlockTotal = tilingData->postTilingData.qPostBlockTotal;
    qPostBaseNum = tilingData->postTilingData.qPostBaseNum;
    qPostTailNum = tilingData->postTilingData.qPostTailNum;
    kvPostBlockFactor = tilingData->postTilingData.kvPostBlockFactor;
    kvPostBlockTotal = tilingData->postTilingData.kvPostBlockTotal;
    kvPostBaseNum = tilingData->postTilingData.kvPostBaseNum;
    kvPostTailNum = tilingData->postTilingData.kvPostTailNum;
    qSizeAlign = tilingData->postTilingData.qSizeAlign;
    kvSizeAlign = tilingData->postTilingData.kvSizeAlign;

    k1v1PostBlockFactor = tilingData->postTilingData.k1v1PostBlockFactor;
    k1v1PostBlockTotal = tilingData->postTilingData.k1v1PostBlockTotal;
    k1v1PostBaseNum = tilingData->postTilingData.k1v1PostBaseNum;
    k1v1PostTailNum = tilingData->postTilingData.k1v1PostTailNum;
    k1v1SizeAlign = tilingData->postTilingData.k1v1SizeAlign;

    if constexpr (INPUT_FORMAT == NZ) {
        b = tilingData->postTilingData.b;
        n2 = tilingData->postTilingData.n2;
        g = tilingData->postTilingData.g;
        s1 = tilingData->postTilingData.s1;
        s2 = tilingData->postTilingData.s2;
        d = tilingData->postTilingData.d;
        dAlign = (d + 15) / 16 * 16;
        actual_seq_qlen_addr = nullptr;
        actual_seq_kvlen_addr = nullptr;
    }

    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
    dk1WorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                   tilingData->postTilingData.dk1WorkSpaceOffset / sizeof(float));

    if constexpr (CAST_DV) {
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      tilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
        dv1WorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                       tilingData->postTilingData.dv1WorkSpaceOffset / sizeof(float));
    }

    if constexpr (INPUT_FORMAT == NZ) {
        pipe->InitBuffer(inQueuePing, 1, ubBaseSize * 2 + nzReservedSize);
        pipe->InitBuffer(inQueuePong, 1, ubBaseSize * 2 + nzReservedSize);
        pipe->InitBuffer(outQueuePing, 1, ubBaseSize);
        pipe->InitBuffer(outQueuePong, 1, ubBaseSize);
        pipe->InitBuffer(tmpBufPing, ubBaseSize * 2);
        pipe->InitBuffer(tmpBufPong, ubBaseSize * 2);
    } else {
        pipe->InitBuffer(inQueue, 1, ubBaseSize * 2);
        pipe->InitBuffer(outQueue, 1, ubBaseSize);
    }
}

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT>
__aicore__ inline void FusedFloydAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT>::InitIndex(
    uint64_t startIdx, int64_t curG, int64_t &curS, GM_ADDR seqS)
{
    bIdx = startIdx / (n2 * curG * curS * d);
    uint64_t bTail = startIdx % (n2 * curG * curS * d);
    nIdx = bTail / (curS * d);
    uint64_t nTail = bTail % (curS * d);
    sIdx = nTail / d;
    ComputeDataCopyOffset(curG, curS);
}

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT>
__aicore__ inline void
FusedFloydAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT>::ComputeDataCopyOffset(int64_t curG,
                                                                                                         int64_t &curS)
{
    // src BNSD
    scrOffsetBase = bIdx * n2 * curS * curG * dAlign;
    copyInSrcOffset = scrOffsetBase + nIdx * curS * dAlign + sIdx * C0_SIZE;

    copyOutDstStride = 0;
    copyOutDstOffset = ((bIdx * n2 * curG + nIdx) * curS + sIdx) * d;
}

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT>
__aicore__ inline void FusedFloydAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT>::NZ2ND(
    LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, uint64_t sLen, uint64_t ubOffset,
    uint64_t srcUbOffset)
{
    /*
    Func:
    将NZ转为ND
    */
    CopyRepeatParams nz2ndParams;
    nz2ndParams.srcStride = sLen * C0_SIZE / cal_block_num + 1;
    nz2ndParams.dstStride = C0_SIZE / cal_block_num;
    nz2ndParams.srcRepeatSize = C0_SIZE / cal_block_num;
    nz2ndParams.dstRepeatSize = dAlign / cal_block_num;

    uint16_t c0_repeat = C0_SIZE / cal_block_num;
    uint16_t c1_repeat = dAlign / C0_SIZE / VEC_REPEAT;
    uint16_t c1_remain = dAlign / C0_SIZE % VEC_REPEAT;
    uint16_t n_repeat = sLen;
    for (uint16_t i = 0; i < c0_repeat; ++i) {
        for (uint16_t j = 0; j < c1_repeat; ++j) {
            Copy(dstTensor[ubOffset + i * cal_block_num + j * VEC_REPEAT * C0_SIZE],
                 srcTensor[srcUbOffset + i * cal_block_num + j * VEC_REPEAT * (sLen * C0_SIZE + cal_block_num)],
                 VEC_REPEAT * cal_block_num, n_repeat, nz2ndParams);
        }
        if (c1_remain > 0) {
            Copy(dstTensor[ubOffset + i * cal_block_num + c1_repeat * VEC_REPEAT * C0_SIZE],
                 srcTensor[srcUbOffset + i * cal_block_num + c1_repeat * VEC_REPEAT * (sLen * C0_SIZE + cal_block_num)],
                 VEC_REPEAT * c1_remain, n_repeat, nz2ndParams);
        }
    }
}

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT>
__aicore__ inline void FusedFloydAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT>::NZVecClc(
    GlobalTensor<float> srcGm, GlobalTensor<OUT_TYPE> dstGm, uint64_t dataSize, GM_ADDR seqS, int64_t curG,
    int64_t &curS, bool needMuls, int64_t flag)
{
    if (dataSize == 0) {
        return;
    }

    event_t mte2WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t mte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCommon;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueCommon;
    TBuf<> tmpBufCommon;
    event_t curEventId;
    if (flag) {
        inQueueCommon = inQueuePing;
        outQueueCommon = outQueuePing;
        tmpBufCommon = tmpBufPing;
        curEventId = mte2WaitVPing;
    } else {
        inQueueCommon = inQueuePong;
        outQueueCommon = outQueuePong;
        tmpBufCommon = tmpBufPong;
        curEventId = mte2WaitVPong;
    }

    LocalTensor<float> vecIn = inQueueCommon.template AllocTensor<float>();
    LocalTensor<float> tmpTensor = tmpBufCommon.template Get<float>();
    LocalTensor<OUT_TYPE> vecOut = outQueueCommon.template AllocTensor<OUT_TYPE>();

    uint32_t sClcSize = dataSize / d;

    uint64_t sLen = (sIdx + sClcSize) > curS ? (curS - sIdx) : sClcSize;
    sLen = sLen > 255 ? 255 : sLen;
    uint64_t dataLen = sLen * dAlign;

    uint64_t ubOffset = 0;
    uint64_t inUbOffset = 0;

    while (sClcSize > 0) {
        // Nz copy In
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = dAlign / C0_SIZE;
        intriParams.blockLen = sLen * C0_SIZE * sizeof(float);
        intriParams.srcStride = curS * C0_SIZE * sizeof(float) - intriParams.blockLen;
        intriParams.dstStride = 1; // 间隔一个block，防止bank冲突
        intriParams.rsv = 0;
        DataCopyPad(vecIn[inUbOffset], srcGm[copyInSrcOffset], intriParams, {false, 0, 0, 0});
        sClcSize = sClcSize - sLen;

        inQueueCommon.EnQue(vecIn);
        inQueueCommon.template DeQue<float>();

        if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
            NZ2ND(tmpTensor, vecIn, sLen, ubOffset, inUbOffset);
        } else {
            NZ2ND(vecOut, vecIn, sLen, ubOffset, inUbOffset);
        }

        if (sClcSize <= 0) {
            inQueueCommon.FreeTensor(vecIn);
        }

        PipeBarrier<PIPE_V>();
        if (needMuls) {
            if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
                Muls(tmpTensor[ubOffset], tmpTensor[ubOffset], (float)tilingData->postTilingData.scaleValue,
                     sLen * dAlign);
            } else {
                Muls(vecOut[ubOffset], vecOut[ubOffset], (float)tilingData->postTilingData.scaleValue, sLen * dAlign);
            }

            PipeBarrier<PIPE_V>();
        }

        if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
            Cast(vecOut[ubOffset], tmpTensor[ubOffset], RoundMode::CAST_ROUND, sLen * dAlign);
            PipeBarrier<PIPE_V>();
        }

        outQueueCommon.EnQue(vecOut);
        outQueueCommon.template DeQue<OUT_TYPE>();

        if constexpr (LAYOUT == TND) {
            DataCopyPad(dstGm[copyOutDstOffset], vecOut[ubOffset],
                        {static_cast<uint16_t>(dataLen / dAlign), static_cast<uint32_t>(d * sizeof(OUT_TYPE)), 0,
                         static_cast<uint32_t>((n2 * curG * d - d) * sizeof(OUT_TYPE)), 0});
        } else {
            DataCopyPad(dstGm[copyOutDstOffset], vecOut[ubOffset],
                        {static_cast<uint16_t>(dataLen / dAlign), static_cast<uint32_t>(d * sizeof(OUT_TYPE)), 0,
                         static_cast<uint32_t>(copyOutDstStride * sizeof(OUT_TYPE)), 0});
        }

        if (sLen + sIdx < curS) {
            sIdx += sLen;
        } else if (nIdx == n2 * curG - 1) {
            sIdx = 0;
            nIdx = 0;
            bIdx++;
            if constexpr (LAYOUT == TND) {
                scrOffsetBase += curS * n2 * curG * dAlign;
                dstOffsetBase += curS * n2 * curG * d;
                curS = ((__gm__ int64_t *)seqS)[bIdx] - ((__gm__ int64_t *)seqS)[bIdx - 1];
            }
        } else {
            sIdx = 0;
            nIdx++;
        }
        if constexpr (LAYOUT == TND) {
            copyInSrcOffset = scrOffsetBase + nIdx * curS * dAlign + sIdx * C0_SIZE;
            copyOutDstOffset = dstOffsetBase + (sIdx * n2 * curG + nIdx) * d;
        } else {
            ComputeDataCopyOffset(curG, curS);
        }
        ubOffset += dataLen;
        inUbOffset += dataLen + dAlign / C0_SIZE * cal_block_num;
        sLen = sClcSize > curS ? curS : sClcSize;
        sLen = sLen > (curS - sIdx) ? (curS - sIdx) : sLen;
        sLen = sLen > 255 ? 255 : sLen;
        dataLen = sLen * dAlign;
        if ((sLen > 0) && (inUbOffset + dataLen + dAlign / C0_SIZE * cal_block_num) * sizeof(float) >
                              ubBaseSize * 2 + nzReservedSize) {
            inUbOffset = 0;
            SetFlag<HardEvent::V_MTE2>(curEventId);
            WaitFlag<HardEvent::V_MTE2>(curEventId);
        }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPing);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPong);
    outQueueCommon.FreeTensor(vecOut);
}

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT>
__aicore__ inline void FusedFloydAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT>::NZProcess()
{
    uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
    uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;
    if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
        qEnd = qPostBlockTotal;
    }

    InitIndex(qBegin, g, s1, actual_seq_qlen_addr);

    for (uint64_t i = qBegin; i < qEnd; i = i + 2 * qPostBaseNum) {
        uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
        NZVecClc(dqWorkSpaceGm, dqGm, dataSize, actual_seq_qlen_addr, g, s1, true, 0);
        uint64_t dataSize1 = i + 2 * qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
        dataSize1 = i + qPostBaseNum >= qPostBlockTotal ? 0 : dataSize1;
        NZVecClc(dqWorkSpaceGm, dqGm, dataSize1, actual_seq_qlen_addr, g, s1, true, 1);
    }

    // init k
    uint64_t kvBegin = cBlockIdx * kvPostBlockFactor * kvPostBaseNum;
    uint64_t kvEnd = (cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum;
    if (((cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum) > kvPostBlockTotal) {
        kvEnd = kvPostBlockTotal;
    }
    InitIndex(kvBegin, g, s2, actual_seq_kvlen_addr);
    for (uint64_t i = kvBegin; i < kvEnd; i = i + 2 * kvPostBaseNum) {
        uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
        NZVecClc(dkWorkSpaceGm, dkGm, dataSize, actual_seq_kvlen_addr, g, s2, true, 0);
        uint64_t dataSize1 = i + 2 * kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
        dataSize1 = i + kvPostBaseNum >= kvPostBlockTotal ? 0 : dataSize1;
        NZVecClc(dkWorkSpaceGm, dkGm, dataSize1, actual_seq_kvlen_addr, g, s2, true, 1);
    }

    // init v
    if constexpr (CAST_DV) {
        InitIndex(kvBegin, g, s2, actual_seq_kvlen_addr);
        for (uint64_t i = kvBegin; i < kvEnd; i = i + 2 * kvPostBaseNum) {
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            NZVecClc(dvWorkSpaceGm, dvGm, dataSize, actual_seq_kvlen_addr, g, s2, false, 0);
            uint64_t dataSize1 = i + 2 * kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            dataSize1 = i + kvPostBaseNum >= kvPostBlockTotal ? 0 : dataSize1;
            NZVecClc(dvWorkSpaceGm, dvGm, dataSize1, actual_seq_kvlen_addr, g, s2, false, 1);
        }
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT>
__aicore__ inline void FusedFloydAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT>::Process()
{
    if constexpr (INPUT_FORMAT == NZ) {
        NZProcess();
        return;
    }
    // init q
    uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
    uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;

    if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
        qEnd = qPostBlockTotal;
    }
    for (uint64_t i = qBegin; i < qEnd; i = i + qPostBaseNum) {
        AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
        AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
        uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
        DataCopy(vecIn, dqWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
        inQueue.EnQue(vecIn);
        inQueue.template DeQue<float>();
        if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
            Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();
            DataCopy(dqGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
        } else {
            Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
            PipeBarrier<PIPE_V>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();
            DataCopy(dqGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
        }
        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
    }
    PipeBarrier<PIPE_ALL>();
    // init k
    uint64_t kvBegin = cBlockIdx * kvPostBlockFactor * kvPostBaseNum;
    uint64_t kvEnd = (cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum;
    if (((cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum) > kvPostBlockTotal) {
        kvEnd = kvPostBlockTotal;
    }

    for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
        AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
        AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
        uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
        DataCopy(vecIn, dkWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
        inQueue.EnQue(vecIn);
        inQueue.template DeQue<float>();
        if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
            Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();
            DataCopy(dkGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
        } else {
            Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
            PipeBarrier<PIPE_V>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();
            DataCopy(dkGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
        }
        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
    }
    PipeBarrier<PIPE_ALL>();

    // init v

    if constexpr (CAST_DV && !AscendC::IsSameType<OUT_TYPE, float>::value) {
        for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            DataCopy(vecIn, dvWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            inQueue.EnQue(vecIn);
            inQueue.template DeQue<float>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();
            DataCopy(dvGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
        }
    }
    PipeBarrier<PIPE_ALL>();

    uint64_t k1v1Begin = cBlockIdx * k1v1PostBlockFactor * k1v1PostBaseNum;
    uint64_t k1v1End = (cBlockIdx + 1) * k1v1PostBlockFactor * k1v1PostBaseNum;
    if (((cBlockIdx + 1) * k1v1PostBlockFactor * k1v1PostBaseNum) > k1v1PostBlockTotal) {
        k1v1End = k1v1PostBlockTotal;
    }
    // init k1
    for (uint64_t i = k1v1Begin; i < k1v1End; i = i + k1v1PostBaseNum) {
        AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
        AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
        uint64_t dataSize = i + k1v1PostBaseNum < k1v1PostBlockTotal ? k1v1PostBaseNum : k1v1PostTailNum;
        DataCopy(vecIn, dk1WorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
        inQueue.EnQue(vecIn);
        inQueue.template DeQue<float>();

        Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
        PipeBarrier<PIPE_V>();
        Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
        outQueue.EnQue(vecOut);
        outQueue.template DeQue<OUT_TYPE>();
        DataCopy(dk1Gm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
    }
    // init v1

    if constexpr (CAST_DV && !AscendC::IsSameType<OUT_TYPE, float>::value) {
        for (uint64_t i = k1v1Begin; i < k1v1End; i = i + k1v1PostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
            uint64_t dataSize = i + k1v1PostBaseNum < k1v1PostBlockTotal ? k1v1PostBaseNum : k1v1PostTailNum;
            DataCopy(vecIn, dv1WorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            inQueue.EnQue(vecIn);
            inQueue.template DeQue<float>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();
            DataCopy(dv1Gm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
        }
    }
}

#endif // _fused_floyd_attention_grad_POST_H_
