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
 * \file sparse_flash_attention_grad_post.h
 * \brief common post process
 */

#pragma once
#include "kernel_operator.h"
using namespace AscendC;

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const bool HAS_ROPE>
class SparseFlashAttentionGradPost {
public:
    __aicore__ inline SparseFlashAttentionGradPost(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
                                __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                                __gm__ uint8_t *dq_rope, __gm__ uint8_t *dk_rope, __gm__ uint8_t *workspace,
                                const TILING_TYPE *__restrict ordTilingData, TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void InitIndex(uint64_t startIdx, int64_t curG, int64_t &curS, int64_t headDim,
                                     int64_t headDimAlign, GM_ADDR seqS);
    __aicore__ inline void NZ2ND(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, uint64_t sLen,
                                 uint64_t ubOffset, uint64_t srcUbOffset, int64_t headDimAlign);
    __aicore__ inline void NZVecClc(GlobalTensor<float> srcGm, GlobalTensor<OUT_TYPE> dstGm, uint64_t dataSize,
                                    GM_ADDR seqS, int64_t curG, int64_t &curS, int64_t headDim, int64_t headDimAlign,
                                    bool needMuls, int64_t flag);
    __aicore__ inline void NZProcess();
    __aicore__ inline void ComputeDataCopyOffset(int64_t curG, int64_t &curS, int64_t headDim, int64_t headDimAlign);

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

    AscendC::GlobalTensor<OUT_TYPE> dqGm;
    AscendC::GlobalTensor<OUT_TYPE> dkGm;
    AscendC::GlobalTensor<OUT_TYPE> dvGm;
    AscendC::GlobalTensor<OUT_TYPE> dqRopeGm;
    AscendC::GlobalTensor<OUT_TYPE> dkRopeGm;
    // input
    AscendC::GlobalTensor<float> dqWorkSpaceGm;
    AscendC::GlobalTensor<float> dkWorkSpaceGm;
    AscendC::GlobalTensor<float> dvWorkSpaceGm;

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
    int64_t kPostBlockFactor;
    uint64_t kPostBlockTotal;
    int64_t kPostBaseNum;
    int64_t kPostTailNum;
    uint64_t kSizeAlign;
    int64_t vPostBlockFactor;
    uint64_t vPostBlockTotal;
    int64_t vPostBaseNum;
    int64_t vPostTailNum;
    uint64_t vSizeAlign;

    // org shape info
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t d2;
    int64_t dAlign;
    int64_t d2Align;
    int64_t dimDqk;
    int64_t dimRope;

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
          const uint32_t INPUT_FORMAT, const bool HAS_ROPE>
__aicore__ inline void
SparseFlashAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, HAS_ROPE>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *actual_seq_qlen,
    __gm__ uint8_t *actual_seq_kvlen, __gm__ uint8_t *dq_rope, __gm__ uint8_t *dk_rope, __gm__ uint8_t *workspace,
    const TILING_TYPE *__restrict ordTilingData, TPipe *pipe_in)
{
    cBlockIdx = GetBlockIdx();

    tilingData = ordTilingData;
    pipe = pipe_in;

    dqGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dq);
    dkGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dk);
    dvGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dv);
    dqRopeGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dq_rope);
    dkRopeGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dk_rope);

    // tiling_data
    usedCoreNum = tilingData->postTilingData.coreNum;
    ubBaseSize = tilingData->postTilingData.postUbBaseSize;
    nzReservedSize = tilingData->postTilingData.nzReservedSize;
    qPostBlockFactor = tilingData->postTilingData.qPostBlockFactor;
    qPostBlockTotal = tilingData->postTilingData.qPostBlockTotal;
    qPostBaseNum = tilingData->postTilingData.qPostBaseNum;
    qPostTailNum = tilingData->postTilingData.qPostTailNum;
    kPostBlockFactor = tilingData->postTilingData.kPostBlockFactor;
    kPostBlockTotal = tilingData->postTilingData.kPostBlockTotal;
    kPostBaseNum = tilingData->postTilingData.kPostBaseNum;
    kPostTailNum = tilingData->postTilingData.kPostTailNum;
    vPostBlockFactor = tilingData->postTilingData.vPostBlockFactor;
    vPostBlockTotal = tilingData->postTilingData.vPostBlockTotal;
    vPostBaseNum = tilingData->postTilingData.vPostBaseNum;
    vPostTailNum = tilingData->postTilingData.vPostTailNum;
    qSizeAlign = tilingData->postTilingData.qSizeAlign;
    kSizeAlign = tilingData->postTilingData.kSizeAlign;
    vSizeAlign = tilingData->postTilingData.vSizeAlign;

    dimDqk = tilingData->opInfo.D;
    dimRope = tilingData->opInfo.ropeD;

    if constexpr (INPUT_FORMAT == NZ) {
        b = tilingData->postTilingData.b;
        n2 = tilingData->postTilingData.n2;
        g = tilingData->postTilingData.g;
        s1 = tilingData->postTilingData.s1;
        s2 = tilingData->postTilingData.s2;
        d = tilingData->postTilingData.d;
        d2 = tilingData->postTilingData.d2;
        dAlign = (d + 15) / 16 * 16;
        d2Align = (d2 + 15) / 16 * 16;
        actual_seq_qlen_addr = actual_seq_qlen;
        actual_seq_kvlen_addr = actual_seq_kvlen;
    }

    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
    if constexpr (CAST_DV) {
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      tilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
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
          const uint32_t INPUT_FORMAT, const bool HAS_ROPE>
__aicore__ inline void SparseFlashAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT,
                                                    HAS_ROPE>::InitIndex(uint64_t startIdx, int64_t curG, int64_t &curS,
                                                                         int64_t headDim, int64_t headDimAlign,
                                                                         GM_ADDR seqS)
{
    if constexpr (LAYOUT == TND) {
        uint64_t totalLen = 0;
        for (int64_t bDimIdx = 0; bDimIdx < b; bDimIdx++) {
            totalLen = n2 * curG * ((__gm__ int32_t *)seqS)[bDimIdx] * headDim;
            if (totalLen > startIdx) {
                bIdx = bDimIdx;
                curS = (bIdx == 0) ? ((__gm__ int32_t *)seqS)[bIdx] :
                                     (((__gm__ int32_t *)seqS)[bIdx] - ((__gm__ int32_t *)seqS)[bIdx - 1]);
                uint64_t bTail = startIdx - (totalLen - n2 * curG * curS * headDim);
                nIdx = bTail / (curS * headDim);
                uint64_t nTail = bTail % (curS * headDim);
                sIdx = nTail / headDim;
                break;
            }
        }
        // 计算输入、输出的offset
        dstOffsetBase = totalLen - n2 * curG * curS * headDim;
        scrOffsetBase = dstOffsetBase / headDim * headDimAlign;

        copyInSrcOffset = scrOffsetBase + nIdx * curS * headDimAlign + sIdx * C0_SIZE;
        copyOutDstOffset = dstOffsetBase + (sIdx * n2 * curG + nIdx) * headDim;
    } else { // 补充offset 计算
        bIdx = startIdx / (n2 * curG * curS * headDim);
        uint64_t bTail = startIdx % (n2 * curG * curS * headDim);
        nIdx = bTail / (curS * headDim);
        uint64_t nTail = bTail % (curS * headDim);
        sIdx = nTail / headDim;
        ComputeDataCopyOffset(curG, curS, headDim, headDimAlign);
    }
}

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const bool HAS_ROPE>
__aicore__ inline void SparseFlashAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT,
                                                    HAS_ROPE>::ComputeDataCopyOffset(int64_t curG, int64_t &curS,
                                                                                     int64_t headDim,
                                                                                     int64_t headDimAlign)
{
    // src BNSD
    scrOffsetBase = bIdx * n2 * curS * curG * headDimAlign;
    copyInSrcOffset = scrOffsetBase + nIdx * curS * headDimAlign + sIdx * C0_SIZE;

    if constexpr (LAYOUT == BSNGD) {
        // BSND
        copyOutDstStride = n2 * curG * headDim - headDim;
        copyOutDstOffset = ((bIdx * curS + sIdx) * n2 * curG + nIdx) * headDim;
    } else if constexpr (LAYOUT == SBNGD) {
        // SBND
        copyOutDstStride = b * n2 * curG * headDim - headDim;
        copyOutDstOffset = ((sIdx * b + bIdx) * n2 * curG + nIdx) * headDim;
    } else if constexpr (LAYOUT == BNGSD) {
        // BNSD
        copyOutDstStride = 0;
        copyOutDstOffset = ((bIdx * n2 * curG + nIdx) * curS + sIdx) * headDim;
    }
}

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const bool HAS_ROPE>
__aicore__ inline void SparseFlashAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT,
                                                    HAS_ROPE>::NZ2ND(LocalTensor<float> &dstTensor,
                                                                     LocalTensor<float> &srcTensor, uint64_t sLen,
                                                                     uint64_t ubOffset, uint64_t srcUbOffset,
                                                                     int64_t headDimAlign)
{
    /*
    Func:
    将NZ转为ND
    */
    CopyRepeatParams nz2ndParams;
    nz2ndParams.srcStride = sLen * C0_SIZE / cal_block_num + 1;
    nz2ndParams.dstStride = C0_SIZE / cal_block_num;
    nz2ndParams.srcRepeatSize = C0_SIZE / cal_block_num;
    nz2ndParams.dstRepeatSize = headDimAlign / cal_block_num;

    uint16_t c0_repeat = C0_SIZE / cal_block_num;
    uint16_t c1_repeat = headDimAlign / C0_SIZE / VEC_REPEAT;
    uint16_t c1_remain = headDimAlign / C0_SIZE % VEC_REPEAT;
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
          const uint32_t INPUT_FORMAT, const bool HAS_ROPE>
__aicore__ inline void
SparseFlashAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, HAS_ROPE>::NZVecClc(
    GlobalTensor<float> srcGm, GlobalTensor<OUT_TYPE> dstGm, uint64_t dataSize, GM_ADDR seqS, int64_t curG,
    int64_t &curS, int64_t headDim, int64_t headDimAlign, bool needMuls, int64_t flag)
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

    uint32_t sClcSize = dataSize / headDim;

    uint64_t sLen = (sIdx + sClcSize) > curS ? (curS - sIdx) : sClcSize;
    sLen = sLen > 255 ? 255 : sLen;
    uint64_t dataLen = sLen * headDimAlign;

    uint64_t ubOffset = 0;
    uint64_t inUbOffset = 0;

    while (sClcSize > 0) {
        // Nz copy In
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = headDimAlign / C0_SIZE;
        intriParams.blockLen = sLen * C0_SIZE * sizeof(float);
        intriParams.srcStride = curS * C0_SIZE * sizeof(float) - intriParams.blockLen;
        intriParams.dstStride = 1; // 间隔一个block，防止bank冲突
        intriParams.rsv = 0;
        DataCopyPad(vecIn[inUbOffset], srcGm[copyInSrcOffset], intriParams, {false, 0, 0, 0});
        sClcSize = sClcSize - sLen;

        inQueueCommon.EnQue(vecIn);
        inQueueCommon.template DeQue<float>();

        if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
            NZ2ND(tmpTensor, vecIn, sLen, ubOffset, inUbOffset, headDimAlign);
        } else {
            NZ2ND(vecOut, vecIn, sLen, ubOffset, inUbOffset, headDimAlign);
        }

        if (sClcSize <= 0) {
            inQueueCommon.FreeTensor(vecIn);
        }

        PIPE_BARRIER(PIPE_V);
        if (needMuls) {
            if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
                Muls(tmpTensor[ubOffset], tmpTensor[ubOffset], (float)tilingData->postTilingData.scaleValue,
                     sLen * headDimAlign);
            } else {
                Muls(vecOut[ubOffset], vecOut[ubOffset], (float)tilingData->postTilingData.scaleValue,
                     sLen * headDimAlign);
            }

            PIPE_BARRIER(PIPE_V);
        }

        if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
            Cast(vecOut[ubOffset], tmpTensor[ubOffset], RoundMode::CAST_ROUND, sLen * headDimAlign);
            PIPE_BARRIER(PIPE_V);
        }

        outQueueCommon.EnQue(vecOut);
        outQueueCommon.template DeQue<OUT_TYPE>();

        if constexpr (LAYOUT == TND) {
            DataCopyPad(
                dstGm[copyOutDstOffset], vecOut[ubOffset],
                {static_cast<uint16_t>(dataLen / headDimAlign), static_cast<uint32_t>(headDim * sizeof(OUT_TYPE)), 0,
                 static_cast<uint32_t>((n2 * curG * headDim - headDim) * sizeof(OUT_TYPE)), 0});
        } else {
            DataCopyPad(
                dstGm[copyOutDstOffset], vecOut[ubOffset],
                {static_cast<uint16_t>(dataLen / headDimAlign), static_cast<uint32_t>(headDim * sizeof(OUT_TYPE)), 0,
                 static_cast<uint32_t>(copyOutDstStride * sizeof(OUT_TYPE)), 0});
        }

        if (sLen + sIdx < curS) {
            sIdx += sLen;
        } else if (nIdx == n2 * curG - 1) {
            sIdx = 0;
            nIdx = 0;
            bIdx++;
            if constexpr (LAYOUT == TND) {
                scrOffsetBase += curS * n2 * curG * headDimAlign;
                dstOffsetBase += curS * n2 * curG * headDim;
                curS = ((__gm__ int32_t *)seqS)[bIdx] - ((__gm__ int32_t *)seqS)[bIdx - 1];
            }
        } else {
            sIdx = 0;
            nIdx++;
        }
        if constexpr (LAYOUT == TND) {
            copyInSrcOffset = scrOffsetBase + nIdx * curS * headDimAlign + sIdx * C0_SIZE;
            copyOutDstOffset = dstOffsetBase + (sIdx * n2 * curG + nIdx) * headDim;
        } else {
            ComputeDataCopyOffset(curG, curS, headDim, headDimAlign);
        }
        ubOffset += dataLen;
        inUbOffset += dataLen + headDimAlign / C0_SIZE * cal_block_num;
        sLen = sClcSize > curS ? curS : sClcSize;
        sLen = sLen > (curS - sIdx) ? (curS - sIdx) : sLen;
        sLen = sLen > 255 ? 255 : sLen;
        dataLen = sLen * headDimAlign;
        if ((sLen > 0) && (inUbOffset + dataLen + headDimAlign / C0_SIZE * cal_block_num) * sizeof(float) >
                              ubBaseSize * 2 + nzReservedSize) {
            inUbOffset = 0;
            SET_FLAG(V, MTE2, curEventId);
            WAIT_FLAG(V, MTE2, curEventId);
        }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPing);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPong);
    outQueueCommon.FreeTensor(vecOut);
}

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const bool HAS_ROPE>
__aicore__ inline void
SparseFlashAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, HAS_ROPE>::NZProcess()
{
    uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
    uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;
    if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
        qEnd = qPostBlockTotal;
    }

    InitIndex(qBegin, g, s1, d, dAlign, actual_seq_qlen_addr);

    for (uint64_t i = qBegin; i < qEnd; i = i + 2 * qPostBaseNum) {
        uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
        NZVecClc(dqWorkSpaceGm, dqGm, dataSize, actual_seq_qlen_addr, g, s1, d, dAlign, true, 0);
        uint64_t dataSize1 = i + 2 * qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
        dataSize1 = i + qPostBaseNum >= qPostBlockTotal ? 0 : dataSize1;
        NZVecClc(dqWorkSpaceGm, dqGm, dataSize1, actual_seq_qlen_addr, g, s1, d, dAlign, true, 1);
    }

    // init k
    uint64_t kBegin = cBlockIdx * kPostBlockFactor * kPostBaseNum;
    uint64_t kEnd = (cBlockIdx + 1) * kPostBlockFactor * kPostBaseNum;
    if (((cBlockIdx + 1) * kPostBlockFactor * kPostBaseNum) > kPostBlockTotal) {
        kEnd = kPostBlockTotal;
    }
    InitIndex(kBegin, 1, s2, d, dAlign, actual_seq_kvlen_addr);
    for (uint64_t i = kBegin; i < kEnd; i = i + 2 * kPostBaseNum) {
        uint64_t dataSize = i + kPostBaseNum < kPostBlockTotal ? kPostBaseNum : kPostTailNum;
        NZVecClc(dkWorkSpaceGm, dkGm, dataSize, actual_seq_kvlen_addr, 1, s2, d, dAlign, true, 0);
        uint64_t dataSize1 = i + 2 * kPostBaseNum < kPostBlockTotal ? kPostBaseNum : kPostTailNum;
        dataSize1 = i + kPostBaseNum >= kPostBlockTotal ? 0 : dataSize1;
        NZVecClc(dkWorkSpaceGm, dkGm, dataSize1, actual_seq_kvlen_addr, 1, s2, d, dAlign, true, 1);
    }

    // init v
    if constexpr (CAST_DV) {
        uint64_t vBegin = cBlockIdx * vPostBlockFactor * vPostBaseNum;
        uint64_t vEnd = (cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum;
        if (((cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum) > vPostBlockTotal) {
            vEnd = vPostBlockTotal;
        }
        InitIndex(vBegin, 1, s2, d2, d2Align, actual_seq_kvlen_addr);
        for (uint64_t i = vBegin; i < vEnd; i = i + 2 * vPostBaseNum) {
            uint64_t dataSize = i + vPostBaseNum < vPostBlockTotal ? vPostBaseNum : vPostTailNum;
            NZVecClc(dvWorkSpaceGm, dvGm, dataSize, actual_seq_kvlen_addr, 1, s2, d2, d2Align, false, 0);
            uint64_t dataSize1 = i + 2 * vPostBaseNum < vPostBlockTotal ? vPostBaseNum : vPostTailNum;
            dataSize1 = i + vPostBaseNum >= vPostBlockTotal ? 0 : dataSize1;
            NZVecClc(dvWorkSpaceGm, dvGm, dataSize1, actual_seq_kvlen_addr, 1, s2, d2, d2Align, false, 1);
        }
        PIPE_BARRIER(PIPE_ALL);
    }
}

template <typename OUT_TYPE, typename TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const bool HAS_ROPE>
__aicore__ inline void
SparseFlashAttentionGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, HAS_ROPE>::Process()
{
    if constexpr (INPUT_FORMAT == NZ) {
        NZProcess();
        return;
    }
    uint64_t totalD = dimDqk + dimRope;
    // init q
    uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
    uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;

    if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
        qEnd = qPostBlockTotal;
    }
    uint64_t dqOutGmOffset = cBlockIdx * qPostBlockFactor * (qPostBaseNum / totalD) * dimDqk;
    uint64_t dqRopeOutGmOffset = cBlockIdx * qPostBlockFactor * (qPostBaseNum / totalD) * dimRope;

    SetFlag<HardEvent::V_MTE2>(0);
    for (uint64_t i = qBegin; i < qEnd; i = i + qPostBaseNum) {
        AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
        AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
        uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
        WaitFlag<HardEvent::V_MTE2>(0);
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
            PIPE_BARRIER(PIPE_V);

            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();

            DataCopyParams repeatParams;
            repeatParams.blockCount = dataSize / totalD;
            repeatParams.blockLen = dimDqk * sizeof(OUT_TYPE) / 32;
            repeatParams.srcStride = dimRope * sizeof(OUT_TYPE) / 32;
            repeatParams.dstStride = 0;

            DataCopy(dqGm[dqOutGmOffset], vecOut, repeatParams);
            dqOutGmOffset += repeatParams.blockCount * dimDqk;

            if constexpr (HAS_ROPE) {
                // if ROPE_ENABLED: scatter --> dq & dq_rope
                repeatParams.blockLen = dimRope * sizeof(OUT_TYPE) / 32;
                repeatParams.srcStride = dimDqk * sizeof(OUT_TYPE) / 32;
                DataCopy(dqRopeGm[dqRopeOutGmOffset], vecOut[dimDqk], repeatParams);
                dqRopeOutGmOffset += repeatParams.blockCount * dimRope;
            }
        }
        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
        SetFlag<HardEvent::V_MTE2>(0);
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    PIPE_BARRIER(PIPE_ALL);
    // init k
    uint64_t kBegin = cBlockIdx * kPostBlockFactor * kPostBaseNum;
    uint64_t kEnd = (cBlockIdx + 1) * kPostBlockFactor * kPostBaseNum;
    if (((cBlockIdx + 1) * kPostBlockFactor * kPostBaseNum) > kPostBlockTotal) {
        kEnd = kPostBlockTotal;
    }
    uint64_t dkOutGmOffset = cBlockIdx * kPostBlockFactor * (kPostBaseNum / totalD) * dimDqk;
    uint64_t dkRopeOutGmOffset = cBlockIdx * kPostBlockFactor * (kPostBaseNum / totalD) * dimRope;

    SetFlag<HardEvent::V_MTE2>(0);
    for (uint64_t i = kBegin; i < kEnd; i = i + kPostBaseNum) {
        AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
        AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
        uint64_t dataSize = i + kPostBaseNum < kPostBlockTotal ? kPostBaseNum : kPostTailNum;
        WaitFlag<HardEvent::V_MTE2>(0);
        DataCopy(vecIn, dkWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
        inQueue.EnQue(vecIn);
        inQueue.template DeQue<float>();
        if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
            PIPE_BARRIER(PIPE_ALL);
            Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();
            PIPE_BARRIER(PIPE_ALL);
            DataCopy(dkGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
        } else {
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();

            DataCopyParams repeatParams;
            repeatParams.blockCount = dataSize / totalD;
            repeatParams.blockLen = dimDqk * sizeof(OUT_TYPE) / 32;
            repeatParams.srcStride = dimRope * sizeof(OUT_TYPE) / 32;
            repeatParams.dstStride = 0;

            DataCopy(dkGm[dkOutGmOffset], vecOut, repeatParams);
            dkOutGmOffset += repeatParams.blockCount * dimDqk;

            if constexpr (HAS_ROPE) {
                // if ROPE_ENABLED: scatter --> dk & dk_rope
                repeatParams.blockLen = dimRope * sizeof(OUT_TYPE) / 32;
                repeatParams.srcStride = dimDqk * sizeof(OUT_TYPE) / 32;
                DataCopy(dkRopeGm[dkRopeOutGmOffset], vecOut[dimDqk], repeatParams);
                dkRopeOutGmOffset += repeatParams.blockCount * dimRope;
            }
        }
        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
        SetFlag<HardEvent::V_MTE2>(0);
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    PIPE_BARRIER(PIPE_ALL);

    // init v
    if constexpr (CAST_DV && !AscendC::IsSameType<OUT_TYPE, float>::value) {
        uint64_t vBegin = cBlockIdx * vPostBlockFactor * vPostBaseNum;
        uint64_t vEnd = (cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum;
        if (((cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum) > vPostBlockTotal) {
            vEnd = vPostBlockTotal;
        }

        SetFlag<HardEvent::V_MTE2>(0);
        for (uint64_t i = vBegin; i < vEnd; i = i + vPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
            uint64_t dataSize = i + vPostBaseNum < vPostBlockTotal ? vPostBaseNum : vPostTailNum;
            WaitFlag<HardEvent::V_MTE2>(0);
            DataCopy(vecIn, dvWorkSpaceGm[i], (dataSize + 7) / 8 * 8);
            inQueue.EnQue(vecIn);
            inQueue.template DeQue<float>();

            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();

            DataCopy(dvGm[i], vecOut, (dataSize + 15) / 16 * 16);
            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
            SetFlag<HardEvent::V_MTE2>(0);
        }
        WaitFlag<HardEvent::V_MTE2>(0);
    }
}
