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
 * \file nsa_selected_attention_grad_basic.h
 * \brief
 */

#pragma once
#include "kernel_operator.h"
#include "../basic_modules/nsa_cube_op.h"
#include "../basic_modules/nsa_vec_op.h"
#include "../basic_modules/nsa_selected_attention_grad_common_header.h"
#include "nsa_selected_attention_grad_post.h"

namespace NSAG_BASIC {

template <typename NSAGT>
class SelectedAttentionGradBasic {
    using TILING_CLASS = typename NSAGT::tiling_class;
    using T1 = typename NSAGT::t1;
    static constexpr uint32_t ATTEN_ENABLE = NSAGT::atten_enable;

public:
    __aicore__ inline SelectedAttentionGradBasic(){};
    __aicore__ inline void Process(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                   GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                   GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                   GM_ADDR atten_mask, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                   const TILING_CLASS *__restrict tilingData);

private:
    __aicore__ inline void Init(const TILING_CLASS *__restrict tilingData);
    __aicore__ inline void CubeCompute(CubeOp<NSAGT> &cubeOp);
    __aicore__ inline void VecCompute(VecOp<NSAGT> &vecOp);
    __aicore__ inline void UpdateGmOffset();
    __aicore__ inline void SaveLastInfo();
    __aicore__ inline void GetTndSeqLen(const GM_ADDR actual_seq_qlen_addr, const GM_ADDR actual_seq_kvlen_addr,
                                        const int64_t t1Idx, int64_t &bIdx);

    uint32_t cubeBlockIdx;
    uint32_t subBlockIdx;
    uint32_t formerCoreNum;
    uint32_t processBS1ByCore;
    uint32_t usedCoreNum;
    // shape info
    int64_t dimG;
    int64_t dimN1;
    int64_t dimN2;
    int64_t dimDqk;
    int64_t dimDv;
    int64_t t1Offset;
    int64_t t2Offset{0};
    // attr
    uint32_t selectedBlockCount;
    // gmoffset
    uint64_t queryGmOffset;
    uint64_t dyGmOffset;
    uint64_t indicesGmOffset;
    uint64_t sumGmOffset;
    uint64_t keyGmOffset;
    uint64_t valueGmOffset;
    uint64_t mm12GmOffset;
    uint64_t mm345GmOffset;
    int32_t blkCntOffset;

    uint64_t lastQueryGmOffset;
    uint64_t lastKeyGmOffset;
    uint64_t lastValueGmOffset;
    uint64_t lastDyGmOffset;
    uint64_t lastIndicesGmOffset;
    uint64_t lastmm12GmOffset;
    uint64_t lastmm345GmOffset;
    int32_t lastblkCntOffset;
    // workspace
    uint32_t mm12WorkspaceLen;
    uint32_t mm345WorkspaceLen;
    // Index
    int64_t bIndex{0};
    int64_t s1Index{0};
    int64_t n2Index{0};
    int64_t loopCnt{0};
    uint32_t mmPingPongIdx{0};
    constexpr static const int32_t BLOCK_FP32 = 32 / sizeof(float);
    constexpr static const int32_t SELECTED_CNT_OFFSET = 8;
    // flag
    constexpr static uint32_t CUBE_WAIT_VEC = 0;
    constexpr static uint32_t VEC_WAIT_CUBE = 1;
    constexpr static uint32_t POST_WAIT_CUBE = 2;
    constexpr static uint32_t eventIdPing = 4;
    constexpr static uint32_t eventIdPong = 5;
};

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGradBasic<NSAGT>::Init(const TILING_CLASS *__restrict tilingData)
{
    dimG = tilingData->opInfo.G;
    dimN2 = tilingData->opInfo.N2;
    dimN1 = dimG * dimN2;
    dimDqk = tilingData->opInfo.D;
    dimDv = tilingData->opInfo.D2;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    mm12WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen / 2 / sizeof(float);
    mm345WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen / 2 / sizeof(T1);

    if ASCEND_IS_AIC {
        cubeBlockIdx = GetBlockIdx();
    }
    if ASCEND_IS_AIV {
        cubeBlockIdx = GetBlockIdx() / 2;
        subBlockIdx = GetBlockIdx() % 2;
    }

    formerCoreNum = tilingData->opInfo.formerCoreNum;
    usedCoreNum = tilingData->opInfo.usedCoreNum;
    if (cubeBlockIdx < formerCoreNum) {
        processBS1ByCore = tilingData->opInfo.formerCoreProcessNNum;
    } else {
        processBS1ByCore = tilingData->opInfo.remainCoreProcessNNum;
    }
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGradBasic<NSAGT>::Process(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR atten_mask,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, const TILING_CLASS *__restrict tilingData)
{
    Init(tilingData);

    // AIC Process
    if ASCEND_IS_AIC {
        TPipe pipeCube;
        CubeOp<NSAGT> cubeOp;
        cubeOp.Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
                    actual_seq_qlen, actual_seq_kvlen, atten_mask, dq, dk, dv, workspace, tilingData, &pipeCube);

        for (int32_t i = 0; i < processBS1ByCore; i++) {
            int32_t t1Index = cubeBlockIdx + usedCoreNum * i;
            GetTndSeqLen(actual_seq_qlen, actual_seq_kvlen, t1Index, bIndex);
            for (n2Index = 0; n2Index < dimN2; n2Index++) {
                for (blkCntOffset = 0; blkCntOffset < selectedBlockCount; blkCntOffset += SELECTED_CNT_OFFSET) {
                    UpdateGmOffset();
                    CubeCompute(cubeOp);
                }
            }
        }

        if (cubeBlockIdx < usedCoreNum) {
            CrossCoreWaitFlag<2, PIPE_FIX>(CUBE_WAIT_VEC);
            SET_FLAG(M, MTE1, eventIdPing);
            SET_FLAG(M, MTE1, eventIdPong);
            SET_FLAG(FIX, M, eventIdPing);
            SET_FLAG(FIX, M, eventIdPong);
            cubeOp.cube345Process(lastQueryGmOffset, lastKeyGmOffset, lastDyGmOffset, lastValueGmOffset,
                                  lastIndicesGmOffset, lastmm345GmOffset, lastblkCntOffset, 1 - mmPingPongIdx);
            WAIT_FLAG(M, MTE1, eventIdPing);
            WAIT_FLAG(M, MTE1, eventIdPong);
            WAIT_FLAG(FIX, M, eventIdPing);
            WAIT_FLAG(FIX, M, eventIdPong);
            CrossCoreSetFlag<2, PIPE_FIX>(POST_WAIT_CUBE);
        }
    }

    // AIV Process
    if ASCEND_IS_AIV {
        TPipe pipeVec;
        VecOp<NSAGT> vecOp;
        vecOp.Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
                   actual_seq_qlen, actual_seq_kvlen, atten_mask, dq, dk, dv, workspace, tilingData, &pipeVec);
        SyncAll();
        for (int32_t i = 0; i < processBS1ByCore; i++) {
            int32_t t1Index = cubeBlockIdx + usedCoreNum * i;
            GetTndSeqLen(actual_seq_qlen, actual_seq_kvlen, t1Index, bIndex);
            for (n2Index = 0; n2Index < dimN2; n2Index++) {
                for (blkCntOffset = 0; blkCntOffset < selectedBlockCount; blkCntOffset += SELECTED_CNT_OFFSET) {
                    UpdateGmOffset();
                    VecCompute(vecOp);
                }
            }
        }
        if (cubeBlockIdx < usedCoreNum) {
            CrossCoreWaitFlag<2, PIPE_FIX>(POST_WAIT_CUBE);
        }
        SyncAll();
        pipeVec.Destroy();

        TPipe pipeCast;
        NsaSelectedAttentionGradPost<T1, TILING_CLASS, true, 0, 3> opCast;
        opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, workspace, tilingData, &pipeCast);
        opCast.Process();
    }
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGradBasic<NSAGT>::CubeCompute(CubeOp<NSAGT> &cubeOp)
{
    if (unlikely(loopCnt == 0)) {
        SET_FLAG(M, MTE1, eventIdPing);
        SET_FLAG(M, MTE1, eventIdPong);
        SET_FLAG(FIX, M, eventIdPing);
        SET_FLAG(FIX, M, eventIdPong);
        cubeOp.cube12Process(queryGmOffset, keyGmOffset, dyGmOffset, valueGmOffset, indicesGmOffset, mm12GmOffset,
                             blkCntOffset, mmPingPongIdx);
        WAIT_FLAG(M, MTE1, eventIdPing);
        WAIT_FLAG(M, MTE1, eventIdPong);
        WAIT_FLAG(FIX, M, eventIdPing);
        WAIT_FLAG(FIX, M, eventIdPong);
        CrossCoreSetFlag<2, PIPE_FIX>(VEC_WAIT_CUBE);
        SaveLastInfo();
        return;
    }

    SET_FLAG(M, MTE1, eventIdPing);
    SET_FLAG(M, MTE1, eventIdPong);
    SET_FLAG(FIX, M, eventIdPing);
    SET_FLAG(FIX, M, eventIdPong);
    cubeOp.cube12Process(queryGmOffset, keyGmOffset, dyGmOffset, valueGmOffset, indicesGmOffset, mm12GmOffset,
                         blkCntOffset, mmPingPongIdx);
    WAIT_FLAG(M, MTE1, eventIdPing);
    WAIT_FLAG(M, MTE1, eventIdPong);
    WAIT_FLAG(FIX, M, eventIdPing);
    WAIT_FLAG(FIX, M, eventIdPong);

    CrossCoreSetFlag<2, PIPE_FIX>(VEC_WAIT_CUBE);
    CrossCoreWaitFlag<2, PIPE_FIX>(CUBE_WAIT_VEC);
    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);

    SET_FLAG(M, MTE1, eventIdPing);
    SET_FLAG(M, MTE1, eventIdPong);
    SET_FLAG(FIX, M, eventIdPing);
    SET_FLAG(FIX, M, eventIdPong);
    cubeOp.cube345Process(lastQueryGmOffset, lastKeyGmOffset, lastDyGmOffset, lastValueGmOffset, lastIndicesGmOffset,
                          lastmm345GmOffset, lastblkCntOffset, 1 - mmPingPongIdx);
    WAIT_FLAG(M, MTE1, eventIdPing);
    WAIT_FLAG(M, MTE1, eventIdPong);
    WAIT_FLAG(FIX, M, eventIdPing);
    WAIT_FLAG(FIX, M, eventIdPong);

    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    SaveLastInfo();
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGradBasic<NSAGT>::VecCompute(VecOp<NSAGT> &vecOp)
{
    CrossCoreWaitFlag(VEC_WAIT_CUBE);
    if (subBlockIdx == 0) {
        vecOp.Process(dyGmOffset, sumGmOffset, indicesGmOffset, s1Index, blkCntOffset, mm12GmOffset, mm345GmOffset);
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(CUBE_WAIT_VEC);
    mmPingPongIdx = 1 - mmPingPongIdx;
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGradBasic<NSAGT>::UpdateGmOffset()
{
    /*
     *  query:    B S1 N2 G D
     *  dy/out:   B S1 N2 G D2
     *  indices:  B S1 N2 SELCTED_BLOCK_COUNT
     *  sum/max   B S1 N2 G 8
     *  key:      B S2 N2 D
     *  value:    B S2 N2 D2
     */
    queryGmOffset = t1Offset * (dimN1 * dimDqk) + s1Index * (dimN1 * dimDqk) + n2Index * (dimG * dimDqk);
    dyGmOffset = t1Offset * (dimN1 * dimDv) + s1Index * (dimN1 * dimDv) + n2Index * (dimG * dimDv);
    indicesGmOffset =
        t1Offset * (dimN2 * selectedBlockCount) + s1Index * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount;
    sumGmOffset = t1Offset * dimN1 * BLOCK_FP32 + s1Index * dimN1 * BLOCK_FP32 + n2Index * dimG * BLOCK_FP32;
    keyGmOffset = t2Offset * (dimN2 * dimDqk) + n2Index * dimDqk;
    valueGmOffset = t2Offset * (dimN2 * dimDv) + n2Index * dimDv;

    // worksapce
    mm12GmOffset = mmPingPongIdx * mm12WorkspaceLen;
    mm345GmOffset = mmPingPongIdx * mm345WorkspaceLen;
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGradBasic<NSAGT>::SaveLastInfo()
{
    lastQueryGmOffset = queryGmOffset;
    lastKeyGmOffset = keyGmOffset;
    lastValueGmOffset = valueGmOffset;
    lastDyGmOffset = dyGmOffset;
    lastIndicesGmOffset = indicesGmOffset;
    lastmm12GmOffset = mm12GmOffset;
    lastmm345GmOffset = mm345GmOffset;
    lastblkCntOffset = blkCntOffset;
    mmPingPongIdx = 1 - mmPingPongIdx;
    loopCnt++;
}

template <typename NSAGT>
__aicore__ inline void SelectedAttentionGradBasic<NSAGT>::GetTndSeqLen(const GM_ADDR actual_seq_qlen_addr,
                                                                       const GM_ADDR actual_seq_kvlen_addr,
                                                                       const int64_t t1Idx, int64_t &bIdx)
{
    int64_t curT1 = ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndex];
    while (t1Idx >= curT1) {
        curT1 = ((__gm__ int64_t *)actual_seq_qlen_addr)[++bIndex];
    }

    if (unlikely(bIndex == 0)) {
        t1Offset = 0;
        t2Offset = 0;
    } else {
        t1Offset = ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndex - 1];
        t2Offset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIndex - 1];
    }

    s1Index = t1Idx - t1Offset;
}

} // namespace NSAG_BASIC
