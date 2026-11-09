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
 * \file ori_sparse_smlag.h
 * \brief ori_kv + ori_sparse_indices backward. cmp_kv is absent.
 */

#pragma once
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "basic_modules/sparse_flash_mla_grad_cube_op.h"
#include "basic_modules/sparse_flash_mla_grad_vec_op.h"
#include "basic_modules/sparse_flash_mla_grad_common_header.h"
#include "post.h"

namespace SMLAG_BASIC {

template <typename SMLAGT>
class OriSparseAttentionGrad {
    using TILING_CLASS = typename SMLAGT::tiling_class;
    using T1 = typename SMLAGT::t1;
    static constexpr bool IS_BSND = SMLAGT::is_bsnd;
    static constexpr bool HAS_SEQUSED = SMLAGT::has_seqused;
    static constexpr bool IS_DETER = SMLAGT::is_deter;

public:
    __aicore__ inline OriSparseAttentionGrad(){};
    __aicore__ inline void Process(GM_ADDR query, GM_ADDR ori_kv, GM_ADDR attention_out, GM_ADDR attention_out_grad,
                                   GM_ADDR lse, GM_ADDR ori_sparse_indices, GM_ADDR ori_topk_length,
                                   GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_ori_kv, GM_ADDR seqused_q,
                                   GM_ADDR seqused_ori_kv, GM_ADDR sinks, GM_ADDR dq, GM_ADDR d_ori_kv, GM_ADDR dsinks,
                                   GM_ADDR workspace, const TILING_CLASS* __restrict tilingData);

private:
    __aicore__ inline void Init(const TILING_CLASS* __restrict tilingData);
    __aicore__ inline void GetTndSeqLen(GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_ori_kv, int64_t t1Idx);
    __aicore__ inline void GetActualSelCount();
    __aicore__ inline void UpdateGmOffset(int64_t task, int32_t loop);
    __aicore__ inline void SaveLastInfo();
    __aicore__ inline void CubeCompute(CubeOp<SMLAGT>& cubeOp);
    __aicore__ inline void VecCompute(VecOp<SMLAGT>& vecOp, bool& hasPending);
    __aicore__ inline void DrainCube(CubeOp<SMLAGT>& cubeOp, bool& hasPending);
    __aicore__ inline void DrainVec(VecOp<SMLAGT>& vecOp, bool& hasPending);

    constexpr static uint32_t CUBE_WAIT_VEC_PING = 0;
    constexpr static uint32_t CUBE_WAIT_VEC_PONG = 1;
    constexpr static uint32_t VEC_WAIT_CUBE_PING = 2;
    constexpr static uint32_t VEC_WAIT_CUBE_PONG = 3;
    constexpr static uint32_t CUBE_WAIT_VEC_GATHER_PING = 4;
    constexpr static uint32_t CUBE_WAIT_VEC_GATHER_PONG = 5;
    constexpr static uint32_t SCATTER_SYNC_FLAG = 6;

    uint32_t cubeBlockIdx{0};
    uint32_t subBlockIdx{0};
    uint32_t formerCoreNum{0};
    uint32_t processBS1ByCore{0};
    uint32_t usedCoreNum{0};
    uint32_t formerProcessNNum{0};
    uint32_t remainProcessNNum{0};

    int64_t dimG{0};
    int64_t dimN2{0};
    int64_t dimDqk{0};
    int64_t dimDv{0};
    int64_t dimS1{0};
    int64_t t1Offset{0};
    int64_t t2Offset{0};
    int64_t curS1{0};
    int64_t curS2{0};
    int64_t usedS1{0};
    int64_t usedS2{0};
    int64_t bIndex{0};
    int64_t s1Index{0};
    int64_t n2Index{0};
    int64_t loopCnt{0};

    uint64_t queryGmOffset{0};
    uint64_t dyGmOffset{0};
    uint64_t lseGmOffset{0};
    uint64_t oriKeyGmOffset{0};
    uint64_t indicesGmOffset{0};
    uint64_t mm12GmOffset{0};
    uint64_t mm345GmOffset{0};
    uint64_t selectedKGmOffset{0};

    uint32_t mm12WorkspaceLen{0};
    uint32_t selectedKWorkspaceLen{0};
    int64_t selectedKWspOffset{0};
    uint32_t mmPingPongIdx{0};
    uint32_t selectdKPPPidx{0};
    int32_t blkCntOffset{0};
    int32_t lastblkCntOffset{0};
    int32_t selectedCountOffset{0};
    int32_t selectedBlockCount{0};
    int32_t selectedBlockSize{1};
    int32_t actualSelectedBlockCount{0};
    int32_t sparseLoop{0};
    int32_t sparseTail{0};
    int32_t actualSelCntOffset{0};
    int64_t scatterTaskId{0};
    int64_t roundDbIdx{0};

    RunInfo runInfo[2];
    event_t vWaitMte3Proc;

    GM_ADDR oriTopkLength_{nullptr};
    GM_ADDR sequsedQ_{nullptr};
    GM_ADDR sequsedOriKv_{nullptr};
    bool hasUsedSeqQ{false};
    bool hasUsedSeqOriKV{false};
};

template <typename SMLAGT>
__aicore__ inline void OriSparseAttentionGrad<SMLAGT>::Init(const TILING_CLASS* __restrict tilingData)
{
    dimS1 = tilingData->opInfo.S1;
    dimG = tilingData->opInfo.G;
    dimN2 = tilingData->opInfo.N2;
    dimDqk = tilingData->opInfo.D;
    dimDv = dimDqk;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
    mm12WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen / 2 / sizeof(float);
    if constexpr (IS_BSND) {
        curS1 = tilingData->opInfo.S1;
        curS2 = tilingData->opInfo.S2;
    }
    if ASCEND_IS_AIC {
        cubeBlockIdx = GetBlockIdx();
    }
    if ASCEND_IS_AIV {
        cubeBlockIdx = GetBlockIdx() / 2;
        subBlockIdx = GetBlockIdx() % 2;
    }
    formerCoreNum = tilingData->opInfo.formerCoreNum;
    usedCoreNum = tilingData->opInfo.usedCoreNum;
    formerProcessNNum = tilingData->opInfo.formerCoreProcessNNum;
    remainProcessNNum = tilingData->opInfo.remainCoreProcessNNum;
    processBS1ByCore = cubeBlockIdx < formerCoreNum ? formerProcessNNum : remainProcessNNum;
    selectedKWorkspaceLen = tilingData->opInfo.selectedKWorkspaceLen;
    selectedKWspOffset = selectedKWorkspaceLen / sizeof(T1) / 4;
    selectedCountOffset = tilingData->splitCoreParams.singleN;
    if constexpr (HAS_SEQUSED) {
        hasUsedSeqQ = tilingData->opInfo.hasUsedSeqQ;
        hasUsedSeqOriKV = tilingData->opInfo.hasUsedSeqOriKV;
    }
}

template <typename SMLAGT>
__aicore__ inline void OriSparseAttentionGrad<SMLAGT>::GetTndSeqLen(GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_ori_kv,
                                                                    int64_t t1Idx)
{
    if constexpr (IS_BSND) {
        t1Offset = t1Idx;
        bIndex = t1Idx / curS1;
        s1Index = t1Idx % dimS1;
        t2Offset = bIndex * curS2;
        usedS1 = curS1;
        usedS2 = curS2;
        if constexpr (HAS_SEQUSED) {
            if (hasUsedSeqQ) {
                usedS1 = ((__gm__ int32_t*)sequsedQ_)[bIndex];
            }
            if (hasUsedSeqOriKV) {
                usedS2 = ((__gm__ int32_t*)sequsedOriKv_)[bIndex];
            }
        }
        return;
    }

    int64_t curT1 = ((__gm__ int32_t*)cu_seqlens_q)[bIndex];
    while (t1Idx >= curT1) {
        curT1 = ((__gm__ int32_t*)cu_seqlens_q)[++bIndex];
    }
    if (unlikely(bIndex == 0)) {
        t1Offset = 0;
        t2Offset = 0;
        curS1 = ((__gm__ int32_t*)cu_seqlens_q)[bIndex + 1];
        curS2 = ((__gm__ int32_t*)cu_seqlens_ori_kv)[bIndex + 1];
    } else {
        t1Offset = ((__gm__ int32_t*)cu_seqlens_q)[bIndex - 1];
        t2Offset = ((__gm__ int32_t*)cu_seqlens_ori_kv)[bIndex - 1];
        curS1 = ((__gm__ int32_t*)cu_seqlens_q)[bIndex] - ((__gm__ int32_t*)cu_seqlens_q)[bIndex - 1];
        curS2 = ((__gm__ int32_t*)cu_seqlens_ori_kv)[bIndex] - ((__gm__ int32_t*)cu_seqlens_ori_kv)[bIndex - 1];
    }
    s1Index = t1Idx - t1Offset;
    int64_t batch = bIndex == 0 ? 0 : bIndex - 1;
    usedS1 = curS1;
    usedS2 = curS2;
    if constexpr (HAS_SEQUSED) {
        if (hasUsedSeqQ) {
            usedS1 = ((__gm__ int32_t*)sequsedQ_)[batch];
        }
        if (hasUsedSeqOriKV) {
            usedS2 = ((__gm__ int32_t*)sequsedOriKv_)[batch];
        }
    }
}

template <typename SMLAGT>
__aicore__ inline void OriSparseAttentionGrad<SMLAGT>::GetActualSelCount()
{
    int64_t globalT1 = IS_BSND ? t1Offset : (t1Offset + s1Index);
    int64_t len = ((__gm__ int32_t*)oriTopkLength_)[globalT1 * dimN2 + n2Index];
    if (len < 0) {
        len = 0;
    }
    int64_t cap = Min(static_cast<int64_t>(selectedBlockCount), usedS2);
    actualSelectedBlockCount = static_cast<int32_t>(Min(len, cap));
    sparseLoop = CeilDiv(static_cast<int64_t>(actualSelectedBlockCount),
                         static_cast<int64_t>(selectedCountOffset > 0 ? selectedCountOffset : 1));
    sparseTail = actualSelectedBlockCount % selectedCountOffset;
    if (sparseTail == 0) {
        sparseTail = actualSelectedBlockCount == 0 ? 0 : selectedCountOffset;
    }
}

template <typename SMLAGT>
__aicore__ inline void OriSparseAttentionGrad<SMLAGT>::UpdateGmOffset(int64_t task, int32_t loop)
{
    if constexpr (IS_BSND) {
        queryGmOffset = t1Offset * (dimG * dimN2 * dimDqk) + n2Index * (dimG * dimDqk);
        dyGmOffset = t1Offset * (dimG * dimN2 * dimDv) + n2Index * (dimG * dimDv);
        indicesGmOffset = t1Offset * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount;
        lseGmOffset = bIndex * dimN2 * dimS1 * dimG + n2Index * dimS1 * dimG + s1Index * dimG;
    } else {
        queryGmOffset = (t1Offset + s1Index) * (dimG * dimN2 * dimDqk) + n2Index * (dimG * dimDqk);
        dyGmOffset = (t1Offset + s1Index) * (dimG * dimN2 * dimDv) + n2Index * (dimG * dimDv);
        indicesGmOffset = (t1Offset + s1Index) * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount;
        lseGmOffset = n2Index * dimS1 * dimG + (t1Offset + s1Index) * dimG;
    }
    oriKeyGmOffset = t2Offset * (dimN2 * dimDqk) + n2Index * dimDqk;
    mm12GmOffset = mmPingPongIdx * mm12WorkspaceLen;
    mm345GmOffset = mmPingPongIdx * mm12WorkspaceLen;
    selectedKGmOffset = selectdKPPPidx * selectedKWspOffset;

    if (loop == 0) {
        blkCntOffset = 0;
    } else {
        blkCntOffset += actualSelCntOffset;
    }
    actualSelCntOffset = (loop == sparseLoop - 1) ? sparseTail : selectedCountOffset;

    RunInfo& info = runInfo[mmPingPongIdx];
    info.isOri = false;
    info.task = task;
    info.lseGmOffset = lseGmOffset;
    info.blkCntOffset = blkCntOffset;
    info.queryGmOffset = queryGmOffset;
    info.oriKeyGmOffset = oriKeyGmOffset;
    info.dyGmOffset = dyGmOffset;
    info.indicesGmOffset = indicesGmOffset;
    info.mm12GmOffset = mm12GmOffset;
    info.mm345GmOffset = mm345GmOffset;
    info.dqOutGmOffset = queryGmOffset;
    info.actualSelCntOffset = actualSelCntOffset;
    info.actualSelectedBlockCount = actualSelectedBlockCount;
    info.lastBlockSize = selectedBlockSize;
    info.isLastBasicBlock = (blkCntOffset + actualSelCntOffset >= actualSelectedBlockCount);
    info.scatterTaskId = scatterTaskId;
    info.roundDbIdx = roundDbIdx;
    info.s1Index = s1Index;
    info.curS1 = usedS1;
    info.curS2 = usedS2;
    info.selectedKGmOffset = selectedKGmOffset;
    info.selectedVGmOffset = selectedKGmOffset;
    info.curS1g = dimG;
    info.curS1Basic = 1;
    info.vWaitMte3Proc = vWaitMte3Proc;
    info.changeS1 = false;
}

template <typename SMLAGT>
__aicore__ inline void OriSparseAttentionGrad<SMLAGT>::SaveLastInfo()
{
    lastblkCntOffset = blkCntOffset;
    mmPingPongIdx = 1 - mmPingPongIdx;
    selectdKPPPidx = (selectdKPPPidx + 1) % 4;
    loopCnt++;
}

template <typename SMLAGT>
__aicore__ inline void OriSparseAttentionGrad<SMLAGT>::CubeCompute(CubeOp<SMLAGT>& cubeOp)
{
    int64_t taskMod = runInfo[mmPingPongIdx].task & 1;
    CrossCoreWaitFlag<2, PIPE_MTE2>(taskMod == 0 ? CUBE_WAIT_VEC_GATHER_PING : CUBE_WAIT_VEC_GATHER_PONG);
    cubeOp.cube12Process(runInfo[mmPingPongIdx], blkCntOffset, mmPingPongIdx);
    CrossCoreSetFlag<2, PIPE_FIX>(taskMod == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
    if (loopCnt == 0) {
        SaveLastInfo();
        return;
    }
    CrossCoreWaitFlag<2, PIPE_MTE2>(taskMod == 0 ? CUBE_WAIT_VEC_PONG : CUBE_WAIT_VEC_PING);
    cubeOp.cube345Process(runInfo[1 - mmPingPongIdx], lastblkCntOffset, 1 - mmPingPongIdx);
    SaveLastInfo();
}

template <typename SMLAGT>
__aicore__ inline void OriSparseAttentionGrad<SMLAGT>::VecCompute(VecOp<SMLAGT>& vecOp, bool& hasPending)
{
    int64_t taskMod = runInfo[mmPingPongIdx].task & 1;
    vecOp.GatherOriKV(runInfo[mmPingPongIdx]);
    CrossCoreSetFlag<2, PIPE_MTE3>(taskMod == 0 ? CUBE_WAIT_VEC_GATHER_PING : CUBE_WAIT_VEC_GATHER_PONG);
    if (hasPending) {
        int64_t taskMod1 = runInfo[1 - mmPingPongIdx].task & 1;
        CrossCoreWaitFlag(taskMod1 == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
        vecOp.Process(runInfo[1 - mmPingPongIdx]);
        CrossCoreSetFlag<2, PIPE_MTE3>(taskMod1 == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
    }
    mmPingPongIdx = 1 - mmPingPongIdx;
    selectdKPPPidx = (selectdKPPPidx + 1) % 4;
    hasPending = true;
}

template <typename SMLAGT>
__aicore__ inline void OriSparseAttentionGrad<SMLAGT>::DrainCube(CubeOp<SMLAGT>& cubeOp, bool& hasPending)
{
    if (!hasPending) {
        return;
    }
    int64_t taskMod = runInfo[1 - mmPingPongIdx].task & 1;
    CrossCoreWaitFlag<2, PIPE_MTE2>(taskMod == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
    cubeOp.cube345Process(runInfo[1 - mmPingPongIdx], lastblkCntOffset, 1 - mmPingPongIdx);
    hasPending = false;
    loopCnt = 0;
}

template <typename SMLAGT>
__aicore__ inline void OriSparseAttentionGrad<SMLAGT>::DrainVec(VecOp<SMLAGT>& vecOp, bool& hasPending)
{
    if (!hasPending) {
        return;
    }
    int64_t taskMod = runInfo[1 - mmPingPongIdx].task & 1;
    CrossCoreWaitFlag(taskMod == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
    vecOp.Process(runInfo[1 - mmPingPongIdx]);
    CrossCoreSetFlag<2, PIPE_MTE3>(taskMod == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
    hasPending = false;
}

template <typename SMLAGT>
__aicore__ inline void OriSparseAttentionGrad<SMLAGT>::Process(
    GM_ADDR query, GM_ADDR ori_kv, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR lse,
    GM_ADDR ori_sparse_indices, GM_ADDR ori_topk_length, GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_ori_kv,
    GM_ADDR seqused_q, GM_ADDR seqused_ori_kv, GM_ADDR sinks, GM_ADDR dq, GM_ADDR d_ori_kv, GM_ADDR dsinks,
    GM_ADDR workspace, const TILING_CLASS* __restrict tilingData)
{
    Init(tilingData);
    oriTopkLength_ = ori_topk_length;
    sequsedQ_ = seqused_q;
    sequsedOriKv_ = seqused_ori_kv;

    if ASCEND_IS_AIC {
        TPipe pipeCube;
        CubeOp<SMLAGT> cubeOp;
        cubeOp.Init(query, ori_kv, nullptr, attention_out_grad, workspace, tilingData, &pipeCube);
        AllocEventID();
        int64_t task = 0;
        int32_t rounds = static_cast<int32_t>(processBS1ByCore);
        if constexpr (IS_DETER) {
            rounds = static_cast<int32_t>(formerProcessNNum);
        }
        for (int32_t i = 0; i < rounds; ++i) {
            bool hasPending = false;
            scatterTaskId = i % 2;
            roundDbIdx = scatterTaskId;
            bool hasActual = cubeBlockIdx < usedCoreNum && i < static_cast<int32_t>(processBS1ByCore);
            int64_t t1Index = static_cast<int64_t>(cubeBlockIdx) + usedCoreNum * i;
            bool rowValid = hasActual;
            if (hasActual) {
                GetTndSeqLen(cu_seqlens_q, cu_seqlens_ori_kv, t1Index);
                if constexpr (HAS_SEQUSED) {
                    if (s1Index >= usedS1) {
                        rowValid = false;
                    }
                }
            }
            for (n2Index = 0; n2Index < dimN2; ++n2Index) {
                if (rowValid) {
                    GetActualSelCount();
                } else {
                    sparseLoop = 0;
                }
                for (int32_t loop = 0; loop < sparseLoop; ++loop) {
                    UpdateGmOffset(task, loop);
                    runInfo[mmPingPongIdx].s1Round = i;
                    CubeCompute(cubeOp);
                    hasPending = true;
                    task++;
                }
            }
            DrainCube(cubeOp, hasPending);
            CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
            loopCnt = 0;
        }
        FreeEventID();
    }

    if ASCEND_IS_AIV {
        TPipe pipeVec;
        VecOp<SMLAGT> vecOp;
        vecOp.Init(ori_kv, nullptr, attention_out, attention_out_grad, lse, ori_sparse_indices, sinks, dsinks,
                   cu_seqlens_q, cu_seqlens_ori_kv, nullptr, nullptr, nullptr, workspace, tilingData, &pipeVec);
        SyncAll();
        vWaitMte3Proc = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        SET_FLAG(MTE3, V, vWaitMte3Proc);
        if constexpr (IS_DETER) {
            vecOp.SetOriSparseDeterCtx(ori_topk_length, seqused_q, seqused_ori_kv);
        }

        int64_t task = 0;
        int32_t rounds = static_cast<int32_t>(processBS1ByCore);
        if constexpr (IS_DETER) {
            rounds = static_cast<int32_t>(formerProcessNNum);
        }
        for (int32_t i = 0; i < rounds; ++i) {
            bool hasPending = false;
            scatterTaskId = i % 2;
            roundDbIdx = scatterTaskId;
            bool hasActual = cubeBlockIdx < usedCoreNum && i < static_cast<int32_t>(processBS1ByCore);
            int64_t t1Index = static_cast<int64_t>(cubeBlockIdx) + usedCoreNum * i;
            bool rowValid = hasActual;
            RunInfo rowInfo;
            bool hasRow = false;
            if (hasActual) {
                GetTndSeqLen(cu_seqlens_q, cu_seqlens_ori_kv, t1Index);
                if constexpr (HAS_SEQUSED) {
                    if (s1Index >= usedS1) {
                        rowValid = false;
                    }
                }
            }
            for (n2Index = 0; n2Index < dimN2; ++n2Index) {
                if (rowValid) {
                    GetActualSelCount();
                } else {
                    sparseLoop = 0;
                }
                for (int32_t loop = 0; loop < sparseLoop; ++loop) {
                    UpdateGmOffset(task, loop);
                    runInfo[mmPingPongIdx].s1Round = i;
                    VecCompute(vecOp, hasPending);
                    rowInfo = runInfo[1 - mmPingPongIdx];
                    hasRow = true;
                    task++;
                }
            }
            DrainVec(vecOp, hasPending);
            CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
            if constexpr (IS_DETER) {
                SyncAll();
                vecOp.ScatterAddOriDeter(i, roundDbIdx);
            } else if (hasRow) {
                vecOp.ScatterAddOri(rowInfo);
            }
        }
        WAIT_FLAG(MTE3, V, vWaitMte3Proc);
        vecOp.CopyOutDsinks();
        SyncAll();
        pipeVec.Destroy();

        TPipe pipeCast;
        SparseFlashMlaGradPost<T1, TILING_CLASS, true, IS_BSND ? 2 : 3, 0, SMLAG_ORI_SPARSE_MODE> opCast;
        opCast.Init(dq, d_ori_kv, nullptr, dsinks, workspace, tilingData, &pipeCast);
        opCast.Process();
    }
}

} // namespace SMLAG_BASIC
