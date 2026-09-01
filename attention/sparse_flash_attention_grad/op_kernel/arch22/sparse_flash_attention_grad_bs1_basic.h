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
 * \file sparse_flash_attention_grad_basic.h
 * \brief
 */

#pragma once
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "../../basic_modules/sparse_flash_attention_grad_cube_op.h"
#include "../../basic_modules/sparse_flash_attention_grad_vec_op.h"
#include "../../basic_modules/sparse_flash_attention_grad_common_header.h"
#include "sparse_flash_attention_grad_post.h"

namespace SFAG_BASIC {

template <typename SFAGT>
class SelectedAttentionGradBasic {
    using TILING_CLASS = typename SFAGT::tiling_class;
    using T1 = typename SFAGT::t1;
    static constexpr uint32_t ATTEN_ENABLE = SFAGT::atten_enable;
    static constexpr bool HAS_ROPE = SFAGT::has_rope;
    static constexpr bool IS_BSND = SFAGT::is_bsnd;
    static constexpr bool IS_DETERMINISTIC = SFAGT::is_deterministic;
    static constexpr bool KV_MERGE = SFAGT::kv_merge;

public:
    __aicore__ inline SelectedAttentionGradBasic(){};
    __aicore__ inline void Process(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                   GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                   GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                   GM_ADDR query_rope, GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                   GM_ADDR dq_rope, GM_ADDR dk_rope, GM_ADDR workspace,
                                   const TILING_CLASS *__restrict tilingData);

private:
    __aicore__ inline void Init(const TILING_CLASS *__restrict tilingData);
    __aicore__ inline void ProcessDeterministic(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                                GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                                GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                                GM_ADDR query_rope, GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk,
                                                GM_ADDR dv, GM_ADDR dq_rope, GM_ADDR dk_rope, GM_ADDR workspace,
                                                const TILING_CLASS *__restrict tilingData);
    __aicore__ inline void ProcessNonDeterministic(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                                   GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                                   GM_ADDR topk_indices, GM_ADDR actual_seq_qlen,
                                                   GM_ADDR actual_seq_kvlen, GM_ADDR query_rope, GM_ADDR key_rope,
                                                   GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dq_rope, GM_ADDR dk_rope,
                                                   GM_ADDR workspace, const TILING_CLASS *__restrict tilingData);
    __aicore__ inline void CubeCompute(CubeOp<SFAGT> &cubeOp);
    __aicore__ inline void VecCompute(VecOp<SFAGT> &vecOp, const GM_ADDR actual_seq_qlen,
                                      const GM_ADDR actual_seq_kvlen);
    __aicore__ inline void UpdateGmOffset(int64_t task, bool validflag);
    __aicore__ inline void SaveLastInfo();
    __aicore__ inline void GetTndSeqLen(const GM_ADDR actual_seq_qlen_addr, const GM_ADDR actual_seq_kvlen_addr,
                                        const int64_t t1Idx, int64_t &bIdx);
    __aicore__ inline void GetActualSelCount(const int64_t t1Idx, const int64_t n2Idx, int32_t &actSelBlkCount);

    __aicore__ inline void UpdateScatterInfo(int64_t t1ScatterIndex);
    __aicore__ inline void GetTndSeqLenScatter(const GM_ADDR actual_seq_qlen_addr, const GM_ADDR actual_seq_kvlen_addr,
                                               const int64_t t1Idx, int64_t &bIdx);
    __aicore__ inline void GetActualSelCountScatter(const int64_t t1Idx, const int64_t n2Idx, int32_t &actSelBlkCount);
    __aicore__ inline void ScatterAddByS1(VecOp<SFAGT> &vecOp, const GM_ADDR actual_seq_qlen_addr,
                                          const GM_ADDR actual_seq_kvlen);

    uint32_t cubeBlockIdx;
    uint32_t subBlockIdx;
    uint32_t formerCoreNum;
    uint32_t processBS1ByCore;
    uint32_t usedCoreNum;
    // shape info
    int64_t dimB{0};
    int64_t dimG;
    int64_t dimN1;
    int64_t dimN2;
    int64_t dimDqk;
    int64_t dimDv;
    int64_t dimRope;
    int64_t t1Offset;
    int64_t t2Offset{0};
    int64_t t1OffsetScatterIndex{0};
    int64_t t2OffsetScatterIndex{0};
    int64_t curS1;
    int64_t curS2;
    int64_t curMaxS2;
    int64_t curS1Scatter{0};
    int64_t curS2Scatter{0};
    int64_t curMaxS2Scatter{0};
    int64_t dimS1;
    int64_t totalS1{0};
    int64_t validT1{0};
    // attr
    uint32_t selectedBlockCount;
    bool enableOptimizedScatter{false};
    // gmoffset
    uint64_t queryGmOffset;
    uint64_t queryRopeGmOffset;
    uint64_t dyGmOffset;
    uint64_t indicesGmOffset;
    uint64_t sumGmOffset;
    uint64_t keyGmOffset;
    uint64_t keyRopeGmOffset;
    uint64_t valueGmOffset;
    uint64_t mm12GmOffset;
    uint64_t mm345GmOffset;
    uint64_t selectedKGmOffset;
    int32_t blkCntOffset{0};

    int32_t lastblkCntOffset;
    // workspace
    uint64_t selectedKWorkspaceLen;
    uint64_t mm12WorkspaceLen;
    uint64_t mm345WorkspaceLen;
    // Index
    int64_t bIndex{0};
    int64_t s1Index{0};
    int64_t n2Index{0};
    int64_t loopCnt{0};
    int64_t n2IndexScatter{0};
    int64_t bIndexScatterIndex{0};
    int64_t s1IndexScatterIndex{0};
    // 地址相关
    int64_t selectedKWspOffset{0};
    uint32_t scatterPingPongIdx{0};
    uint32_t mmPingPongIdx{0};
    uint32_t selectdKPPPidx{0};
    int64_t scatterTaskId{0};
    constexpr static const int32_t BLOCK_FP32 = 32 / sizeof(float);
    // selectBlock相关
    int32_t selectedCountOffset{0};
    int32_t actualSelectedBlockCount{0};
    int32_t selectedBlockSize{0};
    int32_t actualSelectedBlockCountScatter{0};
    // flag
    constexpr static uint32_t CUBE_WAIT_VEC_PING = 0;
    constexpr static uint32_t CUBE_WAIT_VEC_PONG = 1;
    constexpr static uint32_t VEC_WAIT_CUBE_PING = 2;
    constexpr static uint32_t VEC_WAIT_CUBE_PONG = 3;
    constexpr static uint32_t CUBE_WAIT_VEC_GATHER_PING = 4;
    constexpr static uint32_t CUBE_WAIT_VEC_GATHER_PONG = 5;
    constexpr static uint32_t SCATTER_SYNC_FLAG = 6;
    constexpr static uint32_t SCATTER_CUBE_SYNC_FLAG = 7;
    constexpr static uint32_t SCATTER_VECTOR_SYNC_FLAG = 8;
    bool isLastBlockSelected = false;
    bool isLastBlockSelectedScatter = false;

    event_t gatherMte2WaitMte3;
    event_t gatherMte2WaitMte3Pong;
    event_t scatterVWaitMte3;
    event_t scatterVWaitMte3Pong;
    event_t scatterTmpKMte2WaitV;
    event_t scatterTmpVMte2WaitV;
    event_t scatterTmpKMte2WaitVPong;
    event_t scatterTmpVMte2WaitVPong;

    RunInfo runInfo[2];
    RunInfo scatterRunInfo;
    RunInfo tmpScatterRunInfo[2];
    RunInfo tmpScatterRunInfoDet;
    bool isValidBegin{false};
    // gm
    GlobalTensor<int32_t> topkIndicesGm;
};

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::Init(const TILING_CLASS *__restrict tilingData)
{
    dimB = tilingData->opInfo.B;
    dimS1 = tilingData->opInfo.S1;
    dimG = tilingData->opInfo.G;
    dimN2 = tilingData->opInfo.N2;
    dimN1 = dimG * dimN2;
    dimDqk = tilingData->opInfo.D;
    dimDv = tilingData->opInfo.D2;
    dimRope = tilingData->opInfo.ropeD;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    enableOptimizedScatter = tilingData->opInfo.enableOptimizedScatter;
    mm12WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen / 2 / sizeof(float);
    mm345WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen / 2 / sizeof(T1);
    totalS1 = dimS1;
    if constexpr (IS_BSND == true) {
        curS1 = tilingData->opInfo.S1;
        curS2 = tilingData->opInfo.S2;
        curS1Scatter = curS1;
        curS2Scatter = curS2;
        totalS1 = dimB * dimS1;
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
    if (cubeBlockIdx < formerCoreNum) {
        processBS1ByCore = tilingData->opInfo.formerCoreProcessNNum;
    } else {
        processBS1ByCore = tilingData->opInfo.remainCoreProcessNNum;
    }

    selectedKWorkspaceLen = tilingData->opInfo.selectedKWorkspaceLen;
    selectedKWspOffset = selectedKWorkspaceLen / sizeof(T1) / 4;
    selectedCountOffset = PER_LOOP_BLOCK_SIZE / tilingData->opInfo.selectedBlockSize;
    if (tilingData->opInfo.selectedBlockSize * tilingData->opInfo.selectedBlockCount <= PER_LOOP_BLOCK_SIZE) {
        selectedCountOffset = tilingData->opInfo.selectedBlockCount;
    }
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::Process(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR query_rope,
    GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dq_rope, GM_ADDR dk_rope, GM_ADDR workspace,
    const TILING_CLASS *__restrict tilingData)
{
    Init(tilingData);
    topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topk_indices);
    if constexpr (IS_BSND == false) {
        validT1 = ((__gm__ int32_t *)actual_seq_qlen)[dimB - 1];
    }

    if constexpr (IS_DETERMINISTIC) {
        ProcessDeterministic(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum,
                             topk_indices, actual_seq_qlen, actual_seq_kvlen, query_rope, key_rope, dq, dk, dv, dq_rope,
                             dk_rope, workspace, tilingData);
    } else {
        ProcessNonDeterministic(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum,
                                topk_indices, actual_seq_qlen, actual_seq_kvlen, query_rope, key_rope, dq, dk, dv,
                                dq_rope, dk_rope, workspace, tilingData);
    }
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::ProcessNonDeterministic(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR query_rope,
    GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dq_rope, GM_ADDR dk_rope, GM_ADDR workspace,
    const TILING_CLASS *__restrict tilingData)
{
    if constexpr (IS_BSND == false) {
        totalS1 = validT1;
    }

    // AIC Process
    if ASCEND_IS_AIC {
        TPipe pipeCube;
        CubeOp<SFAGT> cubeOp;
        cubeOp.Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
                    actual_seq_qlen, actual_seq_kvlen, query_rope, key_rope, dq, dk, dv, workspace, tilingData,
                    &pipeCube);
        AllocEventID();
        int64_t task = 0;
        bool changeS1 = false;
        uint64_t scatterBufferNum = enableOptimizedScatter ? SCATTER_BUFFER_NUM : PING_PONG_BUFFER;
        for (int32_t i = 0; i < processBS1ByCore; i++) {
            scatterTaskId = i % scatterBufferNum;
            int32_t t1Index = cubeBlockIdx + usedCoreNum * i;
            if (!IS_BSND && t1Index >= validT1) {
                break;
            }
            GetTndSeqLen(actual_seq_qlen, actual_seq_kvlen, t1Index, bIndex);
            bool t1HasTask = false;
            for (n2Index = 0; n2Index < dimN2; n2Index++) {
                isLastBlockSelected = false;
                GetActualSelCount(t1Index, n2Index, actualSelectedBlockCount);
                t1HasTask = t1HasTask || actualSelectedBlockCount != 0;
                for (blkCntOffset = 0; blkCntOffset < actualSelectedBlockCount; blkCntOffset += selectedCountOffset) {
                    UpdateGmOffset(task, true);
                    CubeCompute(cubeOp);
                    if (!enableOptimizedScatter && changeS1) {
                        CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
                        changeS1 = false;
                    }
                    task++;
                }
            }
            if (!enableOptimizedScatter && t1HasTask) {
                changeS1 = true;
            }
        }

        if (cubeBlockIdx < usedCoreNum && task > 0) {
            int64_t taskMod = runInfo[1 - mmPingPongIdx].task & 1;
            CrossCoreWaitFlag<2, PIPE_MTE2>(taskMod == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
            runInfo[1 - mmPingPongIdx].noReload = true;
            cubeOp.cube345Process(runInfo[1 - mmPingPongIdx], lastblkCntOffset, 1 - mmPingPongIdx);
            runInfo[1 - mmPingPongIdx].noReload = false;
            CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
        }
        FreeEventID();
    }

    // AIV Process
    if ASCEND_IS_AIV {
        TPipe pipeVec;
        VecOp<SFAGT> vecOp;
        vecOp.Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
                   actual_seq_qlen, actual_seq_kvlen, key_rope, dq, dk, dv, workspace, tilingData, &pipeVec);
        SyncAll();
        tmpScatterRunInfo[0].changeS1 = false;
        tmpScatterRunInfo[1].changeS1 = false;
        scatterRunInfo.changeS1 = false;
        int64_t task = 0;

        gatherMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        gatherMte2WaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        scatterVWaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        scatterVWaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        scatterTmpKMte2WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        scatterTmpVMte2WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        scatterTmpKMte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        scatterTmpVMte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());

        SET_FLAG(MTE3, MTE2, gatherMte2WaitMte3);
        SET_FLAG(MTE3, MTE2, gatherMte2WaitMte3Pong);
        SET_FLAG(MTE3, V, scatterVWaitMte3);
        SET_FLAG(MTE3, V, scatterVWaitMte3Pong);
        SET_FLAG(V, MTE2, scatterTmpKMte2WaitV);
        SET_FLAG(V, MTE2, scatterTmpVMte2WaitV);
        SET_FLAG(V, MTE2, scatterTmpKMte2WaitVPong);
        SET_FLAG(V, MTE2, scatterTmpVMte2WaitVPong);

        uint64_t scatterBufferNum = enableOptimizedScatter ? SCATTER_BUFFER_NUM : PING_PONG_BUFFER;
        for (int32_t i = 0; i < processBS1ByCore; i++) {
            scatterTaskId = i % scatterBufferNum;
            int32_t t1Index = cubeBlockIdx + usedCoreNum * i;
            if (!IS_BSND && t1Index >= validT1) {
                break;
            }
            GetTndSeqLen(actual_seq_qlen, actual_seq_kvlen, t1Index, bIndex);
            bool t1HasTask = false;
            for (n2Index = 0; n2Index < dimN2; n2Index++) {
                isLastBlockSelected = false;
                GetActualSelCount(t1Index, n2Index, actualSelectedBlockCount);
                t1HasTask = t1HasTask || actualSelectedBlockCount != 0;
                if (unlikely(actualSelectedBlockCount == 0)) {
                    UpdateGmOffset(task, false);
                    vecOp.CleanDqRow(runInfo[mmPingPongIdx]);
                }
                for (blkCntOffset = 0; blkCntOffset < actualSelectedBlockCount; blkCntOffset += selectedCountOffset) {
                    UpdateGmOffset(task, true);
                    VecCompute(vecOp, actual_seq_qlen, actual_seq_kvlen);
                    if (enableOptimizedScatter) {
                        tmpScatterRunInfo[scatterPingPongIdx] = runInfo[1 - mmPingPongIdx];
                        tmpScatterRunInfo[scatterPingPongIdx].changeS1 = true;
                        scatterPingPongIdx = 1 - scatterPingPongIdx;
                    }
                    task++;
                }
            }
            if (!enableOptimizedScatter && t1HasTask) {
                tmpScatterRunInfo[0] = runInfo[1 - mmPingPongIdx];
                tmpScatterRunInfo[0].changeS1 = true;
            }
        }
        if (cubeBlockIdx < usedCoreNum && task > 0) {
            if (scatterRunInfo.changeS1) {
                CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                ScatterAddByS1(vecOp, actual_seq_qlen, actual_seq_kvlen);
                scatterRunInfo.changeS1 = false;
            }

            int64_t taskMod1 = runInfo[1 - mmPingPongIdx].task & 1;
            CrossCoreWaitFlag(taskMod1 == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
            vecOp.Process(runInfo[1 - mmPingPongIdx]);
            CrossCoreSetFlag<2, PIPE_MTE3>(taskMod1 == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);

            if (enableOptimizedScatter) {
                if (task > 1) {
                    CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                    scatterRunInfo = tmpScatterRunInfo[scatterPingPongIdx];
                    ScatterAddByS1(vecOp, actual_seq_qlen, actual_seq_kvlen);
                }

                CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                scatterRunInfo = runInfo[1 - mmPingPongIdx];
                ScatterAddByS1(vecOp, actual_seq_qlen, actual_seq_kvlen);
            } else {
                CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                scatterRunInfo = runInfo[1 - mmPingPongIdx];
                ScatterAddByS1(vecOp, actual_seq_qlen, actual_seq_kvlen);
            }
        }

        WAIT_FLAG(MTE3, MTE2, gatherMte2WaitMte3);
        WAIT_FLAG(MTE3, MTE2, gatherMte2WaitMte3Pong);
        WAIT_FLAG(MTE3, V, scatterVWaitMte3);
        WAIT_FLAG(MTE3, V, scatterVWaitMte3Pong);
        WAIT_FLAG(V, MTE2, scatterTmpKMte2WaitV);
        WAIT_FLAG(V, MTE2, scatterTmpVMte2WaitV);
        WAIT_FLAG(V, MTE2, scatterTmpKMte2WaitVPong);
        WAIT_FLAG(V, MTE2, scatterTmpVMte2WaitVPong);
        SyncAll();
        pipeVec.Destroy();

        TPipe pipeCast;
        if constexpr (KV_MERGE) {
            SparseFlashAttentionGradPost<T1, TILING_CLASS, false, 3, 0, HAS_ROPE> opCast;
            opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, dq_rope, dk_rope, workspace, tilingData,
                        &pipeCast);
            opCast.Process();
        } else {
            SparseFlashAttentionGradPost<T1, TILING_CLASS, true, 3, 0, HAS_ROPE> opCast;
            opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, dq_rope, dk_rope, workspace, tilingData,
                        &pipeCast);
            opCast.Process();
        }
    }
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::ProcessDeterministic(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR query_rope,
    GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dq_rope, GM_ADDR dk_rope, GM_ADDR workspace,
    const TILING_CLASS *__restrict tilingData)
{
    // AIC Process
    if ASCEND_IS_AIC {
        TPipe pipeCube;
        CubeOp<SFAGT> cubeOp;
        cubeOp.Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
                    actual_seq_qlen, actual_seq_kvlen, query_rope, key_rope, dq, dk, dv, workspace, tilingData,
                    &pipeCube);
        AllocEventID();
        int64_t task = 0;
        bool changeS1 = false;
        for (int32_t i = 0; i < processBS1ByCore; i++) {
            scatterTaskId = i % PING_PONG_BUFFER;
            int32_t t1Index = cubeBlockIdx + usedCoreNum * i;
            if (!IS_BSND && t1Index >= validT1) {
                break;
            }
            GetTndSeqLen(actual_seq_qlen, actual_seq_kvlen, t1Index, bIndex);
            for (n2Index = 0; n2Index < dimN2; n2Index++) {
                isLastBlockSelected = false;
                GetActualSelCount(t1Index, n2Index, actualSelectedBlockCount);
                for (blkCntOffset = 0; blkCntOffset < actualSelectedBlockCount; blkCntOffset += selectedCountOffset) {
                    UpdateGmOffset(task, true);
                    CubeCompute(cubeOp);
                    if (changeS1) {
                        CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                        CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                        CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
                        changeS1 = false;
                    }
                    task++;
                }
                if (unlikely(actualSelectedBlockCount == 0)) {
                    if (runInfo[1 - mmPingPongIdx].valid) {
                        int64_t taskMod1 = runInfo[1 - mmPingPongIdx].task & 1;
                        CrossCoreWaitFlag<2, PIPE_MTE2>(taskMod1 == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
                        runInfo[1 - mmPingPongIdx].noReload = true;
                        cubeOp.cube345Process(runInfo[1 - mmPingPongIdx], lastblkCntOffset, 1 - mmPingPongIdx);
                        runInfo[1 - mmPingPongIdx].noReload = false;
                        runInfo[1 - mmPingPongIdx].valid = false;
                        CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                        CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                        CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
                    }
                    CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                    CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                    CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
                    changeS1 = false;
                }
            }
            if (likely(actualSelectedBlockCount != 0)) {
                changeS1 = true;
            }
        }

        if (cubeBlockIdx < usedCoreNum && task > 0) {
            int64_t taskMod = runInfo[1 - mmPingPongIdx].task & 1;
            CrossCoreWaitFlag<2, PIPE_MTE2>(taskMod == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
            runInfo[1 - mmPingPongIdx].noReload = true;
            cubeOp.cube345Process(runInfo[1 - mmPingPongIdx], lastblkCntOffset, 1 - mmPingPongIdx);
            runInfo[1 - mmPingPongIdx].noReload = false;
            CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
        }

        if (processBS1ByCore < tilingData->opInfo.formerCoreProcessNNum) {
            CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
        }
        FreeEventID();
    }

    // AIV Process
    if ASCEND_IS_AIV {
        TPipe pipeVec;
        VecOp<SFAGT> vecOp;
        vecOp.Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
                   actual_seq_qlen, actual_seq_kvlen, key_rope, dq, dk, dv, workspace, tilingData, &pipeVec);
        SyncAll();
        bool isCruS1Empty = false;
        tmpScatterRunInfoDet.changeS1 = false;
        scatterRunInfo.changeS1 = false;
        int64_t task = 0;

        for (int32_t i = 0; i < processBS1ByCore; i++) {
            scatterTaskId = i % PING_PONG_BUFFER;
            int32_t t1Index = cubeBlockIdx + usedCoreNum * i;
            if (!IS_BSND && t1Index >= validT1) {
                break;
            }
            GetTndSeqLen(actual_seq_qlen, actual_seq_kvlen, t1Index, bIndex);
            for (n2Index = 0; n2Index < dimN2; n2Index++) {
                isLastBlockSelected = false;
                GetActualSelCount(t1Index, n2Index, actualSelectedBlockCount);
                for (blkCntOffset = 0; blkCntOffset < actualSelectedBlockCount; blkCntOffset += selectedCountOffset) {
                    isValidBegin = true;
                    UpdateGmOffset(task, true);
                    VecCompute(vecOp, actual_seq_qlen, actual_seq_kvlen);
                    task++;
                }
                if (unlikely(actualSelectedBlockCount == 0)) {
                    if (scatterRunInfo.changeS1) {
                        CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                        ScatterAddByS1(vecOp, actual_seq_qlen, actual_seq_kvlen);
                        scatterRunInfo.changeS1 = false;
                    }
                    if (runInfo[1 - mmPingPongIdx].valid) {
                        int64_t taskMod1 = runInfo[1 - mmPingPongIdx].task & 1;
                        CrossCoreWaitFlag(taskMod1 == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
                        vecOp.Process(runInfo[1 - mmPingPongIdx]);
                        runInfo[1 - mmPingPongIdx].valid = false;
                        CrossCoreSetFlag<2, PIPE_MTE3>(taskMod1 == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
                    }
                    if (tmpScatterRunInfoDet.changeS1) {
                        scatterRunInfo = tmpScatterRunInfoDet;
                        tmpScatterRunInfoDet.changeS1 = false;
                    }
                    UpdateGmOffset(task, false);
                    vecOp.CleanDqRow(runInfo[mmPingPongIdx]);

                    if (!isValidBegin) {
                        scatterRunInfo = runInfo[mmPingPongIdx];
                        CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                        ScatterAddByS1(vecOp, actual_seq_qlen, actual_seq_kvlen);
                    } else {
                        tmpScatterRunInfoDet = runInfo[mmPingPongIdx];
                        isCruS1Empty = true;
                    }
                }
            }
            tmpScatterRunInfoDet = isCruS1Empty ? tmpScatterRunInfoDet : runInfo[1 - mmPingPongIdx];
            tmpScatterRunInfoDet.changeS1 = true;
            isCruS1Empty = false;
        }
        if (cubeBlockIdx < usedCoreNum && task > 0) {
            if (scatterRunInfo.changeS1) {
                CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                ScatterAddByS1(vecOp, actual_seq_qlen, actual_seq_kvlen);
                scatterRunInfo.changeS1 = false;
            }

            int64_t taskMod1 = runInfo[1 - mmPingPongIdx].task & 1;
            CrossCoreWaitFlag(taskMod1 == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
            vecOp.Process(runInfo[1 - mmPingPongIdx]);
            CrossCoreSetFlag<2, PIPE_MTE3>(taskMod1 == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);

            CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
            scatterRunInfo = runInfo[1 - mmPingPongIdx];
            ScatterAddByS1(vecOp, actual_seq_qlen, actual_seq_kvlen);
        }

        if (processBS1ByCore < tilingData->opInfo.formerCoreProcessNNum && cubeBlockIdx < usedCoreNum) {
            CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
            int64_t scatterSyncPhaseBegin = (task > 0 ? runInfo[1 - mmPingPongIdx].s1End : totalS1) * dimN2;
            int64_t scatterSyncPhaseEnd = totalS1 * dimN2;
            for (int64_t i = scatterSyncPhaseBegin; i < scatterSyncPhaseEnd; i++) {
                CrossCoreSetFlag<0, PIPE_MTE3>(SCATTER_VECTOR_SYNC_FLAG);
                CrossCoreWaitFlag<0, PIPE_MTE3>(SCATTER_VECTOR_SYNC_FLAG);
            }
        }
        if (cubeBlockIdx >= usedCoreNum) {
            CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
            for (int64_t i = 0; i < totalS1 * dimN2; i++) {
                CrossCoreSetFlag<0, PIPE_MTE3>(SCATTER_VECTOR_SYNC_FLAG);
                CrossCoreWaitFlag<0, PIPE_MTE3>(SCATTER_VECTOR_SYNC_FLAG);
            }
        }
        SyncAll();
        pipeVec.Destroy();

        TPipe pipeCast;
        if constexpr (KV_MERGE) {
            SparseFlashAttentionGradPost<T1, TILING_CLASS, false, 3, 0, HAS_ROPE> opCast;
            opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, dq_rope, dk_rope, workspace, tilingData,
                        &pipeCast);
            opCast.Process();
        } else {
            SparseFlashAttentionGradPost<T1, TILING_CLASS, true, 3, 0, HAS_ROPE> opCast;
            opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, dq_rope, dk_rope, workspace, tilingData,
                        &pipeCast);
            opCast.Process();
        }
    }
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::CubeCompute(CubeOp<SFAGT> &cubeOp)
{
    int64_t taskMod = runInfo[mmPingPongIdx].task & 1;
    // WaitVec for select & gather
    CrossCoreWaitFlag<2, PIPE_MTE2>(taskMod == 0 ? CUBE_WAIT_VEC_GATHER_PING : CUBE_WAIT_VEC_GATHER_PONG);
    // First Block Load query & dy
    if (blkCntOffset == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS);
        WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_DY_EVENTS);
        cubeOp.preloadQueryAndDy(runInfo[mmPingPongIdx]);
    }

    cubeOp.cube12Process(runInfo[mmPingPongIdx], blkCntOffset, mmPingPongIdx);
    CrossCoreSetFlag<2, PIPE_FIX>(taskMod == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);

    if (likely(loopCnt > 0 && runInfo[1 - mmPingPongIdx].valid)) {
        int64_t taskMod1 = runInfo[1 - mmPingPongIdx].task & 1;
        CrossCoreWaitFlag<2, PIPE_MTE2>(taskMod1 == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
        cubeOp.cube345Process(runInfo[1 - mmPingPongIdx], lastblkCntOffset, 1 - mmPingPongIdx);
        if constexpr (!IS_DETERMINISTIC) {
            if (enableOptimizedScatter) {
                CrossCoreSetFlag<2, PIPE_FIX>(SCATTER_SYNC_FLAG);
            }
        }
        runInfo[1 - mmPingPongIdx].valid = false;
    }
    // Backward Sync, s1 block loop end, reload query & dy
    if (blkCntOffset + selectedCountOffset >= actualSelectedBlockCount) {
        SetFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS);
        SetFlag<HardEvent::MTE1_MTE2>(MM_L1_DY_EVENTS);
    }
    SaveLastInfo();
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::ScatterAddByS1(VecOp<SFAGT> &vecOp,
                                                                         const GM_ADDR actual_seq_qlen,
                                                                         const GM_ADDR actual_seq_kvlen)
{
    if (cubeBlockIdx >= usedCoreNum) {
        return;
    }
    if constexpr (IS_DETERMINISTIC) {
        for (int32_t t1ScatterIndex = scatterRunInfo.s1Begin; t1ScatterIndex < scatterRunInfo.s1End; t1ScatterIndex++) {
            GetTndSeqLenScatter(actual_seq_qlen, actual_seq_kvlen, t1ScatterIndex, bIndexScatterIndex);
            for (n2IndexScatter = 0; n2IndexScatter < dimN2; n2IndexScatter++) {
                GetActualSelCountScatter(t1ScatterIndex, n2IndexScatter, actualSelectedBlockCountScatter);
                UpdateScatterInfo(t1ScatterIndex);

                if (likely(actualSelectedBlockCountScatter != 0)) {
                    vecOp.ScatterAdd(scatterRunInfo);
                }
                CrossCoreSetFlag<0, PIPE_MTE3>(SCATTER_VECTOR_SYNC_FLAG);
                CrossCoreWaitFlag<0, PIPE_MTE3>(SCATTER_VECTOR_SYNC_FLAG);
            }
        }
    } else {
        if (!enableOptimizedScatter) {
            vecOp.ScatterAdd(scatterRunInfo);
            return;
        }
        int64_t t1ScatterIndex = scatterRunInfo.s1Begin + cubeBlockIdx;
        if (t1ScatterIndex >= scatterRunInfo.s1End) {
            return;
        }
        int64_t savedScatterTaskId = scatterRunInfo.scatterTaskId;
        int64_t savedBlkCntOffset = scatterRunInfo.blkCntOffset;
        int64_t savedCurBlkCntOffset = blkCntOffset;
        int64_t savedN2IndexScatter = n2IndexScatter;
        int32_t savedActualSelectedBlockCountScatter = actualSelectedBlockCountScatter;
        bool savedIsLastBlockSelectedScatter = isLastBlockSelectedScatter;
        event_t savedScatterVWaitMte3 = scatterRunInfo.scatterVWaitMte3;
        event_t savedScatterVWaitMte3Pong = scatterRunInfo.scatterVWaitMte3Pong;
        event_t savedScatterTmpKMte2WaitV = scatterRunInfo.scatterTmpKMte2WaitV;
        event_t savedScatterTmpVMte2WaitV = scatterRunInfo.scatterTmpVMte2WaitV;
        event_t savedScatterTmpKMte2WaitVPong = scatterRunInfo.scatterTmpKMte2WaitVPong;
        event_t savedScatterTmpVMte2WaitVPong = scatterRunInfo.scatterTmpVMte2WaitVPong;

        GetTndSeqLenScatter(actual_seq_qlen, actual_seq_kvlen, t1ScatterIndex, bIndexScatterIndex);
        n2IndexScatter = (scatterRunInfo.indicesGmOffset / selectedBlockCount) % dimN2;
        isLastBlockSelectedScatter = false;
        GetActualSelCountScatter(t1ScatterIndex, n2IndexScatter, actualSelectedBlockCountScatter);
        blkCntOffset = savedBlkCntOffset;
        UpdateScatterInfo(t1ScatterIndex);
        scatterRunInfo.isLastBasicBlock = (savedBlkCntOffset + selectedCountOffset >= actualSelectedBlockCountScatter);
        scatterRunInfo.scatterTaskId = savedScatterTaskId;
        scatterRunInfo.scatterVWaitMte3 = savedScatterVWaitMte3;
        scatterRunInfo.scatterVWaitMte3Pong = savedScatterVWaitMte3Pong;
        scatterRunInfo.scatterTmpKMte2WaitV = savedScatterTmpKMte2WaitV;
        scatterRunInfo.scatterTmpVMte2WaitV = savedScatterTmpVMte2WaitV;
        scatterRunInfo.scatterTmpKMte2WaitVPong = savedScatterTmpKMte2WaitVPong;
        scatterRunInfo.scatterTmpVMte2WaitVPong = savedScatterTmpVMte2WaitVPong;
        blkCntOffset = savedCurBlkCntOffset;
        n2IndexScatter = savedN2IndexScatter;
        actualSelectedBlockCountScatter = savedActualSelectedBlockCountScatter;
        isLastBlockSelectedScatter = savedIsLastBlockSelectedScatter;
        vecOp.ScatterAddUnDeter(scatterRunInfo);
    }
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::VecCompute(VecOp<SFAGT> &vecOp, const GM_ADDR actual_seq_qlen,
                                                                     const GM_ADDR actual_seq_kvlen)
{
    int64_t taskMod = runInfo[mmPingPongIdx].task & 1;
    if (cubeBlockIdx < usedCoreNum) {
        vecOp.GatherKV(n2Index, t1Offset, runInfo[mmPingPongIdx]);
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(taskMod == 0 ? CUBE_WAIT_VEC_GATHER_PING : CUBE_WAIT_VEC_GATHER_PONG);

    if constexpr (IS_DETERMINISTIC) {
        if (scatterRunInfo.changeS1) {
            CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
            ScatterAddByS1(vecOp, actual_seq_qlen, actual_seq_kvlen);
            scatterRunInfo.changeS1 = false;
        }

        if (runInfo[mmPingPongIdx].task > 0) {
            if (runInfo[1 - mmPingPongIdx].valid) {
                int64_t taskMod1 = runInfo[1 - mmPingPongIdx].task & 1;
                CrossCoreWaitFlag(taskMod1 == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
                vecOp.Process(runInfo[1 - mmPingPongIdx]);
                runInfo[1 - mmPingPongIdx].valid = false;
                CrossCoreSetFlag<2, PIPE_MTE3>(taskMod1 == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
            }

            if (tmpScatterRunInfoDet.changeS1) {
                scatterRunInfo = tmpScatterRunInfoDet;
                tmpScatterRunInfoDet.changeS1 = false;
            }
        }
    } else {
        if (runInfo[mmPingPongIdx].task > 0) {
            if (runInfo[1 - mmPingPongIdx].valid) {
                int64_t taskMod1 = runInfo[1 - mmPingPongIdx].task & 1;
                CrossCoreWaitFlag(taskMod1 == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
                vecOp.Process(runInfo[1 - mmPingPongIdx]);
                runInfo[1 - mmPingPongIdx].valid = false;
                CrossCoreSetFlag<2, PIPE_MTE3>(taskMod1 == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
            }

            if (!enableOptimizedScatter && tmpScatterRunInfo[0].changeS1) {
                scatterRunInfo = tmpScatterRunInfo[0];
                tmpScatterRunInfo[0].changeS1 = false;
            }

            if (scatterRunInfo.changeS1) {
                CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                ScatterAddByS1(vecOp, actual_seq_qlen, actual_seq_kvlen);
                scatterRunInfo.changeS1 = false;
            }

            if (enableOptimizedScatter && runInfo[mmPingPongIdx].task > 1 &&
                tmpScatterRunInfo[scatterPingPongIdx].changeS1) {
                scatterRunInfo = tmpScatterRunInfo[scatterPingPongIdx];
                tmpScatterRunInfo[scatterPingPongIdx].changeS1 = false;
            }
        }
    }

    mmPingPongIdx = 1 - mmPingPongIdx;
    selectdKPPPidx = (selectdKPPPidx + 1) % 4;
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::UpdateScatterInfo(int64_t t1ScatterIndex)
{
    /*
     *  query:    B S1 N2 G D
     *  dy/out:   B S1 N2 G D2
     *  indices:  B S1 N2 SELCTED_BLOCK_COUNT
     *  sum/max   B S1 N2 G 8 --> N2 T1 G or B N2 S1 G
     *  key:      B S2 N2 D
     *  value:    B S2 N2 D2
     */
    if constexpr (IS_BSND) {
        scatterRunInfo.indicesGmOffset =
            t1OffsetScatterIndex * (dimN2 * selectedBlockCount) + n2IndexScatter * selectedBlockCount;
    } else {
        scatterRunInfo.indicesGmOffset = t1OffsetScatterIndex * (dimN2 * selectedBlockCount) +
                                         s1IndexScatterIndex * (dimN2 * selectedBlockCount) +
                                         n2IndexScatter * selectedBlockCount;
    }

    curMaxS2Scatter = ATTEN_ENABLE ? (s1IndexScatterIndex + curS2Scatter - curS1Scatter + 1) : curS2Scatter;

    keyGmOffset = t2OffsetScatterIndex * (dimN2 * dimDqk) + n2IndexScatter * dimDqk;
    keyRopeGmOffset = t2OffsetScatterIndex * (dimN2 * dimRope) + n2IndexScatter * dimRope;
    valueGmOffset = t2OffsetScatterIndex * (dimN2 * dimDv) + n2IndexScatter * dimDv;
    scatterRunInfo.blkCntOffset = blkCntOffset;
    scatterRunInfo.keyGmOffset = keyGmOffset;
    scatterRunInfo.keyRopeGmOffset = keyRopeGmOffset;
    scatterRunInfo.valueGmOffset = valueGmOffset;
    scatterRunInfo.mm4OutGmOffset = keyGmOffset + keyRopeGmOffset;
    scatterRunInfo.mm5OutGmOffset = valueGmOffset;
    scatterRunInfo.actualSelCntOffset = blkCntOffset + selectedCountOffset <= actualSelectedBlockCountScatter ?
                                            selectedCountOffset :
                                            actualSelectedBlockCountScatter - blkCntOffset;
    scatterRunInfo.lastBlockSize = isLastBlockSelectedScatter && curMaxS2Scatter % selectedBlockSize != 0 ?
                                       curMaxS2Scatter % selectedBlockSize :
                                       selectedBlockSize;

    // scatterRunInfo.scatterTaskId = scatterTaskId;
    scatterRunInfo.s1Index = t1ScatterIndex;
    scatterRunInfo.curS2 = curS2Scatter;
    scatterRunInfo.actualSelectedBlockCount = actualSelectedBlockCountScatter;
    scatterRunInfo.isSmallS2 = (curMaxS2Scatter <= actualSelectedBlockCountScatter * selectedBlockSize);
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::UpdateGmOffset(int64_t task, bool validflag)
{
    /*
     *  query:    B S1 N2 G D
     *  dy/out:   B S1 N2 G D2
     *  indices:  B S1 N2 SELCTED_BLOCK_COUNT
     *  sum/max   B S1 N2 G 8 --> N2 T1 G or B N2 S1 G
     *  key:      B S2 N2 D
     *  value:    B S2 N2 D2
     */
    runInfo[mmPingPongIdx].valid = validflag;
    if constexpr (IS_BSND) {
        queryGmOffset = t1Offset * (dimN1 * dimDqk) + n2Index * (dimG * dimDqk);
        queryRopeGmOffset = t1Offset * (dimN1 * dimRope) + n2Index * (dimG * dimRope);
        dyGmOffset = t1Offset * (dimN1 * dimDv) + n2Index * (dimG * dimDv);
        indicesGmOffset = t1Offset * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount;
        sumGmOffset = bIndex * dimN2 * dimS1 * dimG + n2Index * dimS1 * dimG + s1Index * dimG;
        runInfo[mmPingPongIdx].s1Begin = t1Offset / usedCoreNum * usedCoreNum;
        runInfo[mmPingPongIdx].s1End = (t1Offset / usedCoreNum + 1) * usedCoreNum;
    } else {
        queryGmOffset = t1Offset * (dimN1 * dimDqk) + s1Index * (dimN1 * dimDqk) + n2Index * (dimG * dimDqk);
        queryRopeGmOffset = t1Offset * (dimN1 * dimRope) + s1Index * (dimN1 * dimRope) + n2Index * (dimG * dimRope);
        dyGmOffset = t1Offset * (dimN1 * dimDv) + s1Index * (dimN1 * dimDv) + n2Index * (dimG * dimDv);
        indicesGmOffset = t1Offset * (dimN2 * selectedBlockCount) + s1Index * (dimN2 * selectedBlockCount) +
                          n2Index * selectedBlockCount;
        sumGmOffset = n2Index * dimS1 * dimG + (t1Offset + s1Index) * dimG;
        runInfo[mmPingPongIdx].s1Begin = (t1Offset + s1Index) / usedCoreNum * usedCoreNum;
        runInfo[mmPingPongIdx].s1End = ((t1Offset + s1Index) / usedCoreNum + 1) * usedCoreNum;
    }
    keyGmOffset = t2Offset * (dimN2 * dimDqk) + n2Index * dimDqk;
    keyRopeGmOffset = t2Offset * (dimN2 * dimRope) + n2Index * dimRope;
    valueGmOffset = t2Offset * (dimN2 * dimDv) + n2Index * dimDv;
    mm12GmOffset = mmPingPongIdx * mm12WorkspaceLen;
    mm345GmOffset = mmPingPongIdx * mm345WorkspaceLen;
    selectedKGmOffset = selectdKPPPidx * selectedKWspOffset;

    curMaxS2 = ATTEN_ENABLE ? (s1Index + curS2 - curS1 + 1) : curS2;

    runInfo[mmPingPongIdx].task = task;
    runInfo[mmPingPongIdx].sumGmOffset = sumGmOffset;
    runInfo[mmPingPongIdx].blkCntOffset = blkCntOffset;
    runInfo[mmPingPongIdx].queryGmOffset = queryGmOffset;
    runInfo[mmPingPongIdx].queryRopeGmOffset = queryRopeGmOffset;
    runInfo[mmPingPongIdx].keyGmOffset = keyGmOffset;
    runInfo[mmPingPongIdx].keyRopeGmOffset = keyRopeGmOffset;
    runInfo[mmPingPongIdx].dyGmOffset = dyGmOffset;
    runInfo[mmPingPongIdx].valueGmOffset = valueGmOffset;
    runInfo[mmPingPongIdx].indicesGmOffset = indicesGmOffset;
    runInfo[mmPingPongIdx].mm12GmOffset = mm12GmOffset;
    runInfo[mmPingPongIdx].mm345GmOffset = mm345GmOffset;
    runInfo[mmPingPongIdx].mm3OutGmOffset = queryGmOffset + queryRopeGmOffset;
    runInfo[mmPingPongIdx].mm4OutGmOffset = keyGmOffset + keyRopeGmOffset;
    runInfo[mmPingPongIdx].mm5OutGmOffset = valueGmOffset;
    runInfo[mmPingPongIdx].actualSelCntOffset = blkCntOffset + selectedCountOffset <= actualSelectedBlockCount ?
                                                    selectedCountOffset :
                                                    actualSelectedBlockCount - blkCntOffset;
    runInfo[mmPingPongIdx].lastBlockSize =
        isLastBlockSelected && curMaxS2 % selectedBlockSize != 0 ? curMaxS2 % selectedBlockSize : selectedBlockSize;
    runInfo[mmPingPongIdx].isLastBasicBlock = (blkCntOffset + selectedCountOffset >= actualSelectedBlockCount);
    runInfo[mmPingPongIdx].scatterTaskId = scatterTaskId;
    runInfo[mmPingPongIdx].s1Index = s1Index;
    runInfo[mmPingPongIdx].gatherMte2WaitMte3 = gatherMte2WaitMte3;
    runInfo[mmPingPongIdx].gatherMte2WaitMte3Pong = gatherMte2WaitMte3Pong;
    runInfo[mmPingPongIdx].scatterVWaitMte3 = scatterVWaitMte3;
    runInfo[mmPingPongIdx].scatterVWaitMte3Pong = scatterVWaitMte3Pong;
    runInfo[mmPingPongIdx].scatterTmpKMte2WaitV = scatterTmpKMte2WaitV;
    runInfo[mmPingPongIdx].scatterTmpVMte2WaitV = scatterTmpVMte2WaitV;
    runInfo[mmPingPongIdx].scatterTmpKMte2WaitVPong = scatterTmpKMte2WaitVPong;
    runInfo[mmPingPongIdx].scatterTmpVMte2WaitVPong = scatterTmpVMte2WaitVPong;

    if (runInfo[mmPingPongIdx].s1End > totalS1) {
        runInfo[mmPingPongIdx].s1End = totalS1;
    }

    runInfo[mmPingPongIdx].actualSelectedBlockCount = actualSelectedBlockCount;
    runInfo[mmPingPongIdx].curS1 = curS1;
    runInfo[mmPingPongIdx].curS2 = curS2;
    runInfo[mmPingPongIdx].selectedKGmOffset = selectedKGmOffset;
    runInfo[mmPingPongIdx].isSmallS2 = (curMaxS2 <= actualSelectedBlockCount * selectedBlockSize);
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::SaveLastInfo()
{
    lastblkCntOffset = blkCntOffset;
    mmPingPongIdx = 1 - mmPingPongIdx;
    selectdKPPPidx = (selectdKPPPidx + 1) % 4;
    loopCnt++;
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::GetTndSeqLenScatter(const GM_ADDR actual_seq_qlen_addr,
                                                                              const GM_ADDR actual_seq_kvlen_addr,
                                                                              const int64_t t1Idx, int64_t &bIdx)
{
    if constexpr (IS_BSND == false) {
        int64_t curT1 = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndexScatterIndex];
        while (t1Idx >= curT1) {
            curT1 = ((__gm__ int32_t *)actual_seq_qlen_addr)[++bIndexScatterIndex];
        }

        if (unlikely(bIndexScatterIndex == 0)) {
            t1OffsetScatterIndex = 0;
            t2OffsetScatterIndex = 0;
            curS1Scatter = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndexScatterIndex];
            curS2Scatter = ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIndexScatterIndex];
        } else {
            t1OffsetScatterIndex = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndexScatterIndex - 1];
            t2OffsetScatterIndex = ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIndexScatterIndex - 1];
            curS1Scatter = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndexScatterIndex] -
                           ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndexScatterIndex - 1];
            curS2Scatter = ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIndexScatterIndex] -
                           ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIndexScatterIndex - 1];
        }

        s1IndexScatterIndex = t1Idx - t1OffsetScatterIndex;
    } else {
        t1OffsetScatterIndex = t1Idx;
        bIdx = t1Idx / curS1Scatter;
        s1IndexScatterIndex = t1Idx % dimS1;
        t2OffsetScatterIndex = bIdx * curS2Scatter;
    }
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::GetTndSeqLen(const GM_ADDR actual_seq_qlen_addr,
                                                                       const GM_ADDR actual_seq_kvlen_addr,
                                                                       const int64_t t1Idx, int64_t &bIdx)
{
    if constexpr (IS_BSND == false) {
        int64_t curT1 = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex];
        while (t1Idx >= curT1) {
            curT1 = ((__gm__ int32_t *)actual_seq_qlen_addr)[++bIndex];
        }

        if (unlikely(bIndex == 0)) {
            t1Offset = 0;
            t2Offset = 0;
            curS1 = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex];
            curS2 = ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIndex];
        } else {
            t1Offset = ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex - 1];
            t2Offset = ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIndex - 1];
            curS1 =
                ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex] - ((__gm__ int32_t *)actual_seq_qlen_addr)[bIndex - 1];
            curS2 = ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIndex] -
                    ((__gm__ int32_t *)actual_seq_kvlen_addr)[bIndex - 1];
        }

        s1Index = t1Idx - t1Offset;
    } else {
        t1Offset = t1Idx;
        bIdx = t1Idx / curS1;
        s1Index = t1Idx % dimS1;
        t2Offset = bIdx * curS2;
    }
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::GetActualSelCountScatter(const int64_t t1Idx,
                                                                                   const int64_t n2Idx,
                                                                                   int32_t &actSelBlkCount)
{
    isLastBlockSelectedScatter = false;
    int64_t maxS2Blk = (curS2Scatter + selectedBlockSize - 1) / selectedBlockSize;
    if constexpr (ATTEN_ENABLE) {
        int64_t newMaxS2 = Max(curS2Scatter - curS1Scatter + s1IndexScatterIndex + 1, 0);
        maxS2Blk = (newMaxS2 + selectedBlockSize - 1) / selectedBlockSize;
    }
    actualSelectedBlockCountScatter = Min(selectedBlockCount, maxS2Blk);
    if (actualSelectedBlockCountScatter <= 0) {
        return;
    }
    int64_t topkGmOffset =
        t1Idx * (dimN2 * selectedBlockCount) + n2Idx * selectedBlockCount + actualSelectedBlockCountScatter - 1;
    if (topkIndicesGm[topkGmOffset].GetValue(0) == maxS2Blk - 1) {
        isLastBlockSelectedScatter = true;
    }
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradBasic<SFAGT>::GetActualSelCount(const int64_t t1Idx, const int64_t n2Idx,
                                                                            int32_t &actSelBlkCount)
{
    int64_t maxS2Blk = (curS2 + selectedBlockSize - 1) / selectedBlockSize;
    if constexpr (ATTEN_ENABLE) {
        int64_t newMaxS2 = Max(curS2 - curS1 + s1Index + 1, 0);
        maxS2Blk = (newMaxS2 + selectedBlockSize - 1) / selectedBlockSize;
    }
    actualSelectedBlockCount = Min(selectedBlockCount, maxS2Blk);
    if (actualSelectedBlockCount <= 0) {
        return;
    }
    int64_t topkGmOffset =
        t1Idx * (dimN2 * selectedBlockCount) + n2Idx * selectedBlockCount + actualSelectedBlockCount - 1;
    if (topkIndicesGm[topkGmOffset].GetValue(0) == maxS2Blk - 1) {
        isLastBlockSelected = true;
    }
}

} // namespace SFAG_BASIC
