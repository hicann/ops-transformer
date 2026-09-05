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
 * \file generic_block_sparse_attention_grad.h
 * \brief Arch35 Process / CubeProcess / VectorProcess for GenericBlockSparseAttentionGrad

 *        (design §3.5: Pre once → Preload[Gather+MM12](N) || NotFirst[Vec+MM345](N-1) → Post).
 */
#pragma once
#include "common_header.h"
#include "addr_compute.h"
#include "cube_op.h"
#include "vec_op.h"
using namespace AscendC;

namespace GSAG_ARC35 {

template <typename INPUT_TYPE, uint32_t INPUT_LAYOUT, class TILING_CLASS, bool DETERMINISTIC_ENABLE>
struct GSAG_TYPE {
    using input_type = INPUT_TYPE;
    static constexpr uint32_t input_layout = INPUT_LAYOUT;
    using tiling_class = TILING_CLASS;
    static constexpr bool deterministic_enable = DETERMINISTIC_ENABLE;
};

template <typename GSAG_TYPE>
class GenericBlockSparseAttentionGradArch35 {
    using INPUT_TYPE = typename GSAG_TYPE::input_type;
    static constexpr uint32_t INPUT_LAYOUT = GSAG_TYPE::input_layout;
    using TILING_CLASS = typename GSAG_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = GSAG_TYPE::deterministic_enable;

private:
    RunTimeInfo runTimeInfo_[2];
    AddrComputeModule<GSAG_TYPE> addr_;
    GlobalTensor<INPUT_TYPE> query_gm_, key_gm_, val_gm_, dout_gm_, attention_out_gm_;
    GlobalTensor<INPUT_TYPE> dq_gm_, dk_gm_, dv_gm_;
    GlobalTensor<float> dq_workspace_, dk_workspace_, dv_workspace_, sftg_workspace_;
    TBuf<TPosition::VECCALC> ub_buffer_;
    TBuf<TPosition::A1> l1_buffer_;
    LocalTensor<float> mm1_res_ub_tensor_ping_, mm1_res_ub_tensor_pong_;
    LocalTensor<float> mm2_res_ub_tensor_ping_, mm2_res_ub_tensor_pong_;
    LocalTensor<float> mm1_res_ub_tensor_, mm2_res_ub_tensor_;
    // mm1/mm2 ping-pong slots are exclusive to Softmax/Cube Fixpipe.
    // Gather ND/NZ uses one shared staging buffer (Preload is serial per task).
    LocalTensor<INPUT_TYPE> gather_q_nd_ub_, gather_dout_nd_ub_;
    LocalTensor<INPUT_TYPE> query_nz_ub_tensor_, dout_nz_ub_tensor_;
    LocalTensor<INPUT_TYPE> p_l1_tensor_ping_, p_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> ds_l1_tensor_ping_, ds_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> p_l1_tensor_, ds_l1_tensor_;
    // Filled by AIV0/AIV1 before AIC starts MM12.
    LocalTensor<INPUT_TYPE> query_l1_tensor_ping_, query_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> dout_l1_tensor_ping_, dout_l1_tensor_pong_;
    GlobalTensor<int32_t> sparse_idx_gm_;
    int32_t q_head_num_{0};
    int32_t head_dim_{0};
    uint32_t vec_ub_matrix_elements_{0};
    uint32_t vec_base_m_{0};
    uint32_t taskId = 0;
    uint32_t ping_pong_idx = 0;
    uint32_t last_ping_pong_idx = 0;
    uint32_t ub_offset_ = 0;
    uint32_t l1_offset_ = 0;
    static constexpr int32_t UB_SIZE = 247 * 1024;
    static constexpr int32_t L1_SIZE = 512 * 1024;

    // AIV0/AIV1 row partition: must match Softmax dualDst (mAlign/2) and CopyQAndDoutToL1.
    // Always split by mAlign/2 (including m < 16): AIV1 may own an all-pad half and still
    // must zero-fill L1 + SetFlag gather so Cube sync never deadlocks.
    __aicore__ inline int32_t GatherAivRowCount(int32_t m, int32_t mAlign, int32_t aivHalfIdx) const
    {
        const int32_t halfAlign = mAlign / 2;
        const int32_t rowBegin = aivHalfIdx * halfAlign;
        int32_t rowCount = m - rowBegin;
        if (rowCount < 0) {
            rowCount = 0;
        }
        if (rowCount > halfAlign) {
            rowCount = halfAlign;
        }
        return rowCount;
    }

    __aicore__ inline bool GatherNeedAiv1(int32_t m, int32_t mAlign) const
    {
        (void)m;
        (void)mAlign;
        // Both AIVs always participate in gather SetFlag (pad half writes zeros).
        return true;
    }

    /** ProcessPreload: Gather(curr) on AIV — Q/dO L1 for Cube MM12. */
    __aicore__ inline void ProcessPreloadVec(VecOp<GSAG_TYPE> &vecOp, uint32_t currIdx, uint32_t loopTaskId)
    {
        (void)vecOp;
        const RunTimeInfo &curr = runTimeInfo_[currIdx];
        if (!curr.need_compute || !curr.use_sparse_gather) {
            return;
        }
        const int32_t m = static_cast<int32_t>(curr.s1Len);
        const int32_t mAlign = static_cast<int32_t>(curr.s1LenAlign);
        const int32_t aivHalfIdx = static_cast<int32_t>(GetSubBlockIdx());
        if (loopTaskId > 0) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE3>(FLAG_C_GATHER_ARM);
        }
        CopyQAndDoutToL1(curr, currIdx);
        // Always signal — including AIV with 0 real rows (zero-filled pad half).
        const uint32_t gatherFlagBase = currIdx == 0 ? FLAG_V_GATHER_C_PING : FLAG_V_GATHER_C_PONG;
        const uint32_t gatherFlag = gatherFlagBase + static_cast<uint32_t>(aivHalfIdx) * AIV0_AIV1_FLAG_OFFSET;
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(gatherFlag);
        (void)m;
        (void)mAlign;
    }

    /** ProcessNotFirst: Softmax(last) on AIV — overlaps Cube MM12(curr). */
    __aicore__ inline void ProcessNotFirstVec(VecOp<GSAG_TYPE> &vecOp, uint32_t lastIdx)
    {
        const RunTimeInfo &last = runTimeInfo_[lastIdx];
        if (!last.need_compute) {
            return;
        }
        LocalTensor<float> mm1Last = lastIdx ? mm1_res_ub_tensor_ping_ : mm1_res_ub_tensor_pong_;
        LocalTensor<float> mm2Last = lastIdx ? mm2_res_ub_tensor_ping_ : mm2_res_ub_tensor_pong_;
        LocalTensor<INPUT_TYPE> pL1Last = lastIdx ? p_l1_tensor_ping_ : p_l1_tensor_pong_;
        LocalTensor<INPUT_TYPE> dsL1Last = lastIdx ? ds_l1_tensor_ping_ : ds_l1_tensor_pong_;

        vecOp.SendVecSftPreProcess(last, lastIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_V>(lastIdx == 0 ? FLAG_C1_V1_PING : FLAG_C1_V1_PONG);
        PipeBarrier<PIPE_ALL>();
        vecOp.SendVecSoftmax(pL1Last, mm1Last, last);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE3>(FLAG_V1_C3);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_V>(lastIdx == 0 ? FLAG_C2_V2_PING : FLAG_C2_V2_PONG);
        vecOp.SendVecSoftmaxGrad(dsL1Last, mm1Last, mm2Last, last);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE3>(FLAG_V2_C45);
    }

    __aicore__ inline void ScatterLastVec(VecOp<GSAG_TYPE> &vecOp, uint32_t lastIdx, TBuf<TPosition::VECCALC> &ubBuffer)
    {
        const RunTimeInfo &last = runTimeInfo_[lastIdx];
        if (!last.need_compute || !last.use_sparse_gather) {
            return;
        }
        const uint32_t mm345DoneFlag =
            FLAG_C_MM345_DONE + static_cast<uint32_t>(GetSubBlockIdx()) * AIV0_AIV1_FLAG_OFFSET;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(mm345DoneFlag);
        vecOp.ScatterDqSel(last, ubBuffer, lastIdx);
    }

    /** ProcessPreload: MM12(curr) on AIC — overlaps Vec Gather/Softmax. */
    __aicore__ inline void ProcessPreloadCube(CubeOp<GSAG_TYPE> &cubeOp, GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                              GM_ADDR dout, uint32_t currIdx, uint32_t loopTaskId)
    {
        const RunTimeInfo &curr = runTimeInfo_[currIdx];
        if (!curr.need_compute) {
            return;
        }
        mm1_res_ub_tensor_ = currIdx ? mm1_res_ub_tensor_ping_ : mm1_res_ub_tensor_pong_;
        mm2_res_ub_tensor_ = currIdx ? mm2_res_ub_tensor_ping_ : mm2_res_ub_tensor_pong_;

        if (curr.use_sparse_gather) {
            const uint32_t gatherFlag = currIdx == 0 ? FLAG_V_GATHER_C_PING : FLAG_V_GATHER_C_PONG;
            const int32_t m = static_cast<int32_t>(curr.s1Len);
            const int32_t mAlign = static_cast<int32_t>(curr.s1LenAlign);
            if (loopTaskId > 0) {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_FIX>(FLAG_C_GATHER_ARM);
            }
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(gatherFlag);
            if (GatherNeedAiv1(m, mAlign)) {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(gatherFlag + AIV0_AIV1_FLAG_OFFSET);
            }
        }
        cubeOp.SendMatmulQK(query_gm_, key_gm_, mm1_res_ub_tensor_, curr, currIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_FIX>(currIdx == 0 ? FLAG_C1_V1_PING : FLAG_C1_V1_PONG);
        cubeOp.SendMatmulDyV(dout_gm_, val_gm_, mm2_res_ub_tensor_, curr, currIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_FIX>(currIdx == 0 ? FLAG_C2_V2_PING : FLAG_C2_V2_PONG);
        (void)query;
        (void)key;
        (void)value;
        (void)dout;
    }

    /** ProcessNotFirst: MM345(last) on AIC — after Vec Softmax Sets V*. */
    __aicore__ inline void ProcessNotFirstCube(CubeOp<GSAG_TYPE> &cubeOp, uint32_t lastIdx)
    {
        const RunTimeInfo &last = runTimeInfo_[lastIdx];
        if (!last.need_compute) {
            return;
        }
        p_l1_tensor_ = lastIdx ? p_l1_tensor_ping_ : p_l1_tensor_pong_;
        ds_l1_tensor_ = lastIdx ? ds_l1_tensor_ping_ : ds_l1_tensor_pong_;

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE1>(FLAG_V1_C3);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE1>(FLAG_V2_C45);
        cubeOp.SendMatmulDq(ds_l1_tensor_, dq_workspace_, last, lastIdx);
        cubeOp.SendMatmulDv(p_l1_tensor_, dv_workspace_, last, lastIdx);
        cubeOp.SendMatmulDk(ds_l1_tensor_, dk_workspace_, last, lastIdx);
        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        if (last.use_sparse_gather) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(FLAG_C_MM345_DONE);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(FLAG_C_MM345_DONE + AIV0_AIV1_FLAG_OFFSET);
        }
    }

public:
    __aicore__ inline GenericBlockSparseAttentionGradArch35(){};

    /**
     * Entry: wire GM, allocate ping-pong UB/L1, dispatch AIC / AIV pipelines.
     * IR order matches host: query,key,value,dout,out,lse,rsvd_idx,rsvd_count,metadata,
     * attenMask?,cuQ?,cuKv?,sequsedQ?,sequsedKv?
     */
    __aicore__ inline void Process(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dout, GM_ADDR attention_out,
                                   GM_ADDR softmaxLse, GM_ADDR rsvdBlockIdx, GM_ADDR rsvdBlockCount, GM_ADDR metadata,
                                   GM_ADDR attenMask, GM_ADDR cuSeqLengthsQ, GM_ADDR cuSeqLengthsKv, GM_ADDR sequsedQ,
                                   GM_ADDR sequsedKv, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                   const TILING_CLASS *tilingData, TPipe *tPipe)
    {
        uint32_t base_m = tilingData->baseM;
        uint32_t base_n = tilingData->baseN;
        uint32_t vec_base_m = base_m / 2;
        uint32_t vec_base_n = base_n;

        query_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)query, tilingData->dqSize);
        key_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)key, tilingData->dkSize);
        val_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)value, tilingData->dkSize);
        dout_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dout, tilingData->dqSize);
        attention_out_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)attention_out, tilingData->dqSize);
        dq_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dq, tilingData->dqSize);
        dk_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dk, tilingData->dkSize);
        dv_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dv, tilingData->dkSize);
        dq_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dqWorkspaceOffset), tilingData->dqSize);
        dk_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dkWorkspaceOffset), tilingData->dkSize);
        dv_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dvWorkspaceOffset), tilingData->dkSize);
        const int64_t lseElems =
            (INPUT_LAYOUT == TND) ?
                static_cast<int64_t>(tilingData->qSeqLen) * tilingData->qHeadNum :
                static_cast<int64_t>(tilingData->batchNum) * tilingData->qHeadNum * tilingData->qSeqLen;
        // Match host GetWorkspaceSize padding (AlignTo 256).
        const int64_t sftgElems = ((lseElems * 8) + 255) / 256 * 256;
        sftg_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->sftgWorkspaceOffset), sftgElems);
        const int64_t sparseIdxElems =
            static_cast<int64_t>(tilingData->batchNum) * tilingData->kvHeadNum * tilingData->numJ * tilingData->maxS1;
        sparse_idx_gm_.SetGlobalBuffer((__gm__ int32_t *)rsvdBlockIdx, sparseIdxElems);
        q_head_num_ = static_cast<int32_t>(tilingData->qHeadNum);
        head_dim_ = static_cast<int32_t>(tilingData->headDim);
        vec_ub_matrix_elements_ = vec_base_m * vec_base_n;
        vec_base_m_ = vec_base_m;

        addr_.Init(tilingData, cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, rsvdBlockIdx, rsvdBlockCount,
                   metadata);
        tPipe->InitBuffer(ub_buffer_, UB_SIZE);
        tPipe->InitBuffer(l1_buffer_, L1_SIZE);

        // Per ping-pong slot: mm1 float MN | mm2 float MN (Softmax / Fixpipe only).
        // Shared gather staging after both slots, no mm1/mm2 overlay.
        static_assert(sizeof(INPUT_TYPE) == 2, "GSAG arch35 expects bf16/fp16");
        const uint32_t mmSlotElems = vec_base_m * vec_base_n;
        const uint32_t nz_ub_elements = (vec_base_m + 1) * vec_base_n;
        const uint32_t mmSlotBytes = 2u * mmSlotElems * static_cast<uint32_t>(sizeof(float));
        const uint32_t gatherStagingBytes = 2u * vec_ub_matrix_elements_ * static_cast<uint32_t>(sizeof(INPUT_TYPE)) +
                                            2u * nz_ub_elements * static_cast<uint32_t>(sizeof(INPUT_TYPE));
        const uint32_t mmUbBase = ub_offset_;
        const uint32_t mmFloatBytes = mmSlotElems * static_cast<uint32_t>(sizeof(float));

        mm1_res_ub_tensor_ping_ = ub_buffer_.GetWithOffset<float>(mmSlotElems, mmUbBase);
        mm2_res_ub_tensor_ping_ = ub_buffer_.GetWithOffset<float>(mmSlotElems, mmUbBase + mmFloatBytes);
        const uint32_t pongBase = mmUbBase + mmSlotBytes;
        mm1_res_ub_tensor_pong_ = ub_buffer_.GetWithOffset<float>(mmSlotElems, pongBase);
        mm2_res_ub_tensor_pong_ = ub_buffer_.GetWithOffset<float>(mmSlotElems, pongBase + mmFloatBytes);

        uint32_t gOff = mmUbBase + 2u * mmSlotBytes;
        gather_q_nd_ub_ = ub_buffer_.GetWithOffset<INPUT_TYPE>(vec_ub_matrix_elements_, gOff);
        gOff += vec_ub_matrix_elements_ * sizeof(INPUT_TYPE);
        gather_dout_nd_ub_ = ub_buffer_.GetWithOffset<INPUT_TYPE>(vec_ub_matrix_elements_, gOff);
        gOff += vec_ub_matrix_elements_ * sizeof(INPUT_TYPE);
        query_nz_ub_tensor_ = ub_buffer_.GetWithOffset<INPUT_TYPE>(nz_ub_elements, gOff);
        gOff += nz_ub_elements * sizeof(INPUT_TYPE);
        dout_nz_ub_tensor_ = ub_buffer_.GetWithOffset<INPUT_TYPE>(nz_ub_elements, gOff);
        ub_offset_ = gOff + nz_ub_elements * sizeof(INPUT_TYPE);

        p_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        p_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        ds_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        ds_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);

        const uint32_t head_dim_align = RoundUp(static_cast<uint32_t>(head_dim_), C0_SIZE);
        query_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align, l1_offset_);
        l1_offset_ += base_m * head_dim_align * sizeof(INPUT_TYPE);
        query_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align, l1_offset_);
        l1_offset_ += base_m * head_dim_align * sizeof(INPUT_TYPE);
        dout_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align, l1_offset_);
        l1_offset_ += base_m * head_dim_align * sizeof(INPUT_TYPE);
        dout_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align, l1_offset_);
        l1_offset_ += base_m * head_dim_align * sizeof(INPUT_TYPE);

        (void)attenMask;

        if ASCEND_IS_AIC {
            CubeProcess(query, key, value, dout, attention_out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata,
                        cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, dq, dk, dv, workspace, tilingData, tPipe);
        }
        if ASCEND_IS_AIV {
            VectorProcess(query, key, value, dout, attention_out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata,
                          cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, dq, dk, dv, workspace, tilingData, tPipe);
        }
    }

    __aicore__ inline int64_t GetSparseTokenGmOffset(const RunTimeInfo &runTimeInfo, const int32_t qTok) const
    {
        if constexpr (INPUT_LAYOUT == TND) {
            return (static_cast<int64_t>(runTimeInfo.last_q_seq_sum) + qTok) * q_head_num_ * head_dim_ +
                   static_cast<int64_t>(runTimeInfo.n1Idx) * head_dim_;
        } else if constexpr (INPUT_LAYOUT == BSND) {
            return static_cast<int64_t>(runTimeInfo.bIdx) * runTimeInfo.cur_q_seq_len * q_head_num_ * head_dim_ +
                   static_cast<int64_t>(qTok) * q_head_num_ * head_dim_ +
                   static_cast<int64_t>(runTimeInfo.n1Idx) * head_dim_;
        } else {
            return static_cast<int64_t>(runTimeInfo.bIdx) * q_head_num_ * runTimeInfo.cur_q_seq_len * head_dim_ +
                   static_cast<int64_t>(runTimeInfo.n1Idx) * runTimeInfo.cur_q_seq_len * head_dim_ +
                   static_cast<int64_t>(qTok) * head_dim_;
        }
    }

    /**
     * AIV0/AIV1 each own s1LenAlign/2 rows (16-row fractal). ND gather lands in
     * UB, TransdataND2NZ writes [K1, vec_base_m+1, C0], then CopyUB2L1-style
     * DataCopy scatters into Cube NZ L1 [K1, s1LenAlign, C0].
     */
    __aicore__ inline void CopyQAndDoutToL1(const RunTimeInfo &runTimeInfo, const uint32_t pingPongIdx)
    {
        const int32_t m = static_cast<int32_t>(runTimeInfo.s1Len);
        const int32_t mAlign = static_cast<int32_t>(runTimeInfo.s1LenAlign);
        if (m <= 0 || mAlign <= 0) {
            return;
        }

        // MIX_AIC_1_2: AIV subblock 0/1. Partition MUST match Cube Fixpipe dualDstCtl=1
        // and Vec Softmax (half = mAlign/2). Align16(m/2) diverges when m%32!=0 (e.g. m=68).
        // For m < 16, AIV1 still owns the pad half [8,16) and writes zeros (do NOT AIV0-only
        // copyRows=mAlign — that DataCopy shape hangs MTE on some SoCs).
        const int32_t aivHalfIdx = static_cast<int32_t>(GetSubBlockIdx());
        const int32_t halfAlign = mAlign / 2;
        const int32_t totalAlign = mAlign;
        const int32_t rowBegin = aivHalfIdx * halfAlign;
        int32_t rowCount = m - rowBegin;
        if (rowCount < 0) {
            rowCount = 0;
        }
        if (rowCount > halfAlign) {
            rowCount = halfAlign;
        }
        const int32_t shardAlign = halfAlign;
        // Dedicated events — avoid EVENT_ID0 (Vector) and EVENT_ID3/4 (vec/cube ping-pong).
        constexpr event_t evtGatherVmte2 = EVENT_ID2;
        constexpr event_t evtGatherMte2v = EVENT_ID5;
        constexpr event_t evtGatherVmte3 = EVENT_ID6;
        constexpr event_t evtGatherMte3mte2 = EVENT_ID7;

        // Shared gather staging — never aliases mm1/mm2 ping-pong slots.
        LocalTensor<INPUT_TYPE> qUb = gather_q_nd_ub_;
        LocalTensor<INPUT_TYPE> doutUb = gather_dout_nd_ub_;
        LocalTensor<INPUT_TYPE> qNz = query_nz_ub_tensor_;
        LocalTensor<INPUT_TYPE> doutNz = dout_nz_ub_tensor_;

        const int32_t ndElems = static_cast<int32_t>(vec_ub_matrix_elements_);
        Duplicate(qUb, static_cast<INPUT_TYPE>(0), ndElems);
        Duplicate(doutUb, static_cast<INPUT_TYPE>(0), ndElems);
        SET_FLAG(V, MTE2, evtGatherVmte2);
        WAIT_FLAG(V, MTE2, evtGatherVmte2);
        for (int32_t row = 0; row < rowCount; ++row) {
            const int32_t qTok = sparse_idx_gm_.GetValue(runTimeInfo.sparseIdxOffset + rowBegin + row);
            if (qTok < 0 || qTok >= runTimeInfo.cur_q_seq_len) {
                continue;
            }
            const int64_t gmOffset = GetSparseTokenGmOffset(runTimeInfo, qTok);
            SET_FLAG(S, MTE2, EVENT_ID1);
            WAIT_FLAG(S, MTE2, EVENT_ID1);
            DataCopy(qUb[row * head_dim_], query_gm_[gmOffset], head_dim_);
            DataCopy(doutUb[row * head_dim_], dout_gm_[gmOffset], head_dim_);
        }
        SET_FLAG(MTE2, V, evtGatherMte2v);
        WAIT_FLAG(MTE2, V, evtGatherMte2v);
        // VF always converts the full vec_base_m ND tile (FAG CopyUB2L1).
        const uint32_t vfM = vec_base_m_;
        TransdataND2NZ<INPUT_TYPE>(qNz, qUb, vfM, head_dim_);
        TransdataND2NZ<INPUT_TYPE>(doutNz, doutUb, vfM, head_dim_);
        SET_FLAG(V, MTE3, evtGatherVmte3);
        WAIT_FLAG(V, MTE3, evtGatherVmte3);
        // Copy a full 16-row-aligned shard into L1 NZ. Rows beyond rowCount
        // were zero-filled in UB, so writing the full aligned shard keeps the
        // L1 fractal layout regular for Cube consumers.
        const int32_t copyRows = shardAlign;
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(head_dim_ / C0_SIZE);
        copyParams.blockLen =
            static_cast<uint16_t>(copyRows * static_cast<int32_t>(C0_SIZE) * static_cast<int32_t>(sizeof(INPUT_TYPE)) /
                                  static_cast<int32_t>(BLOCK_SIZE));
        copyParams.srcStride =
            static_cast<uint16_t>((static_cast<int32_t>(vfM) + 1 - copyRows) * static_cast<int32_t>(C0_SIZE) *
                                  static_cast<int32_t>(sizeof(INPUT_TYPE)) / static_cast<int32_t>(BLOCK_SIZE));
        copyParams.dstStride =
            static_cast<uint16_t>((totalAlign - copyRows) * static_cast<int32_t>(C0_SIZE) *
                                  static_cast<int32_t>(sizeof(INPUT_TYPE)) / static_cast<int32_t>(BLOCK_SIZE));

        LocalTensor<INPUT_TYPE> qL1 = pingPongIdx == 0 ? query_l1_tensor_ping_ : query_l1_tensor_pong_;
        LocalTensor<INPUT_TYPE> doutL1 = pingPongIdx == 0 ? dout_l1_tensor_ping_ : dout_l1_tensor_pong_;
        const int32_t l1RowOffset = rowBegin * static_cast<int32_t>(C0_SIZE);
        DataCopy(qL1[l1RowOffset], qNz, copyParams);
        DataCopy(doutL1[l1RowOffset], doutNz, copyParams);
        SET_FLAG(MTE3, MTE2, evtGatherMte3mte2);
        WAIT_FLAG(MTE3, MTE2, evtGatherMte3mte2);
    }

    __aicore__ inline void CubeProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dout, GM_ADDR attention_out,
                                       GM_ADDR softmaxLse, GM_ADDR rsvdBlockIdx, GM_ADDR rsvdBlockCount,
                                       GM_ADDR metadata, GM_ADDR cuSeqLengthsQ, GM_ADDR cuSeqLengthsKv,
                                       GM_ADDR sequsedQ, GM_ADDR sequsedKv, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                       GM_ADDR workspace, const TILING_CLASS *tilingData, TPipe *tPipe)
    {
        CubeOp<GSAG_TYPE> cubeOp;
        cubeOp.Init(tilingData, tPipe, l1_buffer_, l1_offset_, query_l1_tensor_ping_, query_l1_tensor_pong_,
                    dout_l1_tensor_ping_, dout_l1_tensor_pong_, rsvdBlockIdx, workspace);
        // Boot: no Wait FLAG_CUBE_POST — Cube's first barrier is Wait gather (armed while Vec runs Pre).
        (void)query;
        (void)key;
        (void)value;
        (void)dout;
        (void)attention_out;
        (void)softmaxLse;
        (void)rsvdBlockCount;
        (void)metadata;
        (void)cuSeqLengthsQ;
        (void)cuSeqLengthsKv;
        (void)sequsedQ;
        (void)sequsedKv;
        (void)dq;
        (void)dk;
        (void)dv;
        (void)workspace;

        // Preload(N)[Gather+MM12] then NotFirst(N-1)[Softmax+MM345]; epilogue for last task.
        while (true) {
            ping_pong_idx = taskId % 2; // 2 is the ping_pong_idx
            last_ping_pong_idx = 1 - ping_pong_idx;
            addr_.GetRunTimeInfo(runTimeInfo_[ping_pong_idx]);

            if (runTimeInfo_[ping_pong_idx].need_compute == false) {
                break;
            }

            ProcessPreloadCube(cubeOp, query, key, value, dout, ping_pong_idx, taskId);
            if (taskId > 0) {
                ProcessNotFirstCube(cubeOp, last_ping_pong_idx);
            }
            taskId++;
        }
        // Epilogue: MM345 for the final task (ProcessNotFirst after loop).
        if (taskId > 0) {
            const uint32_t epilogueIdx = (taskId + 1) % 2;
            if (runTimeInfo_[epilogueIdx].need_compute) {
                ProcessNotFirstCube(cubeOp, epilogueIdx);
            }
        }
        cubeOp.Destroy();
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_FIX>(FLAG_CUBE_POST);
    }

    __aicore__ inline void VectorProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dout, GM_ADDR attention_out,
                                         GM_ADDR softmaxLse, GM_ADDR rsvdBlockIdx, GM_ADDR rsvdBlockCount,
                                         GM_ADDR metadata, GM_ADDR cuSeqLengthsQ, GM_ADDR cuSeqLengthsKv,
                                         GM_ADDR sequsedQ, GM_ADDR sequsedKv, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                         GM_ADDR workspace, const TILING_CLASS *tilingData, TPipe *tPipe)
    {
        VecOp<GSAG_TYPE> vecOp;
        vecOp.Init(dout, query, key, value, attention_out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata,
                   cuSeqLengthsQ, cuSeqLengthsKv, dq, dk, dv, workspace, tilingData, ub_buffer_, ub_offset_);

        (void)tPipe;
        vecOp.SendVecPre(dq_workspace_, dk_workspace_, dv_workspace_, dout_gm_, attention_out_gm_, sftg_workspace_,
                         tilingData, ub_buffer_);
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        vecOp.SendVecSftgFront(dout_gm_, attention_out_gm_, sftg_workspace_, tilingData, ub_buffer_);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        vecOp.SetFlag();
        // Preload(N)=Gather(curr) then NotFirst(N-1)=Softmax(last); Scatter after MM345(last).
        while (true) {
            ping_pong_idx = taskId % 2; // 2 is the ping_pong_idx
            last_ping_pong_idx = 1 - ping_pong_idx;
            addr_.GetRunTimeInfo(runTimeInfo_[ping_pong_idx]);

            if (runTimeInfo_[ping_pong_idx].need_compute == false) {
                break;
            }

            ProcessPreloadVec(vecOp, ping_pong_idx, taskId);
            if (taskId > 0 && runTimeInfo_[last_ping_pong_idx].need_compute) {
                ProcessNotFirstVec(vecOp, last_ping_pong_idx);
            }
            if (taskId > 0 && runTimeInfo_[last_ping_pong_idx].need_compute) {
                ScatterLastVec(vecOp, last_ping_pong_idx, ub_buffer_);
            }
            taskId++;
        }
        // Epilogue: Softmax + Scatter for the final task.
        if (taskId > 0) {
            const uint32_t epilogueIdx = (taskId + 1) % 2;
            if (runTimeInfo_[epilogueIdx].need_compute) {
                ProcessNotFirstVec(vecOp, epilogueIdx);
                ScatterLastVec(vecOp, epilogueIdx, ub_buffer_);
            }
        }
        vecOp.WaitFlag();
        PipeBarrier<PIPE_ALL>();

        CrossCoreWaitFlag(FLAG_CUBE_POST);
        SyncAll();
        vecOp.SendVecPost(dq_gm_, dk_gm_, dv_gm_, dq_workspace_, dk_workspace_, dv_workspace_, tilingData, ub_buffer_);
    }
};

} // namespace GSAG_ARC35
