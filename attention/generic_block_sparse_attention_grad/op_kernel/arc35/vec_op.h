/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "common_header.h"
#include "vector_api/vf_cast_nd2nz.h"
using namespace AscendC;

namespace GSAG_ARC35 {

struct EvenCoreInfo {
    uint32_t data_size{0};        // 总共需要处理的元素个数
    uint32_t start_idx{0};        // 每个core处理的起始idx
    uint32_t len{0};              // 每个core处理的元素个数
    uint32_t loop_num{0};         // 每个core需要循环的次数
    uint32_t max_process_size{0}; // 每个core单次循环最大处理的元素个数
    uint32_t align_tail{0};       // 每个core处理的对齐尾块, tail = align_tail + pad_tail
    uint32_t pad_tail{0};         // 每个core处理的pad尾块, tail = align_tail + pad_tail
};

template <typename BSA_TYPE>
class VecOp {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

private:
    uint32_t v_core_num_;
    uint32_t v_core_idx_;
    uint32_t v_sub_core_idx_;
    int32_t batch_num_;
    int32_t q_seq_len_;
    int32_t kv_seq_len_;
    int32_t q_group_;
    int32_t q_head_num_;
    int32_t kv_head_num_;
    int32_t head_dim_;
    int32_t head_dim_align_;
    int32_t ping_pong_flag_inner{0};
    int32_t ping_pong_flag_outer{0};
    float softmax_scale_{0.0f};
    uint32_t base_m;
    uint32_t base_n;
    uint32_t vec_base_m;
    uint32_t vec_base_n;
    GM_ADDR act_seq_q_len;
    GM_ADDR rsvd_block_idx_;
    GlobalTensor<float> lse_gm_;
    GlobalTensor<float> sftg_workspace_;
    GlobalTensor<float> dq_workspace_;
    GlobalTensor<float> dk_workspace_;
    GlobalTensor<float> dv_workspace_;
    GlobalTensor<float> dq_sel_workspace_;
    GlobalTensor<int32_t> sparse_idx_gm_;
    uint64_t dq_sel_workspace_offset_{0};
    int64_t dq_sel_core_elems_{0};
    int32_t q_stride_lse_{1};
    LocalTensor<float> lse_tensor_;
    LocalTensor<float> lse_tensor_ping_;
    LocalTensor<float> lse_tensor_pong_;
    LocalTensor<float> sftg_front_tensor_;
    LocalTensor<float> sftg_front_tensor_ping_;
    LocalTensor<float> sftg_front_tensor_pong_;
    LocalTensor<INPUT_TYPE> softmax_res_nz_tensor_;
    LocalTensor<INPUT_TYPE> sftg_res_nz_tensor_;

    LocalTensor<float> vec_in_ping_;
    LocalTensor<float> vec_in_pong_;
    LocalTensor<INPUT_TYPE> vec_out_ping_;
    LocalTensor<INPUT_TYPE> vec_out_pong_;
    // sftg
    LocalTensor<INPUT_TYPE> dy_in_ping_;
    LocalTensor<INPUT_TYPE> dy_in_pong_;
    LocalTensor<INPUT_TYPE> attention_in_ping_;
    LocalTensor<INPUT_TYPE> attention_in_pong_;
    LocalTensor<float> dy_out_ping_;
    LocalTensor<float> dy_out_pong_;
    LocalTensor<float> attention_out_ping_;
    LocalTensor<float> attention_out_pong_;
    LocalTensor<float> sftg_front_ping_;
    LocalTensor<float> sftg_front_pong_;
    LocalTensor<uint8_t> sftg_tmp_tensor;
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t C0_SIZE = 16;
    static constexpr uint32_t SHAPE_RANK_2D = 2; // ND matrix rank for SoftmaxGradFront SetShapeInfo
    static constexpr uint32_t BLOCK_FP32 = BLOCK_SIZE / sizeof(float);
    static constexpr uint32_t BLOCK_INPUT = BLOCK_SIZE / sizeof(INPUT_TYPE);
    constexpr static uint32_t PRE_TILE_LEN = 60 * 1024;  // pre一次处理元素的个数
    constexpr static uint32_t POST_TILE_LEN = 20 * 1024; // POST一次处理元素的个数
    // runtInfo
    int32_t s1_process_;
    int32_t s1_process_align_;
    int32_t s2_process_align_;
    int32_t half_s1_process_align_;
    int32_t data_size;
    int32_t half_s1_process_real_;
    int64_t lse_gm_offset_;
    int64_t sftg_gm_offset_;
    int64_t l1_offset_;
    int32_t runTimeMaskType_{0};
    TEventID event_ping_ = EVENT_ID3;
    TEventID event_pong_ = EVENT_ID4;
    TEventID event_id;

public:
    __aicore__ inline VecOp(){};

    __aicore__ inline void Init(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                GM_ADDR softmaxLse, GM_ADDR rsvdBlockIdx, GM_ADDR rsvdBlockCount, GM_ADDR metadata,
                                GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                GM_ADDR workspace, const TILING_CLASS *tilingData, TBuf<TPosition::VECCALC> &ub_buffer,
                                uint32_t ub_offset)
    {
        this->v_core_num_ = tilingData->cubeCoreNum * 2; // 2 is the v_core_num_
        this->batch_num_ = tilingData->batchNum;
        this->q_seq_len_ = tilingData->qSeqLen;
        this->kv_seq_len_ = tilingData->kvSeqLen;
        this->q_group_ = tilingData->qGroup;
        this->q_head_num_ = tilingData->qHeadNum;
        this->kv_head_num_ = tilingData->kvHeadNum;
        this->head_dim_ = tilingData->headDim;
        this->softmax_scale_ = tilingData->softmaxScale;
        this->head_dim_align_ = RoundUp(head_dim_, static_cast<int32_t>(C0_SIZE));
        this->base_m = tilingData->baseM;
        this->base_n = tilingData->baseN;
        this->vec_base_m = tilingData->baseM / 2; // 2 is the vec_base_m
        this->vec_base_n = tilingData->baseN;
        this->act_seq_q_len = actualQseqlen;
        this->rsvd_block_idx_ = rsvdBlockIdx;
        v_core_idx_ = GetBlockIdx();
        v_sub_core_idx_ = GetSubBlockIdx();
        q_stride_lse_ = (INPUT_LAYOUT == TND) ? q_head_num_ : 1;

        // gm_tensor — pass explicit sizes so GetValue cannot treat large T offsets as OOB (AIC 264).
        const int64_t sparseIdxElems =
            static_cast<int64_t>(tilingData->batchNum) * tilingData->kvHeadNum * tilingData->numJ * tilingData->maxS1;
        const int64_t lseElems =
            (INPUT_LAYOUT == TND) ?
                static_cast<int64_t>(tilingData->qSeqLen) * tilingData->qHeadNum :
                static_cast<int64_t>(tilingData->batchNum) * tilingData->qHeadNum * tilingData->qSeqLen;
        // Match host GetWorkspaceSize padding (AlignTo 256) so EvenCore DataCopy tails stay in-bound.
        const int64_t sftgElems = ((lseElems * 8) + 255) / 256 * 256;
        lse_gm_.SetGlobalBuffer((__gm__ float *)softmaxLse, lseElems);
        sftg_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->sftgWorkspaceOffset), sftgElems);
        dq_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dqWorkspaceOffset), tilingData->dqSize);
        dk_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dkWorkspaceOffset), tilingData->dkSize);
        dv_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dvWorkspaceOffset), tilingData->dkSize);
        dq_sel_workspace_offset_ = tilingData->dqSelWorkspaceOffset;
        dq_sel_core_elems_ = static_cast<int64_t>(base_m) * head_dim_align_;
        const int64_t cubeBlk = static_cast<int64_t>(GetBlockIdx() / 2);
        dq_sel_workspace_.SetGlobalBuffer(
            (__gm__ float *)(workspace + dq_sel_workspace_offset_) + cubeBlk * 2 * dq_sel_core_elems_,
            2 * dq_sel_core_elems_);
        sparse_idx_gm_.SetGlobalBuffer((__gm__ int32_t *)rsvdBlockIdx, sparseIdxElems);
        (void)rsvdBlockCount;
        (void)metadata;
        (void)dout;
        (void)q;
        (void)k;
        (void)v;
        (void)attention_out;
        (void)actualKvseqlen;
        (void)dq;
        (void)dk;
        (void)dv;
        // local_tensor
        softmax_res_nz_tensor_ = ub_buffer.GetWithOffset<INPUT_TYPE>((vec_base_m + 1) * vec_base_n, ub_offset);
        ub_offset += (vec_base_m + 1) * vec_base_n * sizeof(INPUT_TYPE);
        sftg_res_nz_tensor_ = ub_buffer.GetWithOffset<INPUT_TYPE>((vec_base_m + 1) * vec_base_n, ub_offset);
        ub_offset += (vec_base_m + 1) * vec_base_n * sizeof(INPUT_TYPE);
        lse_tensor_ping_ = ub_buffer.GetWithOffset<float>(vec_base_m * BLOCK_FP32, ub_offset);
        ub_offset += vec_base_m * BLOCK_FP32 * sizeof(float);
        lse_tensor_pong_ = ub_buffer.GetWithOffset<float>(vec_base_m * BLOCK_FP32, ub_offset);
        ub_offset += vec_base_m * BLOCK_FP32 * sizeof(float);
        sftg_front_tensor_ping_ = ub_buffer.GetWithOffset<float>(vec_base_m * BLOCK_FP32, ub_offset);
        ub_offset += vec_base_m * BLOCK_FP32 * sizeof(float);
        sftg_front_tensor_pong_ = ub_buffer.GetWithOffset<float>(vec_base_m * BLOCK_FP32, ub_offset);
        ub_offset += vec_base_m * BLOCK_FP32 * sizeof(float);
    }

    __aicore__ inline void SetFlag()
    {
        SET_FLAG(V, MTE2, event_ping_);
        SET_FLAG(V, MTE2, event_pong_);
    }

    __aicore__ inline void WaitFlag()
    {
        WAIT_FLAG(V, MTE2, event_ping_);
        WAIT_FLAG(V, MTE2, event_pong_);
    }

    __aicore__ inline void SendVecPre(const GlobalTensor<float> &dq_workspace, const GlobalTensor<float> &dk_workspace,
                                      const GlobalTensor<float> &dv_workspace, const GlobalTensor<INPUT_TYPE> &dy_gm,
                                      const GlobalTensor<INPUT_TYPE> &out_gm, const GlobalTensor<float> &sftg_workspace,
                                      const TILING_CLASS *tilingData, TBuf<TPosition::VECCALC> &ub_buffer)
    {
        constexpr static uint32_t PRE_TILE_LEN = 20 * 1024; // PRE一次处理元素的个数
        EvenCoreInfo info;
        vec_in_ping_ = ub_buffer.GetWithOffset<float>(PRE_TILE_LEN, 0);
        Duplicate(vec_in_ping_, (float)0.0, PRE_TILE_LEN);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        ComputeEvenCoreInfo(info, tilingData->dqSize, PRE_TILE_LEN);
        ComputeVecPre(dq_workspace, vec_in_ping_, info);
        ComputeEvenCoreInfo(info, tilingData->dkSize, PRE_TILE_LEN);
        ComputeVecPre(dk_workspace, vec_in_ping_, info);
        ComputeVecPre(dv_workspace, vec_in_ping_, info);
    }

    __aicore__ inline void SendVecSftgFront(const GlobalTensor<INPUT_TYPE> &dy_gm,
                                            const GlobalTensor<INPUT_TYPE> &out_gm,
                                            const GlobalTensor<float> &sftg_workspace, const TILING_CLASS *tilingData,
                                            TBuf<TPosition::VECCALC> &ub_buffer)
    {
        EvenCoreInfo info;
        constexpr static uint32_t SFTG_TILE_LEN = 8 * 1024; // POST一次处理元素的个数
        uint32_t process_s1_size = SFTG_TILE_LEN / head_dim_;
        uint32_t ub_offset = 0;
        uint32_t sftg_ping_pong_idx = 0;
        dy_in_ping_ = ub_buffer.GetWithOffset<INPUT_TYPE>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(INPUT_TYPE);
        dy_in_pong_ = ub_buffer.GetWithOffset<INPUT_TYPE>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(INPUT_TYPE);
        attention_in_ping_ = ub_buffer.GetWithOffset<INPUT_TYPE>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(INPUT_TYPE);
        attention_in_pong_ = ub_buffer.GetWithOffset<INPUT_TYPE>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(INPUT_TYPE);
        dy_out_ping_ = ub_buffer.GetWithOffset<float>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(float);
        dy_out_pong_ = ub_buffer.GetWithOffset<float>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(float);
        attention_out_ping_ = ub_buffer.GetWithOffset<float>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(float);
        attention_out_pong_ = ub_buffer.GetWithOffset<float>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(float);
        sftg_front_ping_ = ub_buffer.GetWithOffset<float>(process_s1_size * 8,
                                                          ub_offset); // 8 is the block size for the sftg_front_tensor
        ub_offset += process_s1_size * 8 * sizeof(float);
        sftg_front_pong_ = ub_buffer.GetWithOffset<float>(process_s1_size * 8,
                                                          ub_offset); // 8 is the block size for the sftg_front_tensor
        ub_offset += process_s1_size * 8 * sizeof(float);
        sftg_tmp_tensor = ub_buffer.GetWithOffset<uint8_t>(tilingData->sftgTmpSpaceSize, ub_offset);
        ub_offset += tilingData->sftgTmpSpaceSize;

        SET_FLAG(MTE3, MTE2, event_ping_);
        SET_FLAG(MTE3, MTE2, event_pong_);
        for (uint32_t b_idx = 0; b_idx < batch_num_; b_idx++) {
            int64_t current_q_seqlen;
            int64_t last_seq_total_len;
            if constexpr (INPUT_LAYOUT == TND) {
                current_q_seqlen = GetSeqLen(b_idx, act_seq_q_len);
                last_seq_total_len = GetSeqPrefix(b_idx, act_seq_q_len);
            } else {
                current_q_seqlen = q_seq_len_;
                last_seq_total_len = 0;
            }
            for (uint32_t n1_idx = 0; n1_idx < q_head_num_; n1_idx++) {
                int64_t in_gm_offset;
                int64_t out_gm_offset;

                if constexpr (INPUT_LAYOUT == BSND) {
                    in_gm_offset = b_idx * current_q_seqlen * q_head_num_ * head_dim_ + n1_idx * head_dim_;
                    out_gm_offset =
                        b_idx * q_head_num_ * current_q_seqlen * BLOCK_FP32 + n1_idx * current_q_seqlen * BLOCK_FP32;
                } else if constexpr (INPUT_LAYOUT == BNSD) {
                    in_gm_offset =
                        b_idx * q_head_num_ * current_q_seqlen * head_dim_ + n1_idx * current_q_seqlen * head_dim_;
                    out_gm_offset =
                        b_idx * q_head_num_ * current_q_seqlen * BLOCK_FP32 + n1_idx * current_q_seqlen * BLOCK_FP32;
                } else if constexpr (INPUT_LAYOUT == TND) {
                    in_gm_offset = last_seq_total_len * q_head_num_ * head_dim_ + n1_idx * head_dim_;
                    out_gm_offset =
                        last_seq_total_len * q_head_num_ * BLOCK_FP32 + n1_idx * current_q_seqlen * BLOCK_FP32;
                }
                TEventID event_id = sftg_ping_pong_idx ? event_ping_ : event_pong_;
                ComputeEvenCoreInfo(info, current_q_seqlen, process_s1_size);

                WAIT_FLAG(MTE3, MTE2, event_id);
                ComputeSoftmaxGradFront(sftg_workspace[out_gm_offset], dy_gm[in_gm_offset], out_gm[in_gm_offset], info,
                                        tilingData, sftg_ping_pong_idx);
                SET_FLAG(MTE3, MTE2, event_id);
                sftg_ping_pong_idx = 1 - sftg_ping_pong_idx;
            }
        }

        WAIT_FLAG(MTE3, MTE2, event_ping_);
        WAIT_FLAG(MTE3, MTE2, event_pong_);
    }

    __aicore__ inline void SendVecSftPreProcess(const RunTimeInfo &runTimeInfo, uint32_t pingpong_idx)
    {
        s1_process_ = runTimeInfo.s1Len;
        s1_process_align_ = runTimeInfo.s1LenAlign;
        s2_process_align_ = runTimeInfo.s2LenAlign;
        // Match Cube dualDstCtl=1 / each AIV owns mAlign/2 (pad rows may be empty).
        half_s1_process_align_ = s1_process_align_ / 2;
        if (v_sub_core_idx_ == 0) {
            half_s1_process_real_ = half_s1_process_align_ < s1_process_ ? half_s1_process_align_ : s1_process_;
        } else {
            half_s1_process_real_ = s1_process_ - half_s1_process_align_;
            if (half_s1_process_real_ < 0) {
                half_s1_process_real_ = 0;
            }
        }
        l1_offset_ = v_sub_core_idx_ * half_s1_process_align_ * C0_SIZE;
        data_size = half_s1_process_align_ * s2_process_align_;

        // Always arm event_id + WAIT so SoftmaxGrad can always SET (incl. pad-only AIV).
        lse_tensor_ = pingpong_idx == 0 ? lse_tensor_ping_ : lse_tensor_pong_;
        sftg_front_tensor_ = pingpong_idx == 0 ? sftg_front_tensor_ping_ : sftg_front_tensor_pong_;
        event_id = pingpong_idx == 0 ? event_ping_ : event_pong_;
        WAIT_FLAG(V, MTE2, event_id);

        // Zero pad / invalid rows so SoftmaxGrad (align-sized) does not see garbage.
        const int32_t packElems = half_s1_process_align_ * static_cast<int32_t>(BLOCK_FP32);
        Duplicate(lse_tensor_, 0.0f, packElems);
        Duplicate(sftg_front_tensor_, 0.0f, packElems);
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);

        if (half_s1_process_real_ > 0) {
            // Prefetch sparse idx tile into UB (reuse softmax NZ staging; not live yet).
            // Softmax VF expects [row, 8] with DIST_BRC on element 0 — gather via MTE2
            // instead of per-element GM GetValue/SetValue (PIPE_S hotspot).
            LocalTensor<int32_t> idxUb = softmax_res_nz_tensor_.template ReinterpretCast<int32_t>();
            const int32_t rowStart = v_sub_core_idx_ * half_s1_process_align_;
            // half_s1_process_align_ is mAlign/2 (16-aligned m → 8-aligned); pad slots are -1.
            DataCopy(idxUb, sparse_idx_gm_[runTimeInfo.sparseIdxOffset + rowStart], half_s1_process_align_);
            SET_FLAG(MTE2, S, EVENT_ID1);
            WAIT_FLAG(MTE2, S, EVENT_ID1);

            for (int32_t r = 0; r < half_s1_process_real_; ++r) {
                const int32_t qTok = idxUb.GetValue(r);
                if (qTok < 0 || qTok >= runTimeInfo.cur_q_seq_len) {
                    continue;
                }
                int64_t lseOff = 0;
                int64_t sftgOff = 0;
                if constexpr (INPUT_LAYOUT == TND) {
                    // LSE user tensor is TND [T, N]; sftg workspace is written as
                    // prefix*N*8 + n1*S*8 + t*8 (same as GetSftgGmOffset / SendVecSftgFront).
                    lseOff =
                        (static_cast<int64_t>(runTimeInfo.last_q_seq_sum) + qTok) * q_head_num_ + runTimeInfo.n1Idx;
                    sftgOff = GetSftgGmOffset<INPUT_LAYOUT>(static_cast<int64_t>(runTimeInfo.last_q_seq_sum),
                                                            static_cast<int64_t>(runTimeInfo.cur_q_seq_len),
                                                            q_head_num_, static_cast<int64_t>(runTimeInfo.bIdx),
                                                            static_cast<int64_t>(qTok),
                                                            static_cast<int64_t>(runTimeInfo.n1Idx));
                } else {
                    lseOff = static_cast<int64_t>(runTimeInfo.bIdx) * q_head_num_ * runTimeInfo.cur_q_seq_len +
                             runTimeInfo.n1Idx * runTimeInfo.cur_q_seq_len + qTok;
                    sftgOff = lseOff * 8; // 8 is the block size for the sftg_front_tensor
                }
                SET_FLAG(S, MTE2, EVENT_ID1);
                WAIT_FLAG(S, MTE2, EVENT_ID1);
                // sftg: 8 fp32 = 32B block — one DataCopy replaces 8x GetValue+SetValue.
                DataCopy(sftg_front_tensor_[r * static_cast<int32_t>(BLOCK_FP32)], sftg_workspace_[sftgOff],
                         BLOCK_FP32);
                // lse: Softmax DIST_BRC only needs slot [r*8]; DataCopyPad 1 fp32 (no 8x SetValue).
                DataCopyPad(lse_tensor_[r * static_cast<int32_t>(BLOCK_FP32)], lse_gm_[lseOff],
                            {static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0},
                            {false, 0, 0, 0});
            }
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }
    }

    /**
     * RIGHT_DOWN_CAUSAL: origin at bottom-right; valid when k <= q + (S2 - S1).
     * Invalid columns form a contiguous tail per row — Duplicate(NEG_INF) on
     * 32B-aligned spans; unaligned head of the tail uses SetValue.
     */
    __aicore__ inline void ApplyCausalAttenMask(const LocalTensor<float> &sUb, const RunTimeInfo &runTimeInfo)
    {
        if (runTimeInfo.mask_type == 0 || half_s1_process_real_ <= 0) {
            return;
        }
        constexpr float NEG_INF = -1.0e30f;
        const int32_t rowStart = v_sub_core_idx_ * half_s1_process_align_;
        const int32_t s2Start = runTimeInfo.s2Idx;
        const int32_t s2Len = static_cast<int32_t>(runTimeInfo.s2Len);
        const int32_t causalOffset = runTimeInfo.cur_kv_seq_len - runTimeInfo.cur_q_seq_len;

        // Prefetch sparse idx (softmax NZ staging not live until CastND2NZ after Softmax).
        LocalTensor<int32_t> idxUb = softmax_res_nz_tensor_.template ReinterpretCast<int32_t>();
        DataCopy(idxUb, sparse_idx_gm_[runTimeInfo.sparseIdxOffset + rowStart], half_s1_process_align_);
        SET_FLAG(MTE2, S, EVENT_ID1);
        WAIT_FLAG(MTE2, S, EVENT_ID1);

        for (int32_t r = 0; r < half_s1_process_real_; ++r) {
            const int32_t qTok = idxUb.GetValue(r);
            // first c where kTok = s2Start+c > qTok+causalOffset
            int32_t firstInvalid = qTok + causalOffset - s2Start + 1;
            if (firstInvalid < 0) {
                firstInvalid = 0;
            }
            if (firstInvalid >= s2Len) {
                continue;
            }
            const int32_t rowBase = r * s2_process_align_;
            // Duplicate requires 32B-aligned UB offset for fp32 → align start up.
            const int32_t firstAlign = RoundUp(firstInvalid, static_cast<int32_t>(BLOCK_FP32));
            for (int32_t c = firstInvalid; c < firstAlign && c < s2Len; ++c) {
                sUb.SetValue(rowBase + c, NEG_INF);
            }
            if (firstAlign >= s2Len) {
                continue;
            }
            int32_t nMask = s2Len - firstAlign;
            int32_t nMaskAlign = RoundUp(nMask, static_cast<int32_t>(BLOCK_FP32));
            if (firstAlign + nMaskAlign > s2_process_align_) {
                nMaskAlign = s2_process_align_ - firstAlign;
            }
            if (nMaskAlign <= 0) {
                continue;
            }
            SET_FLAG(S, V, EVENT_ID1);
            WAIT_FLAG(S, V, EVENT_ID1);
            Duplicate(sUb[rowBase + firstAlign], NEG_INF, nMaskAlign);
        }
    }

    __aicore__ inline void SendVecSoftmax(const LocalTensor<INPUT_TYPE> &dst_l1_tensor,
                                          const LocalTensor<float> &src_ub_tensor, const RunTimeInfo &runTimeInfo)
    {
        // Pad-only AIV (m < 16 → AIV1): still write zeros into this half of P L1.
        if (half_s1_process_real_ <= 0) {
            const int32_t zeroElems = half_s1_process_align_ * s2_process_align_;
            Duplicate(src_ub_tensor, 0.0f, zeroElems);
            PipeBarrier<PIPE_V>();
            CastND2NZ<INPUT_TYPE>(softmax_res_nz_tensor_, src_ub_tensor, half_s1_process_align_, s2_process_align_);
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = s2_process_align_ / C0_SIZE;
            dataCopyParams.blockLen = half_s1_process_align_ * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride =
                (s1_process_align_ - half_s1_process_align_) * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
            DataCopy(dst_l1_tensor[l1_offset_], softmax_res_nz_tensor_, dataCopyParams);
            return;
        }

        runTimeMaskType_ = runTimeInfo.mask_type;
        ApplyCausalAttenMask(src_ub_tensor, runTimeInfo);
        // Mask may end on PIPE_S (idx GetValue / skip-only rows); Softmax is PIPE_V.
        SET_FLAG(S, V, EVENT_ID1);
        WAIT_FLAG(S, V, EVENT_ID1);
        PipeBarrier<PIPE_V>();
        // Softmax on real rows only; pad [real, align) to 0 so CastND2NZ/DataCopy
        // can use half_s1_process_align_ (must match NZ fractal).
        SimpleSoftmax((__ubuf__ float *)src_ub_tensor.GetPhyAddr(), (__ubuf__ float *)src_ub_tensor.GetPhyAddr(),
                      (__ubuf__ float *)lse_tensor_.GetPhyAddr(), half_s1_process_real_, s2_process_align_);
        if (half_s1_process_real_ < half_s1_process_align_) {
            const int32_t padElems = (half_s1_process_align_ - half_s1_process_real_) * s2_process_align_;
            Duplicate(src_ub_tensor[half_s1_process_real_ * s2_process_align_], 0.0f, padElems);
            PipeBarrier<PIPE_V>();
        }

        CastND2NZ<INPUT_TYPE>(softmax_res_nz_tensor_, src_ub_tensor, half_s1_process_align_, s2_process_align_);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = s2_process_align_ / C0_SIZE;
        dataCopyParams.blockLen = half_s1_process_align_ * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride =
            (s1_process_align_ - half_s1_process_align_) * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
        DataCopy(dst_l1_tensor[l1_offset_], softmax_res_nz_tensor_, dataCopyParams);
    }

    __aicore__ inline void SendVecSoftmaxGrad(const LocalTensor<INPUT_TYPE> &dst_l1_tensor,
                                              const LocalTensor<float> &softmax_ub_tensor,
                                              const LocalTensor<float> &src_ub_tensor, const RunTimeInfo &runTimeInfo)
    {
        /*
         * function: Compute softmaxGrad
         * input shape：[s1LenAlign / 2, s2LenAlign]
         * out shape:   [s1LenAlign / 2, s2LenAlign]
         * dtype:       float
         */
        // Pad-only AIV: zero dS L1 half and still SET_FLAG(V,MTE2) for event balance.
        if (half_s1_process_real_ <= 0) {
            const int32_t zeroElems = half_s1_process_align_ * s2_process_align_;
            Duplicate(src_ub_tensor, 0.0f, zeroElems);
            PipeBarrier<PIPE_V>();
            CastND2NZ<INPUT_TYPE>(sftg_res_nz_tensor_, src_ub_tensor, half_s1_process_align_, s2_process_align_);
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = s2_process_align_ / C0_SIZE;
            dataCopyParams.blockLen = half_s1_process_align_ * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride =
                (s1_process_align_ - half_s1_process_align_) * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
            DataCopy(dst_l1_tensor[l1_offset_], sftg_res_nz_tensor_, dataCopyParams);
            SET_FLAG(V, MTE2, event_id);
            return;
        }
        // Pad rows beyond real must be 0 so align-sized SoftmaxGrad/Cast stay in-bounds.
        if (half_s1_process_real_ < half_s1_process_align_) {
            const int32_t padElems = (half_s1_process_align_ - half_s1_process_real_) * s2_process_align_;
            Duplicate(src_ub_tensor[half_s1_process_real_ * s2_process_align_], 0.0f, padElems);
            Duplicate(softmax_ub_tensor[half_s1_process_real_ * s2_process_align_], 0.0f, padElems);
            PipeBarrier<PIPE_V>();
        }
        ComputeSoftmaxGrad((__ubuf__ float *)src_ub_tensor.GetPhyAddr(), (__ubuf__ float *)src_ub_tensor.GetPhyAddr(),
                           (__ubuf__ float *)softmax_ub_tensor.GetPhyAddr(),
                           (__ubuf__ float *)sftg_front_tensor_.GetPhyAddr(), half_s1_process_align_,
                           s2_process_align_);

        CastND2NZ<INPUT_TYPE>(sftg_res_nz_tensor_, src_ub_tensor, half_s1_process_align_, s2_process_align_);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);

        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = s2_process_align_ / C0_SIZE;
        dataCopyParams.blockLen = half_s1_process_align_ * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride =
            (s1_process_align_ - half_s1_process_align_) * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
        DataCopy(dst_l1_tensor[l1_offset_], sftg_res_nz_tensor_, dataCopyParams);
        SET_FLAG(V, MTE2, event_id);
    }

    /**
     * ScatterAdd for dQ_sel: Cube Fixpiped [mAlign,D] to GM scratch;
     * AIV0/AIV1 each take half of real rows, batch rows GM->UB, then AtomicAdd to dq.
     * Event chain: MTE3<->MTE2 (UB reuse) + MTE2<->MTE3 (load->atomic).
     */
    __aicore__ inline void ScatterDqSel(const RunTimeInfo &runTimeInfo, TBuf<TPosition::VECCALC> &ub_buffer,
                                        const uint32_t ping_pong_idx)
    {
        if (!runTimeInfo.use_sparse_gather || !runTimeInfo.need_compute) {
            return;
        }
        const int32_t m = runTimeInfo.s1Len;
        if (m <= 0) {
            return;
        }

        // 16 rows × D=128 fp32 ≈ 8KB; fewer outer loops / MTE syncs than UB_ROW_SIZE=8.
        constexpr int32_t UB_ROW_SIZE = 16;
        // Same half split as Softmax dualDst / Gather (mAlign/2), not m/2.
        const int32_t mAlign = runTimeInfo.s1LenAlign;
        const int32_t halfAlign = mAlign / 2;
        const int32_t aivHalf = static_cast<int32_t>(GetSubBlockIdx());
        const int32_t rowBegin = aivHalf * halfAlign;
        int32_t currentRows = m - rowBegin;
        if (currentRows < 0) {
            currentRows = 0;
        }
        if (currentRows > halfAlign) {
            currentRows = halfAlign;
        }
        if (currentRows <= 0) {
            return;
        }

        // Keep Scatter staging far from mm/gather overlay (offset 0). FIX_SUMMARY: offset-0
        // collided with AIV gather and zeroed dq; 200KB sits in the unused tail of 247KB UB.
        constexpr uint32_t SCATTER_UB_OFFSET = 200 * 1024;
        LocalTensor<float> batchUb = ub_buffer.GetWithOffset<float>(UB_ROW_SIZE * head_dim_align_, SCATTER_UB_OFFSET);

        // Layout bases outside the per-row loop.
        const int64_t n1Dim = static_cast<int64_t>(runTimeInfo.n1Idx) * head_dim_;
        int64_t layoutBase = 0;
        int64_t qTokStride = head_dim_;
        if constexpr (INPUT_LAYOUT == TND) {
            layoutBase = static_cast<int64_t>(runTimeInfo.last_q_seq_sum) * q_head_num_ * head_dim_ + n1Dim;
            qTokStride = static_cast<int64_t>(q_head_num_) * head_dim_;
        } else if constexpr (INPUT_LAYOUT == BSND) {
            layoutBase =
                static_cast<int64_t>(runTimeInfo.bIdx) * runTimeInfo.cur_q_seq_len * q_head_num_ * head_dim_ + n1Dim;
            qTokStride = static_cast<int64_t>(q_head_num_) * head_dim_;
        } else {
            layoutBase = static_cast<int64_t>(runTimeInfo.bIdx) * q_head_num_ * runTimeInfo.cur_q_seq_len * head_dim_ +
                         static_cast<int64_t>(runTimeInfo.n1Idx) * runTimeInfo.cur_q_seq_len * head_dim_;
            qTokStride = head_dim_;
        }

        const int64_t slotOff = static_cast<int64_t>(ping_pong_idx) * dq_sel_core_elems_;
        const int32_t maxLoops = CeilDiv(currentRows, UB_ROW_SIZE);
        const int32_t tailRows = currentRows - (maxLoops - 1) * UB_ROW_SIZE;

        AscendC::SetAtomicAdd<float>();
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        for (int32_t loop = 0; loop < maxLoops; ++loop) {
            const int32_t nRows = (loop == maxLoops - 1) ? tailRows : UB_ROW_SIZE;
            WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
            const int32_t globalRow0 = rowBegin + loop * UB_ROW_SIZE;
            DataCopy(batchUb, dq_sel_workspace_[slotOff + static_cast<int64_t>(globalRow0) * head_dim_],
                     nRows * head_dim_);
            SET_FLAG(MTE2, MTE3, EVENT_ID1);
            WAIT_FLAG(MTE2, MTE3, EVENT_ID1);
            for (int32_t row = 0; row < nRows; ++row) {
                const int32_t r = globalRow0 + row;
                const int32_t qTok = sparse_idx_gm_.GetValue(runTimeInfo.sparseIdxOffset + r);
                if (qTok < 0 || qTok >= runTimeInfo.cur_q_seq_len) {
                    continue;
                }
                const int64_t gmOff = layoutBase + static_cast<int64_t>(qTok) * qTokStride;
                DataCopy(dq_workspace_[gmOff], batchUb[row * head_dim_], head_dim_);
            }
            SET_FLAG(MTE3, MTE2, EVENT_ID0);
        }
        AscendC::SetAtomicNone();
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    }

    __aicore__ inline void SendVecPost(const GlobalTensor<INPUT_TYPE> &dq_out_gm,
                                       const GlobalTensor<INPUT_TYPE> &dk_out_gm,
                                       const GlobalTensor<INPUT_TYPE> &dv_out_gm,
                                       const GlobalTensor<float> &dq_workspace, const GlobalTensor<float> &dk_workspace,
                                       const GlobalTensor<float> &dv_workspace, const TILING_CLASS *tilingData,
                                       TBuf<TPosition::VECCALC> &ub_buffer)
    {
        int64_t dq_size = tilingData->dqSize;
        int64_t dkv_size = tilingData->dkSize;
        uint32_t ub_offset = 0;
        uint32_t post_ping_pong_idx = 0;
        vec_in_ping_ = ub_buffer.GetWithOffset<float>(POST_TILE_LEN, ub_offset);
        ub_offset += POST_TILE_LEN * sizeof(float);
        vec_in_pong_ = ub_buffer.GetWithOffset<float>(POST_TILE_LEN, ub_offset);
        ub_offset += POST_TILE_LEN * sizeof(float);
        vec_out_ping_ = ub_buffer.GetWithOffset<INPUT_TYPE>(POST_TILE_LEN, ub_offset);
        ub_offset += POST_TILE_LEN * sizeof(INPUT_TYPE);
        vec_out_pong_ = ub_buffer.GetWithOffset<INPUT_TYPE>(POST_TILE_LEN, ub_offset);
        ub_offset += POST_TILE_LEN * sizeof(INPUT_TYPE);
        SET_FLAG(MTE3, MTE2, event_ping_);
        SET_FLAG(MTE3, MTE2, event_pong_);

        EvenCoreInfo info;
        ComputeEvenCoreInfo(info, dkv_size, POST_TILE_LEN);
        ComputeVecPost<false>(dv_out_gm, dv_workspace, info, post_ping_pong_idx);
        ComputeVecPost<true>(dk_out_gm, dk_workspace, info, post_ping_pong_idx);
        ComputeEvenCoreInfo(info, dq_size, POST_TILE_LEN);
        ComputeVecPost<true>(dq_out_gm, dq_workspace, info, post_ping_pong_idx);

        WAIT_FLAG(MTE3, MTE2, event_ping_);
        WAIT_FLAG(MTE3, MTE2, event_pong_);
    }

private:
    __aicore__ inline void ComputeVecPre(const GlobalTensor<float> &dst_tensor, const LocalTensor<float> &src_tensor,
                                         const EvenCoreInfo &info)
    {
        if (info.start_idx >= info.data_size) {
            return;
        }
        uint32_t process_size = info.max_process_size;
        for (uint32_t i = 0; i < info.loop_num; i++) {
            if (unlikely(i == info.loop_num - 1)) {
                process_size = info.align_tail;
                if (process_size == 0) {
                    break;
                }
            }

            int64_t gm_offset = (int64_t)info.start_idx + i * info.max_process_size;
            DataCopy(dst_tensor[gm_offset], src_tensor, process_size);
        }

        if (info.pad_tail > 0) {
            int64_t gm_offset = (int64_t)info.start_idx + (info.loop_num - 1) * info.max_process_size + info.align_tail;
            DataCopyParams copyParam;
            copyParam.blockCount = 1;
            copyParam.blockLen = info.pad_tail * sizeof(float);
            copyParam.srcStride = 0;
            copyParam.dstStride = 0;
            DataCopyPad(dst_tensor[gm_offset], src_tensor, copyParam);
        }
    }

    __aicore__ inline void ComputeSoftmaxGradFront(const GlobalTensor<float> &dst_gm,
                                                   const GlobalTensor<INPUT_TYPE> &dy_gm,
                                                   const GlobalTensor<INPUT_TYPE> &out_gm, const EvenCoreInfo &info,
                                                   const TILING_CLASS *tilingData, const uint32_t sftg_ping_pong_idx)
    {
        if (info.start_idx >= info.data_size) {
            return;
        }
        LocalTensor<INPUT_TYPE> dy_tensor = sftg_ping_pong_idx ? dy_in_ping_ : dy_in_pong_;
        LocalTensor<INPUT_TYPE> attention_tensor = sftg_ping_pong_idx ? attention_in_ping_ : attention_in_pong_;
        LocalTensor<float> dy_out_tensor = sftg_ping_pong_idx ? dy_out_ping_ : dy_out_pong_;
        LocalTensor<float> attention_out_tensor = sftg_ping_pong_idx ? attention_out_ping_ : attention_out_pong_;
        LocalTensor<float> sftg_front_tensor = sftg_ping_pong_idx ? sftg_front_ping_ : sftg_front_pong_;
        uint32_t process_size = info.max_process_size;
        uint32_t src_stride;
        if constexpr (INPUT_LAYOUT == BSND) {
            src_stride = q_head_num_ * head_dim_;
        } else if constexpr (INPUT_LAYOUT == BNSD) {
            src_stride = head_dim_;
        } else if constexpr (INPUT_LAYOUT == TND) {
            src_stride = q_head_num_ * head_dim_;
        }

        for (uint32_t i = 0; i < info.loop_num; i++) {
            if (unlikely(i == info.loop_num - 1)) {
                process_size = info.align_tail + info.pad_tail;
                if (process_size == 0) {
                    break;
                }
            }
            int64_t in_gm_offset = (int64_t)(info.start_idx + i * info.max_process_size) * src_stride;

            DataCopyParams copyParam;
            copyParam.blockCount = process_size;
            copyParam.blockLen = head_dim_ * sizeof(INPUT_TYPE);
            copyParam.srcStride = (src_stride - head_dim_) * sizeof(INPUT_TYPE);
            copyParam.dstStride = 0;
            // BNSD tokens are contiguous (srcStride==0). Multi-row DataCopyPad with
            // 0-gap can hang MTE on arch35; use a single contiguous block instead.
            if (copyParam.srcStride == 0) {
                DataCopyParams cont;
                cont.blockCount = 1;
                cont.blockLen = process_size * head_dim_ * sizeof(INPUT_TYPE);
                cont.srcStride = 0;
                cont.dstStride = 0;
                DataCopyPad(dy_tensor, dy_gm[in_gm_offset], cont, {false, 0, 0, 0});
                DataCopyPad(attention_tensor, out_gm[in_gm_offset], cont, {false, 0, 0, 0});
            } else {
                DataCopyPad(dy_tensor, dy_gm[in_gm_offset], copyParam, {false, 0, 0, 0});
                DataCopyPad(attention_tensor, out_gm[in_gm_offset], copyParam, {false, 0, 0, 0});
            }
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);

            Cast(dy_out_tensor, dy_tensor, RoundMode::CAST_NONE, process_size * head_dim_);
            Cast(attention_out_tensor, attention_tensor, RoundMode::CAST_NONE, process_size * head_dim_);
            PipeBarrier<PIPE_V>();

            uint32_t intput_shape_arry[SHAPE_RANK_2D] = {static_cast<uint32_t>(process_size),
                                                         static_cast<uint32_t>(head_dim_)};
            uint32_t out_shape_arry[SHAPE_RANK_2D] = {
                static_cast<uint32_t>(process_size),
                static_cast<uint32_t>(BLOCK_FP32)}; // 8 is the block size for the sftg_front_tensor

            dy_out_tensor.SetShapeInfo(ShapeInfo(SHAPE_RANK_2D, intput_shape_arry, AscendC::DataFormat::ND));
            attention_out_tensor.SetShapeInfo(ShapeInfo(SHAPE_RANK_2D, intput_shape_arry, AscendC::DataFormat::ND));
            sftg_front_tensor.SetShapeInfo(ShapeInfo(SHAPE_RANK_2D, out_shape_arry, AscendC::DataFormat::ND));
            bool isBasicBlock = process_size % 8 == 0; // 8 is the block size for the sftg_front_tensor
            if (likely(isBasicBlock)) {
                SoftmaxGradFront<float, true>(sftg_front_tensor, dy_out_tensor, attention_out_tensor, sftg_tmp_tensor,
                                              tilingData->softmaxGradFrontTilingData);
            } else {
                SoftmaxGradFront<float, false>(sftg_front_tensor, dy_out_tensor, attention_out_tensor, sftg_tmp_tensor,
                                               tilingData->softmaxGradFrontTilingData);
            }
            PipeBarrier<PIPE_V>();

            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            int64_t out_gm_offset = (int64_t)(info.start_idx + i * info.max_process_size) * 8;
            DataCopy(dst_gm[out_gm_offset], sftg_front_tensor, process_size * 8);
            SET_FLAG(MTE3, MTE2, EVENT_ID0);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        }
    }

    template <bool MULS>
    __aicore__ inline void ComputeVecPost(const GlobalTensor<INPUT_TYPE> &dst_tensor,
                                          const GlobalTensor<float> &src_tensor, const EvenCoreInfo &info,
                                          uint32_t &post_ping_pong_idx)
    {
        if (info.start_idx >= info.data_size) {
            return;
        }

        LocalTensor<float> vecIn;
        LocalTensor<INPUT_TYPE> vecOut;
        uint32_t process_size = info.max_process_size;

        for (uint32_t i = 0; i < info.loop_num; i++) {
            if (unlikely(i == info.loop_num - 1)) {
                process_size = info.align_tail;
                if (process_size == 0) {
                    break;
                }
            }
            vecIn = post_ping_pong_idx ? vec_in_ping_ : vec_in_pong_;
            vecOut = post_ping_pong_idx ? vec_out_ping_ : vec_out_pong_;
            event_id = post_ping_pong_idx ? event_ping_ : event_pong_;
            int64_t gm_offset = (int64_t)info.start_idx + i * info.max_process_size;

            WAIT_FLAG(MTE3, MTE2, event_id);
            DataCopy(vecIn, src_tensor[gm_offset], process_size);
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);

            if constexpr (MULS) {
                Muls(vecIn, vecIn, softmax_scale_, process_size);
                PipeBarrier<PIPE_V>();
            }
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_RINT, process_size);
            PipeBarrier<PIPE_V>();

            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            DataCopy(dst_tensor[gm_offset], vecOut, process_size);
            SET_FLAG(MTE3, MTE2, event_id);
            post_ping_pong_idx = 1 - post_ping_pong_idx;
        }

        if (info.pad_tail > 0) {
            vecIn = post_ping_pong_idx ? vec_in_ping_ : vec_in_pong_;
            vecOut = post_ping_pong_idx ? vec_out_ping_ : vec_out_pong_;
            event_id = post_ping_pong_idx ? event_ping_ : event_pong_;
            int64_t gm_offset = (int64_t)info.start_idx + (info.loop_num - 1) * info.max_process_size + info.align_tail;
            uint32_t pad_tail_align = RoundUp<uint32_t>(info.pad_tail, 16);
            DataCopyParams copyParam;
            copyParam.blockCount = 1;
            copyParam.blockLen = info.pad_tail * sizeof(float);
            copyParam.srcStride = 0;
            copyParam.dstStride = 0;

            WAIT_FLAG(MTE3, MTE2, event_id);
            DataCopyPad(vecIn, src_tensor[gm_offset], copyParam, {false, 0, 0, 0});
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
            if constexpr (MULS) {
                Muls(vecIn, vecIn, softmax_scale_, pad_tail_align);
                PipeBarrier<PIPE_V>();
            }

            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_RINT, pad_tail_align);
            PipeBarrier<PIPE_V>();

            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            copyParam.blockLen = info.pad_tail * sizeof(INPUT_TYPE);
            DataCopyPad(dst_tensor[gm_offset], vecOut, copyParam);
            SET_FLAG(MTE3, MTE2, event_id);
            post_ping_pong_idx = 1 - post_ping_pong_idx;
        }
    }

    __simd_vf__ inline void SimpleSoftmax(__ubuf__ float *dstTensor, __ubuf__ float *src0Tensor,
                                          __ubuf__ float *src1Tensor, const uint32_t row, const uint32_t col)
    {
        AscendC::Reg::RegTensor<float> src_reg;
        AscendC::Reg::RegTensor<float> lse_reg;
        AscendC::Reg::RegTensor<float> scale_reg;
        AscendC::Reg::MaskReg msk_reg;
        constexpr static uint16_t repeat_size = 256 / sizeof(float);
        uint16_t repeat_times = (col + repeat_size - 1) / repeat_size;
        uint32_t ub_offset = 0;
        Duplicate(scale_reg, softmax_scale_);

        for (int32_t i = 0; i < row; i++) {
            LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(lse_reg, src1Tensor + i * 8);
            uint32_t count = col;
            ub_offset = i * col;

            for (int32_t j = 0; j < repeat_times; j++) {
                msk_reg = AscendC::Reg::UpdateMask<float>(count);

                LoadAlign(src_reg, src0Tensor + ub_offset);
                Mul(src_reg, src_reg, scale_reg, msk_reg);
                Sub(src_reg, src_reg, lse_reg, msk_reg);
                Exp(src_reg, src_reg, msk_reg);
                StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstTensor + ub_offset, src_reg, msk_reg);
                ub_offset += repeat_size;
            }
        }
    }

    __simd_vf__ inline void ComputeSoftmaxGrad(__ubuf__ float *dstTensor, __ubuf__ float *src0Tensor,
                                               __ubuf__ float *src1Tensor, __ubuf__ float *sftFrontTensor,
                                               const uint32_t row, const uint32_t col)
    {
        AscendC::Reg::RegTensor<float> src_reg;
        AscendC::Reg::RegTensor<float> sft_front_reg;
        AscendC::Reg::RegTensor<float> mul_reg;
        AscendC::Reg::MaskReg msk_reg;
        constexpr static uint16_t repeat_size = 256 / sizeof(float);
        uint16_t repeat_times = (col + repeat_size - 1) / repeat_size;
        uint32_t ub_offset = 0;

        for (int32_t i = 0; i < row; i++) {
            LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(sft_front_reg, sftFrontTensor + i * 8);
            uint32_t count = col;
            ub_offset = i * col;

            for (int32_t j = 0; j < repeat_times; j++) {
                msk_reg = AscendC::Reg::UpdateMask<float>(count);

                LoadAlign(src_reg, src0Tensor + ub_offset);
                Sub(src_reg, src_reg, sft_front_reg, msk_reg);
                LoadAlign(mul_reg, src1Tensor + ub_offset);
                Mul(src_reg, src_reg, mul_reg, msk_reg);
                StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstTensor + ub_offset, src_reg, msk_reg);
                ub_offset += repeat_size;
            }
        }
    }

    __aicore__ inline void ComputeEvenCoreInfo(EvenCoreInfo &info, const uint32_t data_size,
                                               const uint32_t max_process_size)
    {
        /*
         * function: even process data_size
         */
        uint32_t per_core_size = CeilDiv<uint32_t>(data_size, v_core_num_);
        info.start_idx = v_core_idx_ * per_core_size;
        info.max_process_size = max_process_size;
        info.data_size = data_size;
        // Tail cores: start_idx can be >= data_size (e.g. T=1024, 56 AIVs, per=19).
        // Must not do uint32 (data_size - start_idx) underflow.
        if (info.start_idx >= data_size) {
            info.len = 0;
            info.loop_num = 0;
            info.align_tail = 0;
            info.pad_tail = 0;
            return;
        }
        info.len = (info.start_idx + per_core_size > data_size) ? data_size - info.start_idx : per_core_size;
        info.loop_num = CeilDiv<uint32_t>(info.len, max_process_size);

        uint32_t tail = info.len % max_process_size;
        // 由于DataCopyPad最多处理65535，因此tail部分分成align_tail和pad_tail计算
        if (tail == 0) {
            info.align_tail = max_process_size;
            info.pad_tail = 0;
        } else {
            info.align_tail = tail / C0_SIZE * C0_SIZE;
            info.pad_tail = tail - info.align_tail;
        }
    }

    __aicore__ inline void CopyInLSE(const LocalTensor<float> &dstTensor, const GlobalTensor<float> &srcTensor,
                                     const int32_t count)
    {
        /*
         * function: Copy lse from global memory to local memory
         * input shape：(b, n, s, 1) or (t, n, 1)
         * out shape:   (s, 8)
         * dtype:       float
         */
        uint32_t src_stride;
        if constexpr (INPUT_LAYOUT == BSND) {
            src_stride = 0;
        } else if (INPUT_LAYOUT == BNSD) {
            src_stride = 0;
        } else if (INPUT_LAYOUT == TND) {
            src_stride = (q_head_num_ - 1) * sizeof(float);
        }

        DataCopyPad(dstTensor, srcTensor,
                    {static_cast<uint16_t>(count), static_cast<uint32_t>(1 * sizeof(float)),
                     static_cast<uint32_t>(src_stride), 0, 0},
                    {false, 0, 0, 0});
    }
};

} // namespace GSAG_ARC35
