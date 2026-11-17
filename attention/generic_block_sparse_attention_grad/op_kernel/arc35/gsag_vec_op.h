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
#include "gsag_common_header.h"
#include "vector_api/gsag_vf_cast_nd2nz.h"
using namespace AscendC;

namespace GSAG_ARC35 {

struct EvenCoreInfo {
    uint64_t data_size{0};        // 总共需要处理的元素个数
    uint64_t start_idx{0};        // 每个core处理的起始idx
    uint64_t len{0};              // 每个core处理的元素个数
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
    int32_t q_head_num_;
    int32_t kv_head_num_;
    int32_t head_dim_;
    int32_t head_dim_align_;
    float softmax_scale_{0.0f};
    uint32_t base_m;
    uint32_t vec_base_m;
    uint32_t vec_base_n;
    GM_ADDR act_seq_q_len;
    GlobalTensor<float> lse_gm_;
    GlobalTensor<float> sftg_workspace_;
    GlobalTensor<float> dq_workspace_;
    GlobalTensor<float> dq_sel_workspace_;
    GlobalTensor<float> det_dq_workspace_;
    GlobalTensor<int32_t> sparse_idx_gm_;
    uint64_t dq_sel_workspace_offset_{0};
    int64_t dq_sel_core_elems_{0};
    int64_t dq_size_{0};
    uint32_t det_core_num_{0};
    static constexpr uint32_t DETER_INTERVAL_COUNT = DETERMINISTIC_ENABLE ? AIC_CORE_MAX_NUM : 1;
    int64_t det_begin_[DETER_INTERVAL_COUNT]{};
    int64_t det_end_[DETER_INTERVAL_COUNT]{};
    int64_t det_base_[DETER_INTERVAL_COUNT]{};
    int64_t det_elems_{0};
    bool compact_dq_{false};
    bool direct_dq_{false};
    int64_t cube_core_idx_{0};
    uint32_t cube_core_num_{0};
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
    static constexpr uint32_t SHAPE_RANK_2D = 2;
    static constexpr uint32_t BLOCK_FP32 = BLOCK_SIZE / sizeof(float);
    static constexpr uint32_t BLOCK_INPUT = BLOCK_SIZE / sizeof(INPUT_TYPE);
    constexpr static uint32_t POST_TILE_LEN = 20 * 1024; // POST一次处理元素的个数
    int32_t s1_process_;
    int32_t s1_process_align_;
    int32_t s2_process_align_;
    int32_t half_s1_process_align_;
    int32_t half_s1_process_real_;
    int64_t l1_offset_;
    TEventID event_ping_ = EVENT_ID3;
    TEventID event_pong_ = EVENT_ID4;
    TEventID event_id;

public:
    __aicore__ inline VecOp(){};

    __aicore__ inline void Init(GM_ADDR softmaxLse, GM_ADDR sparseBlockIdx, GM_ADDR actualQseqlen, GM_ADDR workspace,
                                const TILING_CLASS* tilingData, TBuf<TPosition::VECCALC>& ub_buffer, uint32_t ub_offset)
    {
        this->v_core_num_ = tilingData->cubeCoreNum * 2; // 2 is the v_core_num_
        this->batch_num_ = tilingData->batchNum;
        this->q_seq_len_ = tilingData->qSeqLen;
        this->q_head_num_ = tilingData->qHeadNum;
        this->kv_head_num_ = tilingData->kvHeadNum;
        this->head_dim_ = tilingData->headDim;
        this->softmax_scale_ = tilingData->softmaxScale;
        this->head_dim_align_ = RoundUp(head_dim_, static_cast<int32_t>(C0_SIZE));
        this->base_m = tilingData->baseM;
        this->vec_base_m = tilingData->baseM / 2; // 2 is the vec_base_m
        this->vec_base_n = tilingData->baseN;
        this->act_seq_q_len = actualQseqlen;
        v_core_idx_ = GetBlockIdx();
        v_sub_core_idx_ = GetSubBlockIdx();
        cube_core_idx_ = static_cast<int64_t>(GetBlockIdx() / 2);
        cube_core_num_ = static_cast<uint32_t>(tilingData->cubeCoreNum);
        dq_size_ = static_cast<int64_t>(tilingData->dqSize);
        if constexpr (DETERMINISTIC_ENABLE) {
            compact_dq_ = static_cast<int64_t>(batch_num_) * kv_head_num_ > 1;
        }

        const int64_t sparseIdxElems =
            static_cast<int64_t>(tilingData->batchNum) * tilingData->kvHeadNum * tilingData->numJ * tilingData->maxS1;
        const int64_t lseElems =
            (INPUT_LAYOUT == TND) ?
                static_cast<int64_t>(tilingData->qSeqLen) * tilingData->qHeadNum :
                static_cast<int64_t>(tilingData->batchNum) * tilingData->qHeadNum * tilingData->qSeqLen;
        const int64_t sftgElems = ((lseElems * 8) + 255) / 256 * 256;
        lse_gm_.SetGlobalBuffer((__gm__ float*)softmaxLse, lseElems);
        sftg_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->sftgWorkspaceOffset), sftgElems);
        dq_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->dqWorkspaceOffset), tilingData->dqSize);
        dq_sel_workspace_offset_ = tilingData->dqSelWorkspaceOffset;
        dq_sel_core_elems_ = static_cast<int64_t>(base_m) * head_dim_align_;
        dq_sel_workspace_.SetGlobalBuffer(
            (__gm__ float*)(workspace + dq_sel_workspace_offset_) + cube_core_idx_ * 2 * dq_sel_core_elems_,
            2 * dq_sel_core_elems_);
        if constexpr (DETERMINISTIC_ENABLE) {
            const int64_t detElems =
                static_cast<int64_t>(tilingData->detDqWorkspaceSize) / static_cast<int64_t>(sizeof(float));
            det_dq_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->detDqWorkspaceOffset), detElems);
            det_core_num_ = static_cast<uint32_t>(tilingData->detCoreNum);
            if (det_core_num_ == 0) {
                det_core_num_ = cube_core_num_;
            }
        }
        sparse_idx_gm_.SetGlobalBuffer((__gm__ int32_t*)sparseBlockIdx, sparseIdxElems);
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

    __aicore__ inline uint32_t GetUsedCubeCoreNum(GM_ADDR metadata)
    {
        uint32_t usedCoreNum = cube_core_num_;
        if (metadata != nullptr) {
            const __gm__ int32_t* metaPtr = reinterpret_cast<__gm__ int32_t*>(metadata);
            usedCoreNum = static_cast<uint32_t>(metaPtr[USED_CORE_NUM]);
        }
        if (usedCoreNum == 0) {
            usedCoreNum = 1;
        }
        if (usedCoreNum > cube_core_num_) {
            usedCoreNum = cube_core_num_;
        }
        return usedCoreNum;
    }

    __aicore__ inline int64_t DeterGroupOffset(int32_t b, int32_t n2)
    {
        const int64_t seq =
            INPUT_LAYOUT == TND ? GetSeqLen(b, reinterpret_cast<__gm__ uint8_t*>(act_seq_q_len)) : q_seq_len_;
        const int64_t prefix = INPUT_LAYOUT == TND ? GetSeqPrefix(b, reinterpret_cast<__gm__ uint8_t*>(act_seq_q_len)) :
                                                     static_cast<int64_t>(b) * q_seq_len_;
        return (prefix * q_head_num_ + static_cast<int64_t>(n2) * (q_head_num_ / kv_head_num_) * seq) * head_dim_;
    }

    __aicore__ inline void InitDeterIntervals(GM_ADDR metadata)
    {
        const __gm__ int32_t* meta = reinterpret_cast<__gm__ int32_t*>(metadata);
        const uint32_t used = GetUsedCubeCoreNum(metadata);
        det_elems_ = 0;
        for (uint32_t c = 0; c < used; ++c) {
            const int32_t first = meta[CORE_TASK_START_OFFSET + c];
            const int32_t end = meta[CORE_TASK_END_OFFSET + c];
            det_base_[c] = det_elems_;
            if (first < end) {
                const int64_t lo = TASK_LIST_OFFSET + static_cast<int64_t>(first) * TASK_ENTRY_SIZE;
                const int64_t hi = TASK_LIST_OFFSET + static_cast<int64_t>(end - 1) * TASK_ENTRY_SIZE;
                det_begin_[c] = DeterGroupOffset(meta[lo + TASK_B], meta[lo + TASK_N2]);
                det_end_[c] = DeterGroupOffset(meta[hi + TASK_B], meta[hi + TASK_N2] + 1);
                det_elems_ += det_end_[c] - det_begin_[c];
            }
        }
    }

    __aicore__ inline void SendVecPre(const GlobalTensor<float>& dq_workspace, const GlobalTensor<float>& dk_workspace,
                                      const GlobalTensor<float>& dv_workspace, const TILING_CLASS* tilingData,
                                      TBuf<TPosition::VECCALC>& ub_buffer, GM_ADDR metadata = nullptr)
    {
        constexpr static uint32_t PRE_TILE_LEN = 20 * 1024; // PRE一次处理元素的个数
        EvenCoreInfo info;
        vec_in_ping_ = ub_buffer.GetWithOffset<float>(PRE_TILE_LEN, 0);
        Duplicate(vec_in_ping_, (float)0.0, PRE_TILE_LEN);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        // dk/dv always need clear (AtomicAdd / Fixpipe accumulate into workspace).
        ComputeEvenCoreInfo(info, tilingData->dkSize, PRE_TILE_LEN);
        ComputeVecPre(dk_workspace, vec_in_ping_, info);
        ComputeVecPre(dv_workspace, vec_in_ping_, info);
        if constexpr (DETERMINISTIC_ENABLE) {
            // VecDeterReducePostDq writes the output directly; dq_workspace is unused.
            uint32_t usedCoreNum = GetUsedCubeCoreNum(metadata);
            if (usedCoreNum > det_core_num_) {
                usedCoreNum = det_core_num_;
            }
            const __gm__ int32_t* meta = reinterpret_cast<__gm__ int32_t*>(metadata);
            if (meta[0] == 0) { // TOTAL_NUM: empty tasks need only one dense zero slot.
                compact_dq_ = false;
            }
            // P0: single cube core (or a single nonempty partition) is already a
            // unique writer for every dQ element — skip ordered multi-core reduce.
            if (usedCoreNum <= 1) {
                direct_dq_ = true;
                det_elems_ = dq_size_;
            } else if (compact_dq_) {
                InitDeterIntervals(metadata);
                // Ordered task partitions own disjoint output head groups iff
                // their canonical intervals do not overlap. No Q can then be
                // updated by different cube cores, regardless of sparse indices.
                direct_dq_ = true;
                int64_t previousEnd = 0;
                uint32_t nonempty = 0;
                for (uint32_t c = 0; c < usedCoreNum; ++c) {
                    if (det_end_[c] <= det_begin_[c]) {
                        continue;
                    }
                    ++nonempty;
                    if (det_begin_[c] < previousEnd) {
                        direct_dq_ = false;
                    }
                    previousEnd = det_end_[c];
                }
                if (nonempty <= 1) {
                    direct_dq_ = true;
                }
                if (direct_dq_) {
                    det_elems_ = dq_size_;
                }
            } else {
                det_elems_ = static_cast<int64_t>(usedCoreNum) * dq_size_;
            }
            ComputeEvenCoreInfo(info, static_cast<uint64_t>(det_elems_), PRE_TILE_LEN);
            ComputeVecPre(det_dq_workspace_, vec_in_ping_, info);
        } else {
            ComputeEvenCoreInfo(info, tilingData->dqSize, PRE_TILE_LEN);
            ComputeVecPre(dq_workspace, vec_in_ping_, info);
        }
    }

    __aicore__ inline void SendVecSftgFront(const GlobalTensor<INPUT_TYPE>& dy_gm,
                                            const GlobalTensor<INPUT_TYPE>& out_gm,
                                            const GlobalTensor<float>& sftg_workspace, const TILING_CLASS* tilingData,
                                            TBuf<TPosition::VECCALC>& ub_buffer)
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
                ComputeEvenCoreInfo(info, current_q_seqlen, process_s1_size, false);

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

    __aicore__ inline void SendVecSftPreProcess(const RunTimeInfo& runTimeInfo, uint32_t pingpong_idx)
    {
        s1_process_ = runTimeInfo.s1Len;
        s1_process_align_ = runTimeInfo.s1LenAlign;
        s2_process_align_ = runTimeInfo.s2LenAlign;
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

        lse_tensor_ = pingpong_idx == 0 ? lse_tensor_ping_ : lse_tensor_pong_;
        sftg_front_tensor_ = pingpong_idx == 0 ? sftg_front_tensor_ping_ : sftg_front_tensor_pong_;
        event_id = pingpong_idx == 0 ? event_ping_ : event_pong_;
        WAIT_FLAG(V, MTE2, event_id);

        const int32_t packElems = half_s1_process_align_ * static_cast<int32_t>(BLOCK_FP32);
        Duplicate(lse_tensor_, 0.0f, packElems);
        Duplicate(sftg_front_tensor_, 0.0f, packElems);
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);

        if (half_s1_process_real_ > 0) {
            // SoftmaxGradFront stores every physical Q token. sequsedQ limits
            // computation, but must not change the per-head workspace stride.
            int64_t sftgQSeqLen = runTimeInfo.cur_q_seq_len;
            if constexpr (INPUT_LAYOUT == TND) {
                sftgQSeqLen = GetSeqLen(runTimeInfo.bIdx, act_seq_q_len);
            }
            LocalTensor<int32_t> idxUb = softmax_res_nz_tensor_.template ReinterpretCast<int32_t>();
            const int32_t rowStart = v_sub_core_idx_ * half_s1_process_align_;
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
                    lseOff =
                        (static_cast<int64_t>(runTimeInfo.last_q_seq_sum) + qTok) * q_head_num_ + runTimeInfo.n1Idx;
                    sftgOff = GetSftgGmOffset<INPUT_LAYOUT>(
                        static_cast<int64_t>(runTimeInfo.last_q_seq_sum), sftgQSeqLen, q_head_num_,
                        static_cast<int64_t>(runTimeInfo.bIdx), static_cast<int64_t>(qTok),
                        static_cast<int64_t>(runTimeInfo.n1Idx));
                } else {
                    lseOff = static_cast<int64_t>(runTimeInfo.bIdx) * q_head_num_ * runTimeInfo.cur_q_seq_len +
                             runTimeInfo.n1Idx * runTimeInfo.cur_q_seq_len + qTok;
                    sftgOff = lseOff * 8; // 8 is the block size for the sftg_front_tensor
                }
                SET_FLAG(S, MTE2, EVENT_ID1);
                WAIT_FLAG(S, MTE2, EVENT_ID1);
                DataCopy(sftg_front_tensor_[r * static_cast<int32_t>(BLOCK_FP32)], sftg_workspace_[sftgOff],
                         BLOCK_FP32);
                DataCopyPad(lse_tensor_[r * static_cast<int32_t>(BLOCK_FP32)], lse_gm_[lseOff],
                            {static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0},
                            {false, 0, 0, 0});
            }
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }
    }

    __aicore__ inline void ApplyCausalAttenMask(const LocalTensor<float>& sUb, const RunTimeInfo& runTimeInfo)
    {
        if (runTimeInfo.mask_type == 0 || half_s1_process_real_ <= 0) {
            return;
        }
        const int32_t rowStart = v_sub_core_idx_ * half_s1_process_align_;
        const int32_t s2Start = runTimeInfo.s2Idx;
        const int32_t s2Len = static_cast<int32_t>(runTimeInfo.s2Len);
        const int32_t causalOffset = runTimeInfo.cur_kv_seq_len - runTimeInfo.cur_q_seq_len;
        constexpr int32_t kMaxHalfS1 = 128;
        int32_t firstInvalidArr[kMaxHalfS1];
        const int32_t nRow = half_s1_process_real_ < kMaxHalfS1 ? half_s1_process_real_ : kMaxHalfS1;

        LocalTensor<int32_t> idxUb = softmax_res_nz_tensor_.template ReinterpretCast<int32_t>();
        DataCopy(idxUb, sparse_idx_gm_[runTimeInfo.sparseIdxOffset + rowStart], half_s1_process_align_);
        SET_FLAG(MTE2, S, EVENT_ID1);
        WAIT_FLAG(MTE2, S, EVENT_ID1);

        for (int32_t r = 0; r < nRow; ++r) {
            const int32_t qTok = idxUb.GetValue(r);
            int32_t firstInvalid = qTok + causalOffset - s2Start + 1;
            if (firstInvalid < 0) {
                firstInvalid = 0;
            }
            if (firstInvalid > s2Len) {
                firstInvalid = s2Len;
            }
            firstInvalidArr[r] = firstInvalid;
        }
        SET_FLAG(S, V, EVENT_ID1);
        WAIT_FLAG(S, V, EVENT_ID1);

        __ubuf__ float* sPtr = (__ubuf__ float*)sUb.GetPhyAddr();
        for (int32_t r = 0; r < nRow; ++r) {
            if (firstInvalidArr[r] >= s2Len) {
                continue;
            }
            FillNegInfRange(sPtr + r * s2_process_align_, static_cast<uint32_t>(firstInvalidArr[r]),
                            static_cast<uint32_t>(s2Len), static_cast<uint32_t>(s2_process_align_));
        }
    }

    __aicore__ inline void SendVecSoftmax(const LocalTensor<INPUT_TYPE>& dst_l1_tensor,
                                          const LocalTensor<float>& src_ub_tensor, const RunTimeInfo& runTimeInfo)
    {
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

        const bool scaleBeforeMask = softmax_scale_ < 1.0e-6f;
        if (scaleBeforeMask) {
            Muls(src_ub_tensor, src_ub_tensor, softmax_scale_,
                 static_cast<uint32_t>(half_s1_process_real_ * s2_process_align_));
            PipeBarrier<PIPE_V>();
        }
        ApplyCausalAttenMask(src_ub_tensor, runTimeInfo);
        PipeBarrier<PIPE_V>();
        if (scaleBeforeMask) {
            SimpleSoftmax<true>((__ubuf__ float*)src_ub_tensor.GetPhyAddr(),
                                (__ubuf__ float*)src_ub_tensor.GetPhyAddr(), (__ubuf__ float*)lse_tensor_.GetPhyAddr(),
                                half_s1_process_real_, s2_process_align_);
        } else {
            SimpleSoftmax<false>((__ubuf__ float*)src_ub_tensor.GetPhyAddr(),
                                 (__ubuf__ float*)src_ub_tensor.GetPhyAddr(), (__ubuf__ float*)lse_tensor_.GetPhyAddr(),
                                 half_s1_process_real_, s2_process_align_);
        }
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

    __aicore__ inline void SendVecSoftmaxGrad(const LocalTensor<INPUT_TYPE>& dst_l1_tensor,
                                              const LocalTensor<float>& softmax_ub_tensor,
                                              const LocalTensor<float>& src_ub_tensor, const RunTimeInfo& runTimeInfo)
    {
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
        if (half_s1_process_real_ < half_s1_process_align_) {
            const int32_t padElems = (half_s1_process_align_ - half_s1_process_real_) * s2_process_align_;
            Duplicate(src_ub_tensor[half_s1_process_real_ * s2_process_align_], 0.0f, padElems);
            Duplicate(softmax_ub_tensor[half_s1_process_real_ * s2_process_align_], 0.0f, padElems);
            PipeBarrier<PIPE_V>();
        }
        ComputeSoftmaxGrad((__ubuf__ float*)src_ub_tensor.GetPhyAddr(), (__ubuf__ float*)src_ub_tensor.GetPhyAddr(),
                           (__ubuf__ float*)softmax_ub_tensor.GetPhyAddr(),
                           (__ubuf__ float*)sftg_front_tensor_.GetPhyAddr(), half_s1_process_align_, s2_process_align_);
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
    }

    __aicore__ inline void ScatterDqSel(const RunTimeInfo& runTimeInfo, TBuf<TPosition::VECCALC>& ub_buffer,
                                        const uint32_t ping_pong_idx)
    {
        if (!runTimeInfo.need_compute) {
            return;
        }
        const int32_t m = runTimeInfo.s1Len;
        if (m <= 0) {
            return;
        }

        constexpr int32_t UB_ROW_SIZE = 16;
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

        constexpr uint32_t SCATTER_UB_OFFSET = 200 * 1024;
        LocalTensor<float> batchUb = ub_buffer.GetWithOffset<float>(UB_ROW_SIZE * head_dim_align_, SCATTER_UB_OFFSET);

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
        const uint32_t coreIdx = static_cast<uint32_t>(cube_core_idx_);

        int64_t compactRowBase = 0;
        if constexpr (DETERMINISTIC_ENABLE) {
            if (compact_dq_ && !direct_dq_) {
                const int64_t seq = INPUT_LAYOUT == TND ?
                                        GetSeqLen(runTimeInfo.bIdx, reinterpret_cast<__gm__ uint8_t*>(act_seq_q_len)) :
                                        q_seq_len_;
                const int64_t prefix = INPUT_LAYOUT == TND ? runTimeInfo.last_q_seq_sum :
                                                             static_cast<int64_t>(runTimeInfo.bIdx) * q_seq_len_;
                compactRowBase = det_base_[coreIdx] - det_begin_[coreIdx] +
                                 (prefix * q_head_num_ + static_cast<int64_t>(runTimeInfo.n1Idx) * seq) * head_dim_;
            }
        }

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
                if constexpr (DETERMINISTIC_ENABLE) {
                    int64_t detOff = static_cast<int64_t>(coreIdx) * dq_size_ + gmOff;
                    if (direct_dq_) {
                        detOff = gmOff;
                    } else if (compact_dq_) {
                        detOff = compactRowBase + static_cast<int64_t>(qTok) * head_dim_;
                    }
                    DataCopy(det_dq_workspace_[detOff], batchUb[row * head_dim_], head_dim_);
                } else {
                    DataCopy(dq_workspace_[gmOff], batchUb[row * head_dim_], head_dim_);
                }
            }
            SET_FLAG(MTE3, MTE2, EVENT_ID0);
        }
        AscendC::SetAtomicNone();
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    }

    __aicore__ inline void LoadDeterDqTile(LocalTensor<float> dst, int64_t offset, uint32_t count)
    {
        if (count * sizeof(float) % BLOCK_SIZE == 0) {
            DataCopy(dst, det_dq_workspace_[offset], count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
            DataCopyPad(dst, det_dq_workspace_[offset], copyParams, padParams);
        }
    }

    __aicore__ inline void VecDeterReducePostDqDense(GM_ADDR metadata, const GlobalTensor<INPUT_TYPE>& dq_out_gm,
                                                     TBuf<TPosition::VECCALC>& ub_buffer)
    {
        if constexpr (DETERMINISTIC_ENABLE) {
            uint32_t usedCoreNum = GetUsedCubeCoreNum(metadata);
            if (usedCoreNum > det_core_num_) {
                usedCoreNum = det_core_num_;
            }

            constexpr uint32_t MAX_REDUCE_TILE_LEN = 16 * 1024;
            constexpr uint32_t STAGING_FLOATS = 54 * 1024;
            const bool batchLoad = usedCoreNum <= STAGING_FLOATS / C0_SIZE &&
                                   static_cast<uint64_t>(dq_size_) * sizeof(float) <= 0xFFFFFFFFULL;
            const uint32_t tilePerCore = STAGING_FLOATS / usedCoreNum / C0_SIZE * C0_SIZE;
            const uint32_t REDUCE_TILE_LEN =
                batchLoad && tilePerCore < MAX_REDUCE_TILE_LEN ? tilePerCore : MAX_REDUCE_TILE_LEN;
            const uint32_t stagingElems = batchLoad ? usedCoreNum * REDUCE_TILE_LEN : 2 * REDUCE_TILE_LEN;
            LocalTensor<float> accumUb = ub_buffer.GetWithOffset<float>(REDUCE_TILE_LEN, 0);
            LocalTensor<float> tempUb =
                ub_buffer.GetWithOffset<float>(REDUCE_TILE_LEN, REDUCE_TILE_LEN * sizeof(float));
            LocalTensor<float> stagingUb = ub_buffer.GetWithOffset<float>(stagingElems, 0);
            LocalTensor<INPUT_TYPE> castUb =
                ub_buffer.GetWithOffset<INPUT_TYPE>(REDUCE_TILE_LEN, stagingElems * sizeof(float));

            EvenCoreInfo info;
            ComputeEvenCoreInfo(info, static_cast<uint64_t>(dq_size_), REDUCE_TILE_LEN);
            if (info.start_idx >= info.data_size || info.len == 0) {
                return;
            }

            uint32_t processSize = info.max_process_size;
            for (uint32_t i = 0; i < info.loop_num; ++i) {
                if (unlikely(i == info.loop_num - 1)) {
                    processSize = info.align_tail + info.pad_tail;
                    if (processSize == 0) {
                        break;
                    }
                }
                const int64_t gmOffset = static_cast<int64_t>(info.start_idx) +
                                         static_cast<int64_t>(i) * static_cast<int64_t>(info.max_process_size);
                const uint32_t alignedCount = RoundUp(processSize, static_cast<uint32_t>(C0_SIZE));

                if (batchLoad) {
                    DataCopyExtParams copyParams{
                        static_cast<uint16_t>(usedCoreNum), static_cast<uint32_t>(processSize * sizeof(float)),
                        static_cast<uint32_t>((dq_size_ - processSize) * sizeof(float)),
                        static_cast<uint32_t>((REDUCE_TILE_LEN - alignedCount) * sizeof(float) / BLOCK_SIZE), 0};
                    DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(alignedCount - processSize), 0};
                    DataCopyPad(stagingUb, det_dq_workspace_[gmOffset], copyParams, padParams);
                } else {
                    LoadDeterDqTile(accumUb, gmOffset, processSize);
                }
                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);

                for (uint32_t c = 1; c < usedCoreNum; ++c) {
                    if (batchLoad) {
                        Add(accumUb, accumUb, stagingUb[c * REDUCE_TILE_LEN], alignedCount);
                        PipeBarrier<PIPE_V>();
                        continue;
                    }
                    SET_FLAG(V, MTE2, EVENT_ID0);
                    WAIT_FLAG(V, MTE2, EVENT_ID0);
                    const int64_t srcOff = static_cast<int64_t>(c) * dq_size_ + gmOffset;
                    LoadDeterDqTile(tempUb, srcOff, processSize);
                    SET_FLAG(MTE2, V, EVENT_ID0);
                    WAIT_FLAG(MTE2, V, EVENT_ID0);
                    Add(accumUb, accumUb, tempUb, alignedCount);
                    PipeBarrier<PIPE_V>();
                }

                Muls(accumUb, accumUb, softmax_scale_, alignedCount);
                PipeBarrier<PIPE_V>();
                Cast(castUb, accumUb, AscendC::RoundMode::CAST_RINT, alignedCount);
                PipeBarrier<PIPE_V>();

                SET_FLAG(V, MTE3, EVENT_ID0);
                WAIT_FLAG(V, MTE3, EVENT_ID0);
                if (processSize * sizeof(INPUT_TYPE) % BLOCK_SIZE == 0) {
                    DataCopy(dq_out_gm[gmOffset], castUb, processSize);
                } else {
                    DataCopyExtParams copyParams{1, static_cast<uint32_t>(processSize * sizeof(INPUT_TYPE)), 0, 0, 0};
                    DataCopyPad(dq_out_gm[gmOffset], castUb, copyParams);
                }
                SET_FLAG(MTE3, MTE2, EVENT_ID0);
                WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
            }
        } else {
            (void)metadata;
            (void)dq_out_gm;
            (void)ub_buffer;
        }
    }

    __aicore__ inline void VecDeterReducePostDq(GM_ADDR metadata, const GlobalTensor<INPUT_TYPE>& dq_out_gm,
                                                TBuf<TPosition::VECCALC>& ub_buffer)
    {
        if constexpr (DETERMINISTIC_ENABLE) {
            if (direct_dq_) {
                return; // SendVecPost reads the shared output-layout buffer.
            }
            uint32_t usedCoreNum = GetUsedCubeCoreNum(metadata);
            if (usedCoreNum > det_core_num_) {
                usedCoreNum = det_core_num_;
            }
            // P0 belt-and-suspenders: one writer → dense single-slot reduce (no cross-core Add).
            if (usedCoreNum <= 1) {
                VecDeterReducePostDqDense(metadata, dq_out_gm, ub_buffer);
                return;
            }
            if (!compact_dq_) {
                VecDeterReducePostDqDense(metadata, dq_out_gm, ub_buffer);
                return;
            }

            constexpr uint32_t MAX_TILE = 16 * 1024;
            constexpr uint32_t STAGING_FLOATS = 54 * 1024;
            const uint32_t tile = (STAGING_FLOATS / (usedCoreNum + 1) / C0_SIZE) * C0_SIZE;
            const uint32_t tileLen = tile < MAX_TILE ? tile : MAX_TILE;
            const uint32_t stagingElems = (usedCoreNum + 1) * tileLen;
            LocalTensor<float> staging = ub_buffer.GetWithOffset<float>(stagingElems, 0);
            LocalTensor<float> accum = staging;
            LocalTensor<float> zero = staging[usedCoreNum * tileLen];
            LocalTensor<INPUT_TYPE> cast = ub_buffer.GetWithOffset<INPUT_TYPE>(tileLen, stagingElems * sizeof(float));
            EvenCoreInfo info;
            ComputeEvenCoreInfo(info, static_cast<uint64_t>(dq_size_), tileLen);
            int64_t offset = static_cast<int64_t>(info.start_idx);
            const int64_t limit = offset + static_cast<int64_t>(info.len);
            int32_t b = 0;
            while (offset < limit) {
                int64_t seq = q_seq_len_;
                int64_t prefix = 0;
                if constexpr (INPUT_LAYOUT == TND) {
                    while (b < batch_num_ && DeterGroupOffset(b, kv_head_num_) <= offset) {
                        ++b;
                    }
                    seq = GetSeqLen(b, reinterpret_cast<__gm__ uint8_t*>(act_seq_q_len));
                    prefix = GetSeqPrefix(b, reinterpret_cast<__gm__ uint8_t*>(act_seq_q_len));
                } else {
                    b = offset / (static_cast<int64_t>(q_head_num_) * q_seq_len_ * head_dim_);
                    prefix = static_cast<int64_t>(b) * q_seq_len_;
                }
                const int64_t inBatch = offset - prefix * q_head_num_ * head_dim_;
                const int64_t head = inBatch / (seq * head_dim_);
                const int64_t inHead = inBatch - head * seq * head_dim_;
                uint32_t count = tileLen;
                if (limit - offset < count) {
                    count = limit - offset;
                }
                if (seq * head_dim_ - inHead < count) {
                    count = seq * head_dim_ - inHead;
                }
                const uint32_t aligned = RoundUp(count, static_cast<uint32_t>(C0_SIZE));
                Duplicate(zero, 0.0f, aligned);
                if (!(det_begin_[0] <= offset && offset < det_end_[0])) {
                    Duplicate(accum, 0.0f, aligned);
                }
                SET_FLAG(V, MTE2, EVENT_ID0);
                WAIT_FLAG(V, MTE2, EVENT_ID0);
                for (uint32_t c = 0; c < usedCoreNum; ++c) {
                    if (det_begin_[c] <= offset && offset < det_end_[c]) {
                        const int64_t src = det_base_[c] + offset - det_begin_[c];
                        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
                        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(aligned - count), 0};
                        DataCopyPad(staging[c * tileLen], det_dq_workspace_[src], copy, pad);
                    }
                }
                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);
                // Adding +0 repeatedly is idempotent. Keep the first add in
                // each absent run (including signed-zero normalization), while
                // preserving the order of every actual core contribution.
                bool addedZero = false;
                for (uint32_t c = 1; c < usedCoreNum; ++c) {
                    if (det_begin_[c] <= offset && offset < det_end_[c]) {
                        Add(accum, accum, staging[c * tileLen], aligned);
                        addedZero = false;
                    } else if (!addedZero) {
                        Add(accum, accum, zero, aligned);
                        addedZero = true;
                    } else {
                        continue;
                    }
                    PipeBarrier<PIPE_V>();
                }
                Muls(accum, accum, softmax_scale_, aligned);
                PipeBarrier<PIPE_V>();
                Cast(cast, accum, AscendC::RoundMode::CAST_RINT, aligned);
                SET_FLAG(V, MTE3, EVENT_ID0);
                WAIT_FLAG(V, MTE3, EVENT_ID0);
                if constexpr (INPUT_LAYOUT == BNSD) {
                    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(INPUT_TYPE)), 0, 0, 0};
                    DataCopyPad(dq_out_gm[offset], cast, copy);
                } else {
                    // A tile may start/end inside one D row due to AIV partitioning.
                    // Split partial rows; full rows use one strided DMA.
                    uint32_t done = 0;
                    while (done < count) {
                        const int64_t rowElem = (inHead + done) % head_dim_;
                        uint32_t elems = head_dim_ - rowElem;
                        uint32_t rows = 1;
                        if (rowElem == 0 && count - done >= static_cast<uint32_t>(head_dim_)) {
                            rows = (count - done) / head_dim_;
                            elems = head_dim_;
                        } else if (count - done < elems) {
                            elems = count - done;
                        }
                        const int64_t token = (inHead + done) / head_dim_;
                        const int64_t dst = (prefix + token) * q_head_num_ * head_dim_ + head * head_dim_ + rowElem;
                        DataCopyExtParams copy{
                            static_cast<uint16_t>(rows), static_cast<uint32_t>(elems * sizeof(INPUT_TYPE)), 0,
                            static_cast<uint32_t>((q_head_num_ * head_dim_ - elems) * sizeof(INPUT_TYPE)), 0};
                        DataCopyPad(dq_out_gm[dst], cast[done], copy);
                        done += rows * elems;
                    }
                }
                SET_FLAG(MTE3, MTE2, EVENT_ID0);
                WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
                offset += count;
            }
        } else {
            (void)metadata;
            (void)dq_out_gm;
            (void)ub_buffer;
        }
    }

    __aicore__ inline void SendVecPost(const GlobalTensor<INPUT_TYPE>& dq_out_gm,
                                       const GlobalTensor<INPUT_TYPE>& dk_out_gm,
                                       const GlobalTensor<INPUT_TYPE>& dv_out_gm,
                                       const GlobalTensor<float>& dq_workspace, const GlobalTensor<float>& dk_workspace,
                                       const GlobalTensor<float>& dv_workspace, const TILING_CLASS* tilingData,
                                       TBuf<TPosition::VECCALC>& ub_buffer)
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
        if constexpr (DETERMINISTIC_ENABLE) {
            if (direct_dq_) {
                ComputeEvenCoreInfo(info, dq_size, POST_TILE_LEN);
                ComputeVecPost<true>(dq_out_gm, det_dq_workspace_, info, post_ping_pong_idx);
            }
        } else {
            ComputeEvenCoreInfo(info, dq_size, POST_TILE_LEN);
            ComputeVecPost<true>(dq_out_gm, dq_workspace, info, post_ping_pong_idx);
        }

        WAIT_FLAG(MTE3, MTE2, event_ping_);
        WAIT_FLAG(MTE3, MTE2, event_pong_);
    }

private:
    __aicore__ inline void ComputeVecPre(const GlobalTensor<float>& dst_tensor, const LocalTensor<float>& src_tensor,
                                         const EvenCoreInfo& info)
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

            const int64_t gm_offset = static_cast<int64_t>(info.start_idx) +
                                      static_cast<int64_t>(i) * static_cast<int64_t>(info.max_process_size);
            DataCopy(dst_tensor[gm_offset], src_tensor, process_size);
        }

        if (info.pad_tail > 0) {
            const int64_t gm_offset =
                static_cast<int64_t>(info.start_idx) +
                static_cast<int64_t>(info.loop_num - 1) * static_cast<int64_t>(info.max_process_size) +
                static_cast<int64_t>(info.align_tail);
            DataCopyParams copyParam;
            copyParam.blockCount = 1;
            copyParam.blockLen = info.pad_tail * sizeof(float);
            copyParam.srcStride = 0;
            copyParam.dstStride = 0;
            DataCopyPad(dst_tensor[gm_offset], src_tensor, copyParam);
        }
    }

    __aicore__ inline void ComputeSoftmaxGradFront(const GlobalTensor<float>& dst_gm,
                                                   const GlobalTensor<INPUT_TYPE>& dy_gm,
                                                   const GlobalTensor<INPUT_TYPE>& out_gm, const EvenCoreInfo& info,
                                                   const TILING_CLASS* tilingData, const uint32_t sftg_ping_pong_idx)
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
            const int64_t in_gm_offset = (static_cast<int64_t>(info.start_idx) +
                                          static_cast<int64_t>(i) * static_cast<int64_t>(info.max_process_size)) *
                                         static_cast<int64_t>(src_stride);

            DataCopyParams copyParam;
            copyParam.blockCount = process_size;
            copyParam.blockLen = head_dim_ * sizeof(INPUT_TYPE);
            copyParam.srcStride = (src_stride - head_dim_) * sizeof(INPUT_TYPE);
            copyParam.dstStride = 0;
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
            uint32_t out_shape_arry[SHAPE_RANK_2D] = {static_cast<uint32_t>(process_size),
                                                      static_cast<uint32_t>(BLOCK_FP32)};

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
            const int64_t out_gm_offset = (static_cast<int64_t>(info.start_idx) +
                                           static_cast<int64_t>(i) * static_cast<int64_t>(info.max_process_size)) *
                                          8;
            DataCopy(dst_gm[out_gm_offset], sftg_front_tensor, process_size * 8);
            SET_FLAG(MTE3, MTE2, EVENT_ID0);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        }
    }

    template <bool MULS>
    __aicore__ inline void ComputeVecPost(const GlobalTensor<INPUT_TYPE>& dst_tensor,
                                          const GlobalTensor<float>& src_tensor, const EvenCoreInfo& info,
                                          uint32_t& post_ping_pong_idx)
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
            const int64_t gm_offset = static_cast<int64_t>(info.start_idx) +
                                      static_cast<int64_t>(i) * static_cast<int64_t>(info.max_process_size);

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
            const int64_t gm_offset =
                static_cast<int64_t>(info.start_idx) +
                static_cast<int64_t>(info.loop_num - 1) * static_cast<int64_t>(info.max_process_size) +
                static_cast<int64_t>(info.align_tail);
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

    // Fill rowBuf[colStart, colEnd) with -inf via Reg (Arange + Compare + Select).
    __simd_vf__ inline void FillNegInfRange(__ubuf__ float* rowBuf, const uint32_t colStart, const uint32_t colEnd,
                                            const uint32_t colAlign)
    {
        if (colStart >= colEnd || colStart >= colAlign) {
            return;
        }
        constexpr float NEG_INF = -1.0e30f;
        constexpr uint32_t VF_ELEMS = 256U / sizeof(float);
        const uint32_t fillEnd = colEnd < colAlign ? colEnd : colAlign;
        const uint32_t blockStart = colStart & ~(VF_ELEMS - 1U);

        AscendC::Reg::MaskReg maskGeStart;
        AscendC::Reg::MaskReg maskLtEnd;
        AscendC::Reg::MaskReg maskPad;
        AscendC::Reg::MaskReg maskValid;
        AscendC::Reg::RegTensor<int32_t> regIdx;
        AscendC::Reg::RegTensor<float> regOld;
        AscendC::Reg::RegTensor<float> regNeg;
        AscendC::Reg::RegTensor<float> regOut;
        Duplicate(regNeg, NEG_INF);

        for (uint32_t blockBase = blockStart; blockBase < fillEnd; blockBase += VF_ELEMS) {
            uint32_t count = colAlign - blockBase;
            if (count > VF_ELEMS) {
                count = VF_ELEMS;
            }
            if (count == 0) {
                break;
            }
            maskValid = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::Arange(regIdx, static_cast<int32_t>(blockBase));
            LoadAlign(regOld, rowBuf + blockBase);
            AscendC::Reg::Compares<int32_t, CMPMODE::GE>(maskGeStart, regIdx, static_cast<int32_t>(colStart),
                                                         maskValid);
            AscendC::Reg::Compares<int32_t, CMPMODE::LT>(maskLtEnd, regIdx, static_cast<int32_t>(fillEnd), maskValid);
            AscendC::Reg::And(maskPad, maskGeStart, maskLtEnd, maskValid);
            AscendC::Reg::Select(regOut, regNeg, regOld, maskPad);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(rowBuf + blockBase, regOut, maskValid);
        }
    }

    template <bool SCORES_SCALED>
    __simd_vf__ inline void SimpleSoftmax(__ubuf__ float* dstTensor, __ubuf__ float* src0Tensor,
                                          __ubuf__ float* src1Tensor, const uint32_t row, const uint32_t col)
    {
        AscendC::Reg::RegTensor<float> src_reg;
        AscendC::Reg::RegTensor<float> lse_reg;
        AscendC::Reg::RegTensor<float> scale_reg;
        AscendC::Reg::MaskReg msk_reg;
        constexpr static uint16_t repeat_size = 256 / sizeof(float);
        uint16_t repeat_times = (col + repeat_size - 1) / repeat_size;
        uint32_t ub_offset = 0;
        if constexpr (!SCORES_SCALED) {
            Duplicate(scale_reg, softmax_scale_);
        }

        for (int32_t i = 0; i < row; i++) {
            LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(lse_reg, src1Tensor + i * 8);
            uint32_t count = col;
            ub_offset = i * col;

            for (int32_t j = 0; j < repeat_times; j++) {
                msk_reg = AscendC::Reg::UpdateMask<float>(count);

                LoadAlign(src_reg, src0Tensor + ub_offset);
                if constexpr (!SCORES_SCALED) {
                    Mul(src_reg, src_reg, scale_reg, msk_reg);
                }
                Sub(src_reg, src_reg, lse_reg, msk_reg);
                Exp(src_reg, src_reg, msk_reg);
                StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstTensor + ub_offset, src_reg, msk_reg);
                ub_offset += repeat_size;
            }
        }
    }

    __simd_vf__ inline void ComputeSoftmaxGrad(__ubuf__ float* dstTensor, __ubuf__ float* src0Tensor,
                                               __ubuf__ float* src1Tensor, __ubuf__ float* sftFrontTensor,
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

    __aicore__ inline void ComputeEvenCoreInfo(EvenCoreInfo& info, const uint64_t data_size,
                                               const uint32_t max_process_size, const bool align_start = true)
    {
        uint64_t per_core_size = CeilDiv<uint64_t>(data_size, static_cast<uint64_t>(v_core_num_));
        if (align_start) {
            per_core_size = RoundUp<uint64_t>(per_core_size, static_cast<uint64_t>(C0_SIZE));
        }
        info.start_idx = static_cast<uint64_t>(v_core_idx_) * per_core_size;
        info.max_process_size = max_process_size;
        info.data_size = data_size;
        if (info.start_idx >= data_size) {
            info.len = 0;
            info.loop_num = 0;
            info.align_tail = 0;
            info.pad_tail = 0;
            return;
        }
        info.len = (info.start_idx + per_core_size > data_size) ? data_size - info.start_idx : per_core_size;
        info.loop_num = static_cast<uint32_t>(CeilDiv<uint64_t>(info.len, static_cast<uint64_t>(max_process_size)));

        const uint64_t tail = info.len % static_cast<uint64_t>(max_process_size);
        if (tail == 0) {
            info.align_tail = max_process_size;
            info.pad_tail = 0;
        } else {
            info.align_tail = static_cast<uint32_t>(tail / C0_SIZE * C0_SIZE);
            info.pad_tail = static_cast<uint32_t>(tail - info.align_tail);
        }
    }
};

} // namespace GSAG_ARC35
