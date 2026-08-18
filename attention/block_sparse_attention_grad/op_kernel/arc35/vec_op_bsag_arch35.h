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
#include "vector_api/vf_cast_nd2nz.h"
using namespace AscendC;

namespace BSA_ARC35 {

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
    GlobalTensor<float> lse_gm_;
    GlobalTensor<float> sftg_workspace_;
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
    TEventID event_ping_ = EVENT_ID3;
    TEventID event_pong_ = EVENT_ID4;
    TEventID event_id;

public:
    __aicore__ inline VecOp(){};

    __aicore__ inline void Init(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape, GM_ADDR attentionMask,
                                GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                GM_ADDR workspace, const TILING_CLASS *tilingData, TBuf<TPosition::VECCALC> &ub_buffer,
                                uint32_t ub_offset)
    {
        this->v_core_num_ = tilingData->cubeCoreNum * 2;
        this->batch_num_ = tilingData->batchNum;
        this->q_seq_len_ = tilingData->qSeqLen;
        this->kv_seq_len_ = tilingData->kvSeqLen;
        this->q_group_ = tilingData->qGroup;
        this->q_head_num_ = tilingData->qHeadNum;
        this->kv_head_num_ = tilingData->kvHeadNum;
        this->head_dim_ = tilingData->headDim;
        this->softmax_scale_ = tilingData->softmaxScale;
        this->head_dim_align_ = RoundUp(head_dim_, 16);
        this->base_m = tilingData->baseM;
        this->base_n = tilingData->baseN;
        this->vec_base_m = tilingData->baseM / 2;
        this->vec_base_n = tilingData->baseN;
        this->act_seq_q_len = actualQseqlen;
        v_core_idx_ = GetBlockIdx();
        v_sub_core_idx_ = GetSubBlockIdx();

        // gm_tensor
        lse_gm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        sftg_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->sftgWorkspaceOffset));
        // local_tensor
        softmax_res_nz_tensor_ = ub_buffer.GetWithOffset<INPUT_TYPE>(vec_base_m * vec_base_n, ub_offset);
        ub_offset += vec_base_m * vec_base_n * sizeof(INPUT_TYPE);
        sftg_res_nz_tensor_ = ub_buffer.GetWithOffset<INPUT_TYPE>(vec_base_m * vec_base_n, ub_offset);
        ub_offset += vec_base_m * vec_base_n * sizeof(INPUT_TYPE);
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
        sftg_front_ping_ = ub_buffer.GetWithOffset<float>(process_s1_size * 8, ub_offset);
        ub_offset += process_s1_size * 8 * sizeof(float);
        sftg_front_pong_ = ub_buffer.GetWithOffset<float>(process_s1_size * 8, ub_offset);
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
                last_seq_total_len = b_idx == 0 ? 0 : GetSeqTotalLen(b_idx - 1, act_seq_q_len);
            } else {
                current_q_seqlen = q_seq_len_;
                last_seq_total_len = 0;
            }
            for (uint32_t n1_idx = 0; n1_idx < q_head_num_; n1_idx++) {
                int64_t in_gm_offset;
                int64_t out_gm_offset;

                if constexpr (INPUT_LAYOUT == BSND) {
                    in_gm_offset = b_idx * current_q_seqlen * q_head_num_ * head_dim_ + n1_idx * head_dim_;
                    out_gm_offset = b_idx * q_head_num_ * current_q_seqlen * 8 + n1_idx * current_q_seqlen * 8;
                } else if constexpr (INPUT_LAYOUT == BNSD) {
                    in_gm_offset =
                        b_idx * q_head_num_ * current_q_seqlen * head_dim_ + n1_idx * current_q_seqlen * head_dim_;
                    out_gm_offset = b_idx * q_head_num_ * current_q_seqlen * 8 + n1_idx * current_q_seqlen * 8;
                } else if constexpr (INPUT_LAYOUT == TND) {
                    in_gm_offset = last_seq_total_len * q_head_num_ * head_dim_ + n1_idx * head_dim_;
                    out_gm_offset = last_seq_total_len * q_head_num_ * 8 + n1_idx * current_q_seqlen * 8;
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
        half_s1_process_align_ = s1_process_align_ / 2;
        data_size = half_s1_process_align_ * s2_process_align_;
        half_s1_process_real_ = v_sub_core_idx_ == 0 ?
                                    half_s1_process_align_ :
                                    (s1_process_ - half_s1_process_align_); // GM CopyIn仍采用实际的mProcess
        sftg_gm_offset_ = runTimeInfo.sftgGmOffset + v_sub_core_idx_ * half_s1_process_align_ * 8;
        if constexpr (INPUT_LAYOUT == TND) {
            lse_gm_offset_ = runTimeInfo.lseGmOffset + v_sub_core_idx_ * half_s1_process_align_ * q_head_num_;
        } else {
            lse_gm_offset_ = runTimeInfo.lseGmOffset + v_sub_core_idx_ * half_s1_process_align_;
        }
        l1_offset_ = v_sub_core_idx_ * half_s1_process_align_ * C0_SIZE;

        if (half_s1_process_real_ > 0) {
            lse_tensor_ = pingpong_idx == 0 ? lse_tensor_ping_ : lse_tensor_pong_;
            sftg_front_tensor_ = pingpong_idx == 0 ? sftg_front_tensor_ping_ : sftg_front_tensor_pong_;
            event_id = pingpong_idx == 0 ? event_ping_ : event_pong_;

            WAIT_FLAG(V, MTE2, event_id);
            CopyInLSE(lse_tensor_, lse_gm_[lse_gm_offset_], half_s1_process_real_);
            DataCopy(sftg_front_tensor_, sftg_workspace_[sftg_gm_offset_], half_s1_process_real_ * 8);
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }
    }

    __aicore__ inline void SendVecSoftmax(const LocalTensor<INPUT_TYPE> &dst_l1_tensor,
                                          const LocalTensor<float> &src_ub_tensor, const RunTimeInfo &runTimeInfo)
    { /*
       * function: Compute simple softmax
       * input shape：[s1LenAlign / 2, s2LenAlign]
       * out shape:   [s1LenAlign / 2, s2LenAlign]
       * dtype:       float
       */
        if (half_s1_process_real_ <= 0) {
            return;
        }

        SimpleSoftmax((__ubuf__ float *)src_ub_tensor.GetPhyAddr(), (__ubuf__ float *)src_ub_tensor.GetPhyAddr(),
                      (__ubuf__ float *)lse_tensor_.GetPhyAddr(), half_s1_process_align_, s2_process_align_);

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
        if (half_s1_process_real_ <= 0) {
            return;
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
        uint32_t process_size = info.max_process_size; // 表示处理S1方向的元素个数
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
            DataCopyPad(dy_tensor, dy_gm[in_gm_offset], copyParam, {false, 0, 0, 0});
            DataCopyPad(attention_tensor, out_gm[in_gm_offset], copyParam, {false, 0, 0, 0});
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);

            Cast(dy_out_tensor, dy_tensor, RoundMode::CAST_NONE, process_size * head_dim_);
            Cast(attention_out_tensor, attention_tensor, RoundMode::CAST_NONE, process_size * head_dim_);
            PipeBarrier<PIPE_V>();

            uint32_t intput_shape_arry[2] = {static_cast<uint32_t>(process_size), static_cast<uint32_t>(head_dim_)};
            uint32_t out_shape_arry[2] = {static_cast<uint32_t>(process_size), static_cast<uint32_t>(8)};

            dy_out_tensor.SetShapeInfo(ShapeInfo(2, intput_shape_arry, AscendC::DataFormat::ND));
            attention_out_tensor.SetShapeInfo(ShapeInfo(2, intput_shape_arry, AscendC::DataFormat::ND));
            sftg_front_tensor.SetShapeInfo(ShapeInfo(2, out_shape_arry, AscendC::DataFormat::ND));
            bool isBasicBlock = process_size % 8 == 0;
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
        info.len = (info.start_idx + per_core_size > data_size) ? data_size - info.start_idx : per_core_size;
        info.max_process_size = max_process_size;
        info.data_size = data_size;
        info.loop_num = CeilDiv<uint32_t>(info.len, max_process_size);

        uint32_t tail = info.len % max_process_size;
        // 由于DataCopyPad最多处理65535，因此tail部分分成align_tail和pad_tail计算
        if (tail == 0) {
            info.align_tail = max_process_size;
            info.pad_tail = 0;
        } else {
            info.align_tail = tail / 16 * 16;
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

} // namespace BSA_ARC35
