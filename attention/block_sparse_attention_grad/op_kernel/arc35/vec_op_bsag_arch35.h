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
#include "vector_api/bsa_vf_cast_nd2nz.h"
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
    using ATTENTION_TYPE = typename BSA_TYPE::attention_type;
    using OUTPUT_TYPE = typename BSA_TYPE::output_type;
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
    static constexpr bool HIF8_ENABLE = BSA_TYPE::hif8_enable;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;

private:
    bool softmax_copy_pending_{false};
    bool combined_row_metadata_{false};
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
    GlobalTensor<INPUT_TYPE> refine_q_, refine_k_, refine_do_, refine_v_;
    LocalTensor<INPUT_TYPE> refine_bytes_a_, refine_bytes_b_;
    LocalTensor<float> refine_a_, refine_b_, refine_sum_;
    uint32_t refine_q_stride_, refine_k_stride_;
    float q_descale_{1.0f};
    float k_descale_{1.0f};
    float v_descale_{1.0f};
    float do_descale_{1.0f};
    float p_scale_{1.0f};
    float ds_scale_{1.0f};
    float score_scale_{0.0f};
    float dp_scale_{1.0f};
    float dv_post_scale_{1.0f};
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
    LocalTensor<uint8_t> hif8_gather_indexes_;
    LocalTensor<float> ds_amax_tensor_;
    GlobalTensor<float> ds_scale_workspace_;

    LocalTensor<float> vec_in_ping_;
    LocalTensor<float> vec_in_pong_;
    LocalTensor<OUTPUT_TYPE> vec_out_ping_;
    LocalTensor<OUTPUT_TYPE> vec_out_pong_;
    // sftg
    LocalTensor<INPUT_TYPE> dy_in_ping_;
    LocalTensor<INPUT_TYPE> dy_in_pong_;
    LocalTensor<ATTENTION_TYPE> attention_in_ping_;
    LocalTensor<ATTENTION_TYPE> attention_in_pong_;
    LocalTensor<float> dy_out_ping_;
    LocalTensor<float> dy_out_pong_;
    LocalTensor<float> attention_out_ping_;
    LocalTensor<float> attention_out_pong_;
    LocalTensor<float> sftg_front_ping_;
    LocalTensor<float> sftg_front_pong_;
    LocalTensor<uint8_t> sftg_tmp_tensor;
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t C0_SIZE = BLOCK_SIZE / sizeof(INPUT_TYPE);
    // DCache clean/invalidate works on 64-byte cache lines.  Give the two
    // AIV partial maxima and the shared descale independent cache lines.
    static constexpr uint32_t SCALE_SLOT_FLOATS = 64;
    static constexpr uint32_t SCALE_AMAX_STRIDE = 16;
    static constexpr uint32_t SCALE_DESCALE_INDEX = 32;
    static constexpr float HIF8_DYNAMIC_TARGET = 16384.0f;
    static constexpr uint32_t BLOCK_FP32 = BLOCK_SIZE / sizeof(float);
    static constexpr uint32_t BLOCK_INPUT = BLOCK_SIZE / sizeof(INPUT_TYPE);
    constexpr static uint32_t PRE_TILE_LEN = 60 * 1024;  // pre一次处理元素的个数
    constexpr static uint32_t POST_TILE_LEN = 20 * 1024; // POST一次处理元素的个数
    // runtInfo
    int32_t s1_process_align_;
    int32_t s2_process_align_;
    int32_t half_s1_process_align_;
    int32_t data_size;
    int32_t half_s1_process_real_;
    int64_t l1_offset_;
    TEventID event_ping_ = EVENT_ID3;
    TEventID event_pong_ = EVENT_ID4;
    TEventID event_id;

    bool direct_dkv_{false};
    GlobalTensor<float> direct_dk_zero_, direct_dv_zero_;

public:
    __aicore__ inline void SetDirectDkv(bool enabled, GM_ADDR dk, GM_ADDR dv)
    {
        direct_dkv_ = enabled;
        direct_dk_zero_.SetGlobalBuffer((__gm__ float*)dk);
        direct_dv_zero_.SetGlobalBuffer((__gm__ float*)dv);
    }

    __aicore__ inline VecOp(){};

    uint32_t dp_ready_flag_{FLAG_C2_V2};
    __aicore__ inline void SetDpReadyFlag(uint32_t flag)
    {
        dp_ready_flag_ = flag;
    }

    __aicore__ inline void Init(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape, GM_ADDR attentionMask,
                                GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR qDescale, GM_ADDR kDescale,
                                GM_ADDR vDescale, GM_ADDR doDescale, GM_ADDR pScale, GM_ADDR dsScale, GM_ADDR dq,
                                GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, const TILING_CLASS* tilingData,
                                TBuf<TPosition::VECCALC>& ub_buffer, uint32_t ub_offset)
    {
        this->v_core_num_ = tilingData->cubeCoreNum * 2;
        this->batch_num_ = tilingData->batchNum;
        this->q_seq_len_ = tilingData->qSeqLen;
        this->kv_seq_len_ = tilingData->kvSeqLen;
        this->q_group_ = tilingData->qGroup;
        this->q_head_num_ = tilingData->qHeadNum;
        this->kv_head_num_ = tilingData->kvHeadNum;
        this->head_dim_ = tilingData->headDim;
        // Select from immutable tiling data and sequence-length inputs. Do not
        // read NonZero's generated shape before the preparation barrier.
        // TND locality depends on each sequence, not the concatenated T.
        int64_t metadataSeqLen = tilingData->qSeqLen;
        if constexpr (HIF8_ENABLE && INPUT_LAYOUT == TND) {
            if (tilingData->batchNum > 1) {
                metadataSeqLen = 0;
                int64_t previous = 0;
                for (int64_t b = 0; b < tilingData->batchNum; ++b) {
                    int64_t end = ((__gm__ int64_t*)actualQseqlen)[b];
                    metadataSeqLen = (metadataSeqLen > end - previous ? metadataSeqLen : end - previous);
                    previous = end;
                }
            }
        }
        combined_row_metadata_ = HIF8_ENABLE && head_dim_ == 128 && (INPUT_LAYOUT == BNSD || metadataSeqLen < 131072) &&
                                 metadataSeqLen >= 4096;
        this->softmax_scale_ = tilingData->softmaxScale;
        if constexpr (HIF8_ENABLE) {
            refine_q_.SetGlobalBuffer((__gm__ INPUT_TYPE*)q);
            refine_k_.SetGlobalBuffer((__gm__ INPUT_TYPE*)k);
            refine_do_.SetGlobalBuffer((__gm__ INPUT_TYPE*)dout);
            refine_v_.SetGlobalBuffer((__gm__ INPUT_TYPE*)v);
            refine_q_stride_ = INPUT_LAYOUT == BNSD ? head_dim_ : q_head_num_ * head_dim_;
            refine_k_stride_ = INPUT_LAYOUT == BNSD ? head_dim_ : kv_head_num_ * head_dim_;
            q_descale_ = *((__gm__ float*)qDescale);
            k_descale_ = *((__gm__ float*)kDescale);
            v_descale_ = *((__gm__ float*)vDescale);
            do_descale_ = *((__gm__ float*)doDescale);
            p_scale_ = *((__gm__ float*)pScale);
            ds_scale_ = *((__gm__ float*)dsScale);
            ds_scale_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->dsScaleWorkspaceOffset));
        }
        // Apply attention scaling after the paired input descales. The order
        // changes FP32 rounding and therefore HiF8 decisions at large logits.
        score_scale_ = q_descale_ * k_descale_ * softmax_scale_;
        dp_scale_ = do_descale_ * v_descale_;
        dv_post_scale_ = do_descale_ / p_scale_;
        this->head_dim_align_ = RoundUp(head_dim_, 16);
        this->base_m = tilingData->baseM;
        this->base_n = tilingData->baseN;
        this->vec_base_m = tilingData->baseM / 2;
        this->vec_base_n = tilingData->baseN;
        this->act_seq_q_len = actualQseqlen;
        v_core_idx_ = GetBlockIdx();
        v_sub_core_idx_ = GetSubBlockIdx();
        // gm_tensor
        lse_gm_.SetGlobalBuffer((__gm__ float*)softmaxLse);
        sftg_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->sftgWorkspaceOffset));
        // local_tensor
        softmax_res_nz_tensor_ = ub_buffer.GetWithOffset<INPUT_TYPE>((vec_base_m + 1) * vec_base_n, ub_offset);
        ub_offset += (vec_base_m + 1) * vec_base_n * sizeof(INPUT_TYPE);
        sftg_res_nz_tensor_ = ub_buffer.GetWithOffset<INPUT_TYPE>((vec_base_m + 1) * vec_base_n, ub_offset);
        ub_offset += (vec_base_m + 1) * vec_base_n * sizeof(INPUT_TYPE);
        if (!combined_row_metadata_) {
            lse_tensor_ping_ = ub_buffer.GetWithOffset<float>(vec_base_m * BLOCK_FP32, ub_offset);
            ub_offset += vec_base_m * BLOCK_FP32 * sizeof(float);
            lse_tensor_pong_ = ub_buffer.GetWithOffset<float>(vec_base_m * BLOCK_FP32, ub_offset);
            ub_offset += vec_base_m * BLOCK_FP32 * sizeof(float);
        }
        sftg_front_tensor_ping_ = ub_buffer.GetWithOffset<float>(vec_base_m * BLOCK_FP32, ub_offset);
        ub_offset += vec_base_m * BLOCK_FP32 * sizeof(float);
        sftg_front_tensor_pong_ = ub_buffer.GetWithOffset<float>(vec_base_m * BLOCK_FP32, ub_offset);
        ub_offset += vec_base_m * BLOCK_FP32 * sizeof(float);
        if (combined_row_metadata_) {
            lse_tensor_ping_ = sftg_front_tensor_ping_[1];
            lse_tensor_pong_ = sftg_front_tensor_pong_[1];
        }
        if constexpr (HIF8_ENABLE) {
            hif8_gather_indexes_ = ub_buffer.GetWithOffset<uint8_t>(128, ub_offset);
            ub_offset += 128 * sizeof(uint8_t);
            ds_amax_tensor_ = ub_buffer.GetWithOffset<float>(BLOCK_FP32, ub_offset);
            ub_offset += BLOCK_FP32 * sizeof(float);
            refine_bytes_a_ = ub_buffer.GetWithOffset<INPUT_TYPE>(128, ub_offset);
            ub_offset += 128;
            refine_bytes_b_ = ub_buffer.GetWithOffset<INPUT_TYPE>(128, ub_offset);
            ub_offset += 128;
            refine_a_ = ub_buffer.GetWithOffset<float>(128, ub_offset);
            ub_offset += 512;
            refine_b_ = ub_buffer.GetWithOffset<float>(128, ub_offset);
            ub_offset += 512;
            refine_sum_ = ub_buffer.GetWithOffset<float>(8, ub_offset);
        }
    }

    __aicore__ inline void DrainSoftmaxOutputCopy()
    {
        if constexpr (HIF8_ENABLE) {
            if (softmax_copy_pending_) {
                WAIT_FLAG(MTE3, V, EVENT_ID6);
                softmax_copy_pending_ = false;
            }
        }
    }

    __aicore__ inline void PrepareHif8Workspace()
    {
        if constexpr (HIF8_ENABLE) {
            // FP32->8-bit casts place valid bytes at even byte positions.
            // This is the same gather table used by the Ascend 950 FAG FP8
            // path; a four-way permutation reads padding bytes as HiF8 data.
            // Pack eight byte indices into each scalar UB store.
            // Little-endian bytes remain exactly 0, 2, ..., 254.
            auto packedIndexes = hif8_gather_indexes_.template ReinterpretCast<uint64_t>();
            for (uint32_t i = 0; i < 16; i++) {
                packedIndexes.SetValue(i, 0x0e0c0a0806040200ULL + i * 0x1010101010101010ULL);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void SendPreprocess(const GlobalTensor<float>& dq_workspace,
                                          const GlobalTensor<float>& dk_workspace,
                                          const GlobalTensor<float>& dv_workspace,
                                          const GlobalTensor<INPUT_TYPE>& dy_gm,
                                          const GlobalTensor<ATTENTION_TYPE>& out_gm,
                                          const GlobalTensor<float>& sftg_workspace, const TILING_CLASS* tilingData,
                                          TBuf<TPosition::VECCALC>& ub_buffer)
    {
        const auto& nz = tilingData->mNonZeroTilingData;
        const uint64_t maskElements = nz.realCoreNum * nz.numPerCore + nz.numTailCore;
        const bool overlap = HIF8_ENABLE && !BSA_TYPE::deterministic_enable && ds_scale_ > 0.0f && head_dim_ == 128 &&
                             q_seq_len_ >= 4096 && maskElements <= 32768 && v_core_num_ > 1;
        const uint32_t savedCore = v_core_idx_;
        const uint32_t savedCount = v_core_num_;
        // AIV0 builds SmallMaskIndices before entering this operator. Let the
        // other AIVs cover the entire pre-processing range while it does so.
        if (overlap && savedCore == 0)
            return;
        if (overlap) {
            v_core_idx_ = savedCore - 1;
            v_core_num_ = savedCount - 1;
        }
        SendVecPre(dq_workspace, dk_workspace, dv_workspace, dy_gm, out_gm, sftg_workspace, tilingData, ub_buffer);
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        SendVecSftgFront(dy_gm, out_gm, sftg_workspace, tilingData, ub_buffer);
        // Main-loop and post-processing ownership still uses all original AIVs.
        v_core_idx_ = savedCore;
        v_core_num_ = savedCount;
    }

    __aicore__ inline void SendVecPre(const GlobalTensor<float>& dq_workspace, const GlobalTensor<float>& dk_workspace,
                                      const GlobalTensor<float>& dv_workspace, const GlobalTensor<INPUT_TYPE>& dy_gm,
                                      const GlobalTensor<ATTENTION_TYPE>& out_gm,
                                      const GlobalTensor<float>& sftg_workspace, const TILING_CLASS* tilingData,
                                      TBuf<TPosition::VECCALC>& ub_buffer)
    {
        constexpr static uint32_t PRE_TILE_LEN = 20 * 1024; // PRE一次处理元素的个数
        EvenCoreInfo info;
        vec_in_ping_ = ub_buffer.GetWithOffset<float>(PRE_TILE_LEN, 0);
        Duplicate(vec_in_ping_, (float)0.0, PRE_TILE_LEN);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        ComputeEvenCoreInfo(info, tilingData->dqSize, PRE_TILE_LEN);
        ComputeVecPre(dq_workspace, vec_in_ping_, info);
        if (direct_dkv_) {
            // Head dimension 128 guarantees pairs of BF16 elements. Writing
            // float zeros clears exactly the BF16 output allocation, including
            // rows with no active sparse block; unused FP32 workspaces are untouched.
            ComputeEvenCoreInfo(info, tilingData->dkSize / 2, PRE_TILE_LEN);
            ComputeVecPre(direct_dk_zero_, vec_in_ping_, info);
            ComputeVecPre(direct_dv_zero_, vec_in_ping_, info);
        } else {
            ComputeEvenCoreInfo(info, tilingData->dkSize, PRE_TILE_LEN);
            ComputeVecPre(dk_workspace, vec_in_ping_, info);
            ComputeVecPre(dv_workspace, vec_in_ping_, info);
        }
    }

    __aicore__ inline void SendVecSftgFront(const GlobalTensor<INPUT_TYPE>& dy_gm,
                                            const GlobalTensor<ATTENTION_TYPE>& out_gm,
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
        attention_in_ping_ = ub_buffer.GetWithOffset<ATTENTION_TYPE>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(ATTENTION_TYPE);
        attention_in_pong_ = ub_buffer.GetWithOffset<ATTENTION_TYPE>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(ATTENTION_TYPE);
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

        // Partition complete row tiles across heads/sequences. Splitting every
        // head independently creates up to one short tile per AIV per head.
        const bool tileSchedule =
            HIF8_ENABLE && !BSA_TYPE::deterministic_enable && head_dim_ == 128 && ds_scale_ > 0.0f;
        uint64_t totalRowTiles = 0;
        if (tileSchedule) {
            for (uint32_t b = 0; b < batch_num_; ++b) {
                int64_t seq = q_seq_len_;
                if constexpr (INPUT_LAYOUT == TND)
                    seq = GetSeqLen(b, act_seq_q_len);
                totalRowTiles += CeilDiv<uint64_t>(seq, process_s1_size) * q_head_num_;
            }
        }
        const uint64_t coreTiles = CeilDiv<uint64_t>(totalRowTiles, v_core_num_);
        const uint64_t coreTileStart = v_core_idx_ * coreTiles;
        const uint64_t coreTileEnd =
            coreTileStart + coreTiles < totalRowTiles ? coreTileStart + coreTiles : totalRowTiles;
        uint64_t headTileBase = 0;
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
                if (tileSchedule) {
                    const uint64_t headTiles = CeilDiv<uint64_t>(current_q_seqlen, process_s1_size);
                    const uint64_t begin = coreTileStart > headTileBase ? coreTileStart : headTileBase;
                    const uint64_t end =
                        coreTileEnd < headTileBase + headTiles ? coreTileEnd : headTileBase + headTiles;
                    const uint64_t first = begin > headTileBase ? begin - headTileBase : 0;
                    headTileBase += headTiles;
                    if (begin >= end)
                        continue;
                    info.start_idx = first * process_s1_size;
                    const uint64_t rowEnd = (first + end - begin) * process_s1_size;
                    info.len = (rowEnd < current_q_seqlen ? rowEnd : current_q_seqlen) - info.start_idx;
                    info.data_size = current_q_seqlen;
                    info.max_process_size = process_s1_size;
                    info.loop_num = CeilDiv<uint32_t>(info.len, process_s1_size);
                    const uint32_t tail = info.len % process_s1_size;
                    info.align_tail = tail ? tail / 16 * 16 : process_s1_size;
                    info.pad_tail = tail % 16;
                } else {
                    ComputeEvenCoreInfo(info, current_q_seqlen, process_s1_size);
                }

                WAIT_FLAG(MTE3, MTE2, event_id);
                const int64_t lseBase = INPUT_LAYOUT == TND ?
                                            last_seq_total_len * q_head_num_ + n1_idx :
                                            (HIF8_ENABLE && INPUT_LAYOUT == BSND ?
                                                 b_idx * current_q_seqlen * q_head_num_ + n1_idx :
                                                 b_idx * current_q_seqlen * q_head_num_ + n1_idx * current_q_seqlen);
                ComputeSoftmaxGradFront(sftg_workspace[out_gm_offset], dy_gm[in_gm_offset], out_gm[in_gm_offset], info,
                                        tilingData, sftg_ping_pong_idx, lse_gm_[lseBase]);
                SET_FLAG(MTE3, MTE2, event_id);
                sftg_ping_pong_idx = 1 - sftg_ping_pong_idx;
            }
        }

        WAIT_FLAG(MTE3, MTE2, event_ping_);
        WAIT_FLAG(MTE3, MTE2, event_pong_);
    }

    __aicore__ inline void SendVecSftPreProcess(const RunTimeInfo& runTimeInfo, const RunTimeInfo& nextRunTimeInfo,
                                                uint32_t pingpong_idx, bool first)
    {
        int64_t lse_gm_offset_;
        int64_t sftg_gm_offset_;
        s1_process_align_ = runTimeInfo.s1LenAlign;
        s2_process_align_ = runTimeInfo.s2LenAlign;
        half_s1_process_align_ = s1_process_align_ / 2;
        data_size = half_s1_process_align_ * s2_process_align_;
        if (runTimeInfo.s1Len <= half_s1_process_align_) {
            half_s1_process_real_ = v_sub_core_idx_ == 0 ? runTimeInfo.s1Len : 0;
        } else {
            half_s1_process_real_ =
                v_sub_core_idx_ == 0 ? half_s1_process_align_ : (runTimeInfo.s1Len - half_s1_process_align_);
        }
        l1_offset_ = v_sub_core_idx_ * half_s1_process_align_ * C0_SIZE;

        if (half_s1_process_real_ > 0) {
            lse_tensor_ = pingpong_idx == 0 ? lse_tensor_ping_ : lse_tensor_pong_;
            sftg_front_tensor_ = pingpong_idx == 0 ? sftg_front_tensor_ping_ : sftg_front_tensor_pong_;
            event_id = pingpong_idx == 0 ? event_ping_ : event_pong_;
            if (combined_row_metadata_)
                lse_tensor_ = sftg_front_tensor_[1];

            if (unlikely(first)) {
                WAIT_FLAG(V, MTE2, event_id);
                sftg_gm_offset_ = runTimeInfo.sftgGmOffset + v_sub_core_idx_ * half_s1_process_align_ * 8;
                if constexpr (INPUT_LAYOUT == TND || (HIF8_ENABLE && INPUT_LAYOUT == BSND)) {
                    lse_gm_offset_ = runTimeInfo.lseGmOffset + v_sub_core_idx_ * half_s1_process_align_ * q_head_num_;
                } else {
                    lse_gm_offset_ = runTimeInfo.lseGmOffset + v_sub_core_idx_ * half_s1_process_align_;
                }

                if (!combined_row_metadata_) {
                    CopyInLSE(lse_tensor_, lse_gm_[lse_gm_offset_], half_s1_process_real_);
                }
                DataCopy(sftg_front_tensor_, sftg_workspace_[sftg_gm_offset_], half_s1_process_real_ * 8);
            }
        }

        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);

        if (nextRunTimeInfo.need_compute) {
            int32_t tmp_half_s1_process_align_ = nextRunTimeInfo.s1LenAlign / 2;
            int32_t tmp_half_s1_process_real_ = 0;
            if (nextRunTimeInfo.s1Len <= tmp_half_s1_process_align_) {
                tmp_half_s1_process_real_ = v_sub_core_idx_ == 0 ? nextRunTimeInfo.s1Len : 0;
            } else {
                tmp_half_s1_process_real_ = v_sub_core_idx_ == 0 ? tmp_half_s1_process_align_ :
                                                                   (nextRunTimeInfo.s1Len - tmp_half_s1_process_align_);
            }
            if (tmp_half_s1_process_real_ > 0) {
                LocalTensor<float> tmp_lse_tensor_ = pingpong_idx == 1 ? lse_tensor_ping_ : lse_tensor_pong_;
                LocalTensor<float> tmp_sftg_front_tensor_ =
                    pingpong_idx == 1 ? sftg_front_tensor_ping_ : sftg_front_tensor_pong_;
                TEventID tmp_event_id = pingpong_idx == 1 ? event_ping_ : event_pong_;
                sftg_gm_offset_ = nextRunTimeInfo.sftgGmOffset + v_sub_core_idx_ * tmp_half_s1_process_align_ * 8;
                if constexpr (INPUT_LAYOUT == TND || (HIF8_ENABLE && INPUT_LAYOUT == BSND)) {
                    lse_gm_offset_ =
                        nextRunTimeInfo.lseGmOffset + v_sub_core_idx_ * tmp_half_s1_process_align_ * q_head_num_;
                } else {
                    lse_gm_offset_ = nextRunTimeInfo.lseGmOffset + v_sub_core_idx_ * tmp_half_s1_process_align_;
                }

                WAIT_FLAG(V, MTE2, tmp_event_id);
                if (!combined_row_metadata_) {
                    CopyInLSE(tmp_lse_tensor_, lse_gm_[lse_gm_offset_], tmp_half_s1_process_real_);
                }
                DataCopy(tmp_sftg_front_tensor_, sftg_workspace_[sftg_gm_offset_], tmp_half_s1_process_real_ * 8);
            }
        }
    }
    // Error-free pair additions retain the low part of each partial sum.
    // Six butterfly stages sum all 128 FP32 products without losing small
    // terms next to a large outlier; the result is rounded once to FP32.
    __simd_callee__ inline void CompensatedSum128(Reg::RegTensor<float>& a, Reg::RegTensor<float>& b,
                                                  Reg::RegTensor<float>& total)
    {
        Reg::RegTensor<float> hi, lo, bv, tmp, err, peer, peerLo;
        Reg::RegTensor<uint32_t> lane, step, index;
        auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Add(hi, a, b, all);
        Sub(bv, hi, a, all);
        Sub(tmp, hi, bv, all);
        Sub(lo, a, tmp, all);
        Sub(tmp, b, bv, all);
        Add(lo, lo, tmp, all);
        Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
        for (uint16_t width = 1; width < 64; width *= 2) {
            Duplicate(step, uint32_t(width));
            Xor(index, lane, step, all);
            Reg::Gather<float>(peer, hi, index);
            Reg::Gather<float>(peerLo, lo, index);
            Add(total, hi, peer, all);
            Sub(bv, total, hi, all);
            Sub(tmp, total, bv, all);
            Sub(err, hi, tmp, all);
            Sub(tmp, peer, bv, all);
            Add(err, err, tmp, all);
            Add(lo, lo, peerLo, all);
            Add(lo, lo, err, all);
            Copy(hi, total, all);
        }
        Add(total, hi, lo, all);
    }

    __simd_vf__ inline void RefineDot128(__ubuf__ float* a, __ubuf__ float* b, __ubuf__ float* out)
    {
        Reg::RegTensor<float> a0, a1, b0, b1, total;
        auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        auto one = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
        LoadAlign(a0, a);
        LoadAlign(a1, a + 64);
        LoadAlign(b0, b);
        LoadAlign(b1, b + 64);
        Mul(a0, a0, b0, all);
        Mul(a1, a1, b1, all);
        CompensatedSum128(a0, a1, total);
        StoreAlign(out, total, one);
    }

    __aicore__ inline void AccurateRowSum128(__ubuf__ float* dout, __ubuf__ float* out, __ubuf__ float* dst,
                                             uint32_t rows)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            RefineDot128(dout + row * 128, out + row * 128, dst + row * 8);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void RefineHif8DotProducts(const LocalTensor<float>& dst, const RunTimeInfo& rt, bool qk)
    {
        if constexpr (HIF8_ENABLE) {
            if (head_dim_ != 128)
                return;
            SET_FLAG(MTE2, S, EVENT_ID0);
            WAIT_FLAG(MTE2, S, EVENT_ID0);
            for (uint32_t row = 0; row < half_s1_process_real_; ++row) {
                // Quantization boundaries amplify dot-product error even at small logits.
                const uint64_t qoff =
                    rt.queryGmOffset + uint64_t(v_sub_core_idx_ * half_s1_process_align_ + row) * refine_q_stride_;
                const auto left = qk ? refine_q_ : refine_do_;
                const auto right = qk ? refine_k_ : refine_v_;
                DataCopy(refine_bytes_a_, left[qoff], 128);
                SET_FLAG(MTE2, V, EVENT_ID7);
                WAIT_FLAG(MTE2, V, EVENT_ID7);
                Cast(refine_a_, refine_bytes_a_, RoundMode::CAST_NONE, 128);
                PipeBarrier<PIPE_V>();
                for (uint32_t col = 0; col < rt.s2Len; ++col) {
                    const uint64_t koff = rt.keyGmOffset + uint64_t(col) * refine_k_stride_;
                    DataCopy(refine_bytes_b_, right[koff], 128);
                    SET_FLAG(MTE2, V, EVENT_ID7);
                    WAIT_FLAG(MTE2, V, EVENT_ID7);
                    Cast(refine_b_, refine_bytes_b_, RoundMode::CAST_NONE, 128);
                    PipeBarrier<PIPE_V>();
                    RefineDot128((__ubuf__ float*)refine_a_.GetPhyAddr(), (__ubuf__ float*)refine_b_.GetPhyAddr(),
                                 (__ubuf__ float*)refine_sum_.GetPhyAddr());
                    SET_FLAG(V, S, EVENT_ID7);
                    WAIT_FLAG(V, S, EVENT_ID7);
                    dst.SetValue(row * s2_process_align_ + col, refine_sum_.GetValue(0));
                    SET_FLAG(V, MTE2, EVENT_ID7);
                    WAIT_FLAG(V, MTE2, EVENT_ID7);
                }
            }
            SET_FLAG(S, V, EVENT_ID7);
            WAIT_FLAG(S, V, EVENT_ID7);
        }
    }

    __aicore__ inline void SendVecSoftmax(const LocalTensor<INPUT_TYPE>& p_l1_tensor,
                                          const LocalTensor<INPUT_TYPE>& ds_l1_tensor,
                                          const LocalTensor<float>& p_ub_tensor, const LocalTensor<float>& dp_ub_tensor,
                                          const RunTimeInfo& runTimeInfo)
    { /*
       * function: Compute simple softmax
       * input shape：[s1LenAlign / 2, s2LenAlign]
       * out shape:   [s1LenAlign / 2, s2LenAlign]
       * dtype:       float
       */
        // Both P and dS reuse the same NZ UB allocation in the next tile.
        // Early QK publication can otherwise let V overwrite P while the
        // previous UB->L1 transfer is still reading it.
        DrainSoftmaxOutputCopy();
        bool dsAlreadyNz = false;
        if (half_s1_process_real_ <= 0) {
            if constexpr (HIF8_ENABLE) {
                if (ds_scale_ > 0.0f)
                    CrossCoreWaitFlag<2, PIPE_V>(dp_ready_flag_);
                Hif8ScaleHandshake(runTimeInfo, dp_ub_tensor, 0);
            }
            return;
        }

        if constexpr (HIF8_ENABLE) {
            RefineHif8DotProducts(p_ub_tensor, runTimeInfo, true);
            if (ds_scale_ > 0.0f) {
                if (s2_process_align_ == 128 && half_s1_process_real_ == half_s1_process_align_) {
                    FusedStaticPNZ(p_ub_tensor, half_s1_process_align_);
                } else {
                    StaticSoftmaxP((__ubuf__ float*)p_ub_tensor.GetPhyAddr(), (__ubuf__ float*)lse_tensor_.GetPhyAddr(),
                                   half_s1_process_real_, s2_process_align_);
                    CastND2NZ<INPUT_TYPE>(softmax_res_nz_tensor_, p_ub_tensor, hif8_gather_indexes_,
                                          half_s1_process_align_, s2_process_align_);
                }
                CrossCoreWaitFlag<2, PIPE_V>(dp_ready_flag_);
                RefineHif8DotProducts(dp_ub_tensor, runTimeInfo, false);
                dsAlreadyNz = s2_process_align_ == 128 && half_s1_process_real_ == half_s1_process_align_;
                if (dsAlreadyNz) {
                    FusedStaticDsNZ(p_ub_tensor, dp_ub_tensor, half_s1_process_align_);
                } else {
                    StaticSoftmaxDs(
                        (__ubuf__ float*)p_ub_tensor.GetPhyAddr(), (__ubuf__ float*)dp_ub_tensor.GetPhyAddr(),
                        (__ubuf__ float*)sftg_front_tensor_.GetPhyAddr(), half_s1_process_real_, s2_process_align_);
                }
            } else {
                RefineHif8DotProducts(dp_ub_tensor, runTimeInfo, false);
                SimpleSoftmax((__ubuf__ float*)p_ub_tensor.GetPhyAddr(), (__ubuf__ float*)dp_ub_tensor.GetPhyAddr(),
                              (__ubuf__ float*)lse_tensor_.GetPhyAddr(),
                              (__ubuf__ float*)sftg_front_tensor_.GetPhyAddr(), half_s1_process_real_,
                              s2_process_align_);
                Hif8ScaleHandshake(runTimeInfo, dp_ub_tensor, half_s1_process_real_ * s2_process_align_);
                CastND2NZ<INPUT_TYPE>(softmax_res_nz_tensor_, p_ub_tensor, hif8_gather_indexes_, half_s1_process_align_,
                                      s2_process_align_);
            }
        } else {
            SimpleSoftmax((__ubuf__ float*)p_ub_tensor.GetPhyAddr(), (__ubuf__ float*)dp_ub_tensor.GetPhyAddr(),
                          (__ubuf__ float*)lse_tensor_.GetPhyAddr(), (__ubuf__ float*)sftg_front_tensor_.GetPhyAddr(),
                          half_s1_process_align_, s2_process_align_);
            CastND2NZ<INPUT_TYPE>(softmax_res_nz_tensor_, p_ub_tensor, hif8_gather_indexes_, half_s1_process_align_,
                                  s2_process_align_);
        }
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = s2_process_align_ / C0_SIZE;
        dataCopyParams.blockLen = half_s1_process_align_ * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
        dataCopyParams.srcStride = 1;
        dataCopyParams.dstStride =
            (s1_process_align_ - half_s1_process_align_) * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
        if (!dsAlreadyNz) {
            CastND2NZ<INPUT_TYPE>(sftg_res_nz_tensor_, dp_ub_tensor, hif8_gather_indexes_, half_s1_process_align_,
                                  s2_process_align_);
        }
        // Both casts write independent UB buffers. Publish them together and
        // remove the redundant Vector->MTE3 round trip between P and dS.
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        DataCopy(p_l1_tensor[l1_offset_], softmax_res_nz_tensor_, dataCopyParams);
        DataCopy(ds_l1_tensor[l1_offset_], sftg_res_nz_tensor_, dataCopyParams);
        if constexpr (HIF8_ENABLE) {
            if (ds_scale_ > 0.0f) {
                SET_FLAG(MTE3, V, EVENT_ID6);
                softmax_copy_pending_ = true;
            }
        }
        SET_FLAG(V, MTE2, event_id);
    }

    __aicore__ inline void SendVecPost(const GlobalTensor<OUTPUT_TYPE>& dq_out_gm,
                                       const GlobalTensor<OUTPUT_TYPE>& dk_out_gm,
                                       const GlobalTensor<OUTPUT_TYPE>& dv_out_gm,
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
        vec_out_ping_ = ub_buffer.GetWithOffset<OUTPUT_TYPE>(POST_TILE_LEN, ub_offset);
        ub_offset += POST_TILE_LEN * sizeof(OUTPUT_TYPE);
        vec_out_pong_ = ub_buffer.GetWithOffset<OUTPUT_TYPE>(POST_TILE_LEN, ub_offset);
        ub_offset += POST_TILE_LEN * sizeof(OUTPUT_TYPE);
        SET_FLAG(MTE3, MTE2, event_ping_);
        SET_FLAG(MTE3, MTE2, event_pong_);

        // Static dS has one descale for every tile. Apply the common
        // dQ/dK factors after FP32 accumulation, in the existing Vector pass.
        // Dynamic dS remains tile-scaled by AIC Fixpipe before accumulation.
        float dqPostScale = HIF8_ENABLE ? 1.0f : softmax_scale_;
        float dkPostScale = dqPostScale;
        if constexpr (HIF8_ENABLE) {
            if (ds_scale_ > 0.0f) {
                const float dsDescale = 1.0f / ds_scale_;
                dqPostScale = softmax_scale_ * k_descale_ * dsDescale;
                dkPostScale = softmax_scale_ * q_descale_ * dsDescale;
            }
        }
        EvenCoreInfo info;
        if (!direct_dkv_) {
            ComputeEvenCoreInfo(info, dkv_size, POST_TILE_LEN);
            ComputeVecPost<HIF8_ENABLE>(dv_out_gm, dv_workspace, info, post_ping_pong_idx, dv_post_scale_);
            ComputeVecPost<true>(dk_out_gm, dk_workspace, info, post_ping_pong_idx, dkPostScale);
        }
        ComputeEvenCoreInfo(info, dq_size, POST_TILE_LEN);
        ComputeVecPost<true>(dq_out_gm, dq_workspace, info, post_ping_pong_idx, dqPostScale);

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

    __aicore__ inline void FusedHif8RowSum128(const LocalTensor<float>& dst, const LocalTensor<float>& dout,
                                              const LocalTensor<float>& out, const LocalTensor<float>& lseTensor,
                                              uint32_t rows)
    {
        auto dy = (__ubuf__ float*)dout.GetPhyAddr();
        auto att = (__ubuf__ float*)out.GetPhyAddr();
        auto result = (__ubuf__ float*)dst.GetPhyAddr();
        auto lse = (__ubuf__ float*)lseTensor.GetPhyAddr();
        const float descale = do_descale_;
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> d0, d1, o0, o1, sum, scale, rowLse, packed, unused;
            auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            uint32_t recordWidth = 8;
            auto recordMask = Reg::UpdateMask<float>(recordWidth);
            Duplicate(scale, descale);
            for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
                Reg::LoadAlign(d0, dy + row * 128);
                Reg::LoadAlign(d1, dy + row * 128 + 64);
                Reg::LoadAlign(o0, att + row * 128);
                Reg::LoadAlign(o1, att + row * 128 + 64);
                Mul(d0, d0, scale, mask);
                Mul(d1, d1, scale, mask);
                Mul(d0, d0, o0, mask);
                Mul(d1, d1, o1, mask);
                CompensatedSum128(d0, d1, sum);
                // One 32-byte record per row: delta in lane 0, LSE in lane 1.
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rowLse, lse + row * 8);
                Reg::Interleave(packed, unused, sum, rowLse);
                Reg::StoreAlign(result + row * 8, packed, recordMask);
            }
        }
    }

    __aicore__ inline void ComputeSoftmaxGradFront(const GlobalTensor<float>& dst_gm,
                                                   const GlobalTensor<INPUT_TYPE>& dy_gm,
                                                   const GlobalTensor<ATTENTION_TYPE>& out_gm, const EvenCoreInfo& info,
                                                   const TILING_CLASS* tilingData, const uint32_t sftg_ping_pong_idx,
                                                   const GlobalTensor<float>& srcLse)
    {
        if (info.start_idx >= info.data_size) {
            return;
        }
        LocalTensor<INPUT_TYPE> dy_tensor = sftg_ping_pong_idx ? dy_in_ping_ : dy_in_pong_;
        LocalTensor<ATTENTION_TYPE> attention_tensor = sftg_ping_pong_idx ? attention_in_ping_ : attention_in_pong_;
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
            copyParam.blockLen = head_dim_ * sizeof(ATTENTION_TYPE);
            copyParam.srcStride = (src_stride - head_dim_) * sizeof(ATTENTION_TYPE);
            DataCopyPad(attention_tensor, out_gm[in_gm_offset], copyParam, {false, 0, 0, 0});
            if (combined_row_metadata_) {
                const uint32_t lseStride =
                    (INPUT_LAYOUT == TND || (HIF8_ENABLE && INPUT_LAYOUT == BSND)) ? q_head_num_ : 1;
                const uint64_t row = static_cast<uint64_t>(info.start_idx) + i * info.max_process_size;
                CopyInLSE(sftg_tmp_tensor.template ReinterpretCast<float>(), srcLse[row * lseStride], process_size);
            }
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);

            Cast(dy_out_tensor, dy_tensor, RoundMode::CAST_NONE, process_size * head_dim_);
            Cast(attention_out_tensor, attention_tensor, RoundMode::CAST_NONE, process_size * head_dim_);
            PipeBarrier<PIPE_V>();
            if (combined_row_metadata_) {
                FusedHif8RowSum128(sftg_front_tensor, dy_out_tensor, attention_out_tensor,
                                   sftg_tmp_tensor.template ReinterpretCast<float>(), process_size);
                PipeBarrier<PIPE_V>();
            } else {
                if constexpr (HIF8_ENABLE) {
                    Muls(dy_out_tensor, dy_out_tensor, do_descale_, process_size * head_dim_);
                    PipeBarrier<PIPE_V>();
                }

                uint32_t intput_shape_arry[2] = {static_cast<uint32_t>(process_size), static_cast<uint32_t>(head_dim_)};
                uint32_t out_shape_arry[2] = {static_cast<uint32_t>(process_size), static_cast<uint32_t>(8)};

                dy_out_tensor.SetShapeInfo(ShapeInfo(2, intput_shape_arry, AscendC::DataFormat::ND));
                attention_out_tensor.SetShapeInfo(ShapeInfo(2, intput_shape_arry, AscendC::DataFormat::ND));
                sftg_front_tensor.SetShapeInfo(ShapeInfo(2, out_shape_arry, AscendC::DataFormat::ND));
                bool isBasicBlock = process_size % 8 == 0;
                if constexpr (HIF8_ENABLE) {
                    AccurateRowSum128((__ubuf__ float*)dy_out_tensor.GetPhyAddr(),
                                      (__ubuf__ float*)attention_out_tensor.GetPhyAddr(),
                                      (__ubuf__ float*)sftg_front_tensor.GetPhyAddr(), process_size);
                } else if (likely(isBasicBlock)) {
                    SoftmaxGradFront<float, true>(sftg_front_tensor, dy_out_tensor, attention_out_tensor,
                                                  sftg_tmp_tensor, tilingData->softmaxGradFrontTilingData);
                } else {
                    SoftmaxGradFront<float, false>(sftg_front_tensor, dy_out_tensor, attention_out_tensor,
                                                   sftg_tmp_tensor, tilingData->softmaxGradFrontTilingData);
                }
                PipeBarrier<PIPE_V>();
            }

            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            int64_t out_gm_offset = (int64_t)(info.start_idx + i * info.max_process_size) * 8;
            DataCopy(dst_gm[out_gm_offset], sftg_front_tensor, process_size * 8);
            SET_FLAG(MTE3, MTE2, EVENT_ID0);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        }
    }

    template <bool MULS>
    __aicore__ inline void ComputeVecPost(const GlobalTensor<OUTPUT_TYPE>& dst_tensor,
                                          const GlobalTensor<float>& src_tensor, const EvenCoreInfo& info,
                                          uint32_t& post_ping_pong_idx, float output_scale)
    {
        if (info.start_idx >= info.data_size) {
            return;
        }

        LocalTensor<float> vecIn;
        LocalTensor<OUTPUT_TYPE> vecOut;
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
                Muls(vecIn, vecIn, output_scale, process_size);
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
                Muls(vecIn, vecIn, output_scale, pad_tail_align);
                PipeBarrier<PIPE_V>();
            }

            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_RINT, pad_tail_align);
            PipeBarrier<PIPE_V>();

            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            copyParam.blockLen = info.pad_tail * sizeof(OUTPUT_TYPE);
            DataCopyPad(dst_tensor[gm_offset], vecOut, copyParam);
            SET_FLAG(MTE3, MTE2, event_id);
            post_ping_pong_idx = 1 - post_ping_pong_idx;
        }
    }

    __aicore__ inline void FusedStaticPNZ(const LocalTensor<float>& pTensor, uint32_t rows)
    {
        uint64_t srcAddr = pTensor.GetPhyAddr();
        auto pFp32 = (__ubuf__ float*)pTensor.GetPhyAddr();
        auto lse = (__ubuf__ float*)lse_tensor_.GetPhyAddr();
        uint64_t dstAddr = softmax_res_nz_tensor_.GetPhyAddr();
        uint64_t indexAddr = hif8_gather_indexes_.GetPhyAddr();
        const uint32_t blockStride = rows + 1;
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> even, odd, rowLse, scale, pScale;
            Reg::RegTensor<hifloat8_t> castEven, castOdd, packed, result;
            Reg::RegTensor<uint8_t> indexes;
            Reg::MaskReg fp32Mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg byteMask = Reg::CreateMask<hifloat8_t, Reg::MaskPattern::ALL>();
            uint32_t valid = 128;
            Reg::MaskReg validMask = Reg::UpdateMask<hifloat8_t>(valid);
            Duplicate(scale, score_scale_);
            Duplicate(pScale, p_scale_);
            Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_NORM>(indexes, ((__ubuf__ uint8_t*&)indexAddr));
            for (uint16_t i = 0; i < rows; ++i) {
                LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rowLse, lse + i * 8);
                Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                    even, odd, ((__ubuf__ float*&)srcAddr), 128);
                Mul(even, even, scale, fp32Mask);
                Mul(odd, odd, scale, fp32Mask);
                Sub(even, even, rowLse, fp32Mask);
                Exp(even, even, fp32Mask);
                Sub(odd, odd, rowLse, fp32Mask);
                Exp(odd, odd, fp32Mask);
                // Scale the probability after Exp. Folding log(pScale) into
                // LSE introduces cancellation that changes HiF8 midpoint
                // rounding even for exactly representable uniform P.
                Mul(even, even, pScale, fp32Mask);
                Mul(odd, odd, pScale, fp32Mask);
                // dS consumes unquantized FP32 P after dOV is published.
                Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(pFp32 + i * 128, even, odd, fp32Mask);
                ClampHif8Finite(even, fp32Mask);
                ClampHif8Finite(odd, fp32Mask);
                Reg::Cast<hifloat8_t, float, castTraitFp322Hif8Even>(castEven, even, fp32Mask);
                Reg::Cast<hifloat8_t, float, castTraitFp322Hif8Odd>(castOdd, odd, fp32Mask);
                Or((Reg::RegTensor<uint8_t>&)packed, (Reg::RegTensor<uint8_t>&)castEven,
                   (Reg::RegTensor<uint8_t>&)castOdd, byteMask);
                Reg::Gather<uint8_t>((Reg::RegTensor<uint8_t>&)result, (Reg::RegTensor<uint8_t>&)packed, indexes);
                Reg::DataCopy<hifloat8_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ hifloat8_t*&)dstAddr), result, blockStride, 1, validMask);
            }
        }
    }

    __aicore__ inline void FusedStaticDsNZ(const LocalTensor<float>& pTensor, const LocalTensor<float>& dpTensor,
                                           uint32_t rows)
    {
        // P is deliberately kept in FP32: dS must use the pre-quantized P,
        // exactly as in the original softmax-gradient path.
        uint64_t pAddr = pTensor.GetPhyAddr(), dpAddr = dpTensor.GetPhyAddr();
        uint64_t dstAddr = sftg_res_nz_tensor_.GetPhyAddr();
        uint64_t indexAddr = hif8_gather_indexes_.GetPhyAddr();
        auto front = (__ubuf__ float*)sftg_front_tensor_.GetPhyAddr();
        const uint32_t blockStride = rows + 1;
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> pEven, pOdd, dpEven, dpOdd, delta, dpScale, dsPost;
            Reg::RegTensor<hifloat8_t> castEven, castOdd, packed, result;
            Reg::RegTensor<uint8_t> indexes;
            Reg::MaskReg fp32Mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg byteMask = Reg::CreateMask<hifloat8_t, Reg::MaskPattern::ALL>();
            uint32_t valid = 128;
            Reg::MaskReg validMask = Reg::UpdateMask<hifloat8_t>(valid);
            Duplicate(dpScale, dp_scale_);
            Duplicate(dsPost, (1.0f / p_scale_) * ds_scale_);
            Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_NORM>(indexes, ((__ubuf__ uint8_t*&)indexAddr));
            for (uint16_t i = 0; i < rows; ++i) {
                LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(delta, front + i * 8);
                Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                    pEven, pOdd, ((__ubuf__ float*&)pAddr), 128);
                Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                    dpEven, dpOdd, ((__ubuf__ float*&)dpAddr), 128);
                Mul(dpEven, dpEven, dpScale, fp32Mask);
                Mul(dpOdd, dpOdd, dpScale, fp32Mask);
                Sub(dpEven, dpEven, delta, fp32Mask);
                Sub(dpOdd, dpOdd, delta, fp32Mask);
                Mul(dpEven, dpEven, pEven, fp32Mask);
                Mul(dpOdd, dpOdd, pOdd, fp32Mask);
                Mul(dpEven, dpEven, dsPost, fp32Mask);
                Mul(dpOdd, dpOdd, dsPost, fp32Mask);
                ClampHif8Finite(dpEven, fp32Mask);
                ClampHif8Finite(dpOdd, fp32Mask);
                Reg::Cast<hifloat8_t, float, castTraitFp322Hif8Even>(castEven, dpEven, fp32Mask);
                Reg::Cast<hifloat8_t, float, castTraitFp322Hif8Odd>(castOdd, dpOdd, fp32Mask);
                Or((Reg::RegTensor<uint8_t>&)packed, (Reg::RegTensor<uint8_t>&)castEven,
                   (Reg::RegTensor<uint8_t>&)castOdd, byteMask);
                Reg::Gather<uint8_t>((Reg::RegTensor<uint8_t>&)result, (Reg::RegTensor<uint8_t>&)packed, indexes);
                Reg::DataCopy<hifloat8_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ hifloat8_t*&)dstAddr), result, blockStride, 1, validMask);
            }
        }
    }

    __simd_vf__ inline void StaticSoftmaxP(__ubuf__ float* p, __ubuf__ float* lse, uint32_t rows, uint32_t cols)
    {
        Reg::RegTensor<float> value, rowLse, scale, pScale;
        Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Duplicate(scale, score_scale_);
        Duplicate(pScale, p_scale_);
        for (uint16_t i = 0; i < rows; ++i) {
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(rowLse, lse + i * 8);
            uint32_t count = cols;
            for (uint16_t j = 0; j < (cols + 63) / 64; ++j) {
                Reg::MaskReg mask = Reg::UpdateMask<float>(count);
                LoadAlign(value, p + i * cols + j * 64);
                Mul(value, value, scale, mask);
                Sub(value, value, rowLse, mask);
                Exp(value, value, mask);
                Mul(value, value, pScale, mask);
                StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(p + i * cols + j * 64, value, mask);
            }
        }
    }

    __simd_vf__ inline void StaticSoftmaxDs(__ubuf__ float* p, __ubuf__ float* dp, __ubuf__ float* front, uint32_t rows,
                                            uint32_t cols)
    {
        Reg::RegTensor<float> prob, grad, delta, scale, postScale;
        Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Duplicate(scale, dp_scale_);
        Duplicate(postScale, (1.0f / p_scale_) * ds_scale_);
        for (uint16_t i = 0; i < rows; ++i) {
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(delta, front + i * 8);
            uint32_t count = cols;
            for (uint16_t j = 0; j < (cols + 63) / 64; ++j) {
                Reg::MaskReg mask = Reg::UpdateMask<float>(count);
                LoadAlign(prob, p + i * cols + j * 64);
                LoadAlign(grad, dp + i * cols + j * 64);
                Mul(grad, grad, scale, mask);
                Sub(grad, grad, delta, mask);
                Mul(grad, grad, prob, mask);
                Mul(grad, grad, postScale, mask);
                StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(dp + i * cols + j * 64, grad, mask);
            }
        }
    }

    // Two-float arithmetic keeps polynomial rounding error below an FP32 ULP.
    // This matters when dynamic dS scaling lands on a HiF8 rounding midpoint.
    __simd_callee__ inline void ExpPairAdd(Reg::RegTensor<float>& oh, Reg::RegTensor<float>& ol,
                                           Reg::RegTensor<float>& ah, Reg::RegTensor<float>& al,
                                           Reg::RegTensor<float>& bh, Reg::RegTensor<float>& bl, Reg::MaskReg& mask)
    {
        Reg::RegTensor<float> s, v, t, e;
        Add(s, ah, bh, mask);
        Sub(v, s, ah, mask);
        Sub(t, s, v, mask);
        Sub(e, ah, t, mask);
        Sub(t, bh, v, mask);
        Add(e, e, t, mask);
        Add(e, e, al, mask);
        Add(e, e, bl, mask);
        Add(oh, s, e, mask);
        Sub(t, oh, s, mask);
        Sub(ol, e, t, mask);
    }

    __simd_callee__ inline void ExpPairMul(Reg::RegTensor<float>& oh, Reg::RegTensor<float>& ol,
                                           Reg::RegTensor<float>& ah, Reg::RegTensor<float>& al,
                                           Reg::RegTensor<float>& bh, Reg::RegTensor<float>& bl, Reg::MaskReg& mask)
    {
        Reg::RegTensor<float> a, b, ar, br, p, e, t;
        Reg::RegTensor<uint32_t> split;
        Duplicate(split, uint32_t(0xfffff000U));
        And((Reg::RegTensor<uint32_t>&)a, (Reg::RegTensor<uint32_t>&)ah, split, mask);
        And((Reg::RegTensor<uint32_t>&)b, (Reg::RegTensor<uint32_t>&)bh, split, mask);
        Sub(ar, ah, a, mask);
        Sub(br, bh, b, mask);
        Mul(p, ah, bh, mask);
        Mul(e, a, b, mask);
        Sub(e, e, p, mask);
        Mul(t, a, br, mask);
        Add(e, e, t, mask);
        Mul(t, ar, b, mask);
        Add(e, e, t, mask);
        Mul(t, ar, br, mask);
        Add(e, e, t, mask);
        Mul(t, ah, bl, mask);
        Add(e, e, t, mask);
        Mul(t, al, bh, mask);
        Add(e, e, t, mask);
        Add(oh, p, e, mask);
        Sub(t, oh, p, mask);
        Sub(ol, e, t, mask);
    }

    __simd_callee__ inline void AccurateHif8Exp(Reg::RegTensor<float>& value, Reg::MaskReg& mask)
    {
        // exp(x) = 2^n * exp(r), |r| <= log(2)/2. A degree-10 Taylor
        // polynomial evaluated as two floats avoids the instruction's 1-ULP
        // error at quantization boundaries. Preserve hardware exceptional and
        // underflow behavior outside the normal softmax interval.
        Reg::RegTensor<float> original, x, bound, n, zero, ch, cl, rh, rl;
        Reg::RegTensor<float> ph, pl, th, tl, result, factor;
        Reg::RegTensor<int32_t> ni, shift;
        Reg::MaskReg loMask, hiMask, normalMask;
        original = value;
        Duplicate(bound, -80.0f);
        Max(x, value, bound, mask);
        Duplicate(bound, 80.0f);
        Min(x, x, bound, mask);
        Duplicate(ch, 1.4426950408889634f);
        Mul(n, x, ch, mask);
        Reg::Truncate<float, RoundMode::CAST_RINT>(n, n, mask);
        Duplicate(zero, 0.0f);
        Duplicate(ch, -0.693147182464599609375f);
        Duplicate(cl, 1.904654323148236e-9f);
        ExpPairMul(th, tl, n, zero, ch, cl, mask);
        ExpPairAdd(rh, rl, x, zero, th, tl, mask);
        Duplicate(ph, 2.755731998149713036e-07f);
        Duplicate(pl, -7.575112209051194923e-15f);
        ExpPairMul(th, tl, ph, pl, rh, rl, mask);
        Duplicate(ch, 2.755731884462875314e-06f);
        Duplicate(cl, 3.793571224297229150e-14f);
        ExpPairAdd(ph, pl, th, tl, ch, cl, mask);
        ExpPairMul(th, tl, ph, pl, rh, rl, mask);
        Duplicate(ch, 2.480158764228690416e-05f);
        Duplicate(cl, -3.406996093666819769e-13f);
        ExpPairAdd(ph, pl, th, tl, ch, cl, mask);
        ExpPairMul(th, tl, ph, pl, rh, rl, mask);
        Duplicate(ch, 1.984127011382952332e-04f);
        Duplicate(cl, -2.725596874933455815e-12f);
        ExpPairAdd(ph, pl, th, tl, ch, cl, mask);
        ExpPairMul(th, tl, ph, pl, rh, rl, mask);
        Duplicate(ch, 1.388888922519981861e-03f);
        Duplicate(cl, -3.363109443710321500e-11f);
        ExpPairAdd(ph, pl, th, tl, ch, cl, mask);
        ExpPairMul(th, tl, ph, pl, rh, rl, mask);
        Duplicate(ch, 8.333333767950534821e-03f);
        Duplicate(cl, -4.346172033375950150e-10f);
        ExpPairAdd(ph, pl, th, tl, ch, cl, mask);
        ExpPairMul(th, tl, ph, pl, rh, rl, mask);
        Duplicate(ch, 4.166666790843009949e-02f);
        Duplicate(cl, -1.241763469828072175e-09f);
        ExpPairAdd(ph, pl, th, tl, ch, cl, mask);
        ExpPairMul(th, tl, ph, pl, rh, rl, mask);
        Duplicate(ch, 1.666666716337203979e-01f);
        Duplicate(cl, -4.967053879312288700e-09f);
        ExpPairAdd(ph, pl, th, tl, ch, cl, mask);
        ExpPairMul(th, tl, ph, pl, rh, rl, mask);
        Duplicate(ch, 5.000000000000000000e-01f);
        Duplicate(cl, 0.000000000000000000e+00f);
        ExpPairAdd(ph, pl, th, tl, ch, cl, mask);
        ExpPairMul(th, tl, ph, pl, rh, rl, mask);
        Duplicate(ch, 1.000000000000000000e+00f);
        Duplicate(cl, 0.000000000000000000e+00f);
        ExpPairAdd(ph, pl, th, tl, ch, cl, mask);
        ExpPairMul(th, tl, ph, pl, rh, rl, mask);
        Duplicate(ch, 1.000000000000000000e+00f);
        Duplicate(cl, 0.000000000000000000e+00f);
        ExpPairAdd(ph, pl, th, tl, ch, cl, mask);
        Add(result, ph, pl, mask);
        static constexpr Reg::CastTrait toInt = {Reg::RegLayout::UNKNOWN, Reg::SatMode::SAT,
                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        Reg::Cast<int32_t, float, toInt>(ni, n, mask);
        Reg::Adds(ni, ni, int32_t(127), mask);
        Duplicate(shift, int32_t(23));
        Reg::ShiftLeft(ni, ni, shift, mask);
        factor = (Reg::RegTensor<float>&)ni;
        Mul(result, result, factor, mask);
        Exp(value, original, mask);
        Reg::CompareScalar<float, CMPMODE::GE>(loMask, original, -80.0f, mask);
        Reg::CompareScalar<float, CMPMODE::LE>(hiMask, original, 80.0f, mask);
        Reg::And(normalMask, loMask, hiMask, mask);
        Reg::Select(value, result, value, normalMask);
    }

    __simd_vf__ inline void SimpleSoftmax(__ubuf__ float* pTensor, __ubuf__ float* dpTensor, __ubuf__ float* lseTensor,
                                          __ubuf__ float* sftFrontTensor, const uint32_t row, const uint32_t col)
    {
        AscendC::Reg::RegTensor<float> p_reg;
        AscendC::Reg::RegTensor<float> p_quant_reg;
        AscendC::Reg::RegTensor<float> lse_reg;
        AscendC::Reg::RegTensor<float> sft_front_reg;
        AscendC::Reg::RegTensor<float> scale_reg;
        AscendC::Reg::RegTensor<float> dp_scale_reg;
        AscendC::Reg::RegTensor<float> p_scale_reg;
        AscendC::Reg::RegTensor<float> dp_reg;
        AscendC::Reg::MaskReg msk_reg;
        constexpr static uint16_t repeat_size = 256 / sizeof(float);
        uint16_t repeat_times = (col + repeat_size - 1) / repeat_size;
        uint32_t ub_offset = 0;
        Duplicate(scale_reg, score_scale_);
        if constexpr (HIF8_ENABLE) {
            Duplicate(dp_scale_reg, dp_scale_);
            Duplicate(p_scale_reg, p_scale_);
        }

        for (int32_t i = 0; i < row; i++) {
            LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(lse_reg, lseTensor + i * 8);
            LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(sft_front_reg, sftFrontTensor + i * 8);
            uint32_t count = col;
            ub_offset = i * col;

            for (int32_t j = 0; j < repeat_times; j++) {
                msk_reg = AscendC::Reg::UpdateMask<float>(count);

                LoadAlign(p_reg, pTensor + ub_offset);
                Mul(p_reg, p_reg, scale_reg, msk_reg);
                Sub(p_reg, p_reg, lse_reg, msk_reg);
                if constexpr (HIF8_ENABLE) {
                    AccurateHif8Exp(p_reg, msk_reg);
                } else {
                    Exp(p_reg, p_reg, msk_reg);
                }
                if constexpr (HIF8_ENABLE) {
                    // Match the static path at HiF8 rounding midpoints.
                    Mul(p_quant_reg, p_reg, p_scale_reg, msk_reg);
                    StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(pTensor + ub_offset, p_quant_reg,
                                                                              msk_reg);
                } else {
                    StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(pTensor + ub_offset, p_reg, msk_reg);
                }
                LoadAlign(dp_reg, dpTensor + ub_offset);
                if constexpr (HIF8_ENABLE) {
                    Mul(dp_reg, dp_reg, dp_scale_reg, msk_reg);
                }
                Sub(dp_reg, dp_reg, sft_front_reg, msk_reg);
                Mul(dp_reg, dp_reg, p_reg, msk_reg);
                StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dpTensor + ub_offset, dp_reg, msk_reg);
                ub_offset += repeat_size;
            }
        }
    }

    __simd_vf__ inline void AbsReduceMax(__ubuf__ float* srcTensor, __ubuf__ float* dstTensor, uint32_t rows,
                                         uint32_t stride, uint32_t validCols)
    {
        AscendC::Reg::RegTensor<float> inputReg;
        AscendC::Reg::RegTensor<float> absReg;
        AscendC::Reg::RegTensor<float> chunkMaxReg;
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::UnalignReg storeReg;
        constexpr uint32_t REPEAT = 256 / sizeof(float);
        Duplicate(maxReg, 0.0f, allMask);
        // Padded KV columns are not part of dS. Their softmax values can
        // contain unrelated L0 padding, and must never select the tile scale.
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t count = validCols;
            uint32_t offset = row * stride;
            while (count > 0) {
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(count);
                LoadAlign(inputReg, srcTensor + offset);
                Abs(absReg, inputReg, mask);
                ReduceMax<float, AscendC::Reg::MaskMergeMode::ZEROING>(chunkMaxReg, absReg, mask);
                Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(maxReg, maxReg, chunkMaxReg, oneMask);
                offset += REPEAT;
            }
        }
        vstus(storeReg, 1, maxReg, dstTensor, POST_UPDATE);
        vstas(storeReg, dstTensor, 0, POST_UPDATE);
    }

    __aicore__ inline void Hif8ScaleHandshake(const RunTimeInfo& runTimeInfo, const LocalTensor<float>& dsTensor,
                                              uint32_t count)
    {
        if (ds_scale_ > 0.0f) {
            // The static multiplier is fused into SimpleSoftmax's existing
            // dS/pScale multiply. This function only owns the dynamic
            // two-lane scale handshake now.
            return;
        }
        const int64_t scaleOffset = static_cast<int64_t>(runTimeInfo.taskId) * SCALE_SLOT_FLOATS;
        // A mixed AIC/AIV kernel exposes the same block index to the two AIV
        // sub-cores paired with one AIC.  Use the sub-block index to keep the
        // two partial maxima in distinct cache-line-sized slots.
        const uint32_t scaleLane = v_sub_core_idx_;
        float amax = 0.0f;
        Duplicate(ds_amax_tensor_, 0.0f, BLOCK_FP32);
        PipeBarrier<PIPE_V>();
        if (ds_scale_ == 0.0f && count > 0) {
            AbsReduceMax((__ubuf__ float*)dsTensor.GetPhyAddr(), (__ubuf__ float*)ds_amax_tensor_.GetPhyAddr(),
                         count / runTimeInfo.s2LenAlign, runTimeInfo.s2LenAlign, runTimeInfo.s2Len);
            PipeBarrier<PIPE_V>();
            SET_FLAG(V, S, EVENT_ID0);
            WAIT_FLAG(V, S, EVENT_ID0);
            amax = ds_amax_tensor_.GetValue(0);
        }
        // Keep the two AIV sub-cores' scalar stores in separate 64-byte cache lines.
        GlobalTensor<float> laneAmaxBlock = ds_scale_workspace_[scaleOffset + scaleLane * SCALE_AMAX_STRIDE];
        laneAmaxBlock.SetValue(0, amax);
        Barrier();
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(laneAmaxBlock);
        Barrier();
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_V2_C45);
        // Both GM maxima must be visible before either scalar load.
        CrossCoreWaitFlag<2, PIPE_S>(FLAG_C2_V2);
        GlobalTensor<float> amaxBlock0 = ds_scale_workspace_[scaleOffset];
        GlobalTensor<float> amaxBlock1 = ds_scale_workspace_[scaleOffset + SCALE_AMAX_STRIDE];
        Barrier();
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(amaxBlock0);
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(amaxBlock1);
        Barrier();
        float amax0 = amaxBlock0.GetValue(0);
        float amax1 = amaxBlock1.GetValue(0);
        float globalAmax = amax0 > amax1 ? amax0 : amax1;
        // Keep one exponent bit of headroom. Encoding an exact 32768 peak
        // reaches the HiF8 reserved boundary on C310; the paired Fixpipe
        // descale compensates this factor exactly.
        float quantScale = globalAmax > 1.0e-12f ? HIF8_DYNAMIC_TARGET / globalAmax : 1.0f;
        // Lane 0 publishes the descale even when its own tail is empty; lane 1
        // may still contain the valid rows of this sparse tile.
        if (scaleLane == 0) {
            GlobalTensor<float> descaleBlock = ds_scale_workspace_[scaleOffset + SCALE_DESCALE_INDEX];
            descaleBlock.SetValue(0, 1.0f / quantScale);
            Barrier();
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(descaleBlock);
            Barrier();
            PipeBarrier<PIPE_ALL>();
        }
        if (count > 0) {
            Muls(dsTensor, dsTensor, quantScale, count);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeEvenCoreInfo(EvenCoreInfo& info, const uint32_t data_size,
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
        info.align_tail = tail / 16 * 16;
        info.pad_tail = tail - info.align_tail;
        if (tail == 0) {
            info.align_tail = max_process_size;
            info.pad_tail = 0;
        }
    }

    __aicore__ inline void CopyInLSE(const LocalTensor<float>& dstTensor, const GlobalTensor<float>& srcTensor,
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
            src_stride = HIF8_ENABLE ? (q_head_num_ - 1) * sizeof(float) : 0;
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
