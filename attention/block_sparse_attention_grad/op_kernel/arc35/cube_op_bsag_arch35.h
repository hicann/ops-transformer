/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
using namespace AscendC;

namespace BSA_ARC35 {

template <typename BSA_TYPE>
class CubeOp {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    static constexpr bool HIF8_ENABLE = BSA_TYPE::hif8_enable;
    static constexpr uint32_t INPUT_C0_SIZE = 32 / sizeof(INPUT_TYPE);
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;

private:
    // Pending full-tile dK/dV prefixes must be flushed before a tail resets L0C.
    // FIX->M events 1/2 protect shared ping/pong L0C, 5/6 protect dK/dV.
    // Events 0 and 3/4 belong to the legacy immediate/launch protocols.
    // Each event is consumed before the same buffer is written again.
    bool l0c_fix_pending_[4]{false, false, false, false};
    bool dk_accum_pending_{false};
    bool dv_accum_pending_{false};
    bool deep_pipeline_{false};
    bool direct_dkv_{false};
    float direct_dv_scale_{1.0f};
    GlobalTensor<bfloat16_t> direct_dk_gm_, direct_dv_gm_;
    LocalTensor<float> dq_transport_ub_;
    uint32_t dq_ready_flag_{0};
    int32_t batch_num_;
    int32_t q_seq_len_;
    int32_t kv_seq_len_;
    int32_t q_group_;
    int32_t q_head_num_;
    int32_t kv_head_num_;
    int32_t head_dim_;
    int32_t head_dim_align_;
    int32_t base_m_;
    int32_t base_n_;
    int32_t event_ping_pong_flag{0};
    uint32_t q_stride_;
    uint32_t kv_stride_;
    float q_descale_{1.0f};
    float k_descale_{1.0f};
    float ds_scale_{1.0f};
    float softmax_scale_{1.0f};
    GlobalTensor<float> ds_scale_workspace_;
    static constexpr uint32_t SCALE_SLOT_FLOATS = 64;
    static constexpr uint32_t SCALE_DESCALE_INDEX = 32;
    // l1_tensor
    LocalTensor<INPUT_TYPE> query_l1_tensor_ping_;
    LocalTensor<INPUT_TYPE> query_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> query_l1_tensor_ding_;
    LocalTensor<INPUT_TYPE> query_l1_tensor_fourth_;
    LocalTensor<INPUT_TYPE> key_l1_tensor_ping_;
    LocalTensor<INPUT_TYPE> key_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> key_l1_tensor_ding_;
    LocalTensor<INPUT_TYPE> key_l1_tensor_fourth_;
    LocalTensor<INPUT_TYPE> value_l1_tensor_ping_;
    LocalTensor<INPUT_TYPE> value_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> value_l1_tensor_ding_;
    LocalTensor<INPUT_TYPE> value_l1_tensor_fourth_;
    LocalTensor<INPUT_TYPE> dy_l1_tensor_ping_;
    LocalTensor<INPUT_TYPE> dy_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> dy_l1_tensor_ding_;
    LocalTensor<INPUT_TYPE> dy_l1_tensor_fourth_;
    // l0_tensor
    LocalTensor<INPUT_TYPE> l0_a_tensor_ping_;
    LocalTensor<INPUT_TYPE> l0_a_tensor_pong_;
    LocalTensor<INPUT_TYPE> l0_b_tensor_ping_;
    LocalTensor<INPUT_TYPE> l0_b_tensor_pong_;
    LocalTensor<float> l0_c_tensor_ping_;
    LocalTensor<float> l0_c_tensor_pong_;
    // dk\dvL0C
    LocalTensor<float> l0_c_dk_tensor_;
    LocalTensor<float> l0_c_dv_tensor_;
    static constexpr int32_t L1_SIZE = 512 * 1024;
    static constexpr int32_t L0_AB_SIZE = 64 * 1024;
    static constexpr int32_t L0_C_SIZE = 256 * 1024;
    static constexpr int32_t BASE_BLOCK_SIZE = 128 * 128;
    TEventID event_ping_ = EVENT_ID3;
    TEventID event_pong_ = EVENT_ID4;

public:
    __aicore__ inline CubeOp(){};
    __aicore__ inline void SetDeepPipeline(bool enabled)
    {
        deep_pipeline_ = enabled;
    }
    __aicore__ inline void SetDirectDkv(bool enabled, GM_ADDR dk, GM_ADDR dv, float dvScale)
    {
        direct_dkv_ = enabled;
        direct_dv_scale_ = dvScale;
        direct_dk_gm_.SetGlobalBuffer((__gm__ bfloat16_t*)dk);
        direct_dv_gm_.SetGlobalBuffer((__gm__ bfloat16_t*)dv);
    }
    __aicore__ inline void SetDqTransport(const LocalTensor<float>& ub, uint32_t readyFlag)
    {
        dq_transport_ub_ = ub;
        dq_ready_flag_ = readyFlag;
    }
    __aicore__ inline void Init(const TILING_CLASS* tilingData, GM_ADDR qDescale, GM_ADDR kDescale, GM_ADDR dsScale,
                                GM_ADDR workspace, TPipe* tPipe, TBuf<TPosition::A1>& l1_buffer, uint32_t l1_offset)
    {
        this->batch_num_ = tilingData->batchNum;
        this->q_seq_len_ = tilingData->qSeqLen;
        this->kv_seq_len_ = tilingData->kvSeqLen;
        this->q_group_ = tilingData->qGroup;
        this->q_head_num_ = tilingData->qHeadNum;
        this->kv_head_num_ = tilingData->kvHeadNum;
        this->head_dim_ = tilingData->headDim;
        this->head_dim_align_ = RoundUp(head_dim_, 16);
        this->base_m_ = tilingData->baseM;
        this->base_n_ = tilingData->baseN;
        this->softmax_scale_ = tilingData->softmaxScale;
        if constexpr (HIF8_ENABLE) {
            this->q_descale_ = *((__gm__ float*)qDescale);
            this->k_descale_ = *((__gm__ float*)kDescale);
            this->ds_scale_ = *((__gm__ float*)dsScale);
            ds_scale_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->dsScaleWorkspaceOffset));
        }
        uint32_t base_m = tilingData->baseM;
        uint32_t base_n = tilingData->baseN;
        if constexpr (INPUT_LAYOUT == BSND) {
            q_stride_ = q_head_num_ * head_dim_;
            kv_stride_ = kv_head_num_ * head_dim_;
        } else if constexpr (INPUT_LAYOUT == BNSD) {
            q_stride_ = head_dim_;
            kv_stride_ = head_dim_;
        } else if constexpr (INPUT_LAYOUT == TND) {
            q_stride_ = q_head_num_ * head_dim_;
            kv_stride_ = kv_head_num_ * head_dim_;
        }
        TBuf<TPosition::A2> l0_a_buffer_;
        TBuf<TPosition::B2> l0_b_buffer_;
        TBuf<TPosition::CO1> l0_c_buffer_;
        tPipe->InitBuffer(l0_a_buffer_, L0_AB_SIZE);
        tPipe->InitBuffer(l0_b_buffer_, L0_AB_SIZE);
        tPipe->InitBuffer(l0_c_buffer_, L0_C_SIZE);
        // query_l1
        query_l1_tensor_ping_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align_, l1_offset);
        l1_offset += base_m * head_dim_align_ * sizeof(INPUT_TYPE);
        query_l1_tensor_pong_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align_, l1_offset);
        l1_offset += base_m * head_dim_align_ * sizeof(INPUT_TYPE);
        query_l1_tensor_ding_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align_, l1_offset);
        l1_offset += base_m * head_dim_align_ * sizeof(INPUT_TYPE);
        if (deep_pipeline_) {
            query_l1_tensor_fourth_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align_, l1_offset);
            l1_offset += base_m * head_dim_align_ * sizeof(INPUT_TYPE);
        }
        // key_l1
        key_l1_tensor_ping_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
        l1_offset += base_n * head_dim_align_ * sizeof(INPUT_TYPE);
        key_l1_tensor_pong_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
        l1_offset += base_n * head_dim_align_ * sizeof(INPUT_TYPE);
        key_l1_tensor_ding_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
        l1_offset += base_n * head_dim_align_ * sizeof(INPUT_TYPE);
        if (deep_pipeline_) {
            key_l1_tensor_fourth_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
            l1_offset += base_n * head_dim_align_ * sizeof(INPUT_TYPE);
        }
        // value_l1
        value_l1_tensor_ping_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
        l1_offset += base_n * head_dim_align_ * sizeof(INPUT_TYPE);
        value_l1_tensor_pong_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
        l1_offset += base_n * head_dim_align_ * sizeof(INPUT_TYPE);
        value_l1_tensor_ding_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
        l1_offset += base_n * head_dim_align_ * sizeof(INPUT_TYPE);
        if (deep_pipeline_) {
            value_l1_tensor_fourth_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_n * head_dim_align_, l1_offset);
            l1_offset += base_n * head_dim_align_ * sizeof(INPUT_TYPE);
        }
        // dy is the left matrix of dY * V^T, so its row count follows Q/base_m.
        dy_l1_tensor_ping_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align_, l1_offset);
        l1_offset += base_m * head_dim_align_ * sizeof(INPUT_TYPE);
        dy_l1_tensor_pong_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align_, l1_offset);
        l1_offset += base_m * head_dim_align_ * sizeof(INPUT_TYPE);
        dy_l1_tensor_ding_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align_, l1_offset);
        l1_offset += base_m * head_dim_align_ * sizeof(INPUT_TYPE);
        if (deep_pipeline_) {
            dy_l1_tensor_fourth_ = l1_buffer.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align_, l1_offset);
            l1_offset += base_m * head_dim_align_ * sizeof(INPUT_TYPE);
        }
        // l0
        l0_a_tensor_ping_ = l0_a_buffer_.GetWithOffset<INPUT_TYPE>(32 * 1024 / sizeof(INPUT_TYPE), 0);
        l0_a_tensor_pong_ = l0_a_buffer_.GetWithOffset<INPUT_TYPE>(32 * 1024 / sizeof(INPUT_TYPE), 32 * 1024);
        l0_b_tensor_ping_ = l0_b_buffer_.GetWithOffset<INPUT_TYPE>(32 * 1024 / sizeof(INPUT_TYPE), 0);
        l0_b_tensor_pong_ = l0_b_buffer_.GetWithOffset<INPUT_TYPE>(32 * 1024 / sizeof(INPUT_TYPE), 32 * 1024);
        l0_c_tensor_ping_ = l0_c_buffer_.GetWithOffset<float>(64 * 1024 / sizeof(float), 0);
        l0_c_tensor_pong_ = l0_c_buffer_.GetWithOffset<float>(64 * 1024 / sizeof(float), 64 * 1024);
        l0_c_dk_tensor_ = l0_c_buffer_.GetWithOffset<float>(64 * 1024 / sizeof(float), 128 * 1024);
        l0_c_dv_tensor_ = l0_c_buffer_.GetWithOffset<float>(64 * 1024 / sizeof(float), 192 * 1024);
    }

    __aicore__ inline bool UseAsyncHif8Fixpipe() const
    {
        // Only optimize the conservative static-scale launch paths. Preserve
        // existing UnitFlag behavior for aligned single-sequence launches and
        // the original dynamic-scale synchronization protocol.
        return HIF8_ENABLE && ds_scale_ > 0.0f &&
               ((INPUT_LAYOUT == TND && batch_num_ > 1) || q_seq_len_ % base_m_ != 0);
    }

    __aicore__ inline void WaitL0cFixpipe(uint32_t slot)
    {
        if (l0c_fix_pending_[slot]) {
            WAIT_FLAG(FIX, M, static_cast<TEventID>(slot < 2 ? slot + 1 : slot + 3));
            l0c_fix_pending_[slot] = false;
        }
    }

    __aicore__ inline void PublishL0cFixpipe(uint32_t slot)
    {
        SET_FLAG(FIX, M, static_cast<TEventID>(slot < 2 ? slot + 1 : slot + 3));
        l0c_fix_pending_[slot] = true;
    }

    __aicore__ inline void DrainPendingFixpipe()
    {
        if constexpr (HIF8_ENABLE) {
            for (uint32_t slot = 0; slot < 4; ++slot)
                WaitL0cFixpipe(slot);
        }
    }

    __aicore__ inline void PrepareDsScale(const RunTimeInfo& runTimeInfo)
    {
        if constexpr (!HIF8_ENABLE) {
            return;
        }
        // Static dS quantization needs no cross-core reduction: both AIC and
        // AIV already receive the same multiplier tensor. Keeping it out of
        // the GM handshake also avoids a needless round trip per sparse tile.
        if (ds_scale_ > 0.0f) {
            return;
        }
        // Wait until both AIV lanes publish their maxima in separate 64-byte
        // cache lines, then release them to compute the shared quantization scale.
        // Scale exchange includes scalar GM loads/stores: wait on the scalar pipeline.
        CrossCoreWaitFlag<2, PIPE_S>(FLAG_V2_C45);
        CrossCoreSetFlag<2, PIPE_FIX>(FLAG_C2_V2);
    }

    __aicore__ inline bool UseHif8UnitFlag(const RunTimeInfo& runTimeInfo) const
    {
        // C310's UnitFlag fast path requires a complete base tile. Keep the
        // explicit event protocol for sequence tails so their padded rows and
        // columns retain the validated behavior.
        // Dynamic dS publishes a per-tile descale through the AIV/AIC
        // handshake. Keep that mode on the explicit event path; advancing the
        // Cube/Fixpipe pipeline before both AIV lanes publish the scale can
        // consume an incomplete tile.
        // Multi-sequence TND can interleave aligned and partial sequences on
        // one core. Keep one explicit synchronization protocol for the launch.
        return HIF8_ENABLE && (INPUT_LAYOUT != TND || batch_num_ == 1) && ds_scale_ > 0.0f &&
               runTimeInfo.cur_q_seq_len % base_m_ == 0 && runTimeInfo.s1Len == base_m_ && runTimeInfo.s2Len == base_n_;
    }

    __aicore__ inline void SendMatmulQK(const GlobalTensor<INPUT_TYPE>& queryGm, const GlobalTensor<INPUT_TYPE>& keyGm,
                                        const LocalTensor<float>& mm1OutUb, const RunTimeInfo& runTimeInfo,
                                        const RunTimeInfo& nextRunTimeInfo, const uint32_t l1_buffer_flag, bool first)
    {
        LocalTensor<INPUT_TYPE> l1_a_tensor =
            l1_buffer_flag == 0 ?
                query_l1_tensor_ping_ :
                (l1_buffer_flag == 1 ? query_l1_tensor_pong_ :
                                       (l1_buffer_flag == 2 ? query_l1_tensor_ding_ : query_l1_tensor_fourth_));
        LocalTensor<INPUT_TYPE> l1_b_tensor =
            runTimeInfo.kv_ping_pong_idx == 0 ?
                key_l1_tensor_ping_ :
                (runTimeInfo.kv_ping_pong_idx == 1 ?
                     key_l1_tensor_pong_ :
                     (runTimeInfo.kv_ping_pong_idx == 2 ? key_l1_tensor_ding_ : key_l1_tensor_fourth_));

        uint32_t next_flag = (l1_buffer_flag + 1) % (deep_pipeline_ ? 4 : 3);
        LocalTensor<INPUT_TYPE> next_l1_a_tensor =
            next_flag == 0 ? query_l1_tensor_ping_ :
                             (next_flag == 1 ? query_l1_tensor_pong_ :
                                               (next_flag == 2 ? query_l1_tensor_ding_ : query_l1_tensor_fourth_));
        LocalTensor<INPUT_TYPE> next_l1_b_tensor =
            nextRunTimeInfo.kv_ping_pong_idx == 0 ?
                key_l1_tensor_ping_ :
                (nextRunTimeInfo.kv_ping_pong_idx == 1 ?
                     key_l1_tensor_pong_ :
                     (nextRunTimeInfo.kv_ping_pong_idx == 2 ? key_l1_tensor_ding_ : key_l1_tensor_fourth_));

        ComputeMM12(queryGm, keyGm, l1_a_tensor, l1_b_tensor, next_l1_a_tensor, next_l1_b_tensor, mm1OutUb, runTimeInfo,
                    nextRunTimeInfo, first);
        event_ping_pong_flag = 1 - event_ping_pong_flag;
    }

    __aicore__ inline void SendMatmulDyV(const GlobalTensor<INPUT_TYPE>& dyGm, const GlobalTensor<INPUT_TYPE>& valueGm,
                                         LocalTensor<float>& mm2OutUb, const RunTimeInfo& runTimeInfo,
                                         const RunTimeInfo& nextRunTimeInfo, const uint32_t l1_buffer_flag, bool first)
    {
        LocalTensor<INPUT_TYPE> l1_a_tensor =
            l1_buffer_flag == 0 ?
                dy_l1_tensor_ping_ :
                (l1_buffer_flag == 1 ? dy_l1_tensor_pong_ :
                                       (l1_buffer_flag == 2 ? dy_l1_tensor_ding_ : dy_l1_tensor_fourth_));
        LocalTensor<INPUT_TYPE> l1_b_tensor =
            runTimeInfo.kv_ping_pong_idx == 0 ?
                value_l1_tensor_ping_ :
                (runTimeInfo.kv_ping_pong_idx == 1 ?
                     value_l1_tensor_pong_ :
                     (runTimeInfo.kv_ping_pong_idx == 2 ? value_l1_tensor_ding_ : value_l1_tensor_fourth_));

        uint32_t next_flag = (l1_buffer_flag + 1) % (deep_pipeline_ ? 4 : 3);
        LocalTensor<INPUT_TYPE> next_l1_a_tensor =
            next_flag == 0 ?
                dy_l1_tensor_ping_ :
                (next_flag == 1 ? dy_l1_tensor_pong_ : (next_flag == 2 ? dy_l1_tensor_ding_ : dy_l1_tensor_fourth_));
        LocalTensor<INPUT_TYPE> next_l1_b_tensor =
            nextRunTimeInfo.kv_ping_pong_idx == 0 ?
                value_l1_tensor_ping_ :
                (nextRunTimeInfo.kv_ping_pong_idx == 1 ?
                     value_l1_tensor_pong_ :
                     (nextRunTimeInfo.kv_ping_pong_idx == 2 ? value_l1_tensor_ding_ : value_l1_tensor_fourth_));

        ComputeMM12(dyGm, valueGm, l1_a_tensor, l1_b_tensor, next_l1_a_tensor, next_l1_b_tensor, mm2OutUb, runTimeInfo,
                    nextRunTimeInfo, first);
        event_ping_pong_flag = 1 - event_ping_pong_flag;
    }

    __aicore__ inline void SendMatmulDq(const LocalTensor<INPUT_TYPE>& ds_l1_tensor, const GlobalTensor<float>& outGm,
                                        const RunTimeInfo& runTimeInfo, const uint32_t l1_buffer_flag)
    {
        LocalTensor<INPUT_TYPE> l1_b_tensor =
            runTimeInfo.kv_ping_pong_idx == 0 ?
                key_l1_tensor_ping_ :
                (runTimeInfo.kv_ping_pong_idx == 1 ?
                     key_l1_tensor_pong_ :
                     (runTimeInfo.kv_ping_pong_idx == 2 ? key_l1_tensor_ding_ : key_l1_tensor_fourth_));

        ComputeMMDQ(ds_l1_tensor, l1_b_tensor, outGm[runTimeInfo.queryGmOffset], runTimeInfo);
        event_ping_pong_flag = 1 - event_ping_pong_flag;
    }

    __aicore__ inline void SendMatmulDk(const LocalTensor<INPUT_TYPE>& ds_l1_tensor, const GlobalTensor<float>& outGm,
                                        const RunTimeInfo& runTimeInfo, const uint32_t l1_buffer_flag)
    {
        LocalTensor<INPUT_TYPE> l1_b_tensor =
            l1_buffer_flag == 0 ?
                query_l1_tensor_ping_ :
                (l1_buffer_flag == 1 ? query_l1_tensor_pong_ :
                                       (l1_buffer_flag == 2 ? query_l1_tensor_ding_ : query_l1_tensor_fourth_));
        ComputeMMDKV<DK>(ds_l1_tensor, l1_b_tensor, outGm[runTimeInfo.keyGmOffset], runTimeInfo);
        event_ping_pong_flag = 1 - event_ping_pong_flag;
    }

    __aicore__ inline void SendMatmulDv(const LocalTensor<INPUT_TYPE>& p_l1_tensor, const GlobalTensor<float>& outGm,
                                        const RunTimeInfo& runTimeInfo, const uint32_t l1_buffer_flag)
    {
        LocalTensor<INPUT_TYPE> l1_b_tensor =
            l1_buffer_flag == 0 ?
                dy_l1_tensor_ping_ :
                (l1_buffer_flag == 1 ? dy_l1_tensor_pong_ :
                                       (l1_buffer_flag == 2 ? dy_l1_tensor_ding_ : dy_l1_tensor_fourth_));
        ComputeMMDKV<DV>(p_l1_tensor, l1_b_tensor, outGm[runTimeInfo.keyGmOffset], runTimeInfo);
        event_ping_pong_flag = 1 - event_ping_pong_flag;
    }

private:
    __aicore__ inline void ComputeMM12(const GlobalTensor<INPUT_TYPE>& leftGm, const GlobalTensor<INPUT_TYPE>& rightGm,
                                       const LocalTensor<INPUT_TYPE>& l1_a_tensor,
                                       const LocalTensor<INPUT_TYPE>& l1_b_tensor,
                                       const LocalTensor<INPUT_TYPE>& next_l1_a_tensor,
                                       const LocalTensor<INPUT_TYPE>& next_l1_b_tensor, const LocalTensor<float>& outUb,
                                       const RunTimeInfo& runTimeInfo, const RunTimeInfo& nextRunTimeInfo, bool first)
    {
        int32_t mProcess = runTimeInfo.s1Len;
        int32_t nProcess = runTimeInfo.s2Len;
        int32_t mProcessAlign = runTimeInfo.s1LenAlign;
        int32_t nProcessAlign = runTimeInfo.s2LenAlign;
        LocalTensor<INPUT_TYPE> l0_a_tensor = event_ping_pong_flag ? l0_a_tensor_ping_ : l0_a_tensor_pong_;
        LocalTensor<INPUT_TYPE> l0_b_tensor = event_ping_pong_flag ? l0_b_tensor_ping_ : l0_b_tensor_pong_;
        LocalTensor<float> l0_c_tensor = event_ping_pong_flag ? l0_c_tensor_ping_ : l0_c_tensor_pong_;
        TEventID evnet_id = event_ping_pong_flag ? event_ping_ : event_pong_;

        if (first) {
            SET_FLAG(MTE1, MTE2, EVENT_ID1);
            WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
            load_data_gm_2_l1(l1_a_tensor, leftGm[runTimeInfo.queryGmOffset], mProcess, head_dim_, mProcessAlign,
                              head_dim_align_, q_stride_);
            load_data_gm_2_l1(l1_b_tensor, rightGm[runTimeInfo.keyGmOffset], nProcess, head_dim_, nProcessAlign,
                              head_dim_align_, kv_stride_);
            SET_FLAG(MTE2, MTE1, EVENT_ID0);
            WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
        }

        if (nextRunTimeInfo.need_compute) {
            SET_FLAG(MTE1, MTE2, EVENT_ID0);
            WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
            load_data_gm_2_l1(next_l1_a_tensor, leftGm[nextRunTimeInfo.queryGmOffset], nextRunTimeInfo.s1Len, head_dim_,
                              nextRunTimeInfo.s1LenAlign, head_dim_align_, q_stride_);
            if (nextRunTimeInfo.is_kv_change) {
                load_data_gm_2_l1(next_l1_b_tensor, rightGm[nextRunTimeInfo.keyGmOffset], nextRunTimeInfo.s2Len,
                                  head_dim_, nextRunTimeInfo.s2LenAlign, head_dim_align_, kv_stride_);
            }
            SET_FLAG(MTE2, MTE1, EVENT_ID2);
        }

        if (!first) {
            WAIT_FLAG(MTE2, MTE1, EVENT_ID2);
        }
        WAIT_FLAG(M, MTE1, evnet_id);
        load_data_l1_2_l0_nz(l0_a_tensor, l1_a_tensor, mProcessAlign, head_dim_align_);
        load_data_l1_2_l0_nz(l0_b_tensor, l1_b_tensor, nProcessAlign, head_dim_align_);

        SET_FLAG(MTE1, M, EVENT_ID0);
        WAIT_FLAG(MTE1, M, EVENT_ID0);

        MmadParams madParams;
        // HiF8 L0/Fixpipe strides use the 32-aligned extent. An unaligned
        // MMAD M selects a different internal stride for the same L0C buffer.
        madParams.m = HIF8_ENABLE ? mProcessAlign : (mProcess == 1 ? 2 : mProcess);
        madParams.n = HIF8_ENABLE ? nProcessAlign : nProcess;
        madParams.k = head_dim_;
        madParams.cmatrixInitVal = true;
        const bool useUnitFlag = UseHif8UnitFlag(runTimeInfo);
        if constexpr (HIF8_ENABLE) {
            madParams.disableGemv = true;
        }
        madParams.unitFlag = HIF8_ENABLE ? (useUnitFlag ? 3 : 0) : 3;
        if (UseAsyncHif8Fixpipe())
            WaitL0cFixpipe(event_ping_pong_flag ? 0 : 1);
        AscendC::Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);
        if constexpr (HIF8_ENABLE) {
            // C310 requires an explicit M-pipeline barrier for a small MMAD
            // result. Both Q*K^T and dO*V^T are 32x32 in the sparse-block
            // case (four L0C fractals), which is below the hardware helper's
            // ten-fractal threshold.
            if (!useUnitFlag && (mProcessAlign / 16) * (nProcessAlign / 16) < 10) {
                PipeBarrier<PIPE_M>();
            }
        }
        SET_FLAG(M, MTE1, evnet_id);

        if (HIF8_ENABLE && !useUnitFlag) {
            // unitFlag is disabled for the independent HiF8 sparse tiles, so
            // explicitly wait until Mmad has finished producing L0C before
            // Fixpipe consumes it.
            SET_FLAG(M, FIX, EVENT_ID0);
            WAIT_FLAG(M, FIX, EVENT_ID0);
        }

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.mSize = mProcessAlign;
        fixpipeParams.nSize = nProcessAlign;
        fixpipeParams.srcStride = mProcessAlign;
        fixpipeParams.dstStride = nProcessAlign;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.unitFlag = HIF8_ENABLE ? (useUnitFlag ? 3 : 0) : 3;
        constexpr static FixpipeConfig ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};
        AscendC::Fixpipe<float, float, ROW_MAJOR_UB>(outUb, l0_c_tensor, fixpipeParams);
        if (HIF8_ENABLE && !useUnitFlag) {
            // Close the explicit pipeline dependency in this call. This keeps
            // the shared EVENT_ID3/4 ping-pong protocol unchanged for legacy
            // FP16/BF16 kernels and also protects the next L0C reuse.
            if (UseAsyncHif8Fixpipe()) {
                PublishL0cFixpipe(event_ping_pong_flag ? 0 : 1);
            } else {
                SET_FLAG(FIX, M, EVENT_ID0);
                WAIT_FLAG(FIX, M, EVENT_ID0);
            }
        }
    }

    template <uint32_t TYPE>
    __aicore__ inline void CopyOutHif8Dkv(const GlobalTensor<float>& outGm, const LocalTensor<float>& l0_c_tensor,
                                          const RunTimeInfo& runTimeInfo, bool useUnitFlag)
    {
        const int32_t nProcess = runTimeInfo.s2Len;
        const int32_t nProcessAlign = runTimeInfo.s2LenAlign;
        if (!useUnitFlag) {
            SET_FLAG(M, FIX, EVENT_ID0);
            WAIT_FLAG(M, FIX, EVENT_ID0);
        }
        // HiF8 matmul on C310 uses a 32-element M0.  Use the native
        // C310 parameter type so Fixpipe keeps the 32-element source
        // stride instead of applying the V220 compatibility layout.
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.mSize = nProcess;
        fixpipeParams.nSize = head_dim_;
        fixpipeParams.srcStride = nProcessAlign;
        fixpipeParams.dstStride = kv_stride_;
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        fixpipeParams.unitFlag = useUnitFlag ? 3 : 0;
        if constexpr (TYPE == DK) {
            if (ds_scale_ <= 0.0f) {
                const int64_t scaleOffset =
                    static_cast<int64_t>(runTimeInfo.taskId) * SCALE_SLOT_FLOATS + SCALE_DESCALE_INDEX;
                GlobalTensor<float> descaleBlock = ds_scale_workspace_[scaleOffset];
                Barrier();
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(descaleBlock);
                Barrier();
                float dsDescale = descaleBlock.GetValue(0);
                float deqScaleValue = softmax_scale_ * q_descale_ * dsDescale;
                fixpipeParams.quantPre = QuantMode_t::QF322F32_PRE;
                fixpipeParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&deqScaleValue));
            }
        }
        if (direct_dkv_) {
            // All partial products for this output block have accumulated in
            // FP32 L0C. Round only once, after applying the final output scale.
            const float scale = TYPE == DK ? softmax_scale_ * q_descale_ * (1.0f / ds_scale_) : direct_dv_scale_;
            fixpipeParams.quantPre = QuantMode_t::QF322BF16_PRE;
            fixpipeParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<const uint32_t*>(&scale));
            constexpr static FixpipeConfig ROW_MAJOR_GM = {CO2Layout::ROW_MAJOR, false};
            const auto dst =
                TYPE == DK ? direct_dk_gm_[runTimeInfo.keyGmOffset] : direct_dv_gm_[runTimeInfo.keyGmOffset];
            AscendC::Fixpipe<bfloat16_t, float, ROW_MAJOR_GM>(dst, l0_c_tensor, fixpipeParams);
        } else if (ds_scale_ > 0.0f && runTimeInfo.exclusive_kv_group) {
            MM345CopyOutC310<false>(outGm, l0_c_tensor, fixpipeParams);
        } else {
            MM345CopyOutC310<true>(outGm, l0_c_tensor, fixpipeParams);
        }
        if (!useUnitFlag) {
            if (UseAsyncHif8Fixpipe()) {
                PublishL0cFixpipe(TYPE == DK ? 2 : 3);
            } else {
                SET_FLAG(FIX, M, EVENT_ID0);
                WAIT_FLAG(FIX, M, EVENT_ID0);
            }
        }
    }

    template <uint32_t TYPE>
    __aicore__ inline void ComputeMMDKV(const LocalTensor<INPUT_TYPE>& l1_a_tensor,
                                        const LocalTensor<INPUT_TYPE>& l1_b_tensor, const GlobalTensor<float>& outGm,
                                        const RunTimeInfo& runTimeInfo)
    {
        int32_t mProcess = runTimeInfo.s1Len;
        int32_t nProcess = runTimeInfo.s2Len;
        int32_t mProcessAlign = runTimeInfo.s1LenAlign;
        int32_t nProcessAlign = runTimeInfo.s2LenAlign;
        LocalTensor<INPUT_TYPE> l0_a_tensor = event_ping_pong_flag ? l0_a_tensor_ping_ : l0_a_tensor_pong_;
        LocalTensor<INPUT_TYPE> l0_b_tensor = event_ping_pong_flag ? l0_b_tensor_ping_ : l0_b_tensor_pong_;
        LocalTensor<float> l0_c_tensor;
        if constexpr (TYPE == DK) {
            l0_c_tensor = l0_c_dk_tensor_;
        } else {
            l0_c_tensor = l0_c_dv_tensor_;
        }
        TEventID evnet_id = event_ping_pong_flag ? event_ping_ : event_pong_;

        WAIT_FLAG(M, MTE1, evnet_id);
        load_data_l1_2_l0_zn(l0_a_tensor, l1_a_tensor, mProcessAlign, nProcessAlign);
        load_data_l1_2_l0_zn(l0_b_tensor, l1_b_tensor, mProcessAlign, head_dim_align_);
        SET_FLAG(MTE1, M, EVENT_ID0);
        WAIT_FLAG(MTE1, M, EVENT_ID0);

        MmadParams madParams;
        madParams.m = HIF8_ENABLE ? nProcessAlign : (nProcess == 1 ? 2 : nProcess);
        madParams.n = head_dim_;
        madParams.k = mProcess;
        // Keep one FP32 partial sum per core-local KV group. A partial tile
        // cannot reset the dedicated L0C buffer until the full-tile prefix
        // has been written. Static dS/P scales are uniform within this group.
        // Dynamic dS keeps the original dK/dV decisions unchanged.
        const bool prefixMode = HIF8_ENABLE && ds_scale_ > 0.0f;
        const bool accumulate =
            !HIF8_ENABLE || (prefixMode ? true :
                                          ((INPUT_LAYOUT != TND || batch_num_ == 1) && TYPE != DK &&
                                           runTimeInfo.cur_q_seq_len % base_m_ == 0 && nProcess == base_n_));
        const bool useUnitFlag = UseHif8UnitFlag(runTimeInfo);
        bool pending = TYPE == DK ? dk_accum_pending_ : dv_accum_pending_;
        if constexpr (HIF8_ENABLE) {
            if (prefixMode && !accumulate && pending) {
                // The prefix belongs to this same KV group: group-end writes
                // clear pending before the next key/batch can be selected.
                CopyOutHif8Dkv<TYPE>(outGm, l0_c_tensor, runTimeInfo, false);
                pending = false;
            }
        }
        madParams.cmatrixInitVal =
            prefixMode ? (accumulate ? !pending : true) : (accumulate ? runTimeInfo.is_kv_change : true);
        if constexpr (HIF8_ENABLE) {
            madParams.disableGemv = true;
        }
        madParams.unitFlag = HIF8_ENABLE ? (useUnitFlag ? ((!accumulate || runTimeInfo.is_kv_end) ? 3 : 2) : 0) :
                                           (runTimeInfo.is_kv_end ? 3 : 2);
        if (UseAsyncHif8Fixpipe())
            WaitL0cFixpipe(TYPE == DK ? 2 : 3);
        AscendC::Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);
        SET_FLAG(M, MTE1, evnet_id);

        const bool copyOutCurrentTile = !accumulate || runTimeInfo.is_kv_end;
        if (copyOutCurrentTile) {
            if constexpr (HIF8_ENABLE) {
                CopyOutHif8Dkv<TYPE>(outGm, l0_c_tensor, runTimeInfo, useUnitFlag);
            } else {
                AscendC::FixpipeParamsV220 fixpipeParamsV220;
                fixpipeParamsV220.mSize = nProcess;
                fixpipeParamsV220.nSize = head_dim_;
                fixpipeParamsV220.srcStride = nProcessAlign;
                fixpipeParamsV220.unitFlag = 3;
                fixpipeParamsV220.dstStride = kv_stride_;
                MM345CopyOut<true>(outGm, l0_c_tensor, fixpipeParamsV220);
            }
        }
        if constexpr (HIF8_ENABLE) {
            // Only a full tile not yet at the KV-group end remains in L0C.
            if constexpr (TYPE == DK) {
                dk_accum_pending_ = prefixMode && accumulate && !copyOutCurrentTile;
            } else {
                dv_accum_pending_ = prefixMode && accumulate && !copyOutCurrentTile;
            }
        }
    }

    __aicore__ inline void ComputeMMDQ(const LocalTensor<INPUT_TYPE>& l1_a_tensor,
                                       const LocalTensor<INPUT_TYPE>& l1_b_tensor, const GlobalTensor<float>& outGm,
                                       const RunTimeInfo& runTimeInfo)
    {
        int32_t mProcess = runTimeInfo.s1Len;
        int32_t nProcess = runTimeInfo.s2Len;
        int32_t mProcessAlign = runTimeInfo.s1LenAlign;
        int32_t nProcessAlign = runTimeInfo.s2LenAlign;
        LocalTensor<INPUT_TYPE> l0_a_tensor = event_ping_pong_flag ? l0_a_tensor_ping_ : l0_a_tensor_pong_;
        LocalTensor<INPUT_TYPE> l0_b_tensor = event_ping_pong_flag ? l0_b_tensor_ping_ : l0_b_tensor_pong_;
        LocalTensor<float> l0_c_tensor = event_ping_pong_flag ? l0_c_tensor_ping_ : l0_c_tensor_pong_;
        TEventID evnet_id = event_ping_pong_flag ? event_ping_ : event_pong_;

        WAIT_FLAG(M, MTE1, evnet_id);
        load_data_l1_2_l0_nz(l0_a_tensor, l1_a_tensor, mProcessAlign, nProcessAlign);
        load_data_l1_2_l0_zn(l0_b_tensor, l1_b_tensor, nProcessAlign, head_dim_align_);
        SET_FLAG(MTE1, M, EVENT_ID0);
        WAIT_FLAG(MTE1, M, EVENT_ID0);

        MmadParams madParams;
        // HiF8 L0/Fixpipe strides use the 32-aligned extent. An unaligned
        // MMAD M selects a different internal stride for the same L0C buffer.
        madParams.m = HIF8_ENABLE ? mProcessAlign : (mProcess == 1 ? 2 : mProcess);
        madParams.n = head_dim_;
        madParams.k = nProcess;
        madParams.cmatrixInitVal = true;
        const bool useUnitFlag = UseHif8UnitFlag(runTimeInfo);
        if constexpr (HIF8_ENABLE) {
            madParams.disableGemv = true;
        }
        madParams.unitFlag = HIF8_ENABLE ? (useUnitFlag ? 3 : 0) : 3;
        if (UseAsyncHif8Fixpipe())
            WaitL0cFixpipe(event_ping_pong_flag ? 0 : 1);
        AscendC::Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);
        SET_FLAG(M, MTE1, evnet_id);

        if constexpr (HIF8_ENABLE) {
            if (!useUnitFlag) {
                SET_FLAG(M, FIX, EVENT_ID0);
                WAIT_FLAG(M, FIX, EVENT_ID0);
            }
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
            fixpipeParams.mSize = mProcess;
            fixpipeParams.nSize = head_dim_;
            fixpipeParams.srcStride = mProcessAlign;
            fixpipeParams.dstStride = q_stride_;
            fixpipeParams.dualDstCtl = 0;
            fixpipeParams.params.ndNum = 1;
            fixpipeParams.params.srcNdStride = 0;
            fixpipeParams.params.dstNdStride = 0;
            fixpipeParams.unitFlag = useUnitFlag ? 3 : 0;
            if (ds_scale_ <= 0.0f) {
                const int64_t scaleOffset =
                    static_cast<int64_t>(runTimeInfo.taskId) * SCALE_SLOT_FLOATS + SCALE_DESCALE_INDEX;
                GlobalTensor<float> descaleBlock = ds_scale_workspace_[scaleOffset];
                Barrier();
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(descaleBlock);
                Barrier();
                float dsDescale = descaleBlock.GetValue(0);
                float deqScaleValue = softmax_scale_ * k_descale_ * dsDescale;
                fixpipeParams.quantPre = QuantMode_t::QF322F32_PRE;
                fixpipeParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&deqScaleValue));
            }
            if (deep_pipeline_) {
                // Dedicated DQ transport allows the next MM12 slot to be
                // written while MTE3 drains the previous gradient.
                fixpipeParams.mSize = mProcessAlign;
                fixpipeParams.dstStride = head_dim_;
                fixpipeParams.dualDstCtl = 1;
                constexpr static FixpipeConfig DQ_UB = {CO2Layout::ROW_MAJOR, true};
                AscendC::Fixpipe<float, float, DQ_UB>(dq_transport_ub_, l0_c_tensor, fixpipeParams);
                CrossCoreSetFlag<2, PIPE_FIX>(dq_ready_flag_);
            } else {
                MM345CopyOutC310<true>(outGm, l0_c_tensor, fixpipeParams);
            }
            if (!useUnitFlag) {
                if (UseAsyncHif8Fixpipe()) {
                    PublishL0cFixpipe(event_ping_pong_flag ? 0 : 1);
                } else {
                    SET_FLAG(FIX, M, EVENT_ID0);
                    WAIT_FLAG(FIX, M, EVENT_ID0);
                }
            }
        } else {
            AscendC::FixpipeParamsV220 fixpipeParamsV220;
            fixpipeParamsV220.mSize = mProcess;
            fixpipeParamsV220.nSize = head_dim_;
            fixpipeParamsV220.srcStride = mProcessAlign;
            fixpipeParamsV220.unitFlag = 3;
            fixpipeParamsV220.dstStride = q_stride_;
            MM345CopyOut<true>(outGm, l0_c_tensor, fixpipeParamsV220);
        }
    }

    __aicore__ inline void load_data_gm_2_l1(const LocalTensor<INPUT_TYPE>& dstL1Tensor,
                                             const GlobalTensor<INPUT_TYPE>& srcGmTensor, const int32_t mSize,
                                             const int32_t kSize, const int32_t mSizeAlign, const int32_t kSizeAlign,
                                             const int32_t srcStride)
    {
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = mSize;
        nd2nzPara.dValue = kSize;
        nd2nzPara.srcDValue = srcStride;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzC0Stride = mSizeAlign;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.dstNzMatrixStride = 0;
        // Keep the standard ND2NZ overload. The boolean overload selects the
        // small-C0 compatibility mode and is not the HiF8/C0=32 switch.
        AscendC::DataCopy(dstL1Tensor, srcGmTensor, nd2nzPara);
    }

    __aicore__ inline void load_data_l1_2_l0_nz(const LocalTensor<INPUT_TYPE>& dstL0Tensor,
                                                const LocalTensor<INPUT_TYPE>& srcL1Tensor, const int32_t mSizeAlign,
                                                const int32_t kSizeAlign)
    {
        /*
         * Load 左矩阵非转置 or 右矩阵转置
         */
        AscendC::LoadData2DParamsV2 loadData2dParams;
        loadData2dParams.mStartPosition = 0;
        loadData2dParams.kStartPosition = 0;
        loadData2dParams.mStep = mSizeAlign / 16;
        loadData2dParams.kStep = kSizeAlign / INPUT_C0_SIZE;
        loadData2dParams.srcStride = mSizeAlign / 16;
        loadData2dParams.dstStride = mSizeAlign / 16;
        loadData2dParams.ifTranspose = false;

        AscendC::LoadData(dstL0Tensor, srcL1Tensor, loadData2dParams);
    }

    __aicore__ inline void load_data_l1_2_l0_zn(const LocalTensor<INPUT_TYPE>& dstL0Tensor,
                                                const LocalTensor<INPUT_TYPE>& srcL1Tensor, const int32_t mSizeAlign,
                                                const int32_t kSizeAlign)
    {
        /*
         * Load 左矩阵转置 or 右矩阵非转置
         */
        AscendC::LoadData2DParamsV2 loadData2dParams;
        loadData2dParams.mStartPosition = 0;
        loadData2dParams.kStartPosition = 0;
        loadData2dParams.ifTranspose = true;
        if constexpr (HIF8_ENABLE) {
            // C310 HiF8 transpose is a single 32-element-M0 load.  Splitting
            // it into FP16-style 16-row loads produces an unaligned MTE
            // address on the second fragment.
            loadData2dParams.mStep = ((mSizeAlign + 31) >> 5) << 1;
            loadData2dParams.kStep = (kSizeAlign * sizeof(INPUT_TYPE) + 31) / 32;
            loadData2dParams.srcStride = (mSizeAlign + 15) >> 4;
            loadData2dParams.dstStride = ((kSizeAlign + 31) >> 5) << 1;
            AscendC::LoadData(dstL0Tensor, srcL1Tensor, loadData2dParams);
        } else {
            loadData2dParams.mStep = 1;
            loadData2dParams.kStep = kSizeAlign / INPUT_C0_SIZE;
            loadData2dParams.srcStride = mSizeAlign / 16;
            loadData2dParams.dstStride = 1;
            for (int32_t i = 0; i < mSizeAlign / 16; i++) {
                loadData2dParams.mStartPosition = i;
                AscendC::LoadData(dstL0Tensor[i * kSizeAlign * 16], srcL1Tensor, loadData2dParams);
            }
        }
    }

    template <bool A_MATRIX>
    __aicore__ inline void load_data_gm_2_l0(const LocalTensor<INPUT_TYPE>& dstL0Tensor,
                                             const LocalTensor<INPUT_TYPE>& dstL1Tensor,
                                             const GlobalTensor<INPUT_TYPE>& srcGmTensor, const int32_t mSize,
                                             const int32_t kSize, const int32_t mSizeAlign, const int32_t kSizeAlign,
                                             const int32_t srcKStride)
    {
        load_data_gm_2_l1(dstL1Tensor, srcGmTensor, mSize, kSize, mSizeAlign, kSizeAlign, srcKStride);
        SET_FLAG(MTE2, MTE1, EVENT_ID0);
        WAIT_FLAG(MTE2, MTE1, EVENT_ID0);

        if constexpr (A_MATRIX) {
            load_data_l1_2_l0_nz(dstL0Tensor, dstL1Tensor, mSizeAlign, kSizeAlign);
        } else {
            load_data_l1_2_l0_zn(dstL0Tensor, dstL1Tensor, mSizeAlign, kSizeAlign);
        }
    }

    template <bool A_MATRIX>
    __aicore__ inline void load_data_gm_2_l0_trans(const LocalTensor<INPUT_TYPE>& dstL0Tensor,
                                                   const LocalTensor<INPUT_TYPE>& dstL1Tensor,
                                                   const GlobalTensor<INPUT_TYPE>& srcGmTensor, const int32_t mSize,
                                                   const int32_t kSize, const int32_t mSizeAlign,
                                                   const int32_t kSizeAlign, const int32_t srcKStride)
    {
        load_data_gm_2_l1(dstL1Tensor, srcGmTensor, mSize, kSize, mSizeAlign, kSizeAlign, srcKStride);
        SET_FLAG(MTE2, MTE1, EVENT_ID0);
        WAIT_FLAG(MTE2, MTE1, EVENT_ID0);

        if constexpr (A_MATRIX) {
            load_data_l1_2_l0_zn(dstL0Tensor, dstL1Tensor, mSizeAlign, kSizeAlign);
        } else {
            load_data_l1_2_l0_nz(dstL0Tensor, dstL1Tensor, mSizeAlign, kSizeAlign);
        }
    }

    template <bool ATMOIC_ADD>
    __aicore__ inline void MM345CopyOut(const GlobalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                        const AscendC::FixpipeParamsV220& fixpipeParamsV220)
    {
        if constexpr (ATMOIC_ADD) {
            AscendC::SetAtomicType<float>();
        }

        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(dstTensor, srcTensor, fixpipeParamsV220);

        if constexpr (ATMOIC_ADD) {
            AscendC::SetAtomicNone();
        }
    }

    template <bool ATMOIC_ADD>
    __aicore__ inline void MM345CopyOutC310(const GlobalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                            const FixpipeParamsC310<CO2Layout::ROW_MAJOR>& fixpipeParams)
    {
        if constexpr (ATMOIC_ADD) {
            // C310 Fixpipe uses the explicit atomic-add API. SetAtomicType is
            // the legacy V220 form and leaves FP32 accumulation undefined here.
            AscendC::SetAtomicAdd<float>();
        }

        constexpr static FixpipeConfig ROW_MAJOR_GM = {CO2Layout::ROW_MAJOR, false};
        if constexpr (ATMOIC_ADD && HIF8_ENABLE) {
            if (fixpipeParams.quantPre == QuantMode_t::QF322F32_PRE) {
                // C310's scalar dequant multiplier keeps 10 fraction bits.
                // Dynamic dS needs the full FP32 scale: split its significand
                // into at most three exactly representable scalar multipliers.
                auto preciseParams = fixpipeParams;
                uint32_t scaleBits = static_cast<uint32_t>(fixpipeParams.deqScalar);
                float remainder = *reinterpret_cast<float*>(&scaleBits);
                for (uint32_t part = 0; part < 3 && remainder != 0.0f; ++part) {
                    uint32_t partBits = *reinterpret_cast<uint32_t*>(&remainder) & 0xffffe000U;
                    const float component = *reinterpret_cast<float*>(&partBits);
                    preciseParams.deqScalar = static_cast<uint64_t>(partBits);
                    AscendC::Fixpipe<float, float, ROW_MAJOR_GM>(dstTensor, srcTensor, preciseParams);
                    remainder -= component;
                }
            } else {
                AscendC::Fixpipe<float, float, ROW_MAJOR_GM>(dstTensor, srcTensor, fixpipeParams);
            }
        } else {
            AscendC::Fixpipe<float, float, ROW_MAJOR_GM>(dstTensor, srcTensor, fixpipeParams);
        }

        if constexpr (ATMOIC_ADD) {
            AscendC::SetAtomicNone();
        }
    }
};

} // namespace BSA_ARC35
