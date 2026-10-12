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
#include "bsa_common_header.h"
#include "bsa_addr_compute.h"
#include "cube_op_bsag_arch35.h"
#include "vec_op_bsag_arch35.h"
using namespace AscendC;

namespace BSA_ARC35 {

template <typename INPUT_TYPE, uint32_t INPUT_LAYOUT, class TILING_CLASS, bool DETERMINISTIC_ENABLE,
          bool INDEX_ENABLE = false>
struct BSA_TYPE {
    using input_type = INPUT_TYPE;
    static constexpr uint32_t input_layout = INPUT_LAYOUT;
    using tiling_class = TILING_CLASS;
    static constexpr bool deterministic_enable = DETERMINISTIC_ENABLE;
    static constexpr bool index_enable = INDEX_ENABLE;
};

template <typename BSA_TYPE>
class BlockSparseAttentionGradArch35 {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

private:
    AddrComputeModule<BSA_TYPE> addr_;
    GlobalTensor<INPUT_TYPE> query_gm_, key_gm_, val_gm_, dout_gm_, attention_out_gm_;
    GlobalTensor<INPUT_TYPE> dq_gm_, dk_gm_, dv_gm_;
    GlobalTensor<float> dq_workspace_, dk_workspace_, dv_workspace_, sftg_workspace_;
    TBuf<TPosition::VECCALC> ub_buffer_;
    TBuf<TPosition::A1> l1_buffer_;
    LocalTensor<float> mm1_res_ub_tensor_ping_, mm1_res_ub_tensor_pong_;
    LocalTensor<float> mm2_res_ub_tensor_ping_, mm2_res_ub_tensor_pong_;
    LocalTensor<float> mm1_res_ub_tensor_, mm2_res_ub_tensor_;
    LocalTensor<INPUT_TYPE> p_l1_tensor_ping_, p_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> ds_l1_tensor_ping_, ds_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> p_l1_tensor_, ds_l1_tensor_;
    uint32_t taskId = 0;
    uint32_t ping_pong_idx = 0;
    uint32_t last_ping_pong_idx = 0;
    uint32_t ub_offset_ = 0;
    uint32_t l1_offset_ = 0;
    uint32_t kv_ping_pong_idx_ = 0;
    static constexpr int32_t UB_SIZE = 247 * 1024;
    static constexpr int32_t L1_SIZE = 512 * 1024;

public:
    __aicore__ inline BlockSparseAttentionGradArch35(){};

    __aicore__ inline void Process(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                   GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape,
                                   GM_ADDR attentionMask, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR dq,
                                   GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, const TILING_CLASS* tilingData,
                                   TPipe* tPipe)
    {
        uint32_t base_m = tilingData->baseM;
        uint32_t base_n = tilingData->baseN;
        uint32_t vec_base_m = base_m / 2;
        uint32_t vec_base_n = base_n;

        query_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE*)q);
        key_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE*)k);
        val_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE*)v);
        dout_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE*)dout);
        attention_out_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE*)attention_out);
        dq_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE*)dq);
        dk_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE*)dk);
        dv_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE*)dv);
        dq_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->dqWorkspaceOffset));
        dk_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->dkWorkspaceOffset));
        dv_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->dvWorkspaceOffset));
        sftg_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->sftgWorkspaceOffset));
        addr_.Init(tilingData, actualQseqlen, actualKvseqlen, blockSparseMask, workspace);
        tPipe->InitBuffer(ub_buffer_, UB_SIZE);
        tPipe->InitBuffer(l1_buffer_, L1_SIZE);
        // ub_buffer [base_m * base_n * sizeof(float) * 2]
        mm1_res_ub_tensor_ping_ = ub_buffer_.GetWithOffset<float>(vec_base_m * vec_base_n, ub_offset_);
        ub_offset_ += vec_base_m * vec_base_n * sizeof(float);
        mm2_res_ub_tensor_ping_ = ub_buffer_.GetWithOffset<float>(vec_base_m * vec_base_n, ub_offset_);
        ub_offset_ += vec_base_m * vec_base_n * sizeof(float);
        mm1_res_ub_tensor_pong_ = ub_buffer_.GetWithOffset<float>(vec_base_m * vec_base_n, ub_offset_);
        ub_offset_ += vec_base_m * vec_base_n * sizeof(float);
        mm2_res_ub_tensor_pong_ = ub_buffer_.GetWithOffset<float>(vec_base_m * vec_base_n, ub_offset_);
        ub_offset_ += vec_base_m * vec_base_n * sizeof(float);
        // l1_buffer [base_m * base_n * sizeof(INPUT_TYPE) * 4]
        p_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        p_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        ds_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        ds_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);

        if ASCEND_IS_AIC {
            CubeProcess(dout, q, k, v, attention_out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                        actualQseqlen, actualKvseqlen, dq, dk, dv, workspace, tilingData, tPipe);
        }
        if ASCEND_IS_AIV {
            VectorProcess(dout, q, k, v, attention_out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                          actualQseqlen, actualKvseqlen, dq, dk, dv, workspace, tilingData, tPipe);
        }
    }

    __aicore__ inline void kv_reuse_check(RunTimeInfo& lastRunTimeInfo, RunTimeInfo& runTimeInfo,
                                          RunTimeInfo& nextRunTimeInfo, bool first)
    {
        if constexpr (DETERMINISTIC_ENABLE) {
            // 确定性路径：dkv group 语义由 latin-square 状态机给出（last valid in group 会跨过组内无效轮次）
            runTimeInfo.is_kv_change = runTimeInfo.need_copy_kv;
            runTimeInfo.is_kv_end = runTimeInfo.is_singlekv_last;
        } else {
            if (nextRunTimeInfo.need_compute == 0) {
                runTimeInfo.is_kv_end = true;
            } else {
                runTimeInfo.is_kv_end = runTimeInfo.keyGmOffset != nextRunTimeInfo.keyGmOffset;
            }
        }

        if (unlikely(first)) {
            runTimeInfo.is_kv_change = true;
            runTimeInfo.kv_ping_pong_idx = kv_ping_pong_idx_;
            kv_ping_pong_idx_ = (kv_ping_pong_idx_ + 1) % 3;
        }

        if constexpr (DETERMINISTIC_ENABLE) {
            // 无效轮次不参与 K/V 搬运，不占用 L1 buffer，避免旋转计数器与有效块序列错位
            nextRunTimeInfo.is_kv_change = nextRunTimeInfo.need_compute ? nextRunTimeInfo.need_copy_kv : false;
        } else {
            nextRunTimeInfo.is_kv_change = nextRunTimeInfo.keyGmOffset != runTimeInfo.keyGmOffset;
        }
        if (nextRunTimeInfo.is_kv_change) {
            nextRunTimeInfo.kv_ping_pong_idx = kv_ping_pong_idx_;
            kv_ping_pong_idx_ = (kv_ping_pong_idx_ + 1) % 3;
        } else {
            nextRunTimeInfo.kv_ping_pong_idx = runTimeInfo.kv_ping_pong_idx;
        }
    }

    __aicore__ inline void CubeProcess(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                       GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape,
                                       GM_ADDR attentionMask, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR dq,
                                       GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, const TILING_CLASS* tilingData,
                                       TPipe* tPipe)
    {
        CubeOp<BSA_TYPE> cubeOp;
        cubeOp.Init(tilingData, tPipe, l1_buffer_, l1_offset_);
        CrossCoreWaitFlag(FLAG_CUBE_POST);
        this->SetEventFlag();
        RunTimeInfo runTimeInfo_[2];
        RunTimeInfo nextRunTimeInfo_;

        while (true) {
            ping_pong_idx = taskId % 2;
            last_ping_pong_idx = 1 - ping_pong_idx;
            mm1_res_ub_tensor_ = ping_pong_idx ? mm1_res_ub_tensor_ping_ : mm1_res_ub_tensor_pong_;
            mm2_res_ub_tensor_ = ping_pong_idx ? mm2_res_ub_tensor_ping_ : mm2_res_ub_tensor_pong_;
            p_l1_tensor_ = last_ping_pong_idx ? p_l1_tensor_ping_ : p_l1_tensor_pong_;
            ds_l1_tensor_ = last_ping_pong_idx ? ds_l1_tensor_ping_ : ds_l1_tensor_pong_;
            if (unlikely(taskId == 0)) {
                addr_.GetRunTimeInfo(runTimeInfo_[ping_pong_idx]);
            } else {
                runTimeInfo_[ping_pong_idx] = nextRunTimeInfo_;
            }

            if (DETERMINISTIC_ENABLE || runTimeInfo_[ping_pong_idx].need_compute) {
                addr_.GetRunTimeInfo(nextRunTimeInfo_);
                kv_reuse_check(runTimeInfo_[last_ping_pong_idx], runTimeInfo_[ping_pong_idx], nextRunTimeInfo_,
                               taskId == 0);
            }

            if (runTimeInfo_[ping_pong_idx].need_compute) {
                const bool first_cur = (taskId == 0) || (runTimeInfo_[last_ping_pong_idx].need_compute == 0);
                // mm12
                cubeOp.SendMatmulQK(query_gm_, key_gm_, mm1_res_ub_tensor_, runTimeInfo_[ping_pong_idx],
                                    nextRunTimeInfo_, taskId % 3, first_cur);
                cubeOp.SendMatmulDyV(dout_gm_, val_gm_, mm2_res_ub_tensor_, runTimeInfo_[ping_pong_idx],
                                     nextRunTimeInfo_, taskId % 3, first_cur);
                CrossCoreSetFlag<2, PIPE_FIX>(FLAG_C1_V1);
            }

            if (taskId > 0 && runTimeInfo_[last_ping_pong_idx].need_compute) {
                CrossCoreWaitFlag<2, PIPE_MTE1>(FLAG_V1_C3);
                cubeOp.SendMatmulDq(ds_l1_tensor_, dq_workspace_, runTimeInfo_[last_ping_pong_idx], (taskId - 1) % 3);
                cubeOp.SendMatmulDk(ds_l1_tensor_, dk_workspace_, runTimeInfo_[last_ping_pong_idx], (taskId - 1) % 3);
                cubeOp.SendMatmulDv(p_l1_tensor_, dv_workspace_, runTimeInfo_[last_ping_pong_idx], (taskId - 1) % 3);
            }
            if constexpr (DETERMINISTIC_ENABLE) {
                if (taskId > 0) {
                    CrossCoreSetFlag<0, PIPE_FIX>(FLAG_DETER_FIX);
                    CrossCoreWaitFlag<0, PIPE_FIX>(FLAG_DETER_FIX);
                }
            }
            if constexpr (DETERMINISTIC_ENABLE) {
                if (taskId >= addr_.GetDeterMaxRound()) {
                    break;
                }
            } else if (runTimeInfo_[ping_pong_idx].need_compute == false) {
                break;
            }
            taskId++;
        }

        this->WaitEventFlag();
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_CUBE_POST);
    }

    __aicore__ inline void VectorProcess(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                         GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape,
                                         GM_ADDR attentionMask, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                         GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                         const TILING_CLASS* tilingData, TPipe* tPipe)
    {
        VecOp<BSA_TYPE> vecOp;
        vecOp.Init(dout, q, k, v, attention_out, softmaxLse, blockSparseMask, blockShape, attentionMask, actualQseqlen,
                   actualKvseqlen, dq, dk, dv, workspace, tilingData, ub_buffer_, ub_offset_);

        vecOp.SendVecPre(dq_workspace_, dk_workspace_, dv_workspace_, dout_gm_, attention_out_gm_, sftg_workspace_,
                         tilingData, ub_buffer_);
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        vecOp.SendVecSftgFront(dout_gm_, attention_out_gm_, sftg_workspace_, tilingData, ub_buffer_);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_CUBE_POST);
        this->SetEventFlag();
        RunTimeInfo runTimeInfo_;
        RunTimeInfo nextRunTimeInfo_;
        bool prev_need_compute = true; // 上一次迭代(cur)是否真的做了计算/预取

        while (true) {
            ping_pong_idx = taskId % 2;
            last_ping_pong_idx = 1 - ping_pong_idx;
            mm1_res_ub_tensor_ = ping_pong_idx ? mm1_res_ub_tensor_ping_ : mm1_res_ub_tensor_pong_;
            mm2_res_ub_tensor_ = ping_pong_idx ? mm2_res_ub_tensor_ping_ : mm2_res_ub_tensor_pong_;
            p_l1_tensor_ = ping_pong_idx ? p_l1_tensor_ping_ : p_l1_tensor_pong_;
            ds_l1_tensor_ = ping_pong_idx ? ds_l1_tensor_ping_ : ds_l1_tensor_pong_;
            if (unlikely(taskId == 0)) {
                addr_.GetRunTimeInfo(runTimeInfo_);
            } else {
                runTimeInfo_ = nextRunTimeInfo_;
            }

            if (DETERMINISTIC_ENABLE || runTimeInfo_.need_compute) {
                addr_.GetRunTimeInfo(nextRunTimeInfo_);
            }

            if (runTimeInfo_.need_compute) {
                const bool first_cur = (taskId == 0) || (!prev_need_compute);
                vecOp.SendVecSftPreProcess(runTimeInfo_, nextRunTimeInfo_, ping_pong_idx, first_cur);
                CrossCoreWaitFlag<2, PIPE_V>(FLAG_C1_V1);
                vecOp.SendVecSoftmax(p_l1_tensor_, ds_l1_tensor_, mm1_res_ub_tensor_, mm2_res_ub_tensor_, runTimeInfo_);
                CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_V1_C3);
            }
            prev_need_compute = runTimeInfo_.need_compute;

            if constexpr (DETERMINISTIC_ENABLE) {
                if (taskId >= addr_.GetDeterMaxRound()) {
                    break;
                }
            } else if (runTimeInfo_.need_compute == false) {
                break;
            }
            taskId++;
        }

        this->WaitEventFlag();
        PipeBarrier<PIPE_ALL>();
        CrossCoreWaitFlag(FLAG_CUBE_POST);
        SyncAll();
        vecOp.SendVecPost(dq_gm_, dk_gm_, dv_gm_, dq_workspace_, dk_workspace_, dv_workspace_, tilingData, ub_buffer_);
    }

    __aicore__ inline void SetEventFlag()
    {
        if ASCEND_IS_AIC {
            SET_FLAG(M, MTE1, EVENT_ID3);
            SET_FLAG(M, MTE1, EVENT_ID4);
            SET_FLAG(FIX, M, EVENT_ID3);
            SET_FLAG(FIX, M, EVENT_ID4);
        } else {
            SET_FLAG(V, MTE2, EVENT_ID3);
            SET_FLAG(V, MTE2, EVENT_ID4);
        }
    }

    __aicore__ inline void WaitEventFlag()
    {
        if ASCEND_IS_AIC {
            WAIT_FLAG(M, MTE1, EVENT_ID3);
            WAIT_FLAG(M, MTE1, EVENT_ID4);
            WAIT_FLAG(FIX, M, EVENT_ID3);
            WAIT_FLAG(FIX, M, EVENT_ID4);
        } else {
            WAIT_FLAG(V, MTE2, EVENT_ID3);
            WAIT_FLAG(V, MTE2, EVENT_ID4);
        }
    }
};

} // namespace BSA_ARC35
