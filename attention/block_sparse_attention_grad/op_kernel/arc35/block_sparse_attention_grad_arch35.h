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
          bool INDEX_ENABLE = false, typename ATTENTION_TYPE = INPUT_TYPE, typename OUTPUT_TYPE = INPUT_TYPE,
          bool HIF8_ENABLE = false>
struct BSA_TYPE {
    using input_type = INPUT_TYPE;
    using attention_type = ATTENTION_TYPE;
    using output_type = OUTPUT_TYPE;
    static constexpr uint32_t input_layout = INPUT_LAYOUT;
    using tiling_class = TILING_CLASS;
    static constexpr bool deterministic_enable = DETERMINISTIC_ENABLE;
    static constexpr bool index_enable = INDEX_ENABLE;
    static constexpr bool hif8_enable = HIF8_ENABLE;
};

template <typename BSA_TYPE>
class BlockSparseAttentionGradArch35 {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    using ATTENTION_TYPE = typename BSA_TYPE::attention_type;
    using OUTPUT_TYPE = typename BSA_TYPE::output_type;
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

private:
    AddrComputeModule<BSA_TYPE> addr_;
    GlobalTensor<INPUT_TYPE> query_gm_, key_gm_, val_gm_, dout_gm_;
    GlobalTensor<ATTENTION_TYPE> attention_out_gm_;
    GlobalTensor<OUTPUT_TYPE> dq_gm_, dk_gm_, dv_gm_;
    GlobalTensor<float> dq_workspace_, dk_workspace_, dv_workspace_, sftg_workspace_;
    TBuf<TPosition::VECCALC> ub_buffer_;
    TBuf<TPosition::A1> l1_buffer_;
    LocalTensor<float> mm1_res_ub_tensor_ping_, mm1_res_ub_tensor_pong_;
    LocalTensor<float> mm2_res_ub_tensor_ping_, mm2_res_ub_tensor_pong_;
    LocalTensor<float> mm1_res_ub_tensor_, mm2_res_ub_tensor_;
    LocalTensor<INPUT_TYPE> p_l1_tensor_ping_, p_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> ds_l1_tensor_ping_, ds_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> p_l1_tensor_, ds_l1_tensor_;
    bool deep_pipeline_{false};
    bool direct_dkv_{false};
    LocalTensor<float> dedicated_dq_ub_;
    uint32_t q_output_stride_{0};
    LocalTensor<float> mm1_res_ub_tensor_third_, mm2_res_ub_tensor_third_;
    LocalTensor<INPUT_TYPE> p_l1_tensor_third_, ds_l1_tensor_third_;
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
                                   GM_ADDR attentionMask, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                   GM_ADDR qDescale, GM_ADDR kDescale, GM_ADDR vDescale, GM_ADDR doDescale,
                                   GM_ADDR pScale, GM_ADDR dsScale, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                   GM_ADDR workspace, const TILING_CLASS* tilingData, TPipe* tPipe)
    {
        if constexpr (BSA_TYPE::hif8_enable && !DETERMINISTIC_ENABLE) {
            // Dedicated DQ transport fits after removing two unused LSE buffers
            // in the combined-row-metadata path. Match VecOp's selection.
            int64_t metadataSeqLen = tilingData->qSeqLen;
            if constexpr (INPUT_LAYOUT == TND) {
                if (tilingData->batchNum > 1) {
                    metadataSeqLen = 0;
                    int64_t previous = 0;
                    for (int64_t b = 0; b < tilingData->batchNum; ++b) {
                        int64_t end = ((__gm__ int64_t*)actualQseqlen)[b];
                        metadataSeqLen = IMax(metadataSeqLen, end - previous);
                        previous = end;
                    }
                }
            }
            const bool combinedMetadata = (INPUT_LAYOUT == BNSD || metadataSeqLen < 131072) && metadataSeqLen >= 4096;
            deep_pipeline_ = combinedMetadata && tilingData->headDim == 128 && tilingData->baseM == 128 &&
                             tilingData->baseN == 128 && tilingData->qSeqLen >= 32768 &&
                             *((__gm__ float*)dsScale) > 0.0f;
        }
        q_output_stride_ = INPUT_LAYOUT == BNSD ? tilingData->headDim : tilingData->qHeadNum * tilingData->headDim;
        uint32_t base_m = tilingData->baseM;
        uint32_t base_n = tilingData->baseN;
        uint32_t vec_base_m = base_m / 2;
        uint32_t vec_base_n = base_n;

        query_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE*)q);
        key_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE*)k);
        val_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE*)v);
        dout_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE*)dout);
        attention_out_gm_.SetGlobalBuffer((__gm__ ATTENTION_TYPE*)attention_out);
        dq_gm_.SetGlobalBuffer((__gm__ OUTPUT_TYPE*)dq);
        dk_gm_.SetGlobalBuffer((__gm__ OUTPUT_TYPE*)dk);
        dv_gm_.SetGlobalBuffer((__gm__ OUTPUT_TYPE*)dv);
        dq_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->dqWorkspaceOffset));
        dk_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->dkWorkspaceOffset));
        dv_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->dvWorkspaceOffset));
        sftg_workspace_.SetGlobalBuffer((__gm__ float*)(workspace + tilingData->sftgWorkspaceOffset));
        if constexpr (BSA_TYPE::hif8_enable) {
            if (tilingData->qSeqLen >= 200000) {
                query_gm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                dout_gm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
        }
        addr_.Init(tilingData, actualQseqlen, actualKvseqlen, blockSparseMask, workspace);
        if constexpr (BSA_TYPE::hif8_enable && !DETERMINISTIC_ENABLE) {
            direct_dkv_ = addr_.CanUseUniqueKvOwner() && *((__gm__ float*)dsScale) > 0.0f;
            addr_.SetUniqueKvOwner(direct_dkv_);
        }
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
        if (deep_pipeline_) {
            mm1_res_ub_tensor_third_ = ub_buffer_.GetWithOffset<float>(vec_base_m * vec_base_n, ub_offset_);
            ub_offset_ += vec_base_m * vec_base_n * sizeof(float);
            mm2_res_ub_tensor_third_ = ub_buffer_.GetWithOffset<float>(vec_base_m * vec_base_n, ub_offset_);
            ub_offset_ += vec_base_m * vec_base_n * sizeof(float);
            dedicated_dq_ub_ = ub_buffer_.GetWithOffset<float>(vec_base_m * 128, ub_offset_);
            ub_offset_ += vec_base_m * 128 * sizeof(float);
        }
        // l1_buffer [base_m * base_n * sizeof(INPUT_TYPE) * 4]
        p_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        p_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        ds_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        ds_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        if (deep_pipeline_) {
            p_l1_tensor_third_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
            l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
            ds_l1_tensor_third_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
            l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        }

        if ASCEND_IS_AIC {
            CubeProcess(dout, q, k, v, attention_out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                        actualQseqlen, actualKvseqlen, qDescale, kDescale, doDescale, pScale, dsScale, dq, dk, dv,
                        workspace, tilingData, tPipe);
        }
        if ASCEND_IS_AIV {
            VectorProcess(dout, q, k, v, attention_out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                          actualQseqlen, actualKvseqlen, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, dq,
                          dk, dv, workspace, tilingData, tPipe);
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
            kv_ping_pong_idx_ = (kv_ping_pong_idx_ + 1) % (deep_pipeline_ ? 4 : 3);
        }

        if constexpr (DETERMINISTIC_ENABLE) {
            // 无效轮次不参与 K/V 搬运，不占用 L1 buffer，避免旋转计数器与有效块序列错位
            nextRunTimeInfo.is_kv_change = nextRunTimeInfo.need_compute ? nextRunTimeInfo.need_copy_kv : false;
        } else {
            nextRunTimeInfo.is_kv_change = nextRunTimeInfo.keyGmOffset != runTimeInfo.keyGmOffset;
        }
        if (nextRunTimeInfo.is_kv_change) {
            nextRunTimeInfo.kv_ping_pong_idx = kv_ping_pong_idx_;
            kv_ping_pong_idx_ = (kv_ping_pong_idx_ + 1) % (deep_pipeline_ ? 4 : 3);
        } else {
            nextRunTimeInfo.kv_ping_pong_idx = runTimeInfo.kv_ping_pong_idx;
        }
    }

    __aicore__ inline uint32_t DqReadyFlag(uint32_t slot) const
    {
        return 3;
    }
    __aicore__ inline void StoreDeepDq(const RunTimeInfo& info, uint32_t slot)
    {
        CrossCoreWaitFlag<2, PIPE_MTE3>(DqReadyFlag(slot));
        const uint32_t half = info.s1LenAlign / 2;
        const uint32_t row = GetSubBlockIdx() * half;
        const uint32_t count = info.s1Len > row ? IMin(half, info.s1Len - row) : 0;
        if (count != 0) {
            DataCopyExtParams copy;
            copy.blockCount = count;
            copy.blockLen = 128 * sizeof(float);
            copy.srcStride = 0;
            copy.dstStride = (q_output_stride_ - 128) * sizeof(float);
            SetAtomicAdd<float>();
            DataCopyPad(dq_workspace_[info.queryGmOffset + row * q_output_stride_], dedicated_dq_ub_, copy);
            SetAtomicNone();
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(0);
    }
    __aicore__ inline LocalTensor<float> Mm1Slot(uint32_t slot)
    {
        return slot == 0 ? mm1_res_ub_tensor_pong_ : (slot == 1 ? mm1_res_ub_tensor_ping_ : mm1_res_ub_tensor_third_);
    }
    __aicore__ inline LocalTensor<float> Mm2Slot(uint32_t slot)
    {
        return slot == 0 ? mm2_res_ub_tensor_pong_ : (slot == 1 ? mm2_res_ub_tensor_ping_ : mm2_res_ub_tensor_third_);
    }
    __aicore__ inline LocalTensor<INPUT_TYPE> PSlot(uint32_t slot)
    {
        return slot == 0 ? p_l1_tensor_pong_ : (slot == 1 ? p_l1_tensor_ping_ : p_l1_tensor_third_);
    }
    __aicore__ inline LocalTensor<INPUT_TYPE> DsSlot(uint32_t slot)
    {
        return slot == 0 ? ds_l1_tensor_pong_ : (slot == 1 ? ds_l1_tensor_ping_ : ds_l1_tensor_third_);
    }
    __aicore__ inline void CubeProcessDeep(CubeOp<BSA_TYPE>& cubeOp)
    {
        // Three independent MM12/PDS slots, four input slots including next-tile prefetch.
        // L0C remains 256 KiB; static dS and non-deterministic D128 only.
        RunTimeInfo info[3]{};
        RunTimeInfo next{};
        uint32_t step = 0;
        uint32_t drainRounds = 0;
        bool draining = false;
        while (true) {
            const uint32_t slot = step % 3;
            const uint32_t previous = (step + 2) % 3;
            if (!draining) {
                if (step == 0)
                    addr_.GetRunTimeInfo(info[slot]);
                else
                    info[slot] = next;
                info[slot].taskId = GetBlockIdx() * 2 + step % 2;
                if (info[slot].need_compute) {
                    addr_.GetRunTimeInfo(next);
                    kv_reuse_check(info[previous], info[slot], next, step == 0);
                    auto mm1 = Mm1Slot(slot);
                    auto mm2 = Mm2Slot(slot);
                    cubeOp.SendMatmulQK(query_gm_, key_gm_, mm1, info[slot], next, step % 4, step == 0);
                    CrossCoreSetFlag<2, PIPE_FIX>(6 + slot * 3);
                    cubeOp.SendMatmulDyV(dout_gm_, val_gm_, mm2, info[slot], next, step % 4, step == 0);
                    CrossCoreSetFlag<2, PIPE_FIX>(7 + slot * 3);
                } else {
                    draining = true;
                    drainRounds = 2;
                }
            } else {
                info[slot].need_compute = 0;
            }
            if (step >= 2) {
                const uint32_t oldSlot = (step - 2) % 3;
                if (info[oldSlot].need_compute) {
                    CrossCoreWaitFlag<2, PIPE_MTE1>(8 + oldSlot * 3);
                    auto p = PSlot(oldSlot);
                    auto ds = DsSlot(oldSlot);
                    // Dedicated dK/dV L0C buffers can progress before the dQ
                    // transport-free wait and MM12 L0C reuse dependency.
                    cubeOp.SendMatmulDk(ds, dk_workspace_, info[oldSlot], (step - 2) % 4);
                    cubeOp.SendMatmulDv(p, dv_workspace_, info[oldSlot], (step - 2) % 4);
                    CrossCoreWaitFlag<2, PIPE_FIX>(0);
                    cubeOp.SetDqTransport(dedicated_dq_ub_, DqReadyFlag(oldSlot));
                    cubeOp.SendMatmulDq(ds, dq_workspace_, info[oldSlot], (step - 2) % 4);
                }
            }
            if (draining && --drainRounds == 0)
                break;
            ++step;
        }
        CrossCoreWaitFlag<2, PIPE_FIX>(0);
    }

    __aicore__ inline void CubeProcess(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                       GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape,
                                       GM_ADDR attentionMask, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                       GM_ADDR qDescale, GM_ADDR kDescale, GM_ADDR doDescale, GM_ADDR pScale,
                                       GM_ADDR dsScale, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                       const TILING_CLASS* tilingData, TPipe* tPipe)
    {
        CubeOp<BSA_TYPE> cubeOp;
        cubeOp.SetDeepPipeline(deep_pipeline_);
        if constexpr (BSA_TYPE::hif8_enable) {
            cubeOp.SetDirectDkv(direct_dkv_, dk, dv, *((__gm__ float*)doDescale) / *((__gm__ float*)pScale));
        }
        cubeOp.Init(tilingData, qDescale, kDescale, dsScale, workspace, tPipe, l1_buffer_, l1_offset_);
        CrossCoreWaitFlag(FLAG_CUBE_POST);
        this->SetEventFlag();
        if (deep_pipeline_) {
            CubeProcessDeep(cubeOp);
            cubeOp.DrainPendingFixpipe();
            this->WaitEventFlag();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_CUBE_POST);
            return;
        }
        bool earlyQk = false;
        if constexpr (BSA_TYPE::hif8_enable)
            earlyQk = *((__gm__ float*)dsScale) > 0.0f;
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

            if constexpr (BSA_TYPE::hif8_enable) {
                runTimeInfo_[ping_pong_idx].taskId = GetBlockIdx() * 2 + ping_pong_idx;
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
                if (earlyQk)
                    CrossCoreSetFlag<2, PIPE_FIX>(FLAG_C1_V1);
                cubeOp.SendMatmulDyV(dout_gm_, val_gm_, mm2_res_ub_tensor_, runTimeInfo_[ping_pong_idx],
                                     nextRunTimeInfo_, taskId % 3, first_cur);
                if (earlyQk) {
                    // FLAG_C2_V2 is unused by the static scale handshake.
                    CrossCoreSetFlag<2, PIPE_FIX>(FLAG_C2_V2);
                } else {
                    CrossCoreSetFlag<2, PIPE_FIX>(FLAG_C1_V1);
                }
                if constexpr (BSA_TYPE::hif8_enable) {
                    cubeOp.PrepareDsScale(runTimeInfo_[ping_pong_idx]);
                }
            }

            if (taskId > 0 && runTimeInfo_[last_ping_pong_idx].need_compute) {
                if constexpr (BSA_TYPE::hif8_enable) {
                    if (!earlyQk) {
                        // Dynamic descale is read by the scalar pipeline in dQ/dK.
                        CrossCoreWaitFlag<2, PIPE_S>(FLAG_V1_C3);
                    } else {
                        CrossCoreWaitFlag<2, PIPE_MTE1>(FLAG_V1_C3);
                    }
                } else {
                    CrossCoreWaitFlag<2, PIPE_MTE1>(FLAG_V1_C3);
                }
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

        cubeOp.DrainPendingFixpipe();
        this->WaitEventFlag();
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_CUBE_POST);
    }

    __aicore__ inline void VectorProcess(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                         GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape,
                                         GM_ADDR attentionMask, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                         GM_ADDR qDescale, GM_ADDR kDescale, GM_ADDR vDescale, GM_ADDR doDescale,
                                         GM_ADDR pScale, GM_ADDR dsScale, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                         GM_ADDR workspace, const TILING_CLASS* tilingData, TPipe* tPipe)
    {
        VecOp<BSA_TYPE> vecOp;
        vecOp.Init(dout, q, k, v, attention_out, softmaxLse, blockSparseMask, blockShape, attentionMask, actualQseqlen,
                   actualKvseqlen, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, dq, dk, dv, workspace,
                   tilingData, ub_buffer_, ub_offset_);

        vecOp.SetDirectDkv(direct_dkv_, dk, dv);
        vecOp.SendPreprocess(dq_workspace_, dk_workspace_, dv_workspace_, dout_gm_, attention_out_gm_, sftg_workspace_,
                             tilingData, ub_buffer_);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        vecOp.PrepareHif8Workspace();
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_CUBE_POST);
        this->SetEventFlag();
        RunTimeInfo runTimeInfo_;
        RunTimeInfo nextRunTimeInfo_;
        RunTimeInfo history[3]{};
        if (deep_pipeline_)
            CrossCoreSetFlag<2, PIPE_MTE3>(0);
        bool prev_need_compute = true; // 上一次迭代(cur)是否真的做了计算/预取

        while (true) {
            ping_pong_idx = taskId % 2;
            last_ping_pong_idx = 1 - ping_pong_idx;
            mm1_res_ub_tensor_ = ping_pong_idx ? mm1_res_ub_tensor_ping_ : mm1_res_ub_tensor_pong_;
            mm2_res_ub_tensor_ = ping_pong_idx ? mm2_res_ub_tensor_ping_ : mm2_res_ub_tensor_pong_;
            p_l1_tensor_ = ping_pong_idx ? p_l1_tensor_ping_ : p_l1_tensor_pong_;
            ds_l1_tensor_ = ping_pong_idx ? ds_l1_tensor_ping_ : ds_l1_tensor_pong_;
            if (deep_pipeline_) {
                const uint32_t slot = taskId % 3;
                mm1_res_ub_tensor_ = Mm1Slot(slot);
                mm2_res_ub_tensor_ = Mm2Slot(slot);
                p_l1_tensor_ = PSlot(slot);
                ds_l1_tensor_ = DsSlot(slot);
                vecOp.SetDpReadyFlag(7 + slot * 3);
            }

            if (unlikely(taskId == 0)) {
                addr_.GetRunTimeInfo(runTimeInfo_);
            } else {
                runTimeInfo_ = nextRunTimeInfo_;
            }

            if constexpr (BSA_TYPE::hif8_enable) {
                // Normalize the physical AIV index to its paired AIC index.
                runTimeInfo_.taskId = (GetBlockIdx() / 2) * 2 + ping_pong_idx;
            }

            if (DETERMINISTIC_ENABLE || runTimeInfo_.need_compute) {
                addr_.GetRunTimeInfo(nextRunTimeInfo_);
            }

            if (runTimeInfo_.need_compute) {
                const bool first_cur = (taskId == 0) || (!prev_need_compute);
                vecOp.SendVecSftPreProcess(runTimeInfo_, nextRunTimeInfo_, ping_pong_idx, first_cur);
                CrossCoreWaitFlag<2, PIPE_V>(deep_pipeline_ ? 6 + (taskId % 3) * 3 : FLAG_C1_V1);
                vecOp.SendVecSoftmax(p_l1_tensor_, ds_l1_tensor_, mm1_res_ub_tensor_, mm2_res_ub_tensor_, runTimeInfo_);
                CrossCoreSetFlag<2, PIPE_MTE3>(deep_pipeline_ ? 8 + (taskId % 3) * 3 : FLAG_V1_C3);
            }
            if (deep_pipeline_) {
                history[taskId % 3] = runTimeInfo_;
                if (taskId >= 2) {
                    uint32_t old = (taskId - 2) % 3;
                    if (history[old].need_compute)
                        StoreDeepDq(history[old], old);
                }
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

        if (deep_pipeline_ && taskId > 0) {
            uint32_t last = (taskId - 1) % 3;
            if (history[last].need_compute)
                StoreDeepDq(history[last], last);
        }
        vecOp.DrainSoftmaxOutputCopy();
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
