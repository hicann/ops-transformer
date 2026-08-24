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
 * \file flash_attention_score_grad_kernel.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_KERNEL_H
#define FLASH_ATTENTION_SCORE_GRAD_KERNEL_H

#include "flash_attention_score_grad_common.h"
#include "flash_attention_score_grad_kernel_base.h"
#include "cube_api/mutex_buffer.h"
#include "flash_attention_score_grad_tiling_data_regbase.h"

namespace FagBaseApi {

template <typename CubeBlockType, typename VecBlockType>

class FlashAttentionScoreGradKernel
    : public FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>,
                                               CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    constexpr static uint8_t PRELOAD_TIMES = 3;
    using BaseClass = FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>,
                                                        CubeBlockType, VecBlockType>;
    __aicore__ inline void Process();
    __aicore__ inline void ProcessPreloadTwoTimes();
    __aicore__ inline void ComputeDqkvBn2gs1s2(FagRunInfo &prevRunInfo, bool &needSyncDkMM, int64_t taskId);
    __aicore__ inline void ComputeDqkvBn2s2(FagRunInfo &prevRunInfo, bool &needSyncDkMM, bool &needSyncDkDvFixUb,
                                            int64_t taskId);
    __aicore__ inline void ComputeDqkvBn2(FagRunInfo &prevRunInfo, bool &needSyncDkMM, int64_t taskId);
    __aicore__ inline void ProcessBn2gs1s2LastVec(FagRunInfo &prevRunInfo, bool &needSyncDkMM);
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessBn2gs1s2LastVec(
    FagRunInfo &prevRunInfo, bool &needSyncDkMM)
{
    LocalTensor<CALC_TYPE> mm1ResTensor =
        this->mm1ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    LocalTensor<CALC_TYPE> mm2ResTensor =
        this->mm2ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    if constexpr (IS_DROP) {
        if ASCEND_IS_AIV {
            if (needSyncDkMM) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG);
            }
        }
        MutexBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
        MutexBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
        this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, this->constInfo,
                                   prevRunInfo); // v3: dropout + cast + nd2nz
        if (unlikely(this->constInfo.isSink)) {
            this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo, prevRunInfo);
        }
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG); // dqk must wait ds copy completely
        } else {
            // wait ds in ub copy to l1
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);
        }
        // compute dq
        this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(this->dqWorkSpaceGm, dSL1Buffer,
                                                                                    this->constInfo,
                                                                                    prevRunInfo); // c3
        // compute dk
        this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(this->dkWorkSpaceGm, dSL1Buffer,
                                                                                    this->constInfo,
                                                                                    prevRunInfo); // c4
        if ASCEND_IS_AIV {
            if (needSyncDkMM) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG);
            }
        } else {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
        }
        this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, this->constInfo, prevRunInfo); // v4: cast + nd2nz
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG); // dv must wait ds copy completely
        } else {
            // wait p in ub copy to l1
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);
        }

        // compute dv
        this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(this->dvWorkSpaceGm, pL1Buffer,
                                                                                    this->constInfo, prevRunInfo); // c5
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
        }
        needSyncDkMM = true;
    } else {
        if (unlikely(this->constInfo.isSink)) {
            this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo, prevRunInfo);
        }
        if ASCEND_IS_AIV {
            if (needSyncDkMM) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG);
            }
        }
        MutexBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
        MutexBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
        this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, this->constInfo, prevRunInfo); // v4: cast + nd2nz

        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG); // dv must wait ds copy completely
            if (needSyncDkMM) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG);
            }
        }
        this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, this->constInfo,
                                   prevRunInfo); // v3: dropout + cast + nd2nz

        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG); // dqk must wait ds copy completely
        } else {
            // wait p in ub copy to l1
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);
        }
        // compute dv
        this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(this->dvWorkSpaceGm, pL1Buffer,
                                                                                    this->constInfo, prevRunInfo); // c5
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
            // wait ds in ub copy to l1
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);
        }
        // compute dq
        this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(this->dqWorkSpaceGm, dSL1Buffer,
                                                                                    this->constInfo,
                                                                                    prevRunInfo); // c3
        // compute dk
        this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(this->dkWorkSpaceGm, dSL1Buffer,
                                                                                    this->constInfo,
                                                                                    prevRunInfo); // c4
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
        }
        needSyncDkMM = true;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ComputeDqkvBn2gs1s2(
    FagRunInfo &prevRunInfo, bool &needSyncDkMM, int64_t taskId)
{
    LocalTensor<CALC_TYPE> mm2ResTensor =
        this->mm2ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    // wait mm2 result
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[(taskId + 1) & 1]);
    }
    this->vecBlock.ProcessVec2(mm2ResTensor, this->constInfo, prevRunInfo); // v2: pse + attenMask + simpleSoftmax
    // wait mm1 result
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[(taskId + 1) & 1]);
    }
    ProcessBn2gs1s2LastVec(prevRunInfo, needSyncDkMM);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ComputeDqkvBn2s2(
    FagRunInfo &prevRunInfo, bool &needSyncDkMM, bool &needSyncDkDvFixUb, int64_t taskId)
{
    LocalTensor<CALC_TYPE> mm1ResTensor =
        this->mm1ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    LocalTensor<CALC_TYPE> mm2ResTensor =
        this->mm2ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    // wait mm2 result
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[(taskId + 1) & 1]);
    }
    this->vecBlock.ProcessVec2(mm2ResTensor, this->constInfo, prevRunInfo); // v2: pse + attenMask + simpleSoftmax
    // wait mm1 result
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[(taskId + 1) & 1]);
    }
    if (unlikely(this->constInfo.isSink && !IS_DROP)) {
        this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo, prevRunInfo);
    }
    if ASCEND_IS_AIV {
        if (needSyncDkMM) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG);
        }
    }
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
    this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, this->constInfo,
                               prevRunInfo); // v3: dropout + cast + nd2nz
    if (unlikely(this->constInfo.isSink && IS_DROP)) {
        this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo, prevRunInfo);
    }
    if ASCEND_IS_AIV {
        if (needSyncDkMM) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG);
        }
    }
    this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, this->constInfo, prevRunInfo); // v4: cast + nd2nz
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG); // dqk must wait ds copy completely
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG); // dv must wait ds copy completely
    }

    if ASCEND_IS_AIC {
        // wait ds in ub copy to l1
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);
        // wait p in ub copy to l1
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);
    }

    // compute dq
    this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(this->dqWorkSpaceGm, dSL1Buffer,
                                                                                this->constInfo,
                                                                                prevRunInfo); // c3
    // compute dk
    if constexpr (BaseClass::IS_DK_WRITE_UB) {
        mm1ResTensor = this->mm1ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
        this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(mm1ResTensor, dSL1Buffer,
                                                                                    this->constInfo,
                                                                                    prevRunInfo); // c4
        if (!prevRunInfo.isNextS2IdxNoChange) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C4_TO_V6_FLAG);
            }
            this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DK_WRITE_UB, DK_IDX>(
                mm1ResTensor, this->constInfo, prevRunInfo); // v6: dk muls + cast
        }
    } else {
        this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(this->dkWorkSpaceGm, dSL1Buffer,
                                                                                    this->constInfo,
                                                                                    prevRunInfo); // c4
        if (!prevRunInfo.isNextS2IdxNoChange) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C4_TO_V6_FLAG);
            }
            this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DK_WRITE_UB, DK_IDX>(
                this->dkWorkSpaceGm, this->constInfo, prevRunInfo); // v5: dk muls + cast
        }
    }
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
    }
    // compute dv
    if constexpr (BaseClass::IS_DV_WRITE_UB) {
        mm2ResTensor = this->mm2ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
        this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(mm2ResTensor, pL1Buffer,
                                                                                    this->constInfo,
                                                                                    prevRunInfo); // c3
        if (!prevRunInfo.isNextS2IdxNoChange) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C3_TO_V5_FLAG);
            }
            this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DV_WRITE_UB, DV_IDX>(
                mm2ResTensor, this->constInfo, prevRunInfo); // v6: dv muls + cast
            if ASCEND_IS_AIV {
                CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SYNC_DETER_FIX_FLAG);
            }
            needSyncDkDvFixUb = true;
        }
    } else {
        this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(this->dvWorkSpaceGm, pL1Buffer,
                                                                                    this->constInfo, prevRunInfo); // c5
        if (!prevRunInfo.isNextS2IdxNoChange) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V5_FLAG);
            }
            this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DV_WRITE_UB, DV_IDX>(
                this->dvWorkSpaceGm, this->constInfo, prevRunInfo); // v6: dv muls + cast
        }
    }
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
    }
    needSyncDkMM = true;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ComputeDqkvBn2(
    FagRunInfo &prevRunInfo, bool &needSyncDkMM, int64_t taskId)
{
    LocalTensor<CALC_TYPE> mm1ResTensor =
        this->mm1ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    LocalTensor<CALC_TYPE> mm2ResTensor =
        this->mm2ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    // wait mm2 result
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[(taskId + 1) & 1]);
    }
    this->vecBlock.ProcessVec2(mm2ResTensor, this->constInfo, prevRunInfo); // v2: pse + attenMask + simpleSoftmax
    // wait mm1 result
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[(taskId + 1) & 1]);
    }
    if (unlikely(this->constInfo.isSink && !IS_DROP)) {
        this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo, prevRunInfo);
    }
    if ASCEND_IS_AIV {
        if (needSyncDkMM) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG);
        }
    }
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
    this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, this->constInfo,
                               prevRunInfo); // v3: dropout + cast + nd2nz
    if (unlikely(this->constInfo.isSink && IS_DROP)) {
        this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo, prevRunInfo);
    }
    if ASCEND_IS_AIV {
        if (needSyncDkMM) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG);
        }
    }
    this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, this->constInfo, prevRunInfo); // v4: cast + nd2nz
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG); // dqk must wait ds copy completely
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG); // dv must wait ds copy completely
    }

    if ASCEND_IS_AIC {
        // wait ds in ub copy to l1
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);
        // wait p in ub copy to l1
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);
    }

    if constexpr (SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK) {
        // compute dq
        if constexpr (BaseClass::IS_DQ_WRITE_UB) {
            mm1ResTensor = this->mm1ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(mm1ResTensor, dSL1Buffer,
                                                                                        this->constInfo,
                                                                                        prevRunInfo); // c3
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C3_TO_V5_FLAG);
            }
            this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB, DQ_IDX>(
                mm1ResTensor, this->constInfo, prevRunInfo); // v5: dq muls + cast
        } else {
            this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(this->dqWorkSpaceGm, dSL1Buffer,
                                                                                        this->constInfo,
                                                                                        prevRunInfo); // c3
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V5_FLAG);
            }
            this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB, DQ_IDX>(
                this->dqWorkSpaceGm, this->constInfo, prevRunInfo); // v5: dq muls + cast
        }
        // compute dk
        if constexpr (BaseClass::IS_DK_WRITE_UB) {
            mm2ResTensor = this->mm2ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(mm2ResTensor, dSL1Buffer,
                                                                                        this->constInfo,
                                                                                        prevRunInfo); // c4
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C4_TO_V6_FLAG);
            }
            this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DK_WRITE_UB, DK_IDX>(
                mm2ResTensor, this->constInfo, prevRunInfo); // v6: dk muls + cast
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
            }
            if ASCEND_IS_AIV {
                CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SYNC_DETER_FIX_FLAG);
            }
        } else {
            this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(this->dkWorkSpaceGm, dSL1Buffer,
                                                                                        this->constInfo,
                                                                                        prevRunInfo); // c4
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C4_TO_V6_FLAG);
            }
            this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DK_WRITE_UB, DK_IDX>(
                this->dkWorkSpaceGm, this->constInfo, prevRunInfo); // v6: dk muls + cast
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
            }
        }
        // compute dv
        this->cubeBlock.template IterateMmPDy<OUTDTYPE, BaseClass::IS_DV_WRITE_UB>(this->dvGm, pL1Buffer,
                                                                                   this->constInfo, prevRunInfo); // c5
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
        }
        needSyncDkMM = true;
    } else if constexpr (IS_BN2_MULTIBLK) {
        // compute dq
        this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(this->dqWorkSpaceGm, dSL1Buffer,
                                                                                    this->constInfo,
                                                                                    prevRunInfo); // c3
        if (prevRunInfo.isLastS1Outer) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V5_FLAG);
            }
            this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB, DQ_IDX>(
                this->dqWorkSpaceGm, this->constInfo, prevRunInfo); // v5: dq muls + cast
        }

        // compute dk
        this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(this->dkWorkSpaceGm, dSL1Buffer,
                                                                                    this->constInfo,
                                                                                    prevRunInfo); // c4
        if (!prevRunInfo.isNextS2IdxNoChange) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C4_TO_V6_FLAG);
            }
            this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DK_WRITE_UB, DK_IDX>(
                this->dkWorkSpaceGm, this->constInfo, prevRunInfo); // v6: dk muls + cast
        }
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
        }

        // compute dv
        this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(this->dvWorkSpaceGm, pL1Buffer,
                                                                                    this->constInfo, prevRunInfo); // c5
        if (!prevRunInfo.isNextS2IdxNoChange) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V5_FLAG);
            }
            this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DV_WRITE_UB, DV_IDX>(
                this->dvWorkSpaceGm, this->constInfo, prevRunInfo); // v6: dv muls + cast
        }
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
        }
        needSyncDkMM = true;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::Process()
{
    if constexpr (BaseClass::IS_PRELOAD_TWO_TIMES) {
        ProcessPreloadTwoTimes(); // for small headDim(<=128)
    } else {
        if (this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] == 0) {
            return;
        }
        int64_t taskId = 0;
        FagRunInfo runInfos[2]; // for cv ping pong
        int64_t nextValidBlockInnerIdx = 0;
        int64_t blockInnerIdx = 0;
        int64_t curLoopIdx = 0; // just for continuous split core
        nextValidBlockInnerIdx = this->GetNextValidIdx(
            runInfos[0], taskId, this->tilingData->s1s2BNGS1S2BlockNumList.blockStarts[this->cBlockIdx], curLoopIdx);
        blockInnerIdx = nextValidBlockInnerIdx;

        FagRunInfo prevRunInfo;
        bool needSyncDkMM = false;
        bool needSyncDkDvFixUb = false;
        while (true) {
            this->isLastLoop = (blockInnerIdx == -1);
            if (taskId > 0) {
                prevRunInfo = runInfos[(taskId + 1) & 1];
                if (likely(!this->constInfo.enablePreSfmg)) {
                    this->vecBlock.ProcessVec1(this->constInfo, prevRunInfo); // v1: softmaxGrad
                }
            }
            if (!this->isLastLoop) {
                // get mm1 mm2 next valid block index and next s2 begin end
                nextValidBlockInnerIdx =
                    this->GetNextValidIdx(runInfos[(taskId + 1) & 1], taskId + 1, blockInnerIdx + 1, curLoopIdx + 1);
                this->SetRunInfo(runInfos[taskId & 1], runInfos[(taskId + 1) & 1], taskId, blockInnerIdx,
                                 nextValidBlockInnerIdx);
                if (this->tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx || IS_TND_SWIZZLE) {
                    curLoopIdx++;
                } else {
                    blockInnerIdx++;
                }

                if constexpr (SPLIT_AXIS == BN2 && BaseClass::IS_DK_WRITE_UB) {
                    if ASCEND_IS_AIC {
                        if (needSyncDkMM) {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_DETER_FIX_FLAG);
                        }
                    }
                }
                if constexpr (SPLIT_AXIS == BN2S2 && BaseClass::IS_DK_WRITE_UB) {
                    if ASCEND_IS_AIC {
                        if (needSyncDkDvFixUb) {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_DETER_FIX_FLAG);
                        }
                    }
                    needSyncDkDvFixUb = false; // BN2S2存在l0c累加，不一定每一次mm345的结果都需要搬到mm1，mm2resbuf上
                }

                LocalTensor<CALC_TYPE> mm2ResTensor =
                    this->mm2ResBuf[runInfos[taskId & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                this->cubeBlock.IterateMmQK(mm2ResTensor, this->constInfo, runInfos[taskId & 1], this->preloadArgs);
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[taskId & 1]);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_TO_V2_FLAG[taskId & 1]);
                }

                LocalTensor<CALC_TYPE> mm1ResTensor =
                    this->mm1ResBuf[runInfos[taskId & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                this->cubeBlock.IterateMmDyV(mm1ResTensor, this->constInfo, runInfos[taskId & 1], this->preloadArgs);
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[taskId & 1]);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[taskId & 1]);
                }

                this->vecBlock.CopyMaxSum(this->constInfo, runInfos[taskId & 1],
                                          taskId); // copy in max and sum double buffer
            }
            if (taskId > 0) {
                if constexpr (SPLIT_AXIS == BN2GS1S2) {
                    ComputeDqkvBn2gs1s2(prevRunInfo, needSyncDkMM, taskId);
                } else if constexpr (SPLIT_AXIS == BN2S2) {
                    ComputeDqkvBn2s2(prevRunInfo, needSyncDkMM, needSyncDkDvFixUb, taskId);
                } else {
                    ComputeDqkvBn2(prevRunInfo, needSyncDkMM, taskId);
                }
            }
            if (blockInnerIdx == -1) {
                break;
            }
            taskId++;
            blockInnerIdx = nextValidBlockInnerIdx;
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessPreloadTwoTimes()
{
    if (this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] == 0) {
        return;
    }
    int64_t taskId = 0;
    FagRunInfo runInfos[PRELOAD_TIMES]; // for cv ping pong
    int64_t nextValidBlockInnerIdx = 0;
    int64_t blockInnerIdx = 0;
    int64_t curLoopIdx = 0; // just for continuous split core
    nextValidBlockInnerIdx = this->GetNextValidIdx(
        runInfos[0], taskId, this->tilingData->s1s2BNGS1S2BlockNumList.blockStarts[this->cBlockIdx], curLoopIdx);
    if (nextValidBlockInnerIdx == -1) {
        return;
    }

    blockInnerIdx = nextValidBlockInnerIdx;
    FagRunInfo prevRunInfo;
    bool needSyncDkDvFixUb = false;
    bool isLastTwoLoop = false;
    uint8_t lastTwoLoopCount = 0;
    while (true) {
        if (taskId > 1) {
            prevRunInfo = runInfos[(taskId + 1) % PRELOAD_TIMES];
        }
        if (taskId > 0 && lastTwoLoopCount <= 1) {
            if (likely(!this->constInfo.enablePreSfmg)) {
                this->vecBlock.ProcessVec1(this->constInfo,
                                           runInfos[(taskId + NUM_TWO) % PRELOAD_TIMES]); // v1: softmaxGrad
            }
        }
        if (!isLastTwoLoop) {
            // get mm1 mm2 next valid block index and next s2 begin end
            nextValidBlockInnerIdx = this->GetNextValidIdx(runInfos[(taskId + 1) % PRELOAD_TIMES], taskId + 1,
                                                           blockInnerIdx + 1, curLoopIdx + 1);
            isLastTwoLoop = (nextValidBlockInnerIdx == -1);
            this->SetRunInfo(runInfos[taskId % PRELOAD_TIMES], runInfos[(taskId + 1) % PRELOAD_TIMES], taskId,
                             blockInnerIdx, nextValidBlockInnerIdx);
            if (this->tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx || IS_TND_SWIZZLE) {
                curLoopIdx++;
            } else {
                blockInnerIdx++;
            }
            // ------------------------ q@k --------------------------
            if ASCEND_IS_AIC {
                if (taskId >= 1) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_V2_TO_C2_FLAG[taskId & 1]);
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_V2_TO_C2_FLAG[taskId & 1]);
                }
            }
            LocalTensor<CALC_TYPE> mm2ResTensor =
                this->mm2ResBuf[runInfos[taskId % PRELOAD_TIMES].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            this->cubeBlock.IterateMmQK(mm2ResTensor, this->constInfo, runInfos[taskId % PRELOAD_TIMES],
                                        this->preloadArgs);
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[taskId & 1]);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_TO_V2_FLAG[taskId & 1]);
            }

            // ------------------------ dx@v --------------------------
            if ASCEND_IS_AIC {
                if (taskId >= 1) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_V2_TO_C1_FLAG[taskId & 1]);
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_V2_TO_C1_FLAG[taskId & 1]);
                }
            }
            LocalTensor<CALC_TYPE> mm1ResTensor =
                this->mm1ResBuf[runInfos[taskId % PRELOAD_TIMES].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            this->cubeBlock.IterateMmDyV(mm1ResTensor, this->constInfo, runInfos[taskId % PRELOAD_TIMES],
                                         this->preloadArgs);
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[taskId & 1]);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[taskId & 1]);
            }

            this->vecBlock.CopyMaxSum(this->constInfo, runInfos[taskId % PRELOAD_TIMES],
                                      taskId); // copy in max and sum double buffer
        }
        if (taskId > 0 && lastTwoLoopCount <= 1) {
            if ASCEND_IS_AIV {
                LocalTensor<CALC_TYPE> mm1ResTensor =
                    this->mm1ResBuf[runInfos[(taskId + NUM_TWO) % PRELOAD_TIMES].commonRunInfo.taskIdMod2]
                        .template Get<CALC_TYPE>();
                LocalTensor<CALC_TYPE> mm2ResTensor =
                    this->mm2ResBuf[runInfos[(taskId + NUM_TWO) % PRELOAD_TIMES].commonRunInfo.taskIdMod2]
                        .template Get<CALC_TYPE>();
                // wait mm2 result
                CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[(taskId + 1) & 1]);
                this->vecBlock.ProcessVec2(
                    mm2ResTensor, this->constInfo,
                    runInfos[(taskId + NUM_TWO) % PRELOAD_TIMES]); // v2: pse + attenMask + simpleSoftmax
                // wait mm1 result
                CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[(taskId + 1) & 1]);
                if (unlikely(this->constInfo.isSink && !IS_DROP)) {
                    this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo,
                                                  runInfos[(taskId + NUM_TWO) % PRELOAD_TIMES]);
                }

                // ------------------------ dv dependency --------------------------
                if (taskId >= NUM_TWO) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG); // p must wait dv write l0a over
                }
                MutexBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
                MutexBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
                this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, this->constInfo,
                                           runInfos[(taskId + NUM_TWO) % PRELOAD_TIMES]); // v4: cast + nd2nz
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG); // dv must wait p copy completely

                // ------------------------ dqk dependency --------------------------
                if (taskId >= NUM_TWO) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG); // ds must wait dqk write l0a over
                }
                this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, this->constInfo,
                                           runInfos[(taskId + NUM_TWO) % PRELOAD_TIMES]); // v3: dropout + cast + nd2nz
                if (unlikely(this->constInfo.isSink && IS_DROP)) {
                    this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo,
                                                  runInfos[(taskId + NUM_TWO) % PRELOAD_TIMES]);
                }
                CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SYNC_V2_TO_C1_FLAG[taskId & 1]);
                CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SYNC_V2_TO_C2_FLAG[taskId & 1]);
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG); // dqk must wait ds copy completely
            }
        }
        if (taskId > 1) {
            if ASCEND_IS_AIC {
                LocalTensor<CALC_TYPE> mm1ResTensor =
                    this->mm1ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                LocalTensor<CALC_TYPE> mm2ResTensor =
                    this->mm2ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();

                // wait p in ub copy to l1
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);

                MutexBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
                MutexBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();

                // compute dv
                this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(
                    this->dvWorkSpaceGm, pL1Buffer, this->constInfo, prevRunInfo); // c5
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);        // v4 must wait dv mte1 copy over
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
                // wait ds in ub copy to l1
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);
                // compute dq
                this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(this->dqWorkSpaceGm,
                                                                                            dSL1Buffer, this->constInfo,
                                                                                            prevRunInfo); // c3
                // compute dk
                this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(this->dkWorkSpaceGm,
                                                                                            dSL1Buffer, this->constInfo,
                                                                                            prevRunInfo); // c4
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG); // v3 must wait dqk mte1 copy over
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
            }
        }
        if (isLastTwoLoop) {
            lastTwoLoopCount++;
            if (lastTwoLoopCount == PRELOAD_TIMES) {
                break;
            }
        }
        taskId++;
        blockInnerIdx = nextValidBlockInnerIdx;
    }
}
} // namespace FagBaseApi

#endif
