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
 * \file flash_attention_score_grad_block_vec_small_sd.h
 * \brief Small-S/Small-D dedicated Vector block interface.
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_SMALL_SD_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_SMALL_SD_H

#include "flash_attention_score_grad_common.h"
#include "flash_attention_score_grad_buffer_small_sd.h"
#include "flash_attention_score_grad_event_small_sd.h"
#include "cube_api/mutex_buffer.h"
#include "cube_api/mutex_buffers_policy.h"
#include "vector_api/cast_softmax_grad.h"
#include "vector_api/pse_atten_mask_muls_simple_softmax.h"
#include "vector_api/vf_broadcast_sub_mul.h"
#include "vector_api/vf_cast_transdata_deconflict.h"

namespace FagBaseApi {

template <typename INPUT_TYPE, typename CALC_TYPE, typename OUTDTYPE, bool IS_TND, uint32_t HEAD_DIM>
class SmallSDVectorBlock {
public:
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "SmallSDVectorBlock only supports D=64 or D=128.");
    static constexpr bool IS_SMALL_SD_DEDICATED_BLOCK = true;
    static constexpr uint32_t SMALL_SD_HEAD_DIM = HEAD_DIM;
    static constexpr uint32_t VECTOR_BASEM = 64;
    static constexpr uint32_t VECTOR_BASEN = 128;
    static constexpr uint32_t INPUT_BLOCK_NUM = 32 / sizeof(INPUT_TYPE);

    __aicore__ inline SmallSDVectorBlock() {}

    __aicore__ inline void Init(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, TPipe *pipeIn,
                                const SmallSDConstInfo *constInfoIn, uint32_t vSubBlockIdxIn,
                                TBuf<> *mm1ResBufIn, TBuf<> *mm2ResBufIn,
                                MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> *dSL1BufIn,
                                MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> *pL1BufIn)
    {
        (void)workspace;
        pipe = pipeIn;
        constInfo = constInfoIn;
        vSubBlockIdx = vSubBlockIdxIn;
        mm1ResBuf = mm1ResBufIn;
        mm2ResBuf = mm2ResBufIn;
        dSL1Buf = dSL1BufIn;
        pL1Buf = pL1BufIn;
        valueGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)value);
        dyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)dy);
        yGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)y);
        softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
        softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
        dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
        dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
        dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
        pipe->InitBuffer(yInQue, 1, VECTOR_BASEM * HEAD_DIM * sizeof(INPUT_TYPE));
        pipe->InitBuffer(dyInQue, 1, VECTOR_BASEM * HEAD_DIM * sizeof(INPUT_TYPE));
        pipe->InitBuffer(dSOutQue, 1, (VECTOR_BASEM + 1) * VECTOR_BASEN * sizeof(OUTDTYPE));
        pipe->InitBuffer(pOutQue, 1, (VECTOR_BASEM + 1) * VECTOR_BASEN * sizeof(OUTDTYPE));
        pipe->InitBuffer(maxSumQue[0], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO);
        pipe->InitBuffer(maxSumQue[1], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO);
        pipe->InitBuffer(softmaxGradResBuf, VECTOR_BASEM * sizeof(CALC_TYPE));
    }

    __aicore__ inline void WaitCubeReady(const SmallSDPipelineSlot &slot)
    {
        const int64_t taskMod2 = slot.taskIdMod2;
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SMALL_SD_CUBE_QK_READY_FLAG[taskMod2]);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SMALL_SD_CUBE_DYV_READY_FLAG[taskMod2]);
        }
    }

    __aicore__ inline void ProduceDsAndP(SmallSDPipelineSlot &slot, bool needWaitL1Reusable)
    {
        if ASCEND_IS_AIV {
            WaitCubeReady(slot);
            WaitSharedL1Reusable(needWaitL1Reusable);
            ComputeSoftmaxAndDs(slot);
            CommitDsPToL1(slot);
        }
    }

    __aicore__ inline void FinalizeGradOutput(SmallSDPipelineSlot &slot)
    {
        if ASCEND_IS_AIV {
            WaitGradReady(slot);
            CommitGradToGm(slot);
        }
    }

    __aicore__ inline void ComputeSoftmaxAndDs(const SmallSDPipelineSlot &slot)
    {
        if (slot.halfS1 == 0) {
            return;
        }
        LocalTensor<CALC_TYPE> mm1ResTensor = mm1ResBuf[slot.taskIdMod2].template Get<CALC_TYPE>();
        LocalTensor<CALC_TYPE> mm2ResTensor = mm2ResBuf[slot.taskIdMod2].template Get<CALC_TYPE>();
        LocalTensor<CALC_TYPE> softmaxGradTensor = softmaxGradResBuf.Get<CALC_TYPE>();
        CopyInDyAndY(slot);
        CalculateDyYSoftmaxGrad(slot, softmaxGradTensor);
        CopyInMaxSum(slot);
        LocalTensor<CALC_TYPE> maxSumTensor = maxSumQue[slot.taskIdMod2].template DeQue<CALC_TYPE>();
        LocalTensor<OUTDTYPE> pseTensor;
        LocalTensor<uint8_t> attenMaskTensor;
        if (slot.actualS2Len > 64) {
            AscendC::MulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, 128, false, false, false>(
                mm2ResTensor, maxSumTensor,
                maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)], mm2ResTensor, pseTensor,
                attenMaskTensor, constInfo->scale, 0.0f, slot.halfS1, slot.actualS2Len);
        } else {
            AscendC::MulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, 64, false, false, false>(
                mm2ResTensor, maxSumTensor,
                maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)], mm2ResTensor, pseTensor,
                attenMaskTensor, constInfo->scale, 0.0f, slot.halfS1, slot.actualS2Len);
        }
        maxSumQue[slot.taskIdMod2].FreeTensor(maxSumTensor);
        LocalTensor<INPUT_TYPE> dsCast = dSOutQue.template AllocTensor<INPUT_TYPE>();
        if (slot.actualS2Len > 64) {
            AscendC::BroadcastSubMul<CALC_TYPE, 128, false>(mm1ResTensor, mm1ResTensor, softmaxGradTensor,
                                                            mm2ResTensor, slot.halfS1, slot.actualS2Len);
        } else {
            AscendC::BroadcastSubMul<CALC_TYPE, 64, false>(mm1ResTensor, mm1ResTensor, softmaxGradTensor, mm2ResTensor,
                                                           slot.halfS1, slot.actualS2Len);
        }
        LocalTensor<uint8_t> selrIndexesTensor;
        AscendC::CastTransdataDeconflict<INPUT_TYPE, CALC_TYPE, VECTOR_BASEN>(dsCast, mm1ResTensor, selrIndexesTensor,
                                                                              VECTOR_BASEM);
        dSOutQue.EnQue(dsCast);
        dSOutQue.template DeQue<INPUT_TYPE>();
        CopyUbNzToL1<INPUT_TYPE, true>(dSL1Buf->Get().template GetTensor<INPUT_TYPE>(), dsCast, slot);
        dSOutQue.FreeTensor(dsCast);

        LocalTensor<INPUT_TYPE> pCast = pOutQue.template AllocTensor<INPUT_TYPE>();
        AscendC::CastTransdataDeconflict<INPUT_TYPE, CALC_TYPE, VECTOR_BASEN>(pCast, mm2ResTensor, selrIndexesTensor,
                                                                              VECTOR_BASEM);
        pOutQue.EnQue(pCast);
        pOutQue.template DeQue<INPUT_TYPE>();
        CopyUbNzToL1<INPUT_TYPE, false>(pL1Buf->Get().template GetTensor<INPUT_TYPE>(), pCast, slot);
        pOutQue.FreeTensor(pCast);
    }

    __aicore__ inline void WaitSharedL1Reusable(bool needWaitL1Reusable)
    {
        if ASCEND_IS_AIV {
            if (needWaitL1Reusable) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_DS_L1_REUSABLE_FLAG);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_DS_L1_REUSABLE_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_P_L1_REUSABLE_FLAG);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_P_L1_REUSABLE_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
            }
        }
    }

    __aicore__ inline void CommitDsPToL1(SmallSDPipelineSlot &slot)
    {
        (void)slot;
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_DS_L1_READY_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_DS_L1_READY_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_P_L1_READY_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_P_L1_READY_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
        }
    }

    __aicore__ inline void WaitGradReady(const SmallSDPipelineSlot &slot)
    {
        (void)slot;
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SMALL_SD_DQ_UB_READY_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SMALL_SD_DK_UB_READY_FLAG);
        }
    }

    __aicore__ inline void CommitGradToGm(SmallSDPipelineSlot &slot)
    {
        CommitOneGrad<DQ_IDX>(mm1ResBuf[slot.taskIdMod2].template Get<CALC_TYPE>(), slot);
        CommitOneGrad<DK_IDX>(mm2ResBuf[slot.taskIdMod2].template Get<CALC_TYPE>(), slot);
        // DV is produced directly by Cube into GM in the SmallSD minimal data path.
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_SLOT_REUSE_READY_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_SLOT_REUSE_READY_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
        }
    }

private:
    __aicore__ inline uint64_t GetQLikeRowGmOffset(const SmallSDPipelineSlot &slot, uint64_t baseOffset,
                                                   uint64_t rowStride) const
    {
        return baseOffset + static_cast<uint64_t>(vSubBlockIdx) * slot.firstHalfS1 * rowStride;
    }

    __aicore__ inline uint64_t GetKLikeRowGmOffset(const SmallSDPipelineSlot &slot, uint64_t baseOffset,
                                                   uint64_t rowStride) const
    {
        return baseOffset + static_cast<uint64_t>(vSubBlockIdx) * slot.firstHalfS2 * rowStride;
    }

    __aicore__ inline void CopyInDyAndY(const SmallSDPipelineSlot &slot)
    {
        LocalTensor<INPUT_TYPE> yTensor = yInQue.template AllocTensor<INPUT_TYPE>();
        LocalTensor<INPUT_TYPE> dyTensor = dyInQue.template AllocTensor<INPUT_TYPE>();
        const uint64_t yOffset = GetQLikeRowGmOffset(slot, slot.qOffset, constInfo->qStrideS);
        const uint64_t dyOffset = GetQLikeRowGmOffset(slot, slot.dyOffset, constInfo->dyStrideS);
        const uint32_t ySrcStride = static_cast<uint32_t>((constInfo->qStrideS - constInfo->d) * sizeof(INPUT_TYPE));
        const uint32_t dySrcStride = static_cast<uint32_t>((constInfo->dyStrideS - constInfo->d) * sizeof(INPUT_TYPE));
        const uint32_t dstStride = static_cast<uint32_t>((HEAD_DIM - constInfo->d) * sizeof(INPUT_TYPE) / 32);
        DataCopyPad(dyTensor, dyGm[dyOffset],
                    {static_cast<uint16_t>(slot.halfS1), static_cast<uint32_t>(constInfo->d * sizeof(INPUT_TYPE)),
                     dySrcStride, dstStride, 0},
                    {true, 0, static_cast<uint8_t>(HEAD_DIM - constInfo->d), 0});
        DataCopyPad(yTensor, yGm[yOffset],
                    {static_cast<uint16_t>(slot.halfS1), static_cast<uint32_t>(constInfo->d * sizeof(INPUT_TYPE)),
                     ySrcStride, dstStride, 0},
                    {true, 0, static_cast<uint8_t>(HEAD_DIM - constInfo->d), 0});
        yInQue.EnQue(yTensor);
        dyInQue.EnQue(dyTensor);
    }

    __aicore__ inline void CalculateDyYSoftmaxGrad(const SmallSDPipelineSlot &slot,
                                                   LocalTensor<CALC_TYPE> &softmaxGradTensor)
    {
        LocalTensor<INPUT_TYPE> yTensor = yInQue.template DeQue<INPUT_TYPE>();
        LocalTensor<INPUT_TYPE> dyTensor = dyInQue.template DeQue<INPUT_TYPE>();
        AscendC::MySoftmaxGradFrontCast<INPUT_TYPE, CALC_TYPE, HEAD_DIM, HEAD_DIM>(
            softmaxGradTensor, yTensor, dyTensor, slot.halfS1, constInfo->d);
        yInQue.FreeTensor(yTensor);
        dyInQue.FreeTensor(dyTensor);
    }

    __aicore__ inline void CopyInMaxSum(const SmallSDPipelineSlot &slot)
    {
        LocalTensor<float> maxSumTensor = maxSumQue[slot.taskIdMod2].template AllocTensor<float>();
        uint64_t offset = 0;
        if (constInfo->tndMaxSumLayout == MAX_SUM_TND) {
            offset = ((slot.qOffset / constInfo->d) +
                      static_cast<uint64_t>(vSubBlockIdx) * slot.firstHalfS1 * constInfo->n2) *
                     MAX_SUM_REDUCE_AXIS_SIZE / sizeof(float);
            const uint32_t srcStride = constInfo->n2 * MAX_SUM_REDUCE_AXIS_SIZE - MAX_SUM_REDUCE_AXIS_SIZE;
            DataCopyPad(maxSumTensor, softmaxSumGm[offset],
                        {static_cast<uint16_t>(slot.halfS1), static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE),
                         srcStride, 0, 0},
                        {false, 0, 0, 0});
            DataCopyPad(maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(float)], softmaxMaxGm[offset],
                        {static_cast<uint16_t>(slot.halfS1), static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE),
                         srcStride, 0, 0},
                        {false, 0, 0, 0});
        } else {
            const uint64_t s1Prefix = IS_TND ? static_cast<uint64_t>(slot.lastBatchTotalS1BOffset) /
                                                   (static_cast<uint64_t>(constInfo->n2) * constInfo->d) :
                                               static_cast<uint64_t>(slot.bIdx) * constInfo->s1;
            offset = ((s1Prefix * constInfo->n2 + slot.n2oIdx * slot.actualS1Len) +
                      static_cast<uint64_t>(vSubBlockIdx) * slot.firstHalfS1) *
                     MAX_SUM_REDUCE_AXIS_SIZE / sizeof(float);
            DataCopyPad(maxSumTensor, softmaxSumGm[offset],
                        {1, static_cast<uint16_t>(slot.halfS1 * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
                        {false, 0, 0, 0});
            DataCopyPad(maxSumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(float)], softmaxMaxGm[offset],
                        {1, static_cast<uint16_t>(slot.halfS1 * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
                        {false, 0, 0, 0});
        }
        maxSumQue[slot.taskIdMod2].EnQue(maxSumTensor);
    }

    template <typename T, bool IS_DQ>
    __aicore__ inline void CopyUbNzToL1(LocalTensor<T> dstTensor, LocalTensor<T> &srcTensor,
                                        const SmallSDPipelineSlot &slot)
    {
        if (slot.halfS1 == 0) {
            return;
        }
        const uint32_t c0 = 32 / sizeof(T);
        const uint32_t scmOffset = vSubBlockIdx == 0 ? 0 : slot.firstHalfS1 * c0;
        const uint32_t s1Align16 = AlignTo16(slot.actualS1Len);
        DataCopyParams params;
        params.blockCount = VECTOR_BASEN / c0;
        params.blockLen = static_cast<uint16_t>(slot.halfS1 * c0 / (32 / sizeof(T)));
        params.srcStride = static_cast<uint16_t>((VECTOR_BASEM + 1 - slot.halfS1) * c0 / (32 / sizeof(T)));
        params.dstStride = static_cast<uint16_t>((s1Align16 - slot.halfS1) * c0 / (32 / sizeof(T)));
        DataCopy(dstTensor[scmOffset], srcTensor, params);
    }

    template <uint8_t MM_IDX>
    __aicore__ inline void CommitOneGrad(LocalTensor<CALC_TYPE> inputTensor, const SmallSDPipelineSlot &slot)
    {
        const uint32_t rows = static_cast<uint32_t>(MM_IDX == DQ_IDX ? slot.halfS1 : slot.halfS2);
        if (rows == 0) {
            return;
        }
        const uint32_t dSize = MM_IDX == DV_IDX ? constInfo->dv : constInfo->d;
        const uint64_t rowStride = MM_IDX == DQ_IDX ? constInfo->dqStrideS : constInfo->dkStrideS;
        uint64_t gmOffset = MM_IDX == DQ_IDX ? GetQLikeRowGmOffset(slot, slot.dqOffset, rowStride) :
                                               GetKLikeRowGmOffset(slot, slot.dkOffset, rowStride);
        DataCopyExtParams params;
        params.blockCount = rows;
        params.blockLen = dSize * sizeof(OUTDTYPE);
        params.srcStride = 0;
        params.dstStride = static_cast<uint32_t>((rowStride - dSize) * sizeof(OUTDTYPE));
        const uint32_t dataSize = rows * AlignTo16(dSize);
        if constexpr (MM_IDX != DV_IDX) {
            Muls(inputTensor, inputTensor, constInfo->scale, dataSize);
        }
        LocalTensor<OUTDTYPE> castTensor = dSOutQue.template AllocTensor<OUTDTYPE>();
        Cast(castTensor, inputTensor, RoundMode::CAST_ROUND, dataSize);
        dSOutQue.EnQue(castTensor);
        dSOutQue.template DeQue<OUTDTYPE>();
        if constexpr (MM_IDX == DQ_IDX) {
            DataCopyPad(dqGm[gmOffset], castTensor, params);
        } else {
            DataCopyPad(dkGm[gmOffset], castTensor, params);
        }
        dSOutQue.FreeTensor(castTensor);
    }

    TPipe *pipe = nullptr;
    const SmallSDConstInfo *constInfo = nullptr;
    uint32_t vSubBlockIdx = 0;
    TBuf<> *mm1ResBuf = nullptr;
    TBuf<> *mm2ResBuf = nullptr;
    MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> *dSL1Buf = nullptr;
    MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> *pL1Buf = nullptr;
    GlobalTensor<INPUT_TYPE> valueGm;
    GlobalTensor<INPUT_TYPE> dyGm;
    GlobalTensor<INPUT_TYPE> yGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<OUTDTYPE> dqGm;
    GlobalTensor<OUTDTYPE> dkGm;
    GlobalTensor<OUTDTYPE> dvGm;
    TQue<QuePosition::VECIN, 1> yInQue;
    TQue<QuePosition::VECIN, 1> dyInQue;
    TQue<QuePosition::VECOUT, 1> dSOutQue;
    TQue<QuePosition::VECOUT, 1> pOutQue;
    TQue<QuePosition::VECIN, 1> maxSumQue[2];
    TBuf<> softmaxGradResBuf;
};

} // namespace FagBaseApi

#endif
