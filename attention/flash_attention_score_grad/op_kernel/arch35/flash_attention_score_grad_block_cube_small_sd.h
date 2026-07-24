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
 * \file flash_attention_score_grad_block_cube_small_sd.h
 * \brief Small-S/Small-D dedicated Cube block interface.
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_SMALL_SD_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_SMALL_SD_H

#include "flash_attention_score_grad_common.h"
#include "flash_attention_score_grad_buffer_small_sd.h"
#include "flash_attention_score_grad_event_small_sd.h"
#include "cube_api/matmul.h"
#include "cube_api/mutex_buffers_policy.h"
#include "../../../common/op_kernel/FixpipeOut.h"

namespace FagBaseApi {

template <typename INPUT_TYPE, typename CALC_TYPE, typename OUTDTYPE, bool IS_TND, uint32_t HEAD_DIM>
class SmallSDCubeBlock {
public:
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "SmallSDCubeBlock only supports D=64 or D=128.");
    static constexpr bool IS_SMALL_SD_DEDICATED_BLOCK = true;
    static constexpr uint32_t SMALL_SD_HEAD_DIM = HEAD_DIM;
    static constexpr uint32_t CUBE_BASEM = 128;
    static constexpr uint32_t CUBE_BASEN = 128;
    static constexpr uint32_t CUBE_BASEK = HEAD_DIM;
    static constexpr uint32_t C0_SIZE = 16;
    static constexpr uint32_t SMALL_SD_L0_SINGLE_BUFFER_SIZE = 32 * 1024;
    static constexpr uint32_t DS_P_L1_BYTES = CUBE_BASEM * CUBE_BASEN * sizeof(INPUT_TYPE);
    static constexpr uint32_t L1_INPUT_TILE_BYTES = CUBE_BASEM * HEAD_DIM * sizeof(INPUT_TYPE);
    static constexpr uint32_t L0C_SCORE_BYTES = CUBE_BASEM * CUBE_BASEN * sizeof(CALC_TYPE);
    static constexpr uint32_t L0C_GRAD_BYTES = CUBE_BASEN * HEAD_DIM * sizeof(CALC_TYPE);
    static constexpr uint32_t L0C_BYTES = L0C_SCORE_BYTES > L0C_GRAD_BYTES ? L0C_SCORE_BYTES : L0C_GRAD_BYTES;

    __aicore__ inline SmallSDCubeBlock() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR dq, GM_ADDR dk,
                                GM_ADDR dv, GM_ADDR workspace, TPipe *pipeIn, const SmallSDConstInfo *constInfoIn,
                                TBuf<> *mm1ResBufIn, TBuf<> *mm2ResBufIn,
                                MutexBufferManager<BufferType::L1> *l1BufferManagerIn,
                                MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> *dSL1BufIn,
                                MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> *pL1BufIn)
    {
        (void)workspace;
        pipe = pipeIn;
        constInfo = constInfoIn;
        mm1ResBuf = mm1ResBufIn;
        mm2ResBuf = mm2ResBufIn;
        l1BufferManager = l1BufferManagerIn;
        dSL1Buf = dSL1BufIn;
        pL1Buf = pL1BufIn;
        queryGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)query);
        keyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)key);
        valueGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)value);
        dyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)dy);
        dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
        dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
        dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
        InitCubeBuffer();
    }

    __aicore__ inline void LoadSmallSDInputs(const SmallSDPipelineSlot &slot)
    {
        (void)slot;
        // Inputs are copied by the concrete matmul stage that consumes them, so the
        // lifetime of each L1 tile stays local to that primitive stage.
    }

    __aicore__ inline void IssueQkAndDyV(SmallSDPipelineSlot &slot)
    {
        slot.state = SmallSDSlotState::CUBE_INFLIGHT;
        if ASCEND_IS_AIC {
            LoadSmallSDInputs(slot);
            MatmulQKOrGradStage(slot);
            CommitCubeResultToSlot(slot);
        }
        slot.state = SmallSDSlotState::READY_FOR_VECTOR;
    }

    __aicore__ inline void IssueDqDkDv(SmallSDPipelineSlot &slot)
    {
        if ASCEND_IS_AIC {
            WaitDsPReady(slot);
            MatmulGradStage(slot);
            CommitGradResultToSlot(slot);
        }
    }

    __aicore__ inline void MatmulQKOrGradStage(const SmallSDPipelineSlot &slot)
    {
        if (slot.actualS1Len == 0 || slot.actualS2Len == 0) {
            return;
        }
        LocalTensor<CALC_TYPE> dyVResTensor = mm1ResBuf[slot.taskIdMod2].template Get<CALC_TYPE>();
        LocalTensor<CALC_TYPE> qkResTensor = mm2ResBuf[slot.taskIdMod2].template Get<CALC_TYPE>();
        IterateSmallSDDyV(dyVResTensor, slot);
        IterateSmallSDQK(qkResTensor, slot);
    }

    __aicore__ inline void MatmulGradStage(const SmallSDPipelineSlot &slot)
    {
        if (slot.actualS1Len == 0 || slot.actualS2Len == 0) {
            return;
        }
        LocalTensor<CALC_TYPE> dqTensor = mm1ResBuf[slot.taskIdMod2].template Get<CALC_TYPE>();
        LocalTensor<CALC_TYPE> dkTensor = mm2ResBuf[slot.taskIdMod2].template Get<CALC_TYPE>();
        MutexBuffer<BufferType::L1, SyncType::NO_SYNC> dsL1Buffer = dSL1Buf->Get();
        MutexBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = pL1Buf->Get();
        IterateSmallSDDsK(dqTensor, dsL1Buffer, slot);
        IterateSmallSDDsQ(dkTensor, dsL1Buffer, slot);
        IterateSmallSDPDy(dvGm, pL1Buffer, slot);
    }

    __aicore__ inline void CommitCubeResultToSlot(SmallSDPipelineSlot &slot)
    {
        const int64_t taskMod2 = slot.taskIdMod2;
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_CUBE_QK_READY_FLAG[taskMod2]);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_CUBE_QK_READY_FLAG[taskMod2] + SMALL_SD_EVENT_MIRROR_OFFSET);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_CUBE_DYV_READY_FLAG[taskMod2]);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_CUBE_DYV_READY_FLAG[taskMod2] + SMALL_SD_EVENT_MIRROR_OFFSET);
        }
    }

    __aicore__ inline void WaitDsPReady(const SmallSDPipelineSlot &slot)
    {
        (void)slot;
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_DS_L1_READY_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_DS_L1_READY_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_P_L1_READY_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_P_L1_READY_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
        }
    }

    __aicore__ inline void CommitGradResultToSlot(SmallSDPipelineSlot &slot)
    {
        (void)slot;
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_DQ_UB_READY_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_DQ_UB_READY_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_DK_UB_READY_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_DK_UB_READY_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_DS_L1_REUSABLE_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_DS_L1_REUSABLE_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_P_L1_REUSABLE_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_P_L1_REUSABLE_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
        }
    }

private:
    __aicore__ inline void InitCubeBuffer()
    {
        qL1Buf.Init(*l1BufferManager, L1_INPUT_TILE_BYTES);
        kL1Buf.Init(*l1BufferManager, L1_INPUT_TILE_BYTES);
        vL1Buf.Init(*l1BufferManager, L1_INPUT_TILE_BYTES);
        dyL1Buf.Init(*l1BufferManager, L1_INPUT_TILE_BYTES);
        l0aBufferManager.Init(pipe, L0_MAX_SIZE);
        l0bBufferManager.Init(pipe, L0_MAX_SIZE);
        l0cBufferManager.Init(pipe, L0C_MAX_SIZE);
        l0aBuf.Init(l0aBufferManager, SMALL_SD_L0_SINGLE_BUFFER_SIZE);
        l0bBuf.Init(l0bBufferManager, SMALL_SD_L0_SINGLE_BUFFER_SIZE);
        l0cBuf.Init(l0cBufferManager, L0C_BYTES);
    }

    __aicore__ inline void CopyGmNdToL1Nz(LocalTensor<INPUT_TYPE> dstTensor, GlobalTensor<INPUT_TYPE> &srcGm,
                                          uint64_t srcOffset, uint32_t rows, uint32_t cols,
                                          uint64_t rowStride) const
    {
        if (rows == 0 || cols == 0) {
            return;
        }
        Nd2NzParams nd2NzParams;
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = rows;
        nd2NzParams.dValue = cols;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = rowStride;
        nd2NzParams.dstNzC0Stride = AlignTo16(rows);
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(dstTensor, srcGm[srcOffset], nd2NzParams);
    }

    __aicore__ inline void IterateSmallSDDyV(LocalTensor<CALC_TYPE> &dstTensor,
                                            const SmallSDPipelineSlot &slot)
    {
        MutexBuffer<BufferType::L1> dyL1Buffer = dyL1Buf.Get();
        MutexBuffer<BufferType::L1> vL1Buffer = vL1Buf.Get();
        dyL1Buffer.LockProd();
        CopyGmNdToL1Nz(dyL1Buffer.template GetTensor<INPUT_TYPE>(), dyGm, slot.dyOffset, slot.actualS1Len,
                       constInfo->d, constInfo->dyStrideS);
        dyL1Buffer.UnlockProd();
        vL1Buffer.LockProd();
        CopyGmNdToL1Nz(vL1Buffer.template GetTensor<INPUT_TYPE>(), valueGm, slot.vOffset, slot.actualS2Len,
                       constInfo->dv, constInfo->vStrideS);
        vL1Buffer.UnlockProd();

        dyL1Buffer.LockCons();
        vL1Buffer.LockCons();
        MutexBuffer<BufferType::L0C, SyncType::INNER_CORE_SYNC> l0cBuffer = l0cBuf.Get();
        MMParam param = {
            static_cast<uint32_t>(slot.actualS1Len),
            static_cast<uint32_t>(slot.actualS2Len),
            static_cast<uint32_t>(constInfo->dv),
            false,
            true,
            true,
            true,
            UNITFLAG_EN_OUTER_LAST
        };
        MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN,
                        SMALL_SD_L0_SINGLE_BUFFER_SIZE / CUBE_BASEN / sizeof(INPUT_TYPE),
                        ABLayout::MK, ABLayout::KN>(
            dyL1Buffer.template GetTensor<INPUT_TYPE>(), vL1Buffer.template GetTensor<INPUT_TYPE>(), l0aBuf, l0bBuf,
            l0cBuffer.GetTensor<CALC_TYPE>(), param);
        FixpipeScoreToUb(dstTensor, l0cBuffer.GetTensor<CALC_TYPE>(), slot.actualS1Len, slot.actualS2Len);
        vL1Buffer.UnlockCons();
        dyL1Buffer.UnlockCons();
    }

    __aicore__ inline void IterateSmallSDQK(LocalTensor<CALC_TYPE> &dstTensor,
                                           const SmallSDPipelineSlot &slot)
    {
        MutexBuffer<BufferType::L1> qL1Buffer = qL1Buf.Get();
        MutexBuffer<BufferType::L1> kL1Buffer = kL1Buf.Get();
        qL1Buffer.LockProd();
        CopyGmNdToL1Nz(qL1Buffer.template GetTensor<INPUT_TYPE>(), queryGm, slot.qOffset, slot.actualS1Len,
                       constInfo->d, constInfo->qStrideS);
        qL1Buffer.UnlockProd();
        kL1Buffer.LockProd();
        CopyGmNdToL1Nz(kL1Buffer.template GetTensor<INPUT_TYPE>(), keyGm, slot.kOffset, slot.actualS2Len,
                       constInfo->d, constInfo->kStrideS);
        kL1Buffer.UnlockProd();

        qL1Buffer.LockCons();
        kL1Buffer.LockCons();
        MutexBuffer<BufferType::L0C, SyncType::INNER_CORE_SYNC> l0cBuffer = l0cBuf.Get();
        MMParam param = {
            static_cast<uint32_t>(slot.actualS1Len),
            static_cast<uint32_t>(slot.actualS2Len),
            static_cast<uint32_t>(constInfo->d),
            false,
            true,
            true,
            true,
            UNITFLAG_EN_OUTER_LAST
        };
        MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN,
                        SMALL_SD_L0_SINGLE_BUFFER_SIZE / CUBE_BASEN / sizeof(INPUT_TYPE),
                        ABLayout::MK, ABLayout::KN>(
            qL1Buffer.template GetTensor<INPUT_TYPE>(), kL1Buffer.template GetTensor<INPUT_TYPE>(), l0aBuf, l0bBuf,
            l0cBuffer.GetTensor<CALC_TYPE>(), param);
        FixpipeScoreToUb(dstTensor, l0cBuffer.GetTensor<CALC_TYPE>(), slot.actualS1Len, slot.actualS2Len);
        kL1Buffer.UnlockCons();
        qL1Buffer.UnlockCons();
    }

    __aicore__ inline void IterateSmallSDDsK(LocalTensor<CALC_TYPE> &dstTensor,
                                            MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dsL1Buffer,
                                            const SmallSDPipelineSlot &slot)
    {
        MutexBuffer<BufferType::L1> kL1Buffer = kL1Buf.Get();
        kL1Buffer.LockProd();
        CopyGmNdToL1Nz(kL1Buffer.template GetTensor<INPUT_TYPE>(), keyGm, slot.kOffset, slot.actualS2Len,
                       constInfo->d, constInfo->kStrideS);
        kL1Buffer.UnlockProd();
        kL1Buffer.LockCons();
        MutexBuffer<BufferType::L0C, SyncType::INNER_CORE_SYNC> l0cBuffer = l0cBuf.Get();
        MMParam param = {
            static_cast<uint32_t>(slot.actualS1Len),
            static_cast<uint32_t>(constInfo->d),
            static_cast<uint32_t>(slot.actualS2Len),
            false,
            false,
            true,
            true,
            UNITFLAG_EN_OUTER_LAST
        };
        MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, CUBE_BASEN,
                        ABLayout::MK, ABLayout::KN>(
            dsL1Buffer.template GetTensor<INPUT_TYPE>(), kL1Buffer.template GetTensor<INPUT_TYPE>(), l0aBuf, l0bBuf,
            l0cBuffer.GetTensor<CALC_TYPE>(), param);
        FixpipeDqDkToUb(dstTensor, l0cBuffer.GetTensor<CALC_TYPE>(), slot.actualS1Len, constInfo->d);
        kL1Buffer.UnlockCons();
    }

    __aicore__ inline void IterateSmallSDDsQ(LocalTensor<CALC_TYPE> &dstTensor,
                                            MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dsL1Buffer,
                                            const SmallSDPipelineSlot &slot)
    {
        MutexBuffer<BufferType::L1> qL1Buffer = qL1Buf.Get();
        qL1Buffer.LockProd();
        CopyGmNdToL1Nz(qL1Buffer.template GetTensor<INPUT_TYPE>(), queryGm, slot.qOffset, slot.actualS1Len,
                       constInfo->d, constInfo->qStrideS);
        qL1Buffer.UnlockProd();
        qL1Buffer.LockCons();
        MutexBuffer<BufferType::L0C, SyncType::INNER_CORE_SYNC> l0cBuffer = l0cBuf.Get();
        MMParam param = {
            static_cast<uint32_t>(slot.actualS2Len),
            static_cast<uint32_t>(constInfo->d),
            static_cast<uint32_t>(slot.actualS1Len),
            true,
            false,
            true,
            true,
            UNITFLAG_EN_OUTER_LAST
        };
        MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, CUBE_BASEN,
                        ABLayout::MK, ABLayout::KN>(
            dsL1Buffer.template GetTensor<INPUT_TYPE>(), qL1Buffer.template GetTensor<INPUT_TYPE>(), l0aBuf, l0bBuf,
            l0cBuffer.GetTensor<CALC_TYPE>(), param);
        FixpipeDqDkToUb(dstTensor, l0cBuffer.GetTensor<CALC_TYPE>(), slot.actualS2Len, constInfo->d);
        qL1Buffer.UnlockCons();
    }

    __aicore__ inline void IterateSmallSDPDy(GlobalTensor<OUTDTYPE> &dstGm,
                                            MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buffer,
                                            const SmallSDPipelineSlot &slot)
    {
        MutexBuffer<BufferType::L1> dyL1Buffer = dyL1Buf.Get();
        dyL1Buffer.LockProd();
        CopyGmNdToL1Nz(dyL1Buffer.template GetTensor<INPUT_TYPE>(), dyGm, slot.dyOffset, slot.actualS1Len,
                       constInfo->dv, constInfo->dyStrideS);
        dyL1Buffer.UnlockProd();
        dyL1Buffer.LockCons();
        MutexBuffer<BufferType::L0C, SyncType::INNER_CORE_SYNC> l0cBuffer = l0cBuf.Get();
        MMParam param = {
            static_cast<uint32_t>(slot.actualS2Len),
            static_cast<uint32_t>(constInfo->dv),
            static_cast<uint32_t>(slot.actualS1Len),
            true,
            false,
            true,
            true,
            UNITFLAG_EN_OUTER_LAST
        };
        MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, CUBE_BASEN,
                        ABLayout::MK, ABLayout::KN>(
            pL1Buffer.template GetTensor<INPUT_TYPE>(), dyL1Buffer.template GetTensor<INPUT_TYPE>(), l0aBuf, l0bBuf,
            l0cBuffer.GetTensor<CALC_TYPE>(), param);
        FixpipeDvToGm(dstGm, l0cBuffer.GetTensor<CALC_TYPE>(), slot);
        dyL1Buffer.UnlockCons();
    }

    __aicore__ inline void FixpipeScoreToUb(LocalTensor<CALC_TYPE> &dstTensor, LocalTensor<CALC_TYPE> l0cTensor,
                                           uint32_t rows, uint32_t cols) const
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = cols;
        fixpipeParams.mSize = (rows + 1) >> 1 << 1;
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        fixpipeParams.dstStride = CUBE_BASEN;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.unitFlag = UNITFLAG_EN_OUTER_LAST;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0cTensor, fixpipeParams);
    }

    __aicore__ inline void FixpipeDqDkToUb(LocalTensor<CALC_TYPE> &dstTensor, LocalTensor<CALC_TYPE> l0cTensor,
                                          uint32_t rows, uint32_t cols) const
    {
        constexpr static FixpipeConfig DQDK_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = cols;
        fixpipeParams.mSize = (rows + 1) >> 1 << 1;
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        fixpipeParams.dstStride = HEAD_DIM;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.unitFlag = UNITFLAG_EN_OUTER_LAST;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<CALC_TYPE, CALC_TYPE, DQDK_FIXPIPE_CONFIG>(dstTensor, l0cTensor, fixpipeParams);
    }

    __aicore__ inline void FixpipeDvToGm(GlobalTensor<OUTDTYPE> &dstGm, LocalTensor<CALC_TYPE> l0cTensor,
                                        const SmallSDPipelineSlot &slot) const
    {
        constexpr static FixpipeConfig DV_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, false};
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = constInfo->dv;
        fixpipeParams.mSize = slot.actualS2Len;
        fixpipeParams.srcStride = AlignTo16(slot.actualS2Len);
        fixpipeParams.dstStride = constInfo->dvStrideS;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.unitFlag = UNITFLAG_EN_OUTER_LAST;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        if constexpr (IsSameType<OUTDTYPE, half>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        } else if constexpr (IsSameType<OUTDTYPE, bfloat16_t>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322BF16;
        }
        Fixpipe<OUTDTYPE, CALC_TYPE, DV_FIXPIPE_CONFIG>(dstGm[slot.dvOffset], l0cTensor, fixpipeParams);
    }

    TPipe *pipe = nullptr;
    const SmallSDConstInfo *constInfo = nullptr;
    TBuf<> *mm1ResBuf = nullptr;
    TBuf<> *mm2ResBuf = nullptr;
    MutexBufferManager<BufferType::L1> *l1BufferManager = nullptr;
    MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> *dSL1Buf = nullptr;
    MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> *pL1Buf = nullptr;
    GlobalTensor<INPUT_TYPE> queryGm;
    GlobalTensor<INPUT_TYPE> keyGm;
    GlobalTensor<INPUT_TYPE> valueGm;
    GlobalTensor<INPUT_TYPE> dyGm;
    GlobalTensor<OUTDTYPE> dqGm;
    GlobalTensor<OUTDTYPE> dkGm;
    GlobalTensor<OUTDTYPE> dvGm;
    MutexBuffersPolicySingleBuffer<BufferType::L1> qL1Buf;
    MutexBuffersPolicySingleBuffer<BufferType::L1> kL1Buf;
    MutexBuffersPolicySingleBuffer<BufferType::L1> vL1Buf;
    MutexBuffersPolicySingleBuffer<BufferType::L1> dyL1Buf;
    MutexBufferManager<BufferType::L0A> l0aBufferManager;
    MutexBufferManager<BufferType::L0B> l0bBufferManager;
    MutexBufferManager<BufferType::L0C> l0cBufferManager;
    MutexBuffersPolicyDB<BufferType::L0A> l0aBuf;
    MutexBuffersPolicyDB<BufferType::L0B> l0bBuf;
    MutexBuffersPolicySingleBuffer<BufferType::L0C, SyncType::INNER_CORE_SYNC> l0cBuf;
};

template <typename T>
struct CubeBlockTraits;

template <typename INPUT_TYPE, typename CALC_TYPE, typename OUTDTYPE, bool IS_TND, uint32_t HEAD_DIM>
struct CubeBlockTraits<SmallSDCubeBlock<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_TND, HEAD_DIM>> {
    using INPUT_TYPE_TRAITS = INPUT_TYPE;
    using CALC_TYPE_TRAITS = CALC_TYPE;
    using OUTDTYPE_TRAITS = OUTDTYPE;
    static constexpr bool IS_ATTEN_MASKTraits = false;
    static constexpr bool IS_PSETraits = false;
    static constexpr bool IS_DROPTraits = false;
    static constexpr bool IS_TNDTraits = IS_TND;
    static constexpr bool IS_BN2_MULTIBLKTraits = false;
    static constexpr uint8_t DETER_SPARSE_TYPETraits = NO_DETER;
    static constexpr bool IS_N_EQUALTraits = true;
    static constexpr bool IS_D_NO_EQUALTraits = false;
    static constexpr bool IS_ROPETraits = false;
    static constexpr bool IS_NZ_OUTTraits = false;
    static constexpr bool IS_TND_SWIZZLETraits = false;
    static constexpr uint8_t SPLIT_AXISTraits = BN2;
    static constexpr S1TemplateType s1TemplateTypeTraits = S1TemplateType::Aligned128;
    static constexpr S2TemplateType s2TemplateTypeTraits = S2TemplateType::Aligned128;
    static constexpr DTemplateType dTemplateTypeTraits =
        HEAD_DIM == 64 ? DTemplateType::Aligned64 : DTemplateType::Aligned128;
};

} // namespace FagBaseApi

#endif
