/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sparse_lightning_indexer_kl_loss_grad_cube_block.h
 * \brief
 */
#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_CUBE_BLOCK_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_CUBE_BLOCK_H

#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "sparse_lightning_indexer_kl_loss_grad_regbase_common.h"
namespace Slig {
using T = float;

TEMPLATES_DEF
class SligBlockCube {
public:
    TPipe *pipe;
    constexpr static uint32_t QUERY_INDEX_BUFFER_SIZE = 128 * 128 * 2;
    constexpr static uint32_t MM4_GATHER_BUFFER_SIZE = 128 * 128 * 2;
    constexpr static uint32_t L0C_SINGLE_BUFFER_SIZE = 64 * 1024;
    constexpr static uint32_t L0_SINGLE_BUFFER_SIZE = 32 * 1024;
    Buffer<BufferType::L1> sYQL1Buffer;
    LocalTensor<INPUT_T> sYQL1Tensor;

    static constexpr uint32_t M_SPLIT_SIZE = 128;     // m方向切分
    static constexpr uint32_t N_SPLIT_SIZE = 128;     // n方向切分
    static constexpr uint32_t K_SPLIT_SIZE = 128;     // k方向切分
    static constexpr uint32_t QUERY_INDEX_SIZE = 128; // k方向切分
    static constexpr uint32_t DROPE_SIZE = 64;

    static constexpr uint8_t SYNC_MODE = 2;
    static constexpr uint32_t C0_SIZE = 16;

    // matmul global tensor
    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> qRopeGm;
    GlobalTensor<INPUT_T> queryIndexGm;
    GlobalTensor<T> mm2ResGm;
    GlobalTensor<INPUT_T> gatherSYResGm;
    GlobalTensor<OUT_T> mm4ResGm;
    GlobalTensor<T> mm3DeterResGm;
    /* =====================LocalBuffer变量==================== */

    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BuffersPolicySingleBuffer<BufferType::L1> sYQueryL1Buf;
    BuffersPolicySingleBuffer<BufferType::L1> sYGatherResBuf; // mm4右矩阵用

    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BuffersPolicyDB<BufferType::L0A> l0aBuf;
    BuffersPolicyDB<BufferType::L0B> l0bBuf;

    BufferManager<BufferType::L0C> l0cBufferManager;
    BuffersPolicy3buff<BufferType::L0C> commonL0CBuf;
    BuffersPolicySingleBuffer<BufferType::L0C> mm4L0CSpecialBuf;

    SLIGradConstInfo constInfo{};

    __aicore__ inline SligBlockCube(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *tPipe, BufferManager<BufferType::L1> *l1BuffMgr);
    __aicore__ inline void InitCubeBuffers();
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR dQuery, GlobalTensor<T> &bmm2Res,
                                            GlobalTensor<INPUT_T> &gatherSYResGm, GlobalTensor<T> &bmm3Res);
    __aicore__ inline void ComputeMmSy(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
                                       BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &sYL1Buf,
                                       SLIGradRunInfo &runInfo, SLIGradConstInfo &constInfo,
                                       SLIGradKRunInfo &syRunInfo);
    __aicore__ inline void ComputeMm3(BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm3Buf,
                                      Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &reluGradResL1Buf,
                                      SLIGradRunInfo &runInfo, SLIGradConstInfo &constInfo, SLIGradKRunInfo &pRunInfo);
    __aicore__ inline void ComputeMm4(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &reluGradResL1Buf,
                                      SLIGradRunInfo &runInfo, SLIGradConstInfo &constInfo, SLIGradKRunInfo &pRunInfo,
                                      bool isFixOut);

private:
    __aicore__ inline void CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor, GlobalTensor<INPUT_T> &gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<INPUT_T> &l1Tensor, const SLIGradRunInfo &info,
                                          SLIGradConstInfo &constInfo);
    __aicore__ inline void CopyInMm1ARopeToL1(LocalTensor<INPUT_T> &l1Tensor, const SLIGradRunInfo &info,
                                              SLIGradConstInfo &constInfo);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<INPUT_T> &l1Tensor, const SLIGradRunInfo &info,
                                          SLIGradConstInfo &constInfo);
    __aicore__ inline void CopyInMm4BToL1(LocalTensor<INPUT_T> &l1Tensor, uint64_t gatherOffset, uint32_t mSizeAct,
                                          uint32_t realDSize, uint32_t headOffset);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::SetCubeBlockParams(TPipe *tPipe,
                                                                        BufferManager<BufferType::L1> *l1BuffMgr)
{
    this->pipe = tPipe;
    this->l1BufferManagerPtr = l1BuffMgr;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::InitCubeBuffers()
{
    // init l1 buffer
    sYQueryL1Buf.Init(*l1BufferManagerPtr, QUERY_INDEX_BUFFER_SIZE);
    sYGatherResBuf.Init(*l1BufferManagerPtr, MM4_GATHER_BUFFER_SIZE);

    // init l0a l0b buffer
    l0aBufferManager.Init(pipe, L0_MAX_SIZE);
    l0bBufferManager.Init(pipe, L0_MAX_SIZE);
    l0aBuf.Init(l0aBufferManager, L0_SINGLE_BUFFER_SIZE);
    l0bBuf.Init(l0bBufferManager, L0_SINGLE_BUFFER_SIZE);

    // init l0c buffer
    l0cBufferManager.Init(pipe, L0C_MAX_SIZE);
    commonL0CBuf.Init(l0cBufferManager, L0C_SINGLE_BUFFER_SIZE);
    mm4L0CSpecialBuf.Init(l0cBufferManager, L0_SINGLE_BUFFER_SIZE);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR query, GM_ADDR dQuery,
                                                                      GlobalTensor<T> &bmm2Res,
                                                                      GlobalTensor<INPUT_T> &gatherSYResGm,
                                                                      GlobalTensor<T> &bmm3Res)
{
    queryIndexGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    mm4ResGm.SetGlobalBuffer((__gm__ OUT_T *)dQuery);
    this->mm2ResGm = bmm2Res;
    this->gatherSYResGm = gatherSYResGm;
    this->mm3DeterResGm = bmm3Res;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor,
                                                                GlobalTensor<INPUT_T> &gmSrcTensor, uint32_t srcN,
                                                                uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN;
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::CopyInMm2AToL1(LocalTensor<INPUT_T> &l1Tensor,
                                                                    const SLIGradRunInfo &info,
                                                                    SLIGradConstInfo &constInfo)
{
    auto srcGm = queryIndexGm[info.queryIndexTensorOffset];
    CopyGmToL1(l1Tensor, srcGm, constInfo.gSizeQueryIndex, constInfo.dSizeQueryIndex, constInfo.dSizeQueryIndex);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::CopyInMm4BToL1(LocalTensor<INPUT_T> &l1Tensor,
                                                                    uint64_t gatherOffset, uint32_t mSizeAct,
                                                                    uint32_t realDSize, uint32_t headOffset)
{
    auto srcGm = gatherSYResGm[gatherOffset];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, realDSize, headOffset);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::ComputeMmSy(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &sYL1Buf, SLIGradRunInfo &runInfo,
    SLIGradConstInfo &constInfo, SLIGradKRunInfo &syRunInfo)
{
    uint32_t dSize = constInfo.dSizeQueryIndex;                          // 128
    uint32_t kSize = syRunInfo.kProcessSize;                             // 2048 / 256
    uint32_t kLoopTimes = CeilDiv(kSize, VEC_SY_BASESIZE);               // 循环次数，循环累计kSize行搬出UB
    uint32_t tailLoopKSize = kSize - (kLoopTimes - 1) * VEC_SY_BASESIZE; // 尾块大小
    LocalTensor<T> outTensor = bmm2ResBuf.template GetTensor<T>();
    MMParam mmParam = {constInfo.gSizeQueryIndex, VEC_SY_BASESIZE, constInfo.dSizeQueryIndex, false, true, true, true,
                       UNITFLAG_DISABLE};
    if (syRunInfo.kTaskId == 0) { // 首次搬运query_index
        sYQL1Buffer = sYQueryL1Buf.Get();
        sYQL1Buffer.Wait<HardEvent::MTE1_MTE2>();
        sYQL1Tensor = sYQL1Buffer.GetTensor<INPUT_T>();
        CopyInMm2AToL1(sYQL1Tensor, runInfo, constInfo);
        sYQL1Buffer.Set<HardEvent::MTE2_MTE1>();
        sYQL1Buffer.Wait<HardEvent::MTE2_MTE1>();
    }
    Buffer<BufferType::L0C> mm2L0CBuffer = commonL0CBuf.Get();
    LocalTensor<T> mm2L0CTensor = mm2L0CBuffer.GetTensor<T>();
    uint64_t gmOffset = syRunInfo.kTaskId * constInfo.syKBaseSize;
    mm2L0CBuffer.Wait<HardEvent::FIX_M>(); // 反向同步
    for (uint32_t kIdx = 0; kIdx < kLoopTimes; kIdx++) {
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> gatherResL1Buf = sYL1Buf.Get();
        // 右矩阵
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_GATHER_TO_MM12_FLAG[syRunInfo.kTaskIdMod2]);
        LocalTensor<INPUT_T> gatherL1Tensor = gatherResL1Buf.GetTensor<INPUT_T>();
        uint64_t l0cOffset = kIdx * VEC_SY_BASESIZE * constInfo.gSizeQueryIndex;
        MatmulBase<INPUT_T, INPUT_T, T, QUERY_INDEX_SIZE, N_SPLIT_SIZE, K_SPLIT_SIZE, ABLayout::MK, ABLayout::KN>(
            sYQL1Tensor, gatherL1Tensor, l0aBuf, l0bBuf, mm2L0CTensor[l0cOffset], mmParam);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_GATHER_TO_MM12_FLAG[syRunInfo.kTaskIdMod2]);
    }
    mm2L0CBuffer.Set<HardEvent::M_FIX>();
    mm2L0CBuffer.Wait<HardEvent::M_FIX>();
    // fix2ub
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[syRunInfo.kTaskIdMod2]);
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2UbParams;
    fixpipe2UbParams.nSize = (kSize + 7) >> 3 << 3;
    fixpipe2UbParams.mSize = (constInfo.gSizeQueryIndex + 1) >> 1 << 1;
    fixpipe2UbParams.srcStride = AlignTo(fixpipe2UbParams.mSize, static_cast<uint16_t>(C0_SIZE));
    // mmResUb上两行之间的间隔，单位：element。
    fixpipe2UbParams.dstStride = AlignTo(constInfo.syKBaseSize, static_cast<int32_t>(C0_SIZE));
    // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipe2UbParams.dualDstCtl = 1;
    fixpipe2UbParams.params.ndNum = 1;
    fixpipe2UbParams.params.srcNdStride = 0;
    fixpipe2UbParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outTensor, mm2L0CTensor, fixpipe2UbParams);
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[syRunInfo.kTaskIdMod2]);
    // fix2gm
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2GmParams;
    fixpipe2GmParams.nSize = kSize;
    fixpipe2GmParams.mSize = constInfo.gSizeQueryIndex;
    fixpipe2GmParams.srcStride = AlignTo(fixpipe2GmParams.mSize, static_cast<uint16_t>(C0_SIZE));
    fixpipe2GmParams.dstStride = AlignTo(constInfo.kSizeAlign128, static_cast<uint32_t>(C0_SIZE));
    fixpipe2GmParams.reluEn = true;
    fixpipe2GmParams.dualDstCtl = 0;
    fixpipe2GmParams.params.ndNum = 1;
    fixpipe2GmParams.params.srcNdStride = 0;
    fixpipe2GmParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_GM>(mm2ResGm[gmOffset], mm2L0CTensor, fixpipe2GmParams);
    mm2L0CBuffer.Set<HardEvent::FIX_M>();
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::ComputeMm3(
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm3Buf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &reluGradResL1Buf, SLIGradRunInfo &runInfo,
    SLIGradConstInfo &constInfo, SLIGradKRunInfo &pRunInfo)
{
    uint32_t kSize = pRunInfo.kProcessSize;                             // 2048
    uint32_t kLoopTimes = CeilDiv(kSize, VEC_P_BASESIZE);               // 循环次数，循环累计kSize行搬出UB
    uint32_t tailLoopKSize = kSize - (kLoopTimes - 1) * VEC_P_BASESIZE; // 尾块大小
    LocalTensor<INPUT_T> reluGradL1Tensor;
    MMParam mmParam = {M_SPLIT_SIZE, constInfo.dSizeQueryIndex, constInfo.gSizeQueryIndex, true, false, true,
                       true,         UNITFLAG_DISABLE};
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V6_TO_C3_FLAG);
    for (uint32_t kIdx = 0; kIdx < kLoopTimes; kIdx++) {
        if (kIdx == kLoopTimes - 1) {
            mmParam.singleM = tailLoopKSize;
        }
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm3ResBuf = bmm3Buf.Get();
        LocalTensor<T> outTensor = bmm3ResBuf.GetTensor<T>();
        reluGradL1Tensor = reluGradResL1Buf.GetTensor<INPUT_T>();
        int64_t reluGradOffset =
            kIdx * VEC_P_BASESIZE * AlignTo(constInfo.gSizeQueryIndex, static_cast<uint32_t>(C0_SIZE));
        // 拷贝左矩阵
        Buffer<BufferType::L0C> mm3L0CBuffer = commonL0CBuf.Get();
        mm3L0CBuffer.Wait<HardEvent::FIX_M>();
        MatmulBase<INPUT_T, INPUT_T, T, M_SPLIT_SIZE, N_SPLIT_SIZE, QUERY_INDEX_SIZE, ABLayout::MK, ABLayout::KN>(
            reluGradL1Tensor[reluGradOffset], sYQL1Tensor, l0aBuf, l0bBuf, mm3L0CBuffer.GetTensor<T>(), mmParam);
        mm3L0CBuffer.Set<HardEvent::M_FIX>();
        mm3L0CBuffer.Wait<HardEvent::M_FIX>();
        // fix2ub
        if constexpr (!Deterministic) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V7_FLAG[pRunInfo.kTaskIdMod2]);
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2UbParams;
            fixpipe2UbParams.nSize = (mmParam.singleN + 7) >> 3 << 3;
            fixpipe2UbParams.mSize = (mmParam.singleM + 1) >> 1 << 1;
            fixpipe2UbParams.srcStride = AlignTo(fixpipe2UbParams.mSize, static_cast<uint16_t>(C0_SIZE));
            // mmResUb上两行之间的间隔，单位：element。
            fixpipe2UbParams.dstStride = AlignTo(constInfo.dSizeQueryIndex, static_cast<uint32_t>(C0_SIZE));
            // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
            fixpipe2UbParams.dualDstCtl = 1;
            fixpipe2UbParams.params.ndNum = 1;
            fixpipe2UbParams.params.srcNdStride = 0;
            fixpipe2UbParams.params.dstNdStride = 0;
            Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outTensor, mm3L0CBuffer.GetTensor<T>(), fixpipe2UbParams);
            mm3L0CBuffer.Set<HardEvent::FIX_M>();
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V7_FLAG[pRunInfo.kTaskIdMod2]);
        } else {
            int64_t gmOffset = (runInfo.taskIdMod2 * constInfo.kSize + pRunInfo.kTaskId * constInfo.pKBaseSize +
                                kIdx * VEC_P_BASESIZE) *
                               constInfo.dSizeQueryIndex;
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2GmParams;
            fixpipe2GmParams.nSize = mmParam.singleN;
            fixpipe2GmParams.mSize = mmParam.singleM;
            fixpipe2GmParams.srcStride = AlignTo(fixpipe2GmParams.mSize, static_cast<uint16_t>(C0_SIZE));
            fixpipe2GmParams.dstStride = AlignTo(constInfo.dSizeQueryIndex, static_cast<uint32_t>(C0_SIZE));
            fixpipe2GmParams.dualDstCtl = 0;
            fixpipe2GmParams.params.ndNum = 1;
            fixpipe2GmParams.params.srcNdStride = 0;
            fixpipe2GmParams.params.dstNdStride = 0;
            Fixpipe<T, T, PFA_CFG_ROW_MAJOR_GM>(mm3DeterResGm[constInfo.bmm3BaseOffset + gmOffset],
                                                mm3L0CBuffer.GetTensor<T>(), fixpipe2GmParams);
            mm3L0CBuffer.Set<HardEvent::FIX_M>();
        }
    }
    if (pRunInfo.kTaskId == runInfo.s2LoopTimes - 1) {
        sYQL1Buffer.Set<HardEvent::MTE1_MTE2>();
    }
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::ComputeMm4(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &reluGradResL1Buf, SLIGradRunInfo &runInfo,
    SLIGradConstInfo &constInfo, SLIGradKRunInfo &pRunInfo, bool isFixOut)
{
    uint32_t kSize = pRunInfo.kProcessSize;                             // 2048
    uint32_t kLoopTimes = CeilDiv(kSize, VEC_P_BASESIZE);               // 循环次数，循环累计kSize行搬出UB
    uint32_t tailLoopKSize = kSize - (kLoopTimes - 1) * VEC_P_BASESIZE; // 尾块大小
    uint32_t dSize = constInfo.dSizeQueryIndex;

    LocalTensor<INPUT_T> reluGradL1Tensor;
    Buffer<BufferType::L1> sYGatherL1Buffer = sYGatherResBuf.Get();
    LocalTensor<INPUT_T> sYGatherResTensor = sYGatherL1Buffer.GetTensor<INPUT_T>();
    MMParam mmParam = {constInfo.gSizeQueryIndex, dSize, K_SPLIT_SIZE, false, false, true, true, UNITFLAG_DISABLE};
    Buffer<BufferType::L0C> mm4L0CBuffer = mm4L0CSpecialBuf.Get();
    for (uint32_t kIdx = 0; kIdx < kLoopTimes; kIdx++) { // 2048 / 128
        if (kIdx == kLoopTimes - 1) {
            mmParam.singleK = tailLoopKSize;
        }

        reluGradL1Tensor = reluGradResL1Buf.GetTensor<INPUT_T>();
        // 拷贝左矩阵
        // 拷贝右矩阵
        int64_t reluGradOffset =
            kIdx * VEC_P_BASESIZE * AlignTo(constInfo.gSizeQueryIndex, static_cast<uint32_t>(C0_SIZE));
        int64_t gatherWorkspaceOffset = (pRunInfo.kTaskId * constInfo.pKBaseSize + kIdx * VEC_P_BASESIZE) * dSize;

        sYGatherL1Buffer.Wait<HardEvent::MTE1_MTE2>();
        CopyInMm4BToL1(sYGatherResTensor, gatherWorkspaceOffset, mmParam.singleK, dSize, dSize);
        sYGatherL1Buffer.Set<HardEvent::MTE2_MTE1>();
        sYGatherL1Buffer.Wait<HardEvent::MTE2_MTE1>();
        if ((pRunInfo.kTaskId == 0) && (kIdx == 0)) {
            mm4L0CBuffer.Wait<HardEvent::FIX_M>();
            mmParam.isOutKFisrt = true;
        } else {
            mmParam.isOutKFisrt = false;
        }
        MatmulBase<INPUT_T, INPUT_T, T, QUERY_INDEX_SIZE, N_SPLIT_SIZE, K_SPLIT_SIZE, ABLayout::MK, ABLayout::KN>(
            reluGradL1Tensor[reluGradOffset], sYGatherResTensor, l0aBuf, l0bBuf, mm4L0CBuffer.GetTensor<T>(), mmParam);
        sYGatherL1Buffer.Set<HardEvent::MTE1_MTE2>();
    }
    mm4L0CBuffer.Set<HardEvent::M_FIX>();
    mm4L0CBuffer.Wait<HardEvent::M_FIX>();
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V6_TO_C3_FLAG);
    if (isFixOut) {
        // fix2gm
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2GmParams;
        fixpipe2GmParams.nSize = dSize;
        fixpipe2GmParams.mSize = constInfo.gSizeQueryIndex;
        fixpipe2GmParams.srcStride = AlignTo(fixpipe2GmParams.mSize, static_cast<uint16_t>(C0_SIZE));
        fixpipe2GmParams.dstStride = AlignTo(dSize, static_cast<uint32_t>(C0_SIZE));
        fixpipe2GmParams.dualDstCtl = 0;
        fixpipe2GmParams.params.ndNum = 1;
        fixpipe2GmParams.params.srcNdStride = 0;
        fixpipe2GmParams.params.dstNdStride = 0;
        if constexpr (std::is_same<OUT_T, half>::value) {
            fixpipe2GmParams.quantPre = QuantMode_t::F322F16;
        } else {
            fixpipe2GmParams.quantPre = QuantMode_t::F322BF16;
        }
        Fixpipe<OUT_T, T, PFA_CFG_ROW_MAJOR_GM>(mm4ResGm[runInfo.queryIndexTensorOffset], mm4L0CBuffer.GetTensor<T>(),
                                                fixpipe2GmParams);
        mm4L0CBuffer.Set<HardEvent::FIX_M>();
    }
};

TEMPLATES_DEF
class SligBlockCubeDummy {
public:
    __aicore__ inline SligBlockCubeDummy(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *tPipe, BufferManager<BufferType::L1> *l1BuffMgr) {};
    __aicore__ inline void InitCubeBuffers() {};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR dQuery, GlobalTensor<T> &bmm2Res,
                                            GlobalTensor<INPUT_T> &gatherSYResGm, GlobalTensor<T> &bmm3Res) {};
    __aicore__ inline void ComputeMmSy(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
                                       BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &sYL1Buf,
                                       SLIGradRunInfo &runInfo, SLIGradConstInfo &constInfo,
                                       SLIGradKRunInfo &syRunInfo) {};
    __aicore__ inline void ComputeMm3(BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm3Buf,
                                      Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &reluGradResL1Buf,
                                      SLIGradRunInfo &runInfo, SLIGradConstInfo &constInfo,
                                      SLIGradKRunInfo &pRunInfo) {};
    __aicore__ inline void ComputeMm4(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &reluGradResL1Buf,
                                      SLIGradRunInfo &runInfo, SLIGradConstInfo &constInfo, SLIGradKRunInfo &pRunInfo,
                                      bool isFixOut) {};
};

template <typename T>
struct CubeBlockTraits; // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    };

DEFINE_CUBE_BLOCK_TRAITS(SligBlockCube);
DEFINE_CUBE_BLOCK_TRAITS(SligBlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
} // namespace Slig
#endif
