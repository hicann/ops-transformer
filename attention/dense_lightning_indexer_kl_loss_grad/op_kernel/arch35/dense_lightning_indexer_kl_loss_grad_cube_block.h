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
 * \file dense_lightning_indexer_kl_loss_grad_cube_block.h
 * \brief
 */
#ifndef DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_CUBE_BLOCK_H
#define DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_CUBE_BLOCK_H

#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "dense_lightning_indexer_kl_loss_grad_regbase_common.h"
namespace Dlikg {
using T = float;

TEMPLATES_DEF
class DlikgBlockCube {
public:
    TPipe *pipe;
    constexpr static uint32_t QUERY_INDEX_BUFFER_SIZE = 128 * 128 * 2;
    constexpr static uint32_t MM4_GATHER_BUFFER_SIZE = 128 * 128 * 2;
    constexpr static uint32_t L0C_SINGLE_BUFFER_SIZE = 64 * 1024;
    constexpr static uint32_t L0_SINGLE_BUFFER_SIZE = 32 * 1024;
    Buffer<BufferType::L1> sYQL1Buffer[3];
    LocalTensor<INPUT_T> sYQL1Tensor[3];
    Buffer<BufferType::L1> sYKL1Buffer[3];
    LocalTensor<INPUT_T> sYKL1Tensor[3];

    static constexpr uint32_t M_SPLIT_SIZE = 128;     // m方向切分
    static constexpr uint32_t N_SPLIT_SIZE = 128;     // n方向切分
    static constexpr uint32_t K_SPLIT_SIZE = 128;     // k方向切分
    static constexpr uint32_t QUERY_INDEX_SIZE = 128; // k方向切分

    static constexpr uint8_t SYNC_MODE = 2;
    static constexpr uint32_t C0_SIZE = 16;

    // matmul global tensor
    GlobalTensor<INPUT_T> queryIndexGm;
    GlobalTensor<INPUT_T> keyIndexGm;
    GlobalTensor<OUT_T> mm4ResGm;
    GlobalTensor<OUT_T> mm3ResGm;
    GlobalTensor<T> deterRes;
    GlobalTensor<T> bmm3Res;
    /* =====================LocalBuffer变量==================== */

    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BuffersPolicy3buff<BufferType::L1> sYKeyL1Buf;
    BuffersPolicy3buff<BufferType::L1> sYQueryL1Buf;

    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BuffersPolicyDB<BufferType::L0A> l0aBuf;
    BuffersPolicyDB<BufferType::L0B> l0bBuf;

    BufferManager<BufferType::L0C> l0cBufferManager;
    BuffersPolicy3buff<BufferType::L0C> commonL0CBuf;
    BuffersPolicySingleBuffer<BufferType::L0C> mm4L0CSpecialBuf;

    DLIKGradConstInfo constInfo{};

    __aicore__ inline DlikgBlockCube(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *tPipe, BufferManager<BufferType::L1> *l1BuffMgr);
    __aicore__ inline void InitCubeBuffers();
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR dQuery, GM_ADDR key, GM_ADDR dKey,
                                            GlobalTensor<T> bmm3Res, GlobalTensor<T> deterRes);
    __aicore__ inline void ComputeMmSy(Buffer<BufferType::UB, SyncType::NO_SYNC> &bmm2ResBuf, DLIKGradRunInfo &runInfo,
                                       DLIKGradConstInfo &constInfo);
    __aicore__ inline void ComputeMmDk(Buffer<BufferType::L1, SyncType::NO_SYNC> &reluGradResL1Buf,
                                       DLIKGradRunInfo &runInfo, DLIKGradConstInfo &constInfo);
    __aicore__ inline void ComputeMmDq(Buffer<BufferType::L1, SyncType::NO_SYNC> &reluGradResL1Buf,
                                       DLIKGradRunInfo &runInfo, DLIKGradConstInfo &constInfo, bool isFixOut);

private:
    __aicore__ inline void CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor, GlobalTensor<INPUT_T> &gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<INPUT_T> &l1Tensor, const DLIKGradRunInfo &info,
                                          DLIKGradConstInfo &constInfo);
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<INPUT_T> &l1Tensor, const DLIKGradRunInfo &info,
                                          DLIKGradConstInfo &constInfo);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockCube<TEMPLATE_ARGS>::SetCubeBlockParams(TPipe *tPipe,
                                                                         BufferManager<BufferType::L1> *l1BuffMgr)
{
    this->pipe = tPipe;
    this->l1BufferManagerPtr = l1BuffMgr;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockCube<TEMPLATE_ARGS>::InitCubeBuffers()
{
    // init l1 buffer
    sYKeyL1Buf.Init(*l1BufferManagerPtr, QUERY_INDEX_BUFFER_SIZE);
    sYQueryL1Buf.Init(*l1BufferManagerPtr, QUERY_INDEX_BUFFER_SIZE);
    // sYGatherResBuf.Init(*l1BufferManagerPtr, MM4_GATHER_BUFFER_SIZE);

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
__aicore__ inline void DlikgBlockCube<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR query, GM_ADDR dQuery, GM_ADDR key,
                                                                       GM_ADDR dKey, GlobalTensor<T> bmm3Res,
                                                                       GlobalTensor<T> deterRes)
{
    queryIndexGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    keyIndexGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    mm4ResGm.SetGlobalBuffer((__gm__ OUT_T *)dQuery);
    mm3ResGm.SetGlobalBuffer((__gm__ OUT_T *)dKey);
    this->bmm3Res = bmm3Res;
    this->deterRes = deterRes;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockCube<TEMPLATE_ARGS>::CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor,
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
__aicore__ inline void DlikgBlockCube<TEMPLATE_ARGS>::CopyInMm2AToL1(LocalTensor<INPUT_T> &l1Tensor,
                                                                     const DLIKGradRunInfo &info,
                                                                     DLIKGradConstInfo &constInfo)
{
    auto srcGm = queryIndexGm[info.queryIndexTensorOffset];
    CopyGmToL1(l1Tensor, srcGm, constInfo.gSizeQueryIndex, constInfo.dSizeQueryIndex, constInfo.dSizeQueryIndex);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockCube<TEMPLATE_ARGS>::CopyInMm2BToL1(LocalTensor<INPUT_T> &l1Tensor,
                                                                     const DLIKGradRunInfo &info,
                                                                     DLIKGradConstInfo &constInfo)
{
    auto srcGm = keyIndexGm[info.keyIndexTensorOffset];
    CopyGmToL1(l1Tensor, srcGm, info.kProcessSize, constInfo.dSizeQueryIndex, constInfo.dSizeQueryIndex);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockCube<TEMPLATE_ARGS>::ComputeMmSy(Buffer<BufferType::UB, SyncType::NO_SYNC> &bmm2ResBuf,
                                                                  DLIKGradRunInfo &runInfo,
                                                                  DLIKGradConstInfo &constInfo)
{
    uint32_t dSize = constInfo.dSizeQueryIndex;
    uint32_t kSize = runInfo.kProcessSize;
    uint32_t kLoopTimes = CeilDiv(kSize, VEC_SY_BASESIZE);
    uint32_t tailLoopKSize = kSize - (kLoopTimes - 1) * VEC_SY_BASESIZE;
    LocalTensor<T> outTensor = bmm2ResBuf.template GetTensor<T>();
    MMParam mmParam = {constInfo.gSizeQueryIndex, kSize, constInfo.dSizeQueryIndex, false, true, true, true,
                       UNITFLAG_DISABLE};

    // 搬运query_index
    if (runInfo.s2Offset == 0) {
        sYQL1Buffer[runInfo.sTaskIdMod3] = sYQueryL1Buf.Get();
        sYQL1Buffer[runInfo.sTaskIdMod3].Wait<HardEvent::MTE1_MTE2>();
        sYQL1Tensor[runInfo.sTaskIdMod3] = sYQL1Buffer[runInfo.sTaskIdMod3].GetTensor<INPUT_T>();
        CopyInMm2AToL1(sYQL1Tensor[runInfo.sTaskIdMod3], runInfo, constInfo);
        sYQL1Buffer[runInfo.sTaskIdMod3].Set<HardEvent::MTE2_MTE1>();
        sYQL1Buffer[runInfo.sTaskIdMod3].Wait<HardEvent::MTE2_MTE1>();
    }
    // 搬运key_index
    sYKL1Buffer[runInfo.taskIdMod3] = sYKeyL1Buf.Get();
    LocalTensor<INPUT_T> syKL1Tensor = sYKL1Buffer[runInfo.taskIdMod3].GetTensor<INPUT_T>();
    sYKL1Buffer[runInfo.taskIdMod3].Wait<HardEvent::MTE1_MTE2>();
    CopyInMm2BToL1(syKL1Tensor, runInfo, constInfo);
    sYKL1Buffer[runInfo.taskIdMod3].Set<HardEvent::MTE2_MTE1>();
    sYKL1Buffer[runInfo.taskIdMod3].Wait<HardEvent::MTE2_MTE1>();
    Buffer<BufferType::L0C> mm2L0CBuffer = commonL0CBuf.Get();
    LocalTensor<T> mm2L0CTensor = mm2L0CBuffer.GetTensor<T>();

    mm2L0CBuffer.Wait<HardEvent::FIX_M>();
    for (uint32_t kIdx = 0; kIdx < kLoopTimes; kIdx++) {
        uint64_t l0cOffset = kIdx * VEC_SY_BASESIZE * constInfo.gSizeQueryIndex;
        MatmulBase<INPUT_T, INPUT_T, T, QUERY_INDEX_SIZE, N_SPLIT_SIZE, K_SPLIT_SIZE, ABLayout::MK, ABLayout::KN>(
            sYQL1Tensor[runInfo.sTaskIdMod3], syKL1Tensor, l0aBuf, l0bBuf, mm2L0CTensor[l0cOffset], mmParam);
    }
    mm2L0CBuffer.Set<HardEvent::M_FIX>();
    mm2L0CBuffer.Wait<HardEvent::M_FIX>();
    // fix2ub
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[runInfo.taskIdMod3]);
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2UbParams;
    fixpipe2UbParams.nSize = (kSize + 7) >> 3 << 3;
    fixpipe2UbParams.mSize = (constInfo.gSizeQueryIndex + 1) >> 1 << 1;
    fixpipe2UbParams.srcStride = AlignTo(fixpipe2UbParams.mSize, static_cast<uint16_t>(C0_SIZE));
    fixpipe2UbParams.dstStride = AlignTo(constInfo.syKBaseSize, static_cast<int32_t>(C0_SIZE));
    fixpipe2UbParams.dualDstCtl = 1;
    fixpipe2UbParams.params.ndNum = 1;
    fixpipe2UbParams.params.srcNdStride = 0;
    fixpipe2UbParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outTensor, mm2L0CTensor, fixpipe2UbParams);
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[runInfo.taskIdMod3]);
    mm2L0CBuffer.Set<HardEvent::FIX_M>();
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockCube<TEMPLATE_ARGS>::ComputeMmDk(
    Buffer<BufferType::L1, SyncType::NO_SYNC> &reluGradResL1Buf, DLIKGradRunInfo &runInfo, DLIKGradConstInfo &constInfo)
{
    uint32_t kSize = runInfo.kProcessSize;
    uint32_t kLoopTimes = CeilDiv(kSize, VEC_P_BASESIZE);
    uint32_t tailLoopKSize = kSize - (kLoopTimes - 1) * VEC_P_BASESIZE;
    LocalTensor<INPUT_T> reluGradL1Tensor;
    MMParam mmParam = {kSize, constInfo.dSizeQueryIndex, constInfo.gSizeQueryIndex, true, false, true,
                       true,  UNITFLAG_DISABLE};
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V2_TO_C3_FLAG[runInfo.taskIdMod3]);
    for (uint32_t kIdx = 0; kIdx < kLoopTimes; kIdx++) {
        if (kIdx == kLoopTimes - 1) {
            mmParam.singleM = tailLoopKSize;
        }
        reluGradL1Tensor = reluGradResL1Buf.GetTensor<INPUT_T>();
        int64_t reluGradOffset =
            kIdx * VEC_P_BASESIZE * AlignTo(constInfo.gSizeQueryIndex, static_cast<uint32_t>(C0_SIZE));
        // 拷贝左矩阵
        Buffer<BufferType::L0C> mm3L0CBuffer = commonL0CBuf.Get();
        mm3L0CBuffer.Wait<HardEvent::FIX_M>();
        MatmulBase<INPUT_T, INPUT_T, T, M_SPLIT_SIZE, N_SPLIT_SIZE, QUERY_INDEX_SIZE, ABLayout::MK, ABLayout::KN>(
            reluGradL1Tensor[reluGradOffset], sYQL1Tensor[runInfo.sTaskIdMod3], l0aBuf, l0bBuf,
            mm3L0CBuffer.GetTensor<T>(), mmParam);
        mm3L0CBuffer.Set<HardEvent::M_FIX>();
        mm3L0CBuffer.Wait<HardEvent::M_FIX>();
        // fix2ub
        if constexpr (!Deterministic) {
            SetAtomicAdd<T>();
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2GmParams;
            fixpipe2GmParams.nSize = mmParam.singleN;
            fixpipe2GmParams.mSize = mmParam.singleM;
            fixpipe2GmParams.srcStride = AlignTo(fixpipe2GmParams.mSize, static_cast<uint16_t>(C0_SIZE));
            fixpipe2GmParams.dstStride = AlignTo(constInfo.dSizeQueryIndex, static_cast<uint32_t>(C0_SIZE));
            fixpipe2GmParams.dualDstCtl = 0;
            fixpipe2GmParams.params.ndNum = 1;
            fixpipe2GmParams.params.srcNdStride = 0;
            fixpipe2GmParams.params.dstNdStride = 0;
            Fixpipe<T, T, PFA_CFG_ROW_MAJOR_GM>(bmm3Res[runInfo.keyIndexTensorOffset], mm3L0CBuffer.GetTensor<T>(),
                                                fixpipe2GmParams);
            mm3L0CBuffer.Set<HardEvent::FIX_M>();
            SetAtomicNone();
        } else {
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2GmParams;
            fixpipe2GmParams.nSize = mmParam.singleN;
            fixpipe2GmParams.mSize = mmParam.singleM;
            fixpipe2GmParams.srcStride = AlignTo(fixpipe2GmParams.mSize, static_cast<uint16_t>(C0_SIZE));
            fixpipe2GmParams.dstStride = AlignTo(constInfo.dSizeQueryIndex, static_cast<uint32_t>(C0_SIZE));
            fixpipe2GmParams.dualDstCtl = 0;
            fixpipe2GmParams.params.ndNum = 1;
            fixpipe2GmParams.params.srcNdStride = 0;
            fixpipe2GmParams.params.dstNdStride = 0;
            Fixpipe<T, T, PFA_CFG_ROW_MAJOR_GM>(deterRes[runInfo.deterResOffset], mm3L0CBuffer.GetTensor<T>(),
                                                fixpipe2GmParams);
            mm3L0CBuffer.Set<HardEvent::FIX_M>();
        }
    }
    if (runInfo.isLastBlock) {
        sYQL1Buffer[runInfo.sTaskIdMod3].Set<HardEvent::MTE1_MTE2>();
    }
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockCube<TEMPLATE_ARGS>::ComputeMmDq(
    Buffer<BufferType::L1, SyncType::NO_SYNC> &reluGradResL1Buf, DLIKGradRunInfo &runInfo, DLIKGradConstInfo &constInfo,
    bool isFixOut)
{
    uint32_t kSize = runInfo.kProcessSize;
    uint32_t kLoopTimes = CeilDiv(kSize, VEC_P_BASESIZE);
    uint32_t tailLoopKSize = kSize - (kLoopTimes - 1) * VEC_P_BASESIZE;
    uint32_t dSize = constInfo.dSizeQueryIndex;

    LocalTensor<INPUT_T> reluGradL1Tensor;
    LocalTensor<INPUT_T> syKL1Tensor = sYKL1Buffer[runInfo.taskIdMod3].GetTensor<INPUT_T>();
    MMParam mmParam = {constInfo.gSizeQueryIndex, dSize, K_SPLIT_SIZE, false, false, true, true, UNITFLAG_DISABLE};
    Buffer<BufferType::L0C> mm4L0CBuffer = mm4L0CSpecialBuf.Get();
    for (uint32_t kIdx = 0; kIdx < kLoopTimes; kIdx++) {
        if (kIdx == kLoopTimes - 1) {
            mmParam.singleK = tailLoopKSize;
        }
        reluGradL1Tensor = reluGradResL1Buf.GetTensor<INPUT_T>();
        int64_t reluGradOffset =
            kIdx * VEC_P_BASESIZE * AlignTo(constInfo.gSizeQueryIndex, static_cast<uint32_t>(C0_SIZE));
        int64_t gatherWorkspaceOffset = (runInfo.s2Offset + kIdx * VEC_P_BASESIZE) * dSize;

        if (runInfo.s2Offset == 0) {
            mm4L0CBuffer.Wait<HardEvent::FIX_M>();
            mmParam.isOutKFisrt = true;
        } else {
            mmParam.isOutKFisrt = false;
        }
        MatmulBase<INPUT_T, INPUT_T, T, QUERY_INDEX_SIZE, N_SPLIT_SIZE, K_SPLIT_SIZE, ABLayout::MK, ABLayout::KN>(
            reluGradL1Tensor[reluGradOffset], syKL1Tensor, l0aBuf, l0bBuf, mm4L0CBuffer.GetTensor<T>(), mmParam);
    }
    mm4L0CBuffer.Set<HardEvent::M_FIX>();
    mm4L0CBuffer.Wait<HardEvent::M_FIX>();
    sYKL1Buffer[runInfo.taskIdMod3].Set<HardEvent::MTE1_MTE2>();
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V2_TO_C3_FLAG[runInfo.taskIdMod3]);
    if (isFixOut) {
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
class DlikgBlockCubeDummy {
public:
    __aicore__ inline DlikgBlockCubeDummy(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *tPipe, BufferManager<BufferType::L1> *l1BuffMgr) {};
    __aicore__ inline void InitCubeBuffers() {};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR dQuery, GM_ADDR key, GM_ADDR dKey,
                                            GlobalTensor<T> bmm3Res, GlobalTensor<T> deterRes) {};
    __aicore__ inline void ComputeMmSy(Buffer<BufferType::UB, SyncType::NO_SYNC> &bmm2ResBuf, DLIKGradRunInfo &runInfo,
                                       DLIKGradConstInfo &constInfo) {};
    __aicore__ inline void ComputeMmDk(Buffer<BufferType::L1, SyncType::NO_SYNC> &reluGradResL1Buf,
                                       DLIKGradRunInfo &runInfo, DLIKGradConstInfo &constInfo) {};
    __aicore__ inline void ComputeMmDq(Buffer<BufferType::L1, SyncType::NO_SYNC> &reluGradResL1Buf,
                                       DLIKGradRunInfo &runInfo, DLIKGradConstInfo &constInfo, bool isFixOut) {};
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

DEFINE_CUBE_BLOCK_TRAITS(DlikgBlockCube);
DEFINE_CUBE_BLOCK_TRAITS(DlikgBlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
} // namespace Dlikg
#endif
