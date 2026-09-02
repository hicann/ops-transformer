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
 * \file sparse_flash_mla_grad_block_cube.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_GRAD_BLOCK_CUBE_H
#define SPARSE_FLASH_MLA_GRAD_BLOCK_CUBE_H

#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"

namespace SfagBaseApi {

TEMPLATES_DEF
class FAGBlockCube {
public:
    constexpr static bool IS_FP8_INPUT =
        IsSameType<INPUT_TYPE, fp8_e5m2_t>::value || IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value;
    constexpr static bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    constexpr static uint32_t CUBE_BASEM = 128;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t CUBE_BASEK = 128;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t C0_SIZE = 16;
    constexpr static uint32_t l1BaseD = 512;
    constexpr static uint32_t L0_SINGLE_BUFFER_SIZE = 32 * 1024;
    constexpr static bool IS_L1_REUSE = false;
    constexpr static bool IS_L1_PRELOAD = false;
    constexpr static SyncType SYNC_TYPE = IS_L1_PRELOAD ? SyncType::NO_SYNC : SyncType::INNER_CORE_SYNC;
    constexpr static bool IS_DKV_RESIDENT_L0C = false;
    constexpr static bool IS_FP32_D_EXCEED_256 = IS_FP32_INPUT && HEAD_DIM_ALIGN > 256;

    constexpr static uint32_t DQ_L0_SPLIT_K = 128;
    constexpr static uint32_t DKV_L0_SPLIT_K = 128;

    // input global mmemory
    GlobalTensor<INPUT_TYPE> queryGm, oriKVGm, cmpKVGm, dyGm;

    TPipe *pipe;
    SFagTilingType tilingData;
    // l1 buffer manage
    BufferManager<BufferType::L1> *l1BufferManagerPtr;

    BuffersPolicySingleBuffer<BufferType::L1> qL1Buf;
    BuffersPolicySingleBuffer<BufferType::L1> dYL1Buf;
    BuffersPolicyDB<BufferType::L1> commonL1Buf;

    // l0ab buffer manage, double buffer
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BuffersPolicyDB<BufferType::L0A> l0aBuf;
    BuffersPolicyDB<BufferType::L0B> l0bBuf;

    BufferManager<BufferType::L0C> l0cBufferManager;
    BuffersPolicyDB<BufferType::L0C> commonl0CBuf;

    bool isDkvL0CResidentForD192Dv128 = false;
    MutexID vL1BufMutexId;

    __aicore__ inline FAGBlockCube(){};
    __aicore__ inline ~FAGBlockCube();
    __aicore__ inline void SetCubeBlockParams(TPipe *pipe, SFagTilingType tilingData,
                                              BufferManager<BufferType::L1> *l1BuffMgr);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR d_out,
                                            GM_ADDR workspace);
    __aicore__ inline void InitCubeBuffer(FagConstInfo &constInfo);
    __aicore__ inline void IterateMmDyV(LocalTensor<CALC_TYPE> &mm1ResTensor,
                                        const GlobalTensor<INPUT_TYPE> &selectedVWorkSpaceGm, FagConstInfo &constInfo,
                                        FagRunInfo &runInfo); // mm1
    __aicore__ inline void IterateMmQK(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                       const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo); // mm2
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsK(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm,
                                        BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo); // mm3 dq
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQ(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo); // mm4 dk
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDy(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo); // mm5 dv
    __aicore__ inline void GetCubeVecSharedParams(FagRunInfo &runInfo);

private:
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsKNormal(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm,
                                              BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                              FagConstInfo &constInfo, FagRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQNormal(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                              FagConstInfo &constInfo, FagRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDyNormal(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
                                              FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline GlobalTensor<INPUT_TYPE> GetKVGlobalTensor(FagRunInfo &runInfo,
                                                                 const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline FAGBlockCube<TEMPLATE_ARGS>::~FAGBlockCube()
{
    if constexpr (IS_L1_PRELOAD) {
        ReleaseMutexID(vL1BufMutexId);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::SetCubeBlockParams(TPipe *pipe, SFagTilingType tilingData,
                                                                       BufferManager<BufferType::L1> *l1BuffMgr)
{
    this->pipe = pipe;
    this->tilingData = tilingData;
    this->l1BufferManagerPtr = l1BuffMgr;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR query, GM_ADDR ori_kv, GM_ADDR cmp_kv,
                                                                     GM_ADDR d_out, GM_ADDR workspace)
{
    queryGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)query);
    oriKVGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)ori_kv);
    cmpKVGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)cmp_kv);
    dyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)d_out);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::InitCubeBuffer(FagConstInfo &constInfo)
{
    qL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
    dYL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
    commonL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * CUBE_BASEN * sizeof(INPUT_TYPE));

    // init l0a l0b buffer
    l0aBufferManager.Init(pipe, L0_MAX_SIZE);
    l0bBufferManager.Init(pipe, L0_MAX_SIZE);
    l0aBuf.Init(l0aBufferManager, L0_SINGLE_BUFFER_SIZE);
    l0bBuf.Init(l0bBufferManager, L0_SINGLE_BUFFER_SIZE);

    // init l0c buffer
    l0cBufferManager.Init(pipe, L0C_MAX_SIZE); // 256 * 1024
    commonl0CBuf.Init(l0cBufferManager, L0C_MAX_SIZE / NUM_TWO);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline GlobalTensor<INPUT_TYPE> FAGBlockCube<TEMPLATE_ARGS>::GetKVGlobalTensor(
    FagRunInfo &runInfo, const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm)
{
    if (runInfo.isOriKV) {
        if constexpr (IsOriKVSparse) {
            return selectedKWorkSpaceGm;
        } else {
            return oriKVGm;
        }
    } else {
        if constexpr (IsCmpKVSparse) {
            return selectedKWorkSpaceGm;
        } else {
            return cmpKVGm;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::GetCubeVecSharedParams(FagRunInfo &runInfo) {}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDyV(LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                                 const GlobalTensor<INPUT_TYPE> &selectedVWorkSpaceGm,
                                                                 FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    Buffer<BufferType::L1> dyL1Buffer = dYL1Buf.Get();
    if (runInfo.commonRunInfo.s2RealSize == 0) {
        if (!runInfo.isS1IdxNoChange) {
            dyL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
        }
        if (!runInfo.isNextS1IdxNoChange) {
            dyL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步
        }
        return;
    }

    GlobalTensor<INPUT_TYPE> kvGlobalTensor = this->GetKVGlobalTensor(runInfo, selectedVWorkSpaceGm);
    Nd2NzParams nd2NzParams;
    if (!runInfo.isS1IdxNoChange) {
        dyL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
        LocalTensor<INPUT_TYPE> dyL1Tensor = dyL1Buffer.GetTensor<INPUT_TYPE>();
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = constInfo.commonConstInfo.gSize;
        nd2NzParams.dValue = constInfo.commonConstInfo.dSizeV;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.dstNzC0Stride = AlignTo16(constInfo.commonConstInfo.gSize);
        nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Ka;
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(dyL1Tensor, this->dyGm[runInfo.dyOffset], nd2NzParams);
        dyL1Buffer.Set<HardEvent::MTE2_MTE1>();
        dyL1Buffer.Wait<HardEvent::MTE2_MTE1>();
    }
    Buffer<BufferType::L0C> mm1L0CBuffer = commonl0CBuf.Get();
    // load l1 to l0ab + mmad
    mm1L0CBuffer.Wait<HardEvent::FIX_M>(); // 反向同步
    uint32_t kLoops = Ceil<int64_t>(constInfo.commonConstInfo.dSizeV, CUBE_BASEK);
    uint32_t realK = CUBE_BASEK;
    for (uint32_t k = 0; k < kLoops; ++k) {
        if (k == kLoops - 1) {
            uint32_t tailSize = constInfo.commonConstInfo.dSizeV % CUBE_BASEK;
            realK = tailSize ? tailSize : CUBE_BASEK;
        }
        uint32_t gmNOffset = k * CUBE_BASEK;
        // load right matrix to L1
        bool isCopyRight = true;
        Buffer<BufferType::L1> vL1Buffer = commonL1Buf.Get();
        vL1Buffer.template Wait<HardEvent::MTE1_MTE2>(); // 反向同步
        if (isCopyRight) {
            LocalTensor<INPUT_TYPE> vL1Tensor = vL1Buffer.template GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s2RealSize;
            nd2NzParams.dValue = realK;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Kb;
            nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.commonRunInfo.s2RealSize);
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(vL1Tensor, kvGlobalTensor[runInfo.kSelectedWsAddr + gmNOffset], nd2NzParams);
            vL1Buffer.template Set<HardEvent::MTE2_MTE1>();
            vL1Buffer.template Wait<HardEvent::MTE2_MTE1>();
        }
        MMParam param = {(uint32_t)constInfo.commonConstInfo.gSize,  // singleM
                         (uint32_t)runInfo.commonRunInfo.s2RealSize, // singleN
                         (uint32_t)realK,                            // singleK
                         false,                                      // isLeftTranspose
                         true,                                       // isRightTranspose
                         true,
                         k == 0};
        MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, CUBE_BASEK, ABLayout::MK, ABLayout::KN>(
            dyL1Buffer.GetTensor<INPUT_TYPE>()[AlignTo16(constInfo.commonConstInfo.gSize) * k * CUBE_BASEK],
            vL1Buffer.template GetTensor<INPUT_TYPE>(), l0aBuf, l0bBuf, mm1L0CBuffer.GetTensor<CALC_TYPE>(), param);
        vL1Buffer.template Set<HardEvent::MTE1_MTE2>(); // 反向同步
    }
    mm1L0CBuffer.Set<HardEvent::M_FIX>();
    mm1L0CBuffer.Wait<HardEvent::M_FIX>();
    if (!runInfo.isNextS1IdxNoChange) {
        dyL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步
    }
    // fixp2ub
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = runInfo.commonRunInfo.s2RealSize;
    fixpipeParams.mSize = (constInfo.commonConstInfo.gSize + 1) >> 1 << 1;
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    fixpipeParams.dstStride = CUBE_BASEN;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(mm1ResTensor, mm1L0CBuffer.GetTensor<CALC_TYPE>(),
                                                        fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm1L0CBuffer.Set<HardEvent::FIX_M>();                               // 反向同步
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmQK(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                                const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm,
                                                                FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    Buffer<BufferType::L1> qL1Buffer = qL1Buf.Get();
    if (runInfo.commonRunInfo.s2RealSize == 0) {
        if (!runInfo.isS1IdxNoChange) {
            qL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
        }
        if (!runInfo.isNextS1IdxNoChange) {
            qL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步
        }
        return;
    }

    GlobalTensor<INPUT_TYPE> kvGlobalTensor = this->GetKVGlobalTensor(runInfo, selectedKWorkSpaceGm);
    Nd2NzParams nd2NzParams;

    // copy current, when IS_L1_PRELOAD=true, only first loop copy current
    nd2NzParams.dstNzC0Stride = AlignTo16(constInfo.commonConstInfo.gSize);
    if (!runInfo.isS1IdxNoChange) {
        qL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
        LocalTensor<INPUT_TYPE> qL1Tensor = qL1Buffer.GetTensor<INPUT_TYPE>();

        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = constInfo.commonConstInfo.gSize;
        nd2NzParams.dValue = constInfo.commonConstInfo.dSize;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = constInfo.mm2Ka;
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(qL1Tensor, this->queryGm[runInfo.commonRunInfo.queryOffset], nd2NzParams);

        qL1Buffer.Set<HardEvent::MTE2_MTE1>();
        qL1Buffer.Wait<HardEvent::MTE2_MTE1>();
    }
    // load l1 to l0ab + mmad
    Buffer<BufferType::L0C> mm2L0CBuffer = commonl0CBuf.Get();
    mm2L0CBuffer.Wait<HardEvent::FIX_M>(); // 反向同步
    uint32_t kLoops = Ceil<int64_t>(constInfo.dTotalSize, CUBE_BASEK);
    uint32_t realK = CUBE_BASEK;
    for (uint32_t k = 0; k < kLoops; ++k) {
        if (k == kLoops - 1) {
            uint32_t tailSize = constInfo.dTotalSize % CUBE_BASEK;
            realK = tailSize ? tailSize : CUBE_BASEK;
        }
        uint32_t gmNOffset = k * CUBE_BASEK;
        // load right matrix to L1
        bool isCopyRight = true;
        Buffer<BufferType::L1> kL1Buffer = commonL1Buf.Get();
        nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.commonRunInfo.s2RealSize);
        if (isCopyRight) {
            kL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
            LocalTensor<INPUT_TYPE> kL1Tensor = kL1Buffer.GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s2RealSize;
            nd2NzParams.dValue = realK;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.mm2Kb;
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(kL1Tensor, kvGlobalTensor[runInfo.kSelectedWsAddr + gmNOffset], nd2NzParams);
            kL1Buffer.Set<HardEvent::MTE2_MTE1>();
            kL1Buffer.Wait<HardEvent::MTE2_MTE1>();
        }

        MMParam param = {
            (uint32_t)constInfo.commonConstInfo.gSize,  // singleM
            (uint32_t)runInfo.commonRunInfo.s2RealSize, // singleN
            (uint32_t)realK,                            // singleK
            false,                                      // isLeftTranspose
            true,                                       // isRightTranspose
            true,
            k == 0 // isOutKFisrt
        };
        MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN,
                   L0_SINGLE_BUFFER_SIZE / CUBE_BASEN / sizeof(INPUT_TYPE), ABLayout::MK, ABLayout::KN>(
            qL1Buffer.GetTensor<INPUT_TYPE>()[AlignTo16(constInfo.commonConstInfo.gSize) * k * CUBE_BASEK],
            kL1Buffer.GetTensor<INPUT_TYPE>(), l0aBuf, l0bBuf, mm2L0CBuffer.GetTensor<CALC_TYPE>(), param);

        kL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步
    }
    if (!runInfo.isNextS1IdxNoChange) {
        qL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步
    }
    mm2L0CBuffer.Set<HardEvent::M_FIX>();
    mm2L0CBuffer.Wait<HardEvent::M_FIX>();

    // fixp2ub
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.nSize = runInfo.commonRunInfo.s2RealSize;
    // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.mSize = (constInfo.commonConstInfo.gSize + 1) >> 1 << 1;
    // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T)
    // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
    // mmResUb上两行之间的间隔，单位：element。
    fixpipeParams.dstStride = CUBE_BASEN;
    // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(mm2ResTensor, mm2L0CBuffer.GetTensor<CALC_TYPE>(),
                                                        fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm2L0CBuffer.Set<HardEvent::FIX_M>();
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsKNormal(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm,
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (runInfo.commonRunInfo.s2RealSize == 0) {
        return;
    }
    Buffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = dSL1Buf.Get();
    GlobalTensor<INPUT_TYPE> kvGlobalTensor = this->GetKVGlobalTensor(runInfo, selectedKWorkSpaceGm);
    constexpr uint32_t baseN = CUBE_BASEN;
    uint32_t realN = baseN;
    uint32_t nLoops = ((uint32_t)constInfo.dTotalSize + baseN - 1) / baseN; // 尾块处理
    for (uint32_t n = 0; n < nLoops; ++n) {
        if (n == nLoops - 1) {
            uint32_t tailSize = (uint32_t)constInfo.dTotalSize % baseN;
            realN = tailSize ? tailSize : baseN;
        }
        Buffer<BufferType::L1> kL1Buffer = commonL1Buf.Get();
        LocalTensor<INPUT_TYPE> kL1Tensor;
        uint64_t gmNOffset = n * baseN;
        Nd2NzParams nd2NzParams;
        // load right matrix to L1
        kL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
        kL1Tensor = kL1Buffer.GetTensor<INPUT_TYPE>();
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = runInfo.commonRunInfo.s2RealSize;
        nd2NzParams.dValue = realN;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = constInfo.mm2Kb;
        nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.commonRunInfo.s2RealSize);
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(kL1Tensor, kvGlobalTensor[runInfo.kSelectedWsAddr + gmNOffset], nd2NzParams);
        kL1Buffer.Set<HardEvent::MTE2_MTE1>();
        kL1Buffer.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm3L0CBuffer = commonl0CBuf.Get();
        // load l1 to l0ab + mmad
        mm3L0CBuffer.Wait<HardEvent::FIX_M>();                       // 反向同步
        MMParam param = {(uint32_t)constInfo.commonConstInfo.gSize,  // singleM
                         (uint32_t)realN,                            // singleN
                         (uint32_t)runInfo.commonRunInfo.s2RealSize, // singleK
                         false,                                      // isLeftTranspose
                         false,                                      // isRightTranspose
                         true,
                         true};
        MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DQ_L0_SPLIT_K, ABLayout::MK,
                   ABLayout::KN>(dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor, l0aBuf, l0bBuf,
                                 mm3L0CBuffer.GetTensor<CALC_TYPE>(), param);

        kL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步
        mm3L0CBuffer.Set<HardEvent::M_FIX>();
        mm3L0CBuffer.Wait<HardEvent::M_FIX>();
        // fixp2GM
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = realN;
        // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
        fixpipeParams.mSize = constInfo.commonConstInfo.gSize;
        // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T)
        // 源Nz矩阵中相邻大Z排布的起始地址偏移
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        // mmResUb上两行之间的间隔，单位：element。
        fixpipeParams.dstStride = constInfo.dTotalSize;
        // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        constexpr static FixpipeConfig DQ_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
        SetAtomicAdd<CALC_TYPE>();
        int64_t kk = runInfo.commonRunInfo.queryOffset + gmNOffset;
        Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(outTensor[runInfo.commonRunInfo.queryOffset + gmNOffset],
                                                 mm3L0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
        SetAtomicNone();
        mm3L0CBuffer.Set<HardEvent::FIX_M>();
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsQNormal(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (runInfo.commonRunInfo.s2RealSize == 0) {
        return;
    }
    event_t eventIDFIXToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_M));
    event_t eventIDMToFIX = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_FIX));
    SetFlag<HardEvent::FIX_M>(eventIDFIXToM);
    Buffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = dSL1Buf.Get();
    constexpr uint32_t baseN = CUBE_BASEN;
    uint32_t nLoops = ((uint32_t)constInfo.dTotalSize + baseN - 1) / baseN; // 尾块处理
    uint32_t realN = baseN;
    bool leftReuse = runInfo.isNextS1IdxNoChange && !runInfo.isSecondLast;
    for (uint32_t n = 0; n < nLoops; ++n) {
        if (n == nLoops - 1) {
            uint32_t tailSize = (uint32_t)constInfo.dTotalSize % baseN;
            realN = tailSize ? tailSize : baseN;
        }
        Buffer<BufferType::L1> qL1Buffer = leftReuse ? qL1Buf.Get() : commonL1Buf.Get();
        uint64_t gmNOffset = n * baseN;
        LocalTensor<INPUT_TYPE> qL1Tensor;
        Nd2NzParams nd2NzParams;
        nd2NzParams.dstNzC0Stride = AlignTo16(constInfo.commonConstInfo.gSize);
        // load right matrix to L1
        int64_t queryL1Offset = 0;
        if (leftReuse) {
            qL1Tensor = qL1Buffer.GetTensor<INPUT_TYPE>();
            queryL1Offset = nd2NzParams.dstNzC0Stride * n * CUBE_BASEN;
        } else {
            qL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
            qL1Tensor = qL1Buffer.GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = constInfo.commonConstInfo.gSize;
            nd2NzParams.dValue = realN;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;

            GlobalTensor<INPUT_TYPE> queryGmTmp;
            nd2NzParams.srcDValue = constInfo.mm2Ka;
            queryGmTmp = this->queryGm[runInfo.commonRunInfo.queryOffset + gmNOffset];
            DataCopy(qL1Tensor, queryGmTmp, nd2NzParams);
            qL1Buffer.Set<HardEvent::MTE2_MTE1>();
            qL1Buffer.Wait<HardEvent::MTE2_MTE1>();
        }

        Buffer<BufferType::L0C> dkL0CBuffer = commonl0CBuf.Get();
        // load l1 to l0ab + mmad
        dkL0CBuffer.Wait<HardEvent::FIX_M>();                        // 反向同步
        MMParam param = {(uint32_t)runInfo.commonRunInfo.s2RealSize, // singleM
                         (uint32_t)realN,                            // singleN
                         (uint32_t)constInfo.commonConstInfo.gSize,  // singleK
                         true,                                       // isLeftTranspose
                         false,                                      // isRightTranspose
                         true,
                         true};
        WaitFlag<HardEvent::FIX_M>(eventIDFIXToM);
        MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                   ABLayout::KN>(dSL1Buffer.GetTensor<INPUT_TYPE>(), qL1Tensor[queryL1Offset], l0aBuf, l0bBuf,
                                 dkL0CBuffer.GetTensor<CALC_TYPE>(), param);

        if (!leftReuse) {
            qL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步
        }
        SetFlag<HardEvent::M_FIX>(eventIDMToFIX);
        WaitFlag<HardEvent::M_FIX>(eventIDMToFIX);
        dkL0CBuffer.Set<HardEvent::M_FIX>();
        dkL0CBuffer.Wait<HardEvent::M_FIX>();

        // fixp2gm
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = realN;
        fixpipeParams.mSize = runInfo.commonRunInfo.s2RealSize;
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        fixpipeParams.dstStride = constInfo.dTotalSize;
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        constexpr static FixpipeConfig DK_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
        Fixpipe<T, CALC_TYPE, DK_FIXPIPE_CONFIG>(outTensor[runInfo.mm4ResWsAddr + n * CUBE_BASEN],
                                                 dkL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);

        dkL0CBuffer.Set<HardEvent::FIX_M>();
        SetFlag<HardEvent::FIX_M>(eventIDFIXToM);
    }
    WaitFlag<HardEvent::FIX_M>(eventIDFIXToM);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmPDyNormal(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buf, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (runInfo.commonRunInfo.s2RealSize == 0) {
        return;
    }
    Buffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = pL1Buf.Get();
    constexpr uint32_t baseN = CUBE_BASEN;
    uint32_t nLoops = ((uint32_t)constInfo.commonConstInfo.dSizeV + baseN - 1) / baseN; // 尾块处理
    uint32_t realN = baseN;
    bool leftReuse = runInfo.isNextS1IdxNoChange && !runInfo.isSecondLast;
    for (uint32_t n = 0; n < nLoops; ++n) {
        if (n == nLoops - 1) {
            uint32_t tailSize = (uint32_t)constInfo.commonConstInfo.dSizeV % baseN;
            realN = tailSize ? tailSize : baseN;
        }
        Buffer<BufferType::L1> dYL1Buffer = leftReuse ? dYL1Buf.Get() : commonL1Buf.Get();
        LocalTensor<INPUT_TYPE> dYL1Tensor;
        uint64_t gmNOffset = n * baseN;
        Nd2NzParams nd2NzParams;
        // load right matrix to L1
        nd2NzParams.dstNzC0Stride = AlignTo16(constInfo.commonConstInfo.gSize);
        int64_t dyL1Offset = 0;
        if (leftReuse) {
            dYL1Tensor = dYL1Buffer.GetTensor<INPUT_TYPE>();
            dyL1Offset = nd2NzParams.dstNzC0Stride * n * CUBE_BASEN;
        } else {
            dYL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
            dYL1Tensor = dYL1Buffer.GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = constInfo.commonConstInfo.gSize;
            nd2NzParams.dValue = realN;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Ka;
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(dYL1Tensor, this->dyGm[runInfo.dyOffset + gmNOffset], nd2NzParams);
            dYL1Buffer.Set<HardEvent::MTE2_MTE1>();
            dYL1Buffer.Wait<HardEvent::MTE2_MTE1>();
        }
        Buffer<BufferType::L0C> dvL0CBuffer = commonl0CBuf.Get();
        // load l1 to l0ab + mmad
        dvL0CBuffer.Wait<HardEvent::FIX_M>();                        // 反向同步
        MMParam param = {(uint32_t)runInfo.commonRunInfo.s2RealSize, // singleM
                         realN,                                      // singleN
                         (uint32_t)constInfo.commonConstInfo.gSize,  // singleK
                         true,                                       // isLeftTranspose
                         false,                                      // isRightTranspose
                         true,
                         true};
        MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                   ABLayout::KN>(pL1Buffer.GetTensor<INPUT_TYPE>(), dYL1Tensor[dyL1Offset], l0aBuf, l0bBuf,
                                 dvL0CBuffer.GetTensor<CALC_TYPE>(), param);

        if (!leftReuse) {
            dYL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步
        }

        dvL0CBuffer.Set<HardEvent::M_FIX>();
        dvL0CBuffer.Wait<HardEvent::M_FIX>();
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.mSize = runInfo.commonRunInfo.s2RealSize;
        fixpipeParams.nSize = realN;
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        fixpipeParams.dstStride = constInfo.commonConstInfo.dSizeV;
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        constexpr static FixpipeConfig DV_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
        Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[runInfo.mm5ResWsAddr + n * CUBE_BASEN],
                                                 dvL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
        dvL0CBuffer.Set<HardEvent::FIX_M>();
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsK(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm,
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    IterateMmDsKNormal<T, IS_WRITE_UB>(outTensor, selectedKWorkSpaceGm, dSL1Buf, constInfo, runInfo);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsQ(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    IterateMmDsQNormal<T, IS_WRITE_UB>(outTensor, dSL1Buf, constInfo, runInfo);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmPDy(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buf, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    IterateMmPDyNormal<T, IS_WRITE_UB>(outTensor, pL1Buf, constInfo, runInfo);
}

TEMPLATES_DEF
class FAGBlockCubeDummy {
public:
    __aicore__ inline FAGBlockCubeDummy(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *pipe, SFagTilingType tilingData,
                                              BufferManager<BufferType::L1> *l1BuffMgr) {};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR d_out,
                                            GM_ADDR workspace) {};
    __aicore__ inline void InitCubeBuffer(FagConstInfo &constInfo) {};
    __aicore__ inline void IterateMmDyV(LocalTensor<CALC_TYPE> &mm1ResTensor,
                                        const GlobalTensor<INPUT_TYPE> &selectedVWorkSpaceGm, FagConstInfo &constInfo,
                                        FagRunInfo &runInfo) {};
    __aicore__ inline void IterateMmQK(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                       const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo) {};
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsK(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm,
                                        BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo) {}; // dq
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQ(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo) {}; // dk
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDy(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo) {}; // dv
    __aicore__ inline void GetCubeVecSharedParams(FagRunInfo &runInfo) {};
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

DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCube);
DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)

} // namespace SfagBaseApi
#endif
