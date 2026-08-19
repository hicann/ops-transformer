/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file quant_sparse_flash_mla_csa_block_cube.h
 * \brief
 */
#ifndef QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H
#define QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H

#include "kernel_operator_list_tensor_intf.h"
#include "util_regbase.h"
#include "quant_sparse_flash_mla_common_arch35.h"

#if __has_include("../../common/op_kernel/offset_calculator.h")
#include "../../common/op_kernel/offset_calculator.h"
#else
#include "../common/offset_calculator.h"
#endif
#if __has_include("../../common/op_kernel/matmul.h")
#include "../../common/op_kernel/matmul.h"
#else
#include "../common/matmul.h"
#endif
#if __has_include("../../common/op_kernel/FixpipeOut.h")
#include "../../common/op_kernel/FixpipeOut.h"
#else
#include "../common/FixpipeOut.h"
#endif
#if __has_include("../../common/op_kernel/CopyInL1.h")
#include "../../common/op_kernel/CopyInL1.h"
#else
#include "../common/CopyInL1.h"
#endif

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace BaseApi {

template <QSMLA_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == QSMLA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

TEMPLATES_DEF
class CSABlockCube {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 512;
    static constexpr uint32_t dBaseMatmulSize = 256;

    __aicore__ inline CSABlockCube(){};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> &l1BufferManager,
                                         __gm__ uint8_t *query);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                         const ConstInfo &constInfo);
    __aicore__ inline void IterateLoadQK(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
                                         Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                         RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
                                       Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
                                       Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                       bool notLastTwoLoop, RunInfo &runInfoNext, RunInfo &runInfo,
                                       ConstInfo &constInfo);

    __aicore__ inline void IterateBmm2(
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
        ConstInfo &constInfo);
    __aicore__ inline void FreeEvent();

private:
    __aicore__ inline void InitLocalBuffer(BufferManager<BufferType::L1> &l1BufferManager);
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                        const ConstInfo &constInfo);

    __aicore__ inline void CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void IterateBmm1CSA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                          Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
                                          Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                          bool notLastTwoLoop, RunInfo &runInfoNext, RunInfo &runInfo,
                                          ConstInfo &constInfo);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2CSA(
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
        ConstInfo &constInfo);
    TPipe *tPipe;
    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr bool Q_WITH_ZERO_HEAD = (LAYOUT_T == QSMLA_LAYOUT::TND);
    FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_WITH_ZERO_HEAD> queryGm;

    /* =====================运行时变量==================== */
    uint32_t l0CBufId = 0;
    uint32_t l1QBufId = 0;
    uint32_t l1KLoadBufId = 0;
    uint32_t l1KMatmul1BufId = 0;
    uint32_t l1KMatmul2BufId = 0;
    uint32_t l0CFixToMFlagId = 0;     // {0, 1}, 用于L0C
    uint32_t l0CMToFixFlagId = 0;     // {0, 1}, 用于L0C
    uint32_t l1QMte1ToMte2FlagId = 0; // {0, 1, 2}, 用于l1Q
    uint32_t l1QMte2ToMte1FlagId = 0; // {0, 1, 2}, 用于l1Q
    uint32_t l1KMte1ToMte2FlagId = 3; // {3, 4, 5}, 用于l1K
    uint32_t l1KMte2ToMte1FlagId = 3; // {3, 4, 5}, 用于l1K
    /* =====================LocalBuffer变量==================== */

    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;

    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers;
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers;

    TBuf<TPosition::A1> l1QBuffers;
    LocalTensor<Q_T> l1QTensor;

    TBuf<TPosition::CO1> mmL0CBuffers;
    LocalTensor<T> mmL0CTensor;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitCubeBlock(TPipe *pipe,
                                                                  BufferManager<BufferType::L1> &l1BufferManager,
                                                                  __gm__ uint8_t *query)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        InitLocalBuffer(l1BufferManager);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitCubeInput(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                                                  const ConstInfo &constInfo)
{
    if ASCEND_IS_AIC {
        InitGmTensor(cuSeqlensQ, sequsedQ, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitLocalBuffer(BufferManager<BufferType::L1> &l1BufferManager)
{
    tPipe->InitBuffer(l1QBuffers, BUFFER_SIZE_96K);
    l1QTensor = l1QBuffers.Get<Q_T>();

    l0aBufferManager.Init(tPipe, L0AB_SHARED_SIZE_64K);
    l0bBufferManager.Init(tPipe, L0AB_SHARED_SIZE_64K);

    mmL0ABuffers.Init(l0aBufferManager, BUFFER_SIZE_16K); // db类型，填入数值是总大小的一半
    mmL0BBuffers.Init(l0bBufferManager, BUFFER_SIZE_32K);

    tPipe->InitBuffer(mmL0CBuffers, BUFFER_SIZE_256K);
    mmL0CTensor = mmL0CBuffers.Get<T>();

    SetFlag<HardEvent::FIX_M>(l0CFixToMFlagId); // {0, 1}, 用于L0C
    SetFlag<HardEvent::FIX_M>(l0CFixToMFlagId + 1);
    SetFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId); // {0, 1, 2}, 用于l1Q
    SetFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId + 1);
    SetFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId + 2);
    SetFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId); // {3, 4, 5}, 用于l1K
    SetFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + 1);
    SetFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + 2);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::FreeEvent()
{
    mmL0ABuffers.Uninit(l0aBufferManager);
    mmL0BBuffers.Uninit(l0bBufferManager);

    WaitFlag<HardEvent::FIX_M>(l0CFixToMFlagId); // {0, 1}, 用于L0C
    WaitFlag<HardEvent::FIX_M>(l0CFixToMFlagId + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId); // {0, 1, 2}, 用于l1Q
    WaitFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId + 2);
    WaitFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId); // {3, 4, 5}, 用于l1K
    WaitFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + 2);
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitGmTensor(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                                                 const ConstInfo &constInfo)
{
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                            constInfo.dSize);
    } else { // QSMLA_LAYOUT::TND
        uint32_t sequsedQSize = (sequsedQ == nullptr) ? 0 : constInfo.bSize;
        ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true> parser;
        parser.Init(cuSeqlensQ, constInfo.bSize + 1, sequsedQ, sequsedQSize);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize, parser);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm1CSA(outputBuf, inputRightBuf, v0ResGm, notLastTwoLoop, runInfoNext, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm2CSA(outputBuf, inputLeftBuffers, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
                                                                 runInfo.s1oIdx * runInfo.qSNumInOneBlock, 0);
    for (uint32_t i = 0; i < 2; i++) {
        uint32_t curL1QBufId = (l1QBufId + i) % 3;
        WaitFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId + curL1QBufId);
        uint64_t curGmOffset = gmOffset + i * (constInfo.dSize >> 1);
        CopyToL1Nd2Nz<Q_T>(l1QTensor[curL1QBufId * BUFFER_SIZE_16K], this->queryGm.gmTensor[curGmOffset],
                           runInfo.mRealSize, constInfo.dSize >> 1, constInfo.mm1Ka);
        SetFlag<HardEvent::MTE2_MTE1>(l1QMte2ToMte1FlagId + curL1QBufId);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateLoadQK(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, RunInfo &runInfo, ConstInfo &constInfo,
    bool isFirstLoop)
{
    if (unlikely(isFirstLoop)) {
        CopyQGmToL1(runInfo, constInfo);
    }
    // 加载当前轮的右矩阵到L1
    WaitFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + l1KLoadBufId);
    LocalTensor<Q_T> dst = inputRightBuf.GetTensor<Q_T>();
    v0ResGm.WaitCrossCore();
    if constexpr (IS_SPLIT_G) {
        CrossCoreSetFlag<0, PIPE_MTE2>(15);
        CrossCoreWaitFlag<0, PIPE_MTE2>(15);
    }
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(dst, v0ResGmTensor, Align32Func(runInfo.s2RealSize) * constInfo.dSize);
    SetFlag<HardEvent::MTE2_MTE1>(l1KMte2ToMte1FlagId + l1KLoadBufId);
    l1KLoadBufId = (l1KLoadBufId + 1) % 3;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm1CSA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    WaitFlag<HardEvent::MTE2_MTE1>(l1KMte2ToMte1FlagId + l1KMatmul1BufId);
    l1KMatmul1BufId = (l1KMatmul1BufId + 1) % 3;
    WaitFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);

    MMParam param = {
        static_cast<uint32_t>(runInfo.mRealSize),    // singleM
        static_cast<uint32_t>(runInfo.s2RealSize),   // singleN
        static_cast<uint32_t>(constInfo.dSize >> 1), // singleK
        0,                                           // isLeftTranspose
        1                                            // isRightTranspose
    };
    uint32_t curL1QBufId = l1QBufId;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(l1QMte2ToMte1FlagId + curL1QBufId);
    }

    // m,n不切，k切256，mm1B直接用tensor的数据
    MatmulK<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QTensor[curL1QBufId * BUFFER_SIZE_16K], inputRightBuf.GetTensor<Q_T>(), mmL0ABuffers, mmL0BBuffers,
        mmL0CTensor[BUFFER_SIZE_32K * l0CBufId], param);

    curL1QBufId = (curL1QBufId + 1) % 3;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(l1QMte2ToMte1FlagId + curL1QBufId);
    }
    param.singleK = constInfo.dSize - param.singleK;
    param.isOutKFisrt = false;

    // m,n不切，k切256, mm1B直接用tensor的数据
    MatmulK<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QTensor[curL1QBufId * BUFFER_SIZE_16K],
        inputRightBuf.GetTensor<Q_T>()[(constInfo.dSize >> 1) * Align32Func(runInfo.s2RealSize)], mmL0ABuffers,
        mmL0BBuffers, mmL0CTensor[BUFFER_SIZE_32K * l0CBufId], param);

    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        SetFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId + l1QBufId);
        SetFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId + curL1QBufId);
        l1QBufId = (l1QBufId + 2) % 3;
        if (notLastTwoLoop) {
            CopyQGmToL1(runInfoNext, constInfo);
        }
    }

    SetFlag<HardEvent::M_FIX>(l0CMToFixFlagId + l0CBufId);
    WaitFlag<HardEvent::M_FIX>(l0CMToFixFlagId + l0CBufId);

    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize);
    // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T) //
    // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到,
                                          // ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    // 将matmul结果从L0C搬运到UB
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mmL0CTensor[BUFFER_SIZE_32K * l0CBufId],
                                        fixpipeParams); // 将matmul结果从L0C搬运到UB
    SetFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);
    l0CBufId ^= 1;
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm2CSA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo, ConstInfo &constInfo)
{
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffer = inputLeftBuffers.Get(); // P直接用无需搬运
    l1PBuffer.WaitCrossCore();

    WaitFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);
    MMParam param = {
        static_cast<uint32_t>(runInfo.mRealSize),  // singleM
        static_cast<uint32_t>(constInfo.dSizeV),   // singleN 512
        static_cast<uint32_t>(runInfo.s2RealSize), // singleK 128
        0,                                         // isLeftTranspose
        0                                          // isRightTranspose
    };
    MatmulN<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1PBuffer.GetTensor<Q_T>(), inputRightBuf.GetTensor<Q_T>(), mmL0ABuffers, mmL0BBuffers,
        mmL0CTensor[BUFFER_SIZE_32K * l0CBufId], param);

    SetFlag<HardEvent::M_FIX>(l0CMToFixFlagId + l0CBufId);
    WaitFlag<HardEvent::M_FIX>(l0CMToFixFlagId + l0CBufId);
    SetFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + l1KMatmul2BufId);
    l1KMatmul2BufId = (l1KMatmul2BufId + 1) % 3;

    outputBuf.WaitCrossCore();                             // 占用
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
    fixpipeParams.nSize =
        Align8Func(constInfo.dSizeV); // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
    fixpipeParams.mSize = Align2Func(
        runInfo
            .mRealSize); // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
    fixpipeParams.srcStride = Align16Func(
        fixpipeParams.mSize); // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeV);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mmL0CTensor[BUFFER_SIZE_32K * l0CBufId],
                                        fixpipeParams); // 将matmul结果从L0C搬运到UB
    SetFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);
    l0CBufId ^= 1;

    outputBuf.SetCrossCore();
}

TEMPLATES_DEF
class CSABlockCubeDummy {
public:
    __aicore__ inline CSABlockCubeDummy(){};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> &l1BufferManager,
                                         __gm__ uint8_t *query)
    {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                         const ConstInfo &constInfo)
    {}
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
    }

DEFINE_CUBE_BLOCK_TRAITS(CSABlockCube);
DEFINE_CUBE_BLOCK_TRAITS(CSABlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
} // namespace BaseApi
#endif // QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H
