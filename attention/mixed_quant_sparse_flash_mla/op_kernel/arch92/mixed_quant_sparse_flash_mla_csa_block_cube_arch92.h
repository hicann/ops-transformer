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
 * \file mixed_quant_sparse_flash_mla_csa_block_cube_arch92.h
 * \brief
 */
#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_ARCH92_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_ARCH92_H
#include "../../common/op_kernel/offset_calculator.h"
#include "../../common/op_kernel/matmul.h"
#include "../../common/op_kernel/FixpipeOut.h"
#include "../../common/op_kernel/CopyInL1.h"
#include "kernel_operator_list_tensor_intf.h"

#include "util_regbase_arch92.h"
#include "mixed_quant_sparse_flash_mla_common_arch92.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace BaseApi {
struct CubeCoordInfo {
    uint32_t curBIdx;
    uint32_t s1Coord;
    uint32_t s2Coord;
};

template <MQSMLA_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == MQSMLA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

TEMPLATES_DEF
class MqsmlaCsaBlockCube {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 512;
    static constexpr uint32_t dBaseMatmulSize = 128;
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 2;
    static constexpr uint32_t crossCoreMte2SyncFlagId = 10;
    static constexpr uint32_t NUM_0 = 0;
    static constexpr uint32_t NUM_1 = 1;

    __aicore__ inline MqsmlaCsaBlockCube(){};
    __aicore__ inline void InitCubeBlock(TPipe* pipe, BufferManager<BufferType::L1>& l1BufferManager,
                                         __gm__ uint8_t* query);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* sequsedQ,
                                         const ConstInfo& constInfo);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& output,
                                       Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& inputRightBuf,
                                       Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm,
                                       RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void FreeEvent();
    __aicore__ inline void IterateBmm2(
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf1,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& inputRightBuf, RunInfo& runInfo,
        ConstInfo& constInfo);

private:
    __aicore__ inline void InitLocalBuffer(BufferManager<BufferType::L1>& l1BufferManager);
    __aicore__ inline void InitGmTensor(__gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* sequsedQ,
                                        const ConstInfo& constInfo);
    __aicore__ inline void CalcS1Coord(RunInfo& runInfo, ConstInfo& constInfo);

    __aicore__ inline void IterateBmm1CSA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf,
                                          Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& inputRightBuf,
                                          Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm,
                                          RunInfo& runInfo, ConstInfo& constInfo);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2CSA(
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf1,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& inputRightBuf, RunInfo& runInfo,
        ConstInfo& constInfo);
    TPipe* tPipe;
    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr bool Q_WITH_ZERO_HEAD = (LAYOUT_T == MQSMLA_LAYOUT::TND);
    FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_WITH_ZERO_HEAD> queryGm;

    /* =====================运行时变量==================== */
    CubeCoordInfo coordInfo[3];
    TEventID mte1ToMte2Id[3];
    TEventID mte2ToMte1Id[3];
    /* =====================LocalBuffer变量==================== */
    BufferManager<BufferType::L1>* l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    // D小于等于256 mm1左矩阵Q，GS1循环内左矩阵复用,
    // GS1循环间开pingpong；D大于256使用单块Buffer，S1循环间驻留；fp32场景单块不驻留
    BuffersPolicySingleBuffer<BufferType::L1> l1QBuffers;

    // L0A
    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers;
    // L0B
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers;
    // L0C
    BuffersPolicyDB<BufferType::L0C> mmL0CBuffers;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockCube<TEMPLATE_ARGS>::InitCubeBlock(TPipe* pipe,
                                                                        BufferManager<BufferType::L1>& l1BufferManager,
                                                                        __gm__ uint8_t* query)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T*)query);
        InitLocalBuffer(l1BufferManager);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockCube<TEMPLATE_ARGS>::InitCubeInput(__gm__ uint8_t* cuSeqlensQ,
                                                                        __gm__ uint8_t* sequsedQ,
                                                                        const ConstInfo& constInfo)
{
    if ASCEND_IS_AIC {
        InitGmTensor(cuSeqlensQ, sequsedQ, constInfo);
        if constexpr (IS_SPLIT_G) {
            mte1ToMte2Id[NUM_0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
            mte1ToMte2Id[NUM_1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
            mte1ToMte2Id[NUM_2] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
            mte2ToMte1Id[NUM_0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            mte2ToMte1Id[NUM_1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            mte2ToMte1Id[NUM_2] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            SetFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[NUM_0]);
            SetFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[NUM_1]);
            SetFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[NUM_2]);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockCube<TEMPLATE_ARGS>::FreeEvent()
{
    mmL0ABuffers.Uninit(l0aBufferManager);
    mmL0BBuffers.Uninit(l0bBufferManager);
    mmL0CBuffers.Uninit(l0cBufferManager);
    if constexpr (IS_SPLIT_G) {
        WaitFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[NUM_0]);
        WaitFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[NUM_1]);
        WaitFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[NUM_2]);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockCube<TEMPLATE_ARGS>::InitLocalBuffer(
    BufferManager<BufferType::L1>& l1BufferManager)
{
    constexpr uint32_t mm1LeftSize = s1BaseSize * dBaseSize * sizeof(Q_T);
    l1QBuffers.Init((l1BufferManager), mm1LeftSize);

    // L0A B C 当前写死，能否通过基础api获取
    l0aBufferManager.Init(tPipe, L0AB_SHARED_SIZE_64K);
    l0bBufferManager.Init(tPipe, L0AB_SHARED_SIZE_64K);
    l0cBufferManager.Init(tPipe, L0C_SHARED_SIZE_256K);

    mmL0ABuffers.Init(l0aBufferManager, BUFFER_SIZE_16K); // db类型，填入数值是总大小的一半
    mmL0BBuffers.Init(l0bBufferManager, BUFFER_SIZE_32K);
    mmL0CBuffers.Init(l0cBufferManager, BUFFER_SIZE_128K);
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockCube<TEMPLATE_ARGS>::InitGmTensor(__gm__ uint8_t* cuSeqlensQ,
                                                                       __gm__ uint8_t* sequsedQ,
                                                                       const ConstInfo& constInfo)
{
    if constexpr (LAYOUT_T == MQSMLA_LAYOUT::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                            constInfo.dSize);
    } else { // MQSMLA_LAYOUT::TND
        GlobalTensor<int32_t> cuSeqlensQGm;
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensQ);
        GlobalTensor<int32_t> sequsedQGm;
        if (sequsedQ != nullptr) {
            sequsedQGm.SetGlobalBuffer((__gm__ int32_t*)sequsedQ);
        }
        uint32_t sequsedQSize = (sequsedQ == nullptr) ? 0 : constInfo.bSize;
        ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true> parser;
        parser.Init(cuSeqlensQGm, sequsedQGm, constInfo.bSize, sequsedQSize);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize, parser);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockCube<TEMPLATE_ARGS>::CalcS1Coord(RunInfo& runInfo, ConstInfo& constInfo)
{
    // 计算s1方向偏移
    coordInfo[runInfo.taskIdMod3].s1Coord = runInfo.s1oIdx * runInfo.qSNumInOneBlock;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockCube<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm, RunInfo& runInfo, ConstInfo& constInfo)
{
    CalcS1Coord(runInfo, constInfo);
    IterateBmm1CSA(outputBuf, inputRightBuf, v0ResGm, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockCube<TEMPLATE_ARGS>::IterateBmm2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf1,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& inputRightBuf, RunInfo& runInfo, ConstInfo& constInfo)
{
    IterateBmm2CSA(outputBuf, outputBuf1, inputLeftBuffers, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockCube<TEMPLATE_ARGS>::IterateBmm1CSA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm, RunInfo& runInfo, ConstInfo& constInfo)
{
    Buffer<BufferType::L1> inputLeftBuf;
    // 左矩阵复用，S2的第一次循环加载左矩阵
    // 加载左矩阵到L1, 全载
    // query对ori_kv, cmp_kv都一样，无需区分
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本块：搬运Q
        inputLeftBuf = l1QBuffers.Get();
        inputLeftBuf.Wait<HardEvent::MTE1_MTE2>(); // 占用L1A
        LocalTensor<Q_T> inputLeftTensor = inputLeftBuf.GetTensor<Q_T>();

        uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
                                                                     coordInfo[runInfo.taskIdMod3].s1Coord, 0);
        CopyToL1Nd2Nz<Q_T>(inputLeftTensor, this->queryGm.gmTensor[gmOffset], runInfo.mRealSize, constInfo.dSize,
                           constInfo.mm1Ka);

        inputLeftBuf.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else {                                      // 非S2的第一次循环直接复用Q
        inputLeftBuf = l1QBuffers.GetPre();
        // 左矩阵复用时，sinner循环内不需要MTE2同步等待
        inputLeftBuf.Set<HardEvent::MTE2_MTE1>(); // 通知
    }

    if constexpr (IS_SPLIT_G) {
        WaitFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[runInfo.taskIdMod3]);
        LocalTensor<Q_T> dst = inputRightBuf.GetTensor<Q_T>();
        v0ResGm.WaitCrossCore(); // N1 > 64 时出核，v0将kv直接搬运至GM
        CrossCoreSetFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
        CrossCoreWaitFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
        GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
        DataCopy(dst, v0ResGmTensor, Align16Func(runInfo.s2RealSize) * constInfo.dSize);
        SetFlag<HardEvent::MTE2_MTE1>(mte1ToMte2Id[runInfo.taskIdMod3]);
        WaitFlag<HardEvent::MTE2_MTE1>(mte1ToMte2Id[runInfo.taskIdMod3]);
    } else {
        // N1 < 64时不出核，v0将kv直接搬运至L1
        inputRightBuf.WaitCrossCore();
    }

    inputLeftBuf.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {
        static_cast<uint32_t>(runInfo.mRealSize),  // singleM
        static_cast<uint32_t>(runInfo.s2RealSize), // singleN
        static_cast<uint32_t>(constInfo.dSize),    // singleK
        0,                                         // isLeftTranspose
        1                                          // isRightTranspose
    };
    MatmulK<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>( // m,n不切，k切128
        inputLeftBuf.GetTensor<Q_T>(), inputRightBuf.GetTensor<Q_T>(), // mm1B直接用tensor的数据
        mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(), param);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        inputLeftBuf.Set<HardEvent::MTE1_MTE2>(); // 释放L1A
    }
    mm1ResL0C.Set<HardEvent::M_FIX>();  // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C
    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    fixpipeParams.nSize = Align8Func(
        runInfo.s2RealSize); // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize); // 有效数据不足16行，只需要输出部分行即可;
                                                         // L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.srcStride = Align16Func(
        fixpipeParams.mSize); // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔),
                              // 单位为16*sizeof(T) // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到,
                                          // ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(),
                                        fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm1ResL0C.Set<HardEvent::FIX_M>();                  // 释放L0C
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockCube<TEMPLATE_ARGS>::IterateBmm2CSA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& outputBuf1,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& inputRightBuf, RunInfo& runInfo, ConstInfo& constInfo)
{
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> inputLeftBuf = inputLeftBuffers.Get(); // P直接用无需搬运
    inputLeftBuf.WaitCrossCore();
    Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
    mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {
        static_cast<uint32_t>(runInfo.mRealSize),  // singleM 64
        static_cast<uint32_t>(constInfo.dSizeV),   // singleN 512
        static_cast<uint32_t>(runInfo.s2RealSize), // singleK 128
        0,                                         // isLeftTranspose
        0                                          // isRightTranspose
    };
    MatmulN<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        inputLeftBuf.GetTensor<Q_T>(), inputRightBuf.GetTensor<Q_T>(), mmL0ABuffers, mmL0BBuffers,
        mm2ResL0C.GetTensor<T>(), param);
    if constexpr (IS_SPLIT_G) {
        SetFlag<HardEvent::MTE1_MTE2>(mte2ToMte1Id[runInfo.taskIdMod3]);
    }
    mm2ResL0C.Set<HardEvent::M_FIX>();                     // 通知
    mm2ResL0C.Wait<HardEvent::M_FIX>();                    // 等待
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    fixpipeParams.nSize =
        Align8Func(constInfo.dSizeV /
                   NUM_4); // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize); // 有效数据不足16行，只需要输出部分行即可;
                                                         // L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.srcStride = Align16Func(
        fixpipeParams.mSize); // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔),
                              // 单位为16*sizeof(T) // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.dstStride = Align16Func(
        constInfo.dSizeV /
        NUM_4); // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到, ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    outputBuf.WaitCrossCore();
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(),
                                        mm2ResL0C.GetTensor<T>(s1BaseSize * constInfo.dSizeV / NUM_4 * 0),
                                        fixpipeParams); // 将matmul结果从L0C搬运到UB
    outputBuf.SetCrossCore();
    outputBuf1.WaitCrossCore();
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf1.template GetTensor<T>(),
                                        mm2ResL0C.GetTensor<T>(s1BaseSize * constInfo.dSizeV / NUM_4 * NUM_1),
                                        fixpipeParams); // 将matmul结果从L0C搬运到UB
    outputBuf1.SetCrossCore();
    outputBuf.WaitCrossCore();
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(),
                                        mm2ResL0C.GetTensor<T>(s1BaseSize * constInfo.dSizeV / NUM_4 * NUM_2),
                                        fixpipeParams); // 将matmul结果从L0C搬运到UB
    outputBuf.SetCrossCore();
    outputBuf1.WaitCrossCore();
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf1.template GetTensor<T>(),
                                        mm2ResL0C.GetTensor<T>(s1BaseSize * constInfo.dSizeV / NUM_4 * NUM_3),
                                        fixpipeParams); // 将matmul结果从L0C搬运到UB
    outputBuf1.SetCrossCore();
    mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放
}

TEMPLATES_DEF
class MqsmlaCsaBlockCubeDummy {
public:
    __aicore__ inline MqsmlaCsaBlockCubeDummy(){};
    __aicore__ inline void InitCubeBlock(TPipe* pipe, BufferManager<BufferType::L1>& l1BufferManager,
                                         __gm__ uint8_t* query)
    {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* sequsedQ,
                                         const ConstInfo& constInfo)
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

DEFINE_CUBE_BLOCK_TRAITS(MqsmlaCsaBlockCube);
DEFINE_CUBE_BLOCK_TRAITS(MqsmlaCsaBlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
} // namespace BaseApi
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H
