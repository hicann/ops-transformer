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
 * \file sparse_flash_attention_grad_block_cube_nle64.h
 * \brief Cube block specialized for gSize/N <= 64 (KV resident in L1).
 */

#ifndef SPARSE_FLASH_ATTENTION_GRAD_BLOCK_CUBE_NLE64_H
#define SPARSE_FLASH_ATTENTION_GRAD_BLOCK_CUBE_NLE64_H

#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"
#include "sparse_flash_attention_grad_common.h"
#include "sparse_flash_attention_grad_arch35_common.h"

using namespace commondef;

namespace SfagBaseApi {

TEMPLATES_DEF
class FAGBlockCubeNLe64 {
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
    GlobalTensor<INPUT_TYPE> queryGm, keyGm, valueGm, dyGm, queryRopeGm, keyRopeGm;

    // output global mmemory
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    // GlobalTensor<INPUT_TYPE> dqGm, dkGm, dvGm;

    TPipe *pipe;
    SFagTilingType tilingData;
    // vector scaleDs通过ssbuf传递给cube
    FagCVSharedParams sharedParams;
    // l1 buffer manage
    BufferManager<BufferType::L1> *l1BufferManagerPtr;

    // Q/Dy 按 S1 ping-pong 双槽；K 按 taskId%2 双槽。切 S1 时 mm12 写新槽、mm345 读旧槽
    BuffersPolicySingleBuffer<BufferType::L1> qL1Buf[2];
    BuffersPolicySingleBuffer<BufferType::L1> dYL1Buf[2];
    BuffersPolicySingleBuffer<BufferType::L1> kvL1Buf[2];

    // l0ab buffer manage, double buffer
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BuffersPolicyDB<BufferType::L0A> l0aBuf;
    BuffersPolicyDB<BufferType::L0B> l0bBuf;

    BufferManager<BufferType::L0C> l0cBufferManager;
    BuffersPolicyDB<BufferType::L0C> commonl0CBuf;
    // N<=64 + Rope：dq 的 512 nope 常驻；mm1/2/4/5 与 dq-rope/dk 走 work ping-pong
    BuffersPolicySingleBuffer<BufferType::L0C> dqNopeL0CBuf;
    BuffersPolicyDB<BufferType::L0C> workL0CBuf;

    bool isDkvL0CResidentForD192Dv128 = false;
    MutexID vL1BufMutexId;

    __aicore__ inline FAGBlockCubeNLe64(){};
    __aicore__ inline ~FAGBlockCubeNLe64();
    __aicore__ inline void SetCubeBlockParams(TPipe *pipe, SFagTilingType tilingData,
                                              BufferManager<BufferType::L1> *l1BuffMgr);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR queryRope,
                                            GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace);
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
                                        BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo); // mm3 dq
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQ(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo); // mm4 dk
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDy(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo); // mm5 dv
private:
    __aicore__ inline uint32_t GetKvNzC0Stride(FagRunInfo &runInfo);
    __aicore__ inline uint32_t GetQDySlot(FagRunInfo &runInfo);
    __aicore__ inline bool IsDqNopeL0CResident();
    __aicore__ inline Buffer<BufferType::L0C> GetWorkL0CBuffer();
    __aicore__ inline uint32_t GetLoopTileSize(uint32_t idx, uint32_t loops, uint32_t total, uint32_t base);
    __aicore__ inline void CopyKVToL1(const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo,
                                      FagRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsKNopeRope(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                                BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                                FagConstInfo &constInfo, FagRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQNopeRope(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                                BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                                FagConstInfo &constInfo, FagRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsKNormal(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm,
                                              BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                              FagConstInfo &constInfo, FagRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQNormal(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                              FagConstInfo &constInfo, FagRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDyNormal(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
                                              FagConstInfo &constInfo, FagRunInfo &runInfo);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline FAGBlockCubeNLe64<TEMPLATE_ARGS>::~FAGBlockCubeNLe64()
{
    if constexpr (IS_L1_PRELOAD) {
        ReleaseMutexID(vL1BufMutexId);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::SetCubeBlockParams(TPipe *pipe, SFagTilingType tilingData,
                                                                            BufferManager<BufferType::L1> *l1BuffMgr)
{
    this->pipe = pipe;
    this->tilingData = tilingData;
    this->l1BufferManagerPtr = l1BuffMgr;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                                                          GM_ADDR dy, GM_ADDR queryRope,
                                                                          GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk,
                                                                          GM_ADDR dv, GM_ADDR workspace)
{
    queryGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)query);
    keyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)key);
    valueGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)value);
    dyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)dy);
    queryRopeGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)queryRope);
    keyRopeGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)keyRope);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline uint32_t FAGBlockCubeNLe64<TEMPLATE_ARGS>::GetKvNzC0Stride(FagRunInfo &runInfo)
{
    if constexpr (IS_FP8_INPUT) {
        return AlignTo32(runInfo.commonRunInfo.s2RealSize);
    }
    return AlignTo16(runInfo.commonRunInfo.s2RealSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline uint32_t FAGBlockCubeNLe64<TEMPLATE_ARGS>::GetQDySlot(FagRunInfo &runInfo)
{
    return static_cast<uint32_t>(runInfo.qDxPingPongIdx) & 1U;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline bool FAGBlockCubeNLe64<TEMPLATE_ARGS>::IsDqNopeL0CResident()
{
    if constexpr (IS_ROPE) {
        return true;
    }
    return false;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline Buffer<BufferType::L0C> FAGBlockCubeNLe64<TEMPLATE_ARGS>::GetWorkL0CBuffer()
{
    if (IsDqNopeL0CResident()) {
        return workL0CBuf.Get();
    }
    return commonl0CBuf.Get();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::CopyKVToL1(
    const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    Buffer<BufferType::L1> kvL1Buffer = kvL1Buf[runInfo.commonRunInfo.taskIdMod2].Get();
    LocalTensor<INPUT_TYPE> kvL1Tensor = kvL1Buffer.GetTensor<INPUT_TYPE>();
    Nd2NzParams nd2NzParams;
    nd2NzParams.ndNum = 1;
    nd2NzParams.nValue = runInfo.commonRunInfo.s2RealSize;
    nd2NzParams.dValue = constInfo.dTotalSize;
    nd2NzParams.srcNdMatrixStride = 0;
    nd2NzParams.srcDValue = constInfo.mm2Kb;
    nd2NzParams.dstNzC0Stride = GetKvNzC0Stride(runInfo);
    nd2NzParams.dstNzNStride = 1;
    nd2NzParams.dstNzMatrixStride = 0;
    DataCopy(kvL1Tensor, selectedKWorkSpaceGm[runInfo.kSelectedWsAddr], nd2NzParams);
    kvL1Buffer.Set<HardEvent::MTE2_MTE1>();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline uint32_t FAGBlockCubeNLe64<TEMPLATE_ARGS>::GetLoopTileSize(uint32_t idx, uint32_t loops,
                                                                             uint32_t total, uint32_t base)
{
    if (idx + 1U == loops) {
        uint32_t tail = total % base;
        return tail == 0 ? base : tail;
    }
    return base;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::InitCubeBuffer(FagConstInfo &constInfo)
{
    uint32_t gAlign = AlignTo16(constInfo.commonConstInfo.gSize);
    uint32_t qL1Size = gAlign * constInfo.dTotalSize * sizeof(INPUT_TYPE);
    uint32_t dyL1Size = gAlign * constInfo.commonConstInfo.dSizeV * sizeof(INPUT_TYPE);
    qL1Buf[0].Init(*l1BufferManagerPtr, qL1Size);
    qL1Buf[1].Init(*l1BufferManagerPtr, qL1Size);
    dYL1Buf[0].Init(*l1BufferManagerPtr, dyL1Size);
    dYL1Buf[1].Init(*l1BufferManagerPtr, dyL1Size);
    uint32_t kvL1Size = SFAG_GATHER_S2_HEAD_N * constInfo.dTotalSize * sizeof(INPUT_TYPE);
    kvL1Buf[0].Init(*l1BufferManagerPtr, kvL1Size);
    kvL1Buf[1].Init(*l1BufferManagerPtr, kvL1Size);

    l0aBufferManager.Init(pipe, L0_MAX_SIZE);
    l0bBufferManager.Init(pipe, L0_MAX_SIZE);
    l0aBuf.Init(l0aBufferManager, L0_SINGLE_BUFFER_SIZE);
    l0bBuf.Init(l0bBufferManager, L0_SINGLE_BUFFER_SIZE);

    l0cBufferManager.Init(pipe, L0C_MAX_SIZE);
    if (IsDqNopeL0CResident()) {
        uint32_t mAlign = AlignTo16(constInfo.commonConstInfo.gSize);
        uint32_t dqNopeSize = mAlign * constInfo.commonConstInfo.dSize * sizeof(CALC_TYPE);
        dqNopeL0CBuf.Init(l0cBufferManager, dqNopeSize);
        workL0CBuf.Init(l0cBufferManager, (L0C_MAX_SIZE - dqNopeSize) / NUM_TWO);
    } else {
        commonl0CBuf.Init(l0cBufferManager, L0C_MAX_SIZE / NUM_TWO);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::IterateMmDyV(
    LocalTensor<CALC_TYPE> &mm1ResTensor, const GlobalTensor<INPUT_TYPE> &selectedVWorkSpaceGm, FagConstInfo &constInfo,
    FagRunInfo &runInfo)
{
    Buffer<BufferType::L1> dyL1Buffer = dYL1Buf[GetQDySlot(runInfo)].Get();
    Nd2NzParams nd2NzParams;
    if (!runInfo.isS1IdxNoChange) {
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
    }
    uint32_t kLoops = Ceil<int64_t>(constInfo.commonConstInfo.dSizeV, CUBE_BASEK);
    CopyKVToL1(selectedVWorkSpaceGm, constInfo, runInfo);
    kvL1Buf[runInfo.commonRunInfo.taskIdMod2].Get().Wait<HardEvent::MTE2_MTE1>();
    if (!runInfo.isS1IdxNoChange) {
        dyL1Buffer.Wait<HardEvent::MTE2_MTE1>();
    }
    Buffer<BufferType::L0C> mm1L0CBuffer = GetWorkL0CBuffer();
    mm1L0CBuffer.Wait<HardEvent::FIX_M>();
    uint32_t realK = CUBE_BASEK;
    Buffer<BufferType::L1> vL1Buffer = kvL1Buf[runInfo.commonRunInfo.taskIdMod2].Get();
    const uint32_t kvNzC0Stride = GetKvNzC0Stride(runInfo);
    for (uint32_t k = 0; k < kLoops; ++k) {
        realK = GetLoopTileSize(k, kLoops, constInfo.commonConstInfo.dSizeV, CUBE_BASEK);
        LocalTensor<INPUT_TYPE> vL1Tensor = vL1Buffer.template GetTensor<INPUT_TYPE>()[kvNzC0Stride * k * CUBE_BASEK];
        MMParam param = {(uint32_t)constInfo.commonConstInfo.gSize,
                         (uint32_t)runInfo.commonRunInfo.s2RealSize,
                         (uint32_t)realK,
                         false,
                         true,
                         true,
                         k == 0};
        MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, CUBE_BASEK, ABLayout::MK, ABLayout::KN>(
            dyL1Buffer.GetTensor<INPUT_TYPE>()[AlignTo16(constInfo.commonConstInfo.gSize) * k * CUBE_BASEK], vL1Tensor,
            l0aBuf, l0bBuf, mm1L0CBuffer.GetTensor<CALC_TYPE>(), param);
    }
    mm1L0CBuffer.Set<HardEvent::M_FIX>();
    mm1L0CBuffer.Wait<HardEvent::M_FIX>();
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
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::IterateMmQK(
    LocalTensor<CALC_TYPE> &mm2ResTensor, const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo,
    FagRunInfo &runInfo)
{
    (void)selectedKWorkSpaceGm;
    Buffer<BufferType::L1> qL1Buffer = qL1Buf[GetQDySlot(runInfo)].Get();
    Nd2NzParams nd2NzParams;

    // copy current, when IS_L1_PRELOAD=true, only first loop copy current
    nd2NzParams.dstNzC0Stride = AlignTo16(constInfo.commonConstInfo.gSize);
    if (!runInfo.isS1IdxNoChange) {
        LocalTensor<INPUT_TYPE> qL1Tensor = qL1Buffer.GetTensor<INPUT_TYPE>();
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = constInfo.commonConstInfo.gSize;
        nd2NzParams.dValue = constInfo.commonConstInfo.dSize;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = constInfo.mm2Ka;
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        if constexpr (IS_ROPE) {
            DataCopy(qL1Tensor, this->queryGm[runInfo.queryOffsetWithRopeForMm12], nd2NzParams);
            nd2NzParams.dValue = ROPE_D_64;
            nd2NzParams.srcDValue = ROPE_D_64;
            DataCopy(qL1Tensor[nd2NzParams.dstNzC0Stride * constInfo.commonConstInfo.dSize],
                     this->queryRopeGm[runInfo.commonRunInfo.qRopeOffset], nd2NzParams);
        } else {
            DataCopy(qL1Tensor, this->queryGm[runInfo.commonRunInfo.queryOffset], nd2NzParams);
        }
        qL1Buffer.Set<HardEvent::MTE2_MTE1>();
    }
    uint32_t kLoops = Ceil<int64_t>(constInfo.dTotalSize, CUBE_BASEK);
    Buffer<BufferType::L0C> mm2L0CBuffer = GetWorkL0CBuffer();
    mm2L0CBuffer.Wait<HardEvent::FIX_M>();
    if (!runInfo.isS1IdxNoChange) {
        qL1Buffer.Wait<HardEvent::MTE2_MTE1>();
    }
    uint32_t realK = CUBE_BASEK;
    Buffer<BufferType::L1> kL1Buffer = kvL1Buf[runInfo.commonRunInfo.taskIdMod2].Get();
    const uint32_t kvNzC0Stride = GetKvNzC0Stride(runInfo);
    for (uint32_t k = 0; k < kLoops; ++k) {
        realK = GetLoopTileSize(k, kLoops, constInfo.dTotalSize, CUBE_BASEK);
        LocalTensor<INPUT_TYPE> kL1Tensor = kL1Buffer.GetTensor<INPUT_TYPE>()[kvNzC0Stride * k * CUBE_BASEK];
        MMParam param = {(uint32_t)constInfo.commonConstInfo.gSize,
                         (uint32_t)runInfo.commonRunInfo.s2RealSize,
                         (uint32_t)realK,
                         false,
                         true,
                         true,
                         k == 0};
        MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN,
                   L0_SINGLE_BUFFER_SIZE / CUBE_BASEN / sizeof(INPUT_TYPE), ABLayout::MK, ABLayout::KN>(
            qL1Buffer.GetTensor<INPUT_TYPE>()[AlignTo16(constInfo.commonConstInfo.gSize) * k * CUBE_BASEK], kL1Tensor,
            l0aBuf, l0bBuf, mm2L0CBuffer.GetTensor<CALC_TYPE>(), param);
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
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::IterateMmDsKNopeRope(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    Buffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = dSL1Buf.Get();
    Buffer<BufferType::L1> kL1Buffer = kvL1Buf[runInfo.commonRunInfo.taskIdMod2].Get();
    LocalTensor<INPUT_TYPE> kL1Tensor = kL1Buffer.GetTensor<INPUT_TYPE>();
    const uint32_t kvNzC0Stride = GetKvNzC0Stride(runInfo);
    const uint32_t nopeN = constInfo.commonConstInfo.dSize;
    constexpr static FixpipeConfig DQ_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};

    Buffer<BufferType::L0C> dqNopeBuf = dqNopeL0CBuf.Get();
    if (!runInfo.isS1IdxNoChange) {
        dqNopeBuf.Wait<HardEvent::FIX_M>();
    }
    MMParam nopeParam = {(uint32_t)constInfo.commonConstInfo.gSize,
                         nopeN,
                         (uint32_t)runInfo.commonRunInfo.s2RealSize,
                         false,
                         false,
                         true,
                         !runInfo.isS1IdxNoChange};
    MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DQ_L0_SPLIT_K, ABLayout::MK, ABLayout::KN>(
        dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor, l0aBuf, l0bBuf, dqNopeBuf.GetTensor<CALC_TYPE>(), nopeParam);
    if (!runInfo.isNextS1IdxNoChange) {
        dqNopeBuf.Set<HardEvent::M_FIX>();
        dqNopeBuf.Wait<HardEvent::M_FIX>();
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = nopeN;
        fixpipeParams.mSize = constInfo.commonConstInfo.gSize;
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        fixpipeParams.dstStride = constInfo.dTotalSize;
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(outTensor[runInfo.queryOffsetWithRope],
                                                 dqNopeBuf.GetTensor<CALC_TYPE>(), fixpipeParams);
        dqNopeBuf.Set<HardEvent::FIX_M>();
    }

    Buffer<BufferType::L0C> ropeBuf = workL0CBuf.Get();
    ropeBuf.Wait<HardEvent::FIX_M>();
    MMParam ropeParam = {(uint32_t)constInfo.commonConstInfo.gSize,
                         ROPE_D_64,
                         (uint32_t)runInfo.commonRunInfo.s2RealSize,
                         false,
                         false,
                         true,
                         true};
    MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DQ_L0_SPLIT_K, ABLayout::MK, ABLayout::KN>(
        dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor[kvNzC0Stride * nopeN], l0aBuf, l0bBuf,
        ropeBuf.GetTensor<CALC_TYPE>(), ropeParam);
    ropeBuf.Set<HardEvent::M_FIX>();
    ropeBuf.Wait<HardEvent::M_FIX>();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> ropeFixpipe;
    ropeFixpipe.nSize = ROPE_D_64;
    ropeFixpipe.mSize = constInfo.commonConstInfo.gSize;
    ropeFixpipe.srcStride = AlignTo16(ropeFixpipe.mSize);
    ropeFixpipe.dstStride = constInfo.dTotalSize;
    ropeFixpipe.dualDstCtl = 0;
    ropeFixpipe.params.ndNum = 1;
    ropeFixpipe.params.srcNdStride = 0;
    ropeFixpipe.params.dstNdStride = 0;
    SetAtomicAdd<CALC_TYPE>();
    Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(outTensor[runInfo.queryOffsetWithRope + nopeN],
                                             ropeBuf.GetTensor<CALC_TYPE>(), ropeFixpipe);
    SetAtomicNone();
    ropeBuf.Set<HardEvent::FIX_M>();
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::IterateMmDsKNormal(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm,
    BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (IsDqNopeL0CResident()) {
        IterateMmDsKNopeRope<T, IS_WRITE_UB>(outTensor, dSL1Buf, constInfo, runInfo);
        return;
    }
    (void)selectedKWorkSpaceGm;
    Buffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = dSL1Buf.Get();
    constexpr uint32_t baseN = CUBE_BASEN;
    uint32_t realN = baseN;
    uint32_t nLoops = ((uint32_t)constInfo.dTotalSize + baseN - 1) / baseN;
    Buffer<BufferType::L1> kL1Buffer = kvL1Buf[runInfo.commonRunInfo.taskIdMod2].Get();
    for (uint32_t n = 0; n < nLoops; ++n) {
        realN = GetLoopTileSize(n, nLoops, constInfo.dTotalSize, baseN);
        LocalTensor<INPUT_TYPE> kL1Tensor = kL1Buffer.GetTensor<INPUT_TYPE>()[GetKvNzC0Stride(runInfo) * n * baseN];
        uint64_t gmNOffset = n * baseN;

        Buffer<BufferType::L0C> mm3L0CBuffer = GetWorkL0CBuffer();
        mm3L0CBuffer.Wait<HardEvent::FIX_M>();
        MMParam param = {(uint32_t)constInfo.commonConstInfo.gSize,
                         (uint32_t)realN,
                         (uint32_t)runInfo.commonRunInfo.s2RealSize,
                         false,
                         false,
                         true,
                         true};
        MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DQ_L0_SPLIT_K, ABLayout::MK,
                   ABLayout::KN>(dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor, l0aBuf, l0bBuf,
                                 mm3L0CBuffer.GetTensor<CALC_TYPE>(), param);

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
        Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(outTensor[runInfo.queryOffsetWithRope + gmNOffset],
                                                 mm3L0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
        SetAtomicNone();
        mm3L0CBuffer.Set<HardEvent::FIX_M>();
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::IterateMmDsQNopeRope(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    Buffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = dSL1Buf.GetPre();
    Buffer<BufferType::L1> qL1Buffer = qL1Buf[GetQDySlot(runInfo)].Get();
    LocalTensor<INPUT_TYPE> qL1Tensor = qL1Buffer.GetTensor<INPUT_TYPE>();
    const uint32_t qStride = AlignTo16(constInfo.commonConstInfo.gSize);
    const uint32_t nopeN = constInfo.commonConstInfo.dSize;
    constexpr static FixpipeConfig DK_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
    // v2.1：每轮 N=128，1 次 MM + 1 次 Fixpipe；4 轮铺满 512，单槽 L0C=32KB 便于 ping-pong
    constexpr uint32_t mm4NopeTileN = CUBE_BASEN;
    uint32_t nopeLoops = (nopeN + mm4NopeTileN - 1) / mm4NopeTileN;
    for (uint32_t n = 0; n < nopeLoops; ++n) {
        uint32_t realN = GetLoopTileSize(n, nopeLoops, nopeN, mm4NopeTileN);
        Buffer<BufferType::L0C> dkBuf = workL0CBuf.Get();
        dkBuf.Wait<HardEvent::FIX_M>();
        MMParam nopeParam = {(uint32_t)runInfo.commonRunInfo.s2RealSize,
                             realN,
                             (uint32_t)constInfo.commonConstInfo.gSize,
                             true,
                             false,
                             true,
                             true};
        MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                   ABLayout::KN>(dSL1Buffer.GetTensor<INPUT_TYPE>(), qL1Tensor[qStride * n * mm4NopeTileN], l0aBuf,
                                 l0bBuf, dkBuf.GetTensor<CALC_TYPE>(), nopeParam);
        dkBuf.Set<HardEvent::M_FIX>();
        dkBuf.Wait<HardEvent::M_FIX>();
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> nopeFixpipe;
        nopeFixpipe.nSize = realN;
        nopeFixpipe.mSize = runInfo.commonRunInfo.s2RealSize;
        nopeFixpipe.srcStride = AlignTo16(nopeFixpipe.mSize);
        nopeFixpipe.dstStride = constInfo.dTotalSize;
        nopeFixpipe.dualDstCtl = 0;
        nopeFixpipe.params.ndNum = 1;
        nopeFixpipe.params.srcNdStride = 0;
        nopeFixpipe.params.dstNdStride = 0;
        Fixpipe<T, CALC_TYPE, DK_FIXPIPE_CONFIG>(outTensor[runInfo.mm4ResWsAddr + n * mm4NopeTileN],
                                                 dkBuf.GetTensor<CALC_TYPE>(), nopeFixpipe);
        dkBuf.Set<HardEvent::FIX_M>();
    }

    Buffer<BufferType::L0C> ropeBuf = workL0CBuf.Get();
    ropeBuf.Wait<HardEvent::FIX_M>();
    MMParam ropeParam = {(uint32_t)runInfo.commonRunInfo.s2RealSize,
                         ROPE_D_64,
                         (uint32_t)constInfo.commonConstInfo.gSize,
                         true,
                         false,
                         true,
                         true};
    MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK, ABLayout::KN>(
        dSL1Buffer.GetTensor<INPUT_TYPE>(), qL1Tensor[qStride * nopeN], l0aBuf, l0bBuf, ropeBuf.GetTensor<CALC_TYPE>(),
        ropeParam);
    ropeBuf.Set<HardEvent::M_FIX>();
    ropeBuf.Wait<HardEvent::M_FIX>();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> ropeFixpipe;
    ropeFixpipe.nSize = ROPE_D_64;
    ropeFixpipe.mSize = runInfo.commonRunInfo.s2RealSize;
    ropeFixpipe.srcStride = AlignTo16(ropeFixpipe.mSize);
    ropeFixpipe.dstStride = constInfo.dTotalSize;
    ropeFixpipe.dualDstCtl = 0;
    ropeFixpipe.params.ndNum = 1;
    ropeFixpipe.params.srcNdStride = 0;
    ropeFixpipe.params.dstNdStride = 0;
    Fixpipe<T, CALC_TYPE, DK_FIXPIPE_CONFIG>(outTensor[runInfo.mm4ResWsAddr + nopeN], ropeBuf.GetTensor<CALC_TYPE>(),
                                             ropeFixpipe);
    ropeBuf.Set<HardEvent::FIX_M>();
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::IterateMmDsQNormal(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (IsDqNopeL0CResident()) {
        IterateMmDsQNopeRope<T, IS_WRITE_UB>(outTensor, dSL1Buf, constInfo, runInfo);
        return;
    }
    Buffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = dSL1Buf.GetPre();
    constexpr uint32_t baseN = CUBE_BASEN;
    uint32_t nLoops = ((uint32_t)constInfo.dTotalSize + baseN - 1) / baseN; // 尾块处理
    uint32_t realN = baseN;
    Buffer<BufferType::L1> qL1Buffer = qL1Buf[GetQDySlot(runInfo)].Get();
    LocalTensor<INPUT_TYPE> qL1Tensor = qL1Buffer.GetTensor<INPUT_TYPE>();
    for (uint32_t n = 0; n < nLoops; ++n) {
        realN = GetLoopTileSize(n, nLoops, constInfo.dTotalSize, baseN);
        Nd2NzParams nd2NzParams;
        nd2NzParams.dstNzC0Stride = AlignTo16(constInfo.commonConstInfo.gSize);
        int64_t queryL1Offset = nd2NzParams.dstNzC0Stride * n * CUBE_BASEN;

        Buffer<BufferType::L0C> dkL0CBuffer = GetWorkL0CBuffer();
        // load l1 to l0ab + mmad
        dkL0CBuffer.Wait<HardEvent::FIX_M>();                        // 反向同步
        MMParam param = {(uint32_t)runInfo.commonRunInfo.s2RealSize, // singleM
                         (uint32_t)realN,                            // singleN
                         (uint32_t)constInfo.commonConstInfo.gSize,  // singleK
                         true,                                       // isLeftTranspose
                         false,                                      // isRightTranspose
                         true,
                         true};
        MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                   ABLayout::KN>(dSL1Buffer.GetTensor<INPUT_TYPE>(), qL1Tensor[queryL1Offset], l0aBuf, l0bBuf,
                                 dkL0CBuffer.GetTensor<CALC_TYPE>(), param);

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
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::IterateMmPDyNormal(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    Buffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = pL1Buf.Get();
    constexpr uint32_t baseN = CUBE_BASEN;
    uint32_t nLoops = ((uint32_t)constInfo.commonConstInfo.dSizeV + baseN - 1) / baseN; // 尾块处理
    uint32_t realN = baseN;
    Buffer<BufferType::L1> dYL1Buffer = dYL1Buf[GetQDySlot(runInfo)].Get();
    LocalTensor<INPUT_TYPE> dYL1Tensor = dYL1Buffer.GetTensor<INPUT_TYPE>();
    for (uint32_t n = 0; n < nLoops; ++n) {
        realN = GetLoopTileSize(n, nLoops, constInfo.commonConstInfo.dSizeV, baseN);
        Nd2NzParams nd2NzParams;
        nd2NzParams.dstNzC0Stride = AlignTo16(constInfo.commonConstInfo.gSize);
        int64_t dyL1Offset = nd2NzParams.dstNzC0Stride * n * CUBE_BASEN;
        Buffer<BufferType::L0C> dvL0CBuffer = GetWorkL0CBuffer();
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
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::IterateMmDsK(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm,
    BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    IterateMmDsKNormal<T, IS_WRITE_UB>(outTensor, selectedKWorkSpaceGm, dSL1Buf, constInfo, runInfo);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::IterateMmDsQ(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    IterateMmDsQNormal<T, IS_WRITE_UB>(outTensor, dSL1Buf, constInfo, runInfo);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCubeNLe64<TEMPLATE_ARGS>::IterateMmPDy(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    IterateMmPDyNormal<T, IS_WRITE_UB>(outTensor, pL1Buf, constInfo, runInfo);
}

} // namespace SfagBaseApi
#endif
