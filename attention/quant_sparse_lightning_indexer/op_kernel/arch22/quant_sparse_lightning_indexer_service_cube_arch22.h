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
 * \file quant_sparse_lightning_indexer_service_cube_arch22.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_SERVICE_CUBE_H
#define QUANT_LIGHTNING_INDEXER_V2_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_sparse_lightning_indexer_common_arch22.h"

namespace QLIV2Kernel {
using namespace QLIV2Common;
struct MmInfo {
    int64_t s2L0LoopId;
    int64_t s1gL0LoopId;
    int64_t s2L0RealSize;
    int64_t s2GmOffset;
};

template <typename QLIV2T>
class QLIV2Matmul {
public:
    using Q_T = typename QLIV2T::queryType;
    using K_T = typename QLIV2T::keyType;

    __aicore__ inline QLIV2Matmul(){};
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t>& blkTableGm, const GlobalTensor<K_T>& keyGm,
                                               const GlobalTensor<Q_T>& queryGm, const GlobalTensor<float>& mm1ResGm,
                                               const GlobalTensor<half>& weightWorkspaceGm);
    __aicore__ inline void InitParams(const ConstInfo& constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const QLIV2Common::RunInfo& runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
    static constexpr uint64_t DOUBLE_BUF_NUM = 2;
    static constexpr uint64_t L0AB_BUF_NUM = 4;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QW_MTE1_MTE2_EVENT = EVENT_ID5; // KEY_MTE1_MTE2_EVENT + DOUBLE_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID0;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t FIX_MTE1_EVENT = EVENT_ID4;

    static constexpr uint64_t S8_BLOCK_CUBE = 32;
    // W4A4(int4b_t): L1->L0 的 _s4 fract 每 32 字节承载 64 个 int4 逻辑元素 (IFA arch22 先例 kSize/64)
    static constexpr uint64_t S4_BLOCK_CUBE = 64;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;

    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S1G_BASIC_BLOCK_L1 = 256;

    static constexpr uint64_t S1G_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t QUERY_BUFFER_OFFSET = S1G_BASIC_BLOCK_L1 * D_BASIC_BLOCK;
    static constexpr uint64_t SL1_BUFFER_OFFSET = S1G_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK;
    static constexpr uint64_t WEIGHT_BUFFER_OFFSET = S1G_BASIC_BLOCK_L1 * BLOCK_CUBE;
    static constexpr uint64_t L0AB_BUFFER_OFFSET_S8_16K = 16 * 1024;
    static constexpr uint64_t L0AB_BUFFER_OFFSET_FP16_16K = 16 * 512;
    // int4b_t 元素 0.5B: LocalTensor 按元素寻址, 16KB 字节步进需 2 倍元素偏移,
    // 与 half 视角 (FP16_16K) 的 S/W buffer 保持同相位 16KB 步进, 否则 Key/Q buffer
    // 与 S/W buffer 错位重叠引发 L0A/L0B read/write conflict (Mmad 并发读同地址)
    static constexpr uint64_t L0AB_BUFFER_OFFSET_S4_16K = L0AB_BUFFER_OFFSET_S8_16K * 2;
    static constexpr uint64_t L0C_BUFFER_OFFSET = 64 * 256;

private:
    __aicore__ inline void WeightDmaCopy(uint64_t s1gL1RealSize, const QLIV2Common::RunInfo& runInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0RealSize);
    __aicore__ inline uint64_t GetL0abElemOffset() const;
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize, uint64_t s1gL0RealSize);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, const QLIV2Common::RunInfo& runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                         const QLIV2Common::RunInfo& runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, const MmInfo& mmInfo, const QLIV2Common::RunInfo& runInfo);
    __aicore__ inline void FixpSToL1(uint64_t s1gL0RealSize, uint64_t s2L0RealSize);
    __aicore__ inline void LoadSToL0b(uint64_t s1gL1RealSize, uint64_t s2L0RealSize, uint64_t sL1BufIdx,
                                      int64_t mStartPt);
    __aicore__ inline void LoadWeightToL0a(uint64_t s1gL1Offset);
    __aicore__ inline void ComputeWs(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, int64_t s1gOffset);
    __aicore__ inline void FixpResToGm(uint64_t s1L0RealCount, uint64_t s2L0RealSize, uint64_t s1GmOffset,
                                       uint64_t s2GmOffset, const QLIV2Common::RunInfo& runInfo);
    __aicore__ inline void ComputeQk(uint64_t s1gL0RealSize, uint64_t s2L0RealSize);
    __aicore__ inline void ProcessWs(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t sL1BufIdx,
                                     const MmInfo& mmInfo, const QLIV2Common::RunInfo& runInfo);
    __aicore__ inline void ProcessQk(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t s1L0LoopCnt,
                                     const MmInfo& mmInfo, const QLIV2Common::RunInfo& runInfo);
    __aicore__ inline void CalcMmInfo(MmInfo& mmInfo, uint64_t loopIdx, uint64_t s1L0LoopCnt, const MmInfo& lastMmInfo,
                                      const QLIV2Common::RunInfo& runInfo);
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIV2T::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIV2T::keyLayout;
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<half> weightGm_;
    GlobalTensor<float> mm1ResGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;
    TBuf<TPosition::A1> bufWeightL1_;
    LocalTensor<half> weightL1_;
    TBuf<TPosition::B1> bufSL1_;
    LocalTensor<half> sL1_;

    TBuf<TPosition::A2> bufL0A_;
    LocalTensor<Q_T> l0a_;
    TBuf<TPosition::B2> bufL0B_;
    LocalTensor<K_T> l0b_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<int32_t> cL0_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t qwL1Mte2BufIdx_ = 0;
    uint64_t sL1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;
    uint64_t l0cBufIdx_ = 0;

    ConstInfo constInfo_;
};

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::InitParams(const ConstInfo& constInfo)
{
    constInfo_ = constInfo;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::InitBuffers(TPipe* pipe)
{
    pipe->InitBuffer(bufQL1_, DOUBLE_BUF_NUM * S1G_BASIC_BLOCK_L1 * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, DOUBLE_BUF_NUM * S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufWeightL1_, DOUBLE_BUF_NUM * S1G_BASIC_BLOCK_L1 * BLOCK_CUBE * sizeof(half));
    weightL1_ = bufWeightL1_.Get<half>();
    pipe->InitBuffer(bufSL1_, DOUBLE_BUF_NUM * S2_BASIC_BLOCK_L0 * S1G_BASIC_BLOCK_L0 * sizeof(half));
    sL1_ = bufSL1_.Get<half>();

    pipe->InitBuffer(bufL0A_, 64 * 1024);
    l0a_ = bufL0A_.Get<Q_T>();
    pipe->InitBuffer(bufL0B_, 64 * 1024);
    l0b_ = bufL0B_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, 128 * 1024);
    cL0_ = bufL0C_.Get<int32_t>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::InitMm1GlobalTensor(const GlobalTensor<int32_t>& blkTableGm,
                                                                const GlobalTensor<K_T>& keyGm,
                                                                const GlobalTensor<Q_T>& queryGm,
                                                                const GlobalTensor<float>& mm1ResGm,
                                                                const GlobalTensor<half>& weightWorkspaceGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
    mm1ResGm_ = mm1ResGm;
    weightGm_ = weightWorkspaceGm;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ProcessWs(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t sL1BufIdx,
                                                      const MmInfo& mmInfo, const QLIV2Common::RunInfo& runInfo)
{
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % DOUBLE_BUF_NUM);
    for (int64_t s1gOffset = 0; s1gOffset < s1gL0RealSize; s1gOffset += constInfo_.gSize) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
        LoadSToL0b(s1gL0RealSize, mmInfo.s2L0RealSize, sL1BufIdx, s1gOffset);
        LoadWeightToL0a(s1gOffset + s1gL1Offset);

        ComputeWs(s1gL0RealSize, mmInfo.s2L0RealSize, s1gOffset);

        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
        l0BufIdx_++;
    }

    FixpResToGm(s1gL0RealSize / constInfo_.gSize, mmInfo.s2L0RealSize, s1gL1Offset / constInfo_.gSize,
                mmInfo.s2L0LoopId * S2_BASIC_BLOCK_L0, runInfo);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % DOUBLE_BUF_NUM);
    l0cBufIdx_++;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ProcessQk(uint64_t s1gL0RealSize, uint64_t s1gL1Offset,
                                                      uint64_t s1L0LoopCnt, const MmInfo& mmInfo,
                                                      const QLIV2Common::RunInfo& runInfo)
{
    if (mmInfo.s1gL0LoopId == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % DOUBLE_BUF_NUM);
        if constexpr (K_LAYOUT_T == LI_LAYOUT::PA_BBND) {
            KeyNd2NzForPA(mmInfo.s2L0RealSize, runInfo.s2Idx * constInfo_.s2BaseSize + mmInfo.s2GmOffset, runInfo);
        } else {
            KeyNd2Nz(mmInfo.s2L0RealSize, mmInfo, runInfo);
        }

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    }

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
    LoadQueryToL0a(s1gL1Offset, runInfo.actMBaseSize, s1gL0RealSize);
    LoadKeyToL0b(mmInfo.s2L0RealSize);

    if (mmInfo.s1gL0LoopId + 1 >= s1L0LoopCnt) {
        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % DOUBLE_BUF_NUM);
        keyL1BufIdx_++;
    }

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % DOUBLE_BUF_NUM);
    ComputeQk(s1gL0RealSize, mmInfo.s2L0RealSize);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);

    FixpSToL1(s1gL0RealSize, mmInfo.s2L0RealSize);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % DOUBLE_BUF_NUM);
    l0BufIdx_++;
    l0cBufIdx_++;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::CalcMmInfo(MmInfo& mmInfo, uint64_t loopIdx, uint64_t s1L0LoopCnt,
                                                       const MmInfo& lastMmInfo, const QLIV2Common::RunInfo& runInfo)
{
    mmInfo.s2L0LoopId = loopIdx / s1L0LoopCnt;
    mmInfo.s1gL0LoopId = loopIdx % s1L0LoopCnt;

    if (mmInfo.s1gL0LoopId == 0) {
        mmInfo.s2GmOffset = mmInfo.s2L0LoopId * S2_BASIC_BLOCK_L0;
        mmInfo.s2L0RealSize = mmInfo.s2GmOffset + S2_BASIC_BLOCK_L0 > runInfo.actualSingleProcessSInnerSize ?
                                  runInfo.actualSingleProcessSInnerSize - mmInfo.s2GmOffset :
                                  S2_BASIC_BLOCK_L0;
    } else {
        mmInfo.s2L0RealSize = lastMmInfo.s2L0RealSize;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ComputeMm1(const QLIV2Common::RunInfo& runInfo)
{
    if (runInfo.isFirstS2InnerLoop) {
        WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM);
        QueryNd2Nz(runInfo.actMBaseSize, runInfo); // 256 * 128 // L1BasicBlock
        WeightDmaCopy(runInfo.actMBaseSize, runInfo);
    }
    int64_t loopIdx = 0;
    int64_t s2L0LoopCnt = CeilDiv(runInfo.actualSingleProcessSInnerSize, S2_BASIC_BLOCK_L0); // 2048取128
    int64_t s1L0LoopCnt = CeilDiv(runInfo.actMBaseSize, S1G_BASIC_BLOCK_L0);                 // 256取128
    int64_t s1gL1Offset[2] = {0, static_cast<int64_t>(S1G_BASIC_BLOCK_L0)};
    int64_t s1gL0RealSize[2] = {s1L0LoopCnt > 1 ? static_cast<int64_t>(S1G_BASIC_BLOCK_L0) : runInfo.actMBaseSize,
                                runInfo.actMBaseSize - s1gL1Offset[1]};
    MmInfo mmInfo[2];
    CalcMmInfo(mmInfo[loopIdx & 1], loopIdx, s1L0LoopCnt, mmInfo[(loopIdx + 1) & 1], runInfo);

    ProcessQk(s1gL0RealSize[mmInfo[loopIdx & 1].s1gL0LoopId % s1L0LoopCnt],
              s1gL1Offset[mmInfo[loopIdx & 1].s1gL0LoopId % s1L0LoopCnt], s1L0LoopCnt, mmInfo[loopIdx & 1], runInfo);

    SetFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % DOUBLE_BUF_NUM);
    sL1BufIdx_++;
    loopIdx++;

    while (loopIdx < s2L0LoopCnt * s1L0LoopCnt) {
        CalcMmInfo(mmInfo[loopIdx & 1], loopIdx, s1L0LoopCnt, mmInfo[(loopIdx + 1) & 1], runInfo);

        ProcessQk(s1gL0RealSize[mmInfo[loopIdx & 1].s1gL0LoopId % s1L0LoopCnt],
                  s1gL1Offset[mmInfo[loopIdx & 1].s1gL0LoopId % s1L0LoopCnt], s1L0LoopCnt, mmInfo[loopIdx & 1],
                  runInfo);

        SetFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % DOUBLE_BUF_NUM);
        sL1BufIdx_++;

        WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % DOUBLE_BUF_NUM);

        ProcessWs(s1gL0RealSize[mmInfo[(loopIdx + 1) & 1].s1gL0LoopId % s1L0LoopCnt],
                  s1gL1Offset[mmInfo[(loopIdx + 1) & 1].s1gL0LoopId % s1L0LoopCnt], sL1BufIdx_,
                  mmInfo[(loopIdx + 1) & 1], runInfo);
        loopIdx++;
    }

    WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + (sL1BufIdx_ + 1) % DOUBLE_BUF_NUM);

    ProcessWs(s1gL0RealSize[mmInfo[(loopIdx + 1) & 1].s1gL0LoopId % s1L0LoopCnt],
              s1gL1Offset[mmInfo[(loopIdx + 1) & 1].s1gL0LoopId % s1L0LoopCnt], sL1BufIdx_ - 1,
              mmInfo[(loopIdx + 1) & 1], runInfo);

    if (runInfo.isLastS2InnerLoop) {
        SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM);
        qwL1Mte2BufIdx_++;
    }
}

// blkNum, blkSize, N2, D
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                          const QLIV2Common::RunInfo& runInfo)
{
    uint64_t s2L1Offset = 0;
    // A11: key 0 轴非连续寻址 — 块基址 = 表值 x keyStride0 (host 下发的真实 stride)。
    // keyStride0 == 0 (旧调用/非 PA 紧凑) 时兜底原紧凑公式, 现网行为 bit 级不变 (R8)
    uint64_t blkStride = constInfo_.keyStride0 != 0 ? static_cast<uint64_t>(constInfo_.keyStride0) :
                                                      static_cast<uint64_t>(constInfo_.kCacheBlockSize) *
                                                          constInfo_.kHeadNum * constInfo_.headDim;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
        // P13 同型防御: 表寻址 bIdx * maxBlockNumPerBatch 为 uint32 乘法, block_table 总项
        // 超 2^32 时截断 — 提前提升 uint64
        uint64_t keyGmOffset =
            blkTableGm_.GetValue(static_cast<uint64_t>(runInfo.bIdx) * constInfo_.maxBlockNumPerBatch + s2BlkId) *
                blkStride +
            s2BlkOffset * constInfo_.headDim;
        uint64_t s2Mte2Size = s2L1RealSize - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ? constInfo_.kCacheBlockSize - s2BlkOffset :
                                                                              s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size; // 行数
        // GM 偏移 (keyGmOffset/blkStride) 保持 int4 逻辑元素寻址 (GlobalTensor<int4b_t> 按 nibble 折算),
        // 仅 Nd2Nz 的 src/dst 列宽按 int8 字节视角折半 (IFA arch22 KVINT4 先例 dValue=headDim/2)
        if constexpr (IsSameType<K_T, int4b_t>::value) {
            nd2nzPara.dValue = constInfo_.headDim / 2;
            nd2nzPara.srcDValue = constInfo_.headDim / 2;
        } else {
            nd2nzPara.dValue = constInfo_.headDim;
            nd2nzPara.srcDValue = constInfo_.headDim;
        }
        nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        // L1 行偏移按物理 32B 行槽守恒 (IFA full_quant PA 先例 blockElementCnt: int8=32/int4=64 元素),
        // int4b_t 的 LocalTensor 按 nibble 寻址 (element=0.5B), 32B 行槽需 64 元素步进
        uint64_t l1RowStep = IsSameType<K_T, int4b_t>::value ? S4_BLOCK_CUBE : S8_BLOCK_CUBE;
        DataCopy(keyL1_[(keyL1BufIdx_ % DOUBLE_BUF_NUM) * KEY_BUFFER_OFFSET + s2L1Offset * l1RowStep],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::KeyNd2Nz(uint64_t s2L1RealSize, const MmInfo& mmInfo,
                                                     const QLIV2Common::RunInfo& runInfo)
{
    uint64_t dStride = constInfo_.headDim;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::BSND || K_LAYOUT_T == LI_LAYOUT::TND) {
        dStride = constInfo_.headDim * constInfo_.kHeadNum; // constInfo_.kHeadNum
    }
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s2L1RealSize; // 行数
    // GM 偏移 (tensorKeyOffset/s2GmOffset) 保持 int4 逻辑元素寻址, 仅列宽按 int8 字节视角折半
    if constexpr (IsSameType<K_T, int4b_t>::value) {
        nd2nzPara.dValue = constInfo_.headDim / 2;
        nd2nzPara.srcDValue = dStride / 2;
    } else {
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = dStride;
    }
    nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(keyL1_[(keyL1BufIdx_ % DOUBLE_BUF_NUM) * KEY_BUFFER_OFFSET],
             keyGm_[runInfo.tensorKeyOffset + mmInfo.s2GmOffset * constInfo_.headDim], nd2nzPara);
}

// batch, s1, g, 1
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::WeightDmaCopy(uint64_t s1gL1RealSize, const QLIV2Common::RunInfo& runInfo)
{
    DataCopyParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = s1gL1RealSize;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopy(weightL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * WEIGHT_BUFFER_OFFSET],
             weightGm_[runInfo.loop % DOUBLE_BUF_NUM * BLOCK_CUBE * constInfo_.mBaseSize], copyInParams);
}

// batch, s1, n2, g, d
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::QueryNd2Nz(uint64_t s1gL1RealSize, const QLIV2Common::RunInfo& runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize; // 行数
    // W4A4: Q 按 int4 打包存储, Nd2Nz 列宽按 int8 字节视角折半; GM 偏移保持逻辑元素寻址
    if constexpr (IsSameType<Q_T, int4b_t>::value) {
        nd2nzPara.dValue = constInfo_.headDim / 2;
        nd2nzPara.srcDValue = constInfo_.headDim / 2;
    } else {
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
    }
    nd2nzPara.dstNzC0Stride = CeilAlign(s1gL1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(queryL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * QUERY_BUFFER_OFFSET], queryGm_[runInfo.tensorQueryOffset],
             nd2nzPara);
}

// s1g, d
template <typename QLIV2T>
__aicore__ inline uint64_t QLIV2Matmul<QLIV2T>::GetL0abElemOffset() const
{
    // Q/K 为 int4b_t 时元素为 0.5B, 16KB 字节步进需 2 倍元素数; half 视角 (S/W) 保持原偏移
    return IsSameType<Q_T, int4b_t>::value ? L0AB_BUFFER_OFFSET_S4_16K : L0AB_BUFFER_OFFSET_S8_16K;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize,
                                                           uint64_t s1gL0RealSize)
{
    LoadData3DParamsV2<Q_T> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = CeilDiv(s1gL1RealSize, BLOCK_CUBE); // Hin=M1=8
    loadData3DParams.l1W = BLOCK_CUBE;                         // Win=M0
    loadData3DParams.channelSize = constInfo_.headDim;         // Cin=K

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    // W4A4: LoadData3D 参数以逻辑元素为单位 (CANN 校验 int4b_t channelSize 允许值 64 = 32B fract),
    // _s4 指令自动按 nibble 处理, kExtension/channelSize 保持 headDim 不折半 (IFA full_quant LoadDataL0A 先例)
    loadData3DParams.kExtension = constInfo_.headDim;                   // K width维度目的
    loadData3DParams.mExtension = CeilAlign(s1gL0RealSize, BLOCK_CUBE); // M height维度目的
    loadData3DParams.mStartPt = s1gL1Offset;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;

    LoadData<Q_T, LOAD3DV2_CONFIG>(l0a_[(l0BufIdx_ % L0AB_BUF_NUM) * GetL0abElemOffset()],
                                   queryL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * QUERY_BUFFER_OFFSET],
                                   loadData3DParams);
}

// s1, g, s2  -->  2 * 64* 128
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadSToL0b(uint64_t s1gL1RealSize, uint64_t s2L0RealSize,
                                                       uint64_t sL1BufIdx, int64_t mStartPt)
{
    LoadData3DParamsV2<half> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = S1G_BASIC_BLOCK_L0 / BLOCK_CUBE;             // Hin=M1=8
    loadData3DParams.l1W = BLOCK_CUBE;                                  // Win=M0
    loadData3DParams.channelSize = CeilAlign(s2L0RealSize, BLOCK_CUBE); // Cin=K

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = constInfo_.gSize;                    // M height维度目的
    loadData3DParams.kExtension = CeilAlign(s2L0RealSize, BLOCK_CUBE); // K   width维度目的
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 1;
    loadData3DParams.fMatrixCtrl = 0;

    loadData3DParams.mStartPt = mStartPt;
    LoadData<half, LOAD3DV2_CONFIG>(
        l0b_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
        sL1_[(sL1BufIdx % DOUBLE_BUF_NUM) * SL1_BUFFER_OFFSET], loadData3DParams);
}

// s1,g,1(16), 2,64,16
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadWeightToL0a(uint64_t s1gL1Offset)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(constInfo_.gSize, BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = true;
    LoadData(l0a_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
             weightL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * WEIGHT_BUFFER_OFFSET + s1gL1Offset * BLOCK_CUBE],
             loadData2DParams);
}

// s2, d -> 128,128
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadKeyToL0b(uint64_t s2L0RealSize)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    if constexpr (IsSameType<K_T, int4b_t>::value) {
        // W4A4: _s4 fract 每 32 字节承载 64 个 int4 逻辑元素, K 维 fract 数减半 (IFA 先例 kSize/64)
        loadData2DParams.repeatTimes = CeilDiv(s2L0RealSize, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, S4_BLOCK_CUBE);
    } else {
        loadData2DParams.repeatTimes = CeilDiv(s2L0RealSize, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, S8_BLOCK_CUBE);
    }
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0b_[(l0BufIdx_ % L0AB_BUF_NUM) * GetL0abElemOffset()],
             keyL1_[(keyL1BufIdx_ % DOUBLE_BUF_NUM) * KEY_BUFFER_OFFSET], loadData2DParams);
}

// A: s1,g,1(16) B: s1,g,s2  C: s1, 1(16), s2
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ComputeWs(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, int64_t s1gOffset)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    MmadParams mmadParams;
    mmadParams.m = BLOCK_CUBE;
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.gSize;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_.template ReinterpretCast<float>()[(l0cBufIdx_ % DOUBLE_BUF_NUM) * L0C_BUFFER_OFFSET +
                                                s1gOffset * S2_BASIC_BLOCK_L0],
         l0a_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
         l0b_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K], mmadParams);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ComputeQk(uint64_t s1gL0RealSize, uint64_t s2L0RealSize)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_[(l0cBufIdx_ % DOUBLE_BUF_NUM) * L0C_BUFFER_OFFSET],
         l0a_[(l0BufIdx_ % L0AB_BUF_NUM) * GetL0abElemOffset()], l0b_[(l0BufIdx_ % L0AB_BUF_NUM) * GetL0abElemOffset()],
         mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::FixpSToL1(uint64_t s1gL0RealSize, uint64_t s2L0RealSize)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    DataCopyCO12DstParams params;
    params.mSize = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    params.nSize = CeilAlign(s2L0RealSize, BLOCK_CUBE);
    params.dstStride = S1G_BASIC_BLOCK_L0;
    params.srcStride = params.mSize;
    params.quantPre = QuantMode_t::DEQF16;
    params.reluPre = 1;
    params.channelSplit = 0;
    params.nz2ndEn = 0;
    SetFixpipePreQuantFlag(0x3a800000);
    DataCopy(sL1_[(sL1BufIdx_ % DOUBLE_BUF_NUM) * SL1_BUFFER_OFFSET],
             cL0_[(l0cBufIdx_ % DOUBLE_BUF_NUM) * L0C_BUFFER_OFFSET], params);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::FixpResToGm(uint64_t s1L0RealCount, uint64_t s2L0RealSize,
                                                        uint64_t s1GmOffset, uint64_t s2GmOffset,
                                                        const QLIV2Common::RunInfo& runInfo)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);

    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = 1;
    intriParams.nSize = s2L0RealSize;
    intriParams.dstStride = constInfo_.s2BaseSize;
    intriParams.srcStride = 16;
    // set mode according to dtype
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.reluPre = 0;
    AscendC::SetFixpipeNz2ndFlag(s1L0RealCount, CeilDiv(constInfo_.gSize, BLOCK_CUBE) * S2_BASIC_BLOCK_L0 / BLOCK_CUBE,
                                 2048);
    AscendC::DataCopy(mm1ResGm_[(runInfo.loop % 2) * constInfo_.mBaseSize / constInfo_.gSize * constInfo_.s2BaseSize +
                                s1GmOffset * intriParams.dstStride + s2GmOffset],
                      cL0_.template ReinterpretCast<float>()[(l0cBufIdx_ % DOUBLE_BUF_NUM) * L0C_BUFFER_OFFSET],
                      intriParams);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}
} // namespace QLIV2Kernel
#endif // QUANT_LIGHTNING_INDEXER_V2_SERVICE_CUBE_H
