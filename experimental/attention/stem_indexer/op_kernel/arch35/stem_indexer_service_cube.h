/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file stem_indexer_service_cube.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef stem_indexer_SERVICE_CUBE_H
#define stem_indexer_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stem_indexer_common.h"

namespace SIKernel {
using namespace SICommon;
template <typename SIT>
class SIMatmul {
public:
    using Q_T = typename SIT::queryType;
    using K_T = typename SIT::keyType;
    using QK_T = float32_t;

    __aicore__ inline SIMatmul(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<Q_T> &queryGm, const GlobalTensor<K_T> &keyGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const SICommon::RunInfo &runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t QUERY_BUF_NUM = 4;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID3; // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID3;

    static constexpr uint64_t M_BASIC_BLOCK = 96;
    static constexpr uint64_t S2_BASIC_BLOCK = 256;

    static constexpr uint64_t M_BASIC_BLOCK_L1 = 96;
    static constexpr uint64_t D_BASIC_BLOCK_L1 = 512;
    static constexpr uint64_t S2_BASIC_BLOCK_L1 = 32;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 96;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 32;

    static constexpr uint64_t BF16_BLOCK_CUBE = 16;
    static constexpr uint32_t PIPE_M_BARRIER_THRESHOLD = 10U;
    // ROW_MAJOR使能NZ2ND并输出ND格式；true表示目的地址位于UB。
    static constexpr FixpipeConfig SI_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1;
    static constexpr uint64_t L0A_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0B_BUFFER_OFFSET = S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK * S2_BASIC_BLOCK;

protected:
    __aicore__ inline void Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                uint64_t s2L0RealSize, const SICommon::RunInfo &runInfo);
    __aicore__ inline void ComuteL0c(uint64_t s1gOffset, uint64_t s2Offset, uint64_t kGmOffset, uint64_t s1gL0RealSize,
                                     uint64_t s2L0RealSize, const SICommon::RunInfo &runInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                        uint64_t kL1Offset, const SICommon::RunInfo &runInfo);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL0Offset, uint64_t s1gL1RealSize, uint64_t s1gL0RealSize,
                                          uint64_t kL1Offset, const SICommon::RunInfo &runInfo);
    __aicore__ inline void CopyQuerySegmentNd2Nz(LocalTensor<Q_T> queryL1Base, uint64_t l1RowOffset,
                                                 uint64_t l1RowAlign, uint64_t gmOffset, uint64_t nValue);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gL1Offset, uint64_t kGmOffset,
                                      const SICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset, uint64_t kGmOffset,
                                    const SICommon::RunInfo &runInfo);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;

    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<Q_T> queryL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<K_T> keyL0_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<float> cL0_;

    TBuf<TPosition::VECCALC> bufUB_;
    LocalTensor<QK_T> mm1ResUB_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t queryBufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;
    uint64_t l0cBufIdx_ = 0;

    ConstInfo constInfo_;

private:
    static constexpr bool PAGE_ATTENTION = SIT::pageAttention;
};

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::InitBuffers(TPipe *pipe)
{
    // 大小：2(开dB) * 2 * 64 * 128 * 4 = 128KB
    pipe->InitBuffer(bufUB_, 2 * ((constInfo_.mBaseSize + 1U) >> 1U) * constInfo_.s2BaseSize * sizeof(QK_T));
    mm1ResUB_ = bufUB_.Get<QK_T>();
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1 * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1 * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * M_BASIC_BLOCK * S2_BASIC_BLOCK * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::InitMm1GlobalTensor(const GlobalTensor<Q_T> &queryGm,
                                                          const GlobalTensor<K_T> &keyGm)
{
    queryGm_ = queryGm;
    keyGm_ = keyGm;
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::ComputeMm1(const SICommon::RunInfo &runInfo)
{
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_VC_EVENT + (runInfo.loop & 1U));
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_VC_EVENT + (runInfo.loop & 1U) +
                                                         SICommon::AIV0_AIV1_OFFSET);
    uint64_t s1gProcessSize = runInfo.actMBaseSize;
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    uint64_t kProcessSize = constInfo_.headDim;

    // s2轴循环
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + (l0cBufIdx_ & (L0_BUF_NUM - 1U)));
    for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += S2_BASIC_BLOCK_L1) {
        uint64_t s2L1RealSize =
            s2GmOffset + S2_BASIC_BLOCK_L1 > s2ProcessSize ? s2ProcessSize - s2GmOffset : S2_BASIC_BLOCK_L1;
        for (uint64_t kGmOffset = 0; kGmOffset < kProcessSize; kGmOffset += D_BASIC_BLOCK_L1) {
            WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            KeyNd2Nz(s2L1RealSize, s2GmOffset, kGmOffset, runInfo);

            SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += M_BASIC_BLOCK_L1) {
                uint64_t s1gL1RealSize =
                    s1gGmOffset + M_BASIC_BLOCK_L1 > s1gProcessSize ? s1gProcessSize - s1gGmOffset : M_BASIC_BLOCK_L1;
                if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                    queryBufIdx_ = kGmOffset / D_BASIC_BLOCK_L1;
                    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + (queryBufIdx_ & (QUERY_BUF_NUM - 1U)));
                    QueryNd2Nz(s1gL1RealSize, s1gGmOffset, kGmOffset, runInfo);
                    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                } else {
                    queryBufIdx_ = kGmOffset / D_BASIC_BLOCK_L1;
                }
                for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                    uint64_t s2L0RealSize =
                        s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
                    for (uint64_t kL1Offset = 0; kL1Offset < D_BASIC_BLOCK_L1; kL1Offset += D_BASIC_BLOCK_L0) {
                        for (uint64_t s1gL1Offset = 0; s1gL1Offset < s1gL1RealSize; s1gL1Offset += M_BASIC_BLOCK_L0) {
                            uint64_t s1gL0RealSize = s1gL1Offset + M_BASIC_BLOCK_L0 > s1gL1RealSize ?
                                                         s1gL1RealSize - s1gL1Offset :
                                                         M_BASIC_BLOCK_L0;
                            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + (l0BufIdx_ & (L0_BUF_NUM - 1U)));
                            LoadQueryToL0a(s1gL1Offset, s1gL1RealSize, s1gL0RealSize, kL1Offset, runInfo);
                            LoadKeyToL0b(s2L1Offset, s2L1RealSize, s2L0RealSize, kL1Offset, runInfo);

                            SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                            WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                            ComuteL0c(s1gGmOffset + s1gL1Offset, s2GmOffset + s2L1Offset, kGmOffset + kL1Offset,
                                      s1gL0RealSize, s2L0RealSize, runInfo);
                            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + (l0BufIdx_ & (L0_BUF_NUM - 1U)));
                            l0BufIdx_++;
                        }
                    }
                }
                if (s2GmOffset + S2_BASIC_BLOCK_L1 >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
                    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + (queryBufIdx_ & (QUERY_BUF_NUM - 1U)));
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            keyL1BufIdx_++;
        }
    }
    Fixp(0, 0, s1gProcessSize, s2ProcessSize, runInfo);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + (l0cBufIdx_ & (L0_BUF_NUM - 1U)));
    l0cBufIdx_++;
    CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_CV_EVENT + (runInfo.loop & 1U));
    CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_CV_EVENT + (runInfo.loop & 1U) +
                                                        SICommon::AIV0_AIV1_OFFSET);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset, uint64_t kGmOffset,
                                               const SICommon::RunInfo &runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s2L1RealSize; // 行数
    nd2nzPara.dValue = D_BASIC_BLOCK_L1;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET],
             keyGm_[runInfo.tensorKeyOffset + s2GmOffset * constInfo_.headDim + kGmOffset], nd2nzPara);
}

// batch, n2, g, s1, d
template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::CopyQuerySegmentNd2Nz(LocalTensor<Q_T> queryL1Base, uint64_t l1RowOffset,
                                                            uint64_t l1RowAlign, uint64_t gmOffset, uint64_t nValue)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = nValue; // 行数
    nd2nzPara.dValue = D_BASIC_BLOCK_L1;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = l1RowAlign; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(queryL1Base[l1RowOffset * BLOCK_CUBE], queryGm_[gmOffset], nd2nzPara);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gGmOffset, uint64_t kGmOffset,
                                                 const SICommon::RunInfo &runInfo)
{
    LocalTensor<Q_T> queryL1Base = queryL1_[(queryBufIdx_ & (QUERY_BUF_NUM - 1U)) * QUERY_BUFFER_OFFSET];
    uint64_t l1RowAlign = CeilAlign(s1gL1RealSize, (uint64_t)BLOCK_CUBE);
    uint64_t logicalMStart = static_cast<uint64_t>(runInfo.gS1Idx) * constInfo_.mBaseSize + s1gGmOffset;
    uint64_t queryBaseOffset = runInfo.tensorQueryOffset;
    if (runInfo.actS1Size == constInfo_.qSeqSize) {
        CopyQuerySegmentNd2Nz(queryL1Base, 0, l1RowAlign,
                              queryBaseOffset + logicalMStart * constInfo_.headDim + kGmOffset, s1gL1RealSize);
        return;
    }

    uint64_t copiedRows = 0;
    uint64_t logicalMEnd = logicalMStart + s1gL1RealSize;
    while (logicalMStart < logicalMEnd) {
        uint64_t globalGIdx = logicalMStart / runInfo.actS1Size;
        uint64_t globalS1Idx = logicalMStart % runInfo.actS1Size;
        uint64_t copyRows = Min(logicalMEnd - logicalMStart, static_cast<uint64_t>(runInfo.actS1Size) - globalS1Idx);
        uint64_t gmOffset =
            queryBaseOffset + (globalGIdx * constInfo_.qSeqSize + globalS1Idx) * constInfo_.headDim + kGmOffset;
        CopyQuerySegmentNd2Nz(queryL1Base, copiedRows, l1RowAlign, gmOffset, copyRows);
        logicalMStart += copyRows;
        copiedRows += copyRows;
    }
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize,
                                                     uint64_t s1gL0RealSize, uint64_t kL1Offset,
                                                     const SICommon::RunInfo &runInfo)
{
    LoadData2DParamsV2 loadData2DParamsV2;
    loadData2DParamsV2.mStartPosition = CeilDiv(s1gL1Offset, BLOCK_CUBE);
    loadData2DParamsV2.kStartPosition = CeilDiv(kL1Offset, BLOCK_CUBE);
    loadData2DParamsV2.mStep = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.kStep = CeilDiv(D_BASIC_BLOCK_L0, BF16_BLOCK_CUBE);
    loadData2DParamsV2.srcStride = CeilDiv(s1gL1RealSize, BLOCK_CUBE);
    loadData2DParamsV2.dstStride = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.ifTranspose = false;

    LoadData(queryL0_[(l0BufIdx_ & (L0_BUF_NUM - 1U)) * L0A_BUFFER_OFFSET],
             queryL1_[(queryBufIdx_ & (QUERY_BUF_NUM - 1U)) * QUERY_BUFFER_OFFSET], loadData2DParamsV2);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                                   uint64_t kL1Offset, const SICommon::RunInfo &runInfo)
{
    LoadData2DParamsV2 loadData2DParamsV2;
    loadData2DParamsV2.mStartPosition = CeilDiv(s2L1Offset, BLOCK_CUBE);
    loadData2DParamsV2.kStartPosition = CeilDiv(kL1Offset, BLOCK_CUBE);
    loadData2DParamsV2.mStep = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.kStep = CeilDiv(D_BASIC_BLOCK_L0, BF16_BLOCK_CUBE);
    loadData2DParamsV2.srcStride = CeilDiv(s2L1RealSize, BLOCK_CUBE);
    loadData2DParamsV2.dstStride = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.ifTranspose = false;

    LoadData(keyL0_[(l0BufIdx_ & (L0_BUF_NUM - 1U)) * L0B_BUFFER_OFFSET],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET], loadData2DParamsV2);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::ComuteL0c(uint64_t s1gOffset, uint64_t s2Offset, uint64_t kGmOffset,
                                                uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                const SICommon::RunInfo &runInfo)
{
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = D_BASIC_BLOCK_L0;
    if (kGmOffset == 0) {
        mmadParams.cmatrixInitVal = true;
    } else {
        mmadParams.cmatrixInitVal = false;
    }
    mmadParams.cmatrixSource = false;
    uint64_t offset = (l0cBufIdx_ & (L0_BUF_NUM - 1U)) * L0C_BUFFER_OFFSET +
                      s2Offset * static_cast<uint64_t>(mmadParams.m) + s1gOffset * S2_BASIC_BLOCK;
    Mmad(cL0_[offset], queryL0_[(l0BufIdx_ & (L0_BUF_NUM - 1U)) * L0A_BUFFER_OFFSET],
         keyL0_[(l0BufIdx_ & (L0_BUF_NUM - 1U)) * L0B_BUFFER_OFFSET], mmadParams);
    if ((mmadParams.m / BLOCK_CUBE) * (mmadParams.n / BLOCK_CUBE) < PIPE_M_BARRIER_THRESHOLD) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                           uint64_t s2L0RealSize, const SICommon::RunInfo &runInfo)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT + (l0cBufIdx_ & (L0_BUF_NUM - 1U)));
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT + (l0cBufIdx_ & (L0_BUF_NUM - 1U)));

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;

    uint32_t mSize = (s1gL0RealSize + 1) >> 1 << 1;
    uint32_t nSize = (s2L0RealSize + 7) >> 3 << 3; // 32B对齐
    fixpipeParams.nSize = nSize;
    fixpipeParams.mSize = mSize;
    fixpipeParams.srcStride = ((fixpipeParams.mSize + BLOCK_CUBE - 1U) / BLOCK_CUBE) * BLOCK_CUBE;
    fixpipeParams.dstStride = constInfo_.s2BaseSize;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    // 将matmul结果从L0C搬运到UB。
    Fixpipe<QK_T, float, SI_CFG_ROW_MAJOR_UB>(
        mm1ResUB_[(runInfo.loop & 1U) * ((constInfo_.mBaseSize + 1U) >> 1U) * constInfo_.s2BaseSize],
        cL0_[(l0cBufIdx_ & (L0_BUF_NUM - 1U)) * L0C_BUFFER_OFFSET], fixpipeParams);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 2);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 3);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);

    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::FreeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 2);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 3);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}
} // namespace SIKernel
#endif
