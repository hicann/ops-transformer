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
 * \file lightning_indexer_service_cube_arch35.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef LIGHTNING_INDEXER_SERVICE_CUBE_ARCH35_H
#define LIGHTNING_INDEXER_SERVICE_CUBE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../lightning_indexer_common.h"

namespace LIKernel {
using namespace LICommon;
template <typename LIT>
class LightningIndexerServiceCube {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using SCORE_T = uint32_t;

    __aicore__ inline LightningIndexerServiceCube(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                               const GlobalTensor<Q_T> &queryGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const LICommon::RunInfo &liCubeRunInfo);

    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5; // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;

    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID3;

    static constexpr uint64_t M_BASIC_BLOCK = 256;
    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S2_BASIC_BLOCK = 128;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t FP16_BLOCK_CUBE = 16;
    static constexpr FixpipeConfig LI_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t L0AB_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;

protected:
    __aicore__ inline void Fixp(uint64_t liS1GmOffset, uint64_t liS2GmOffset, uint64_t s1gL0RealSize,
                                uint64_t s2L0RealSize, const LICommon::RunInfo &liCubeRunInfo);
    __aicore__ inline void ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                      const LICommon::RunInfo &liCubeRunInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                        const LICommon::RunInfo &liCubeRunInfo);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL0Offset, uint64_t s1gL1RealSize,
                                          uint64_t s1gL0RealSize, const LICommon::RunInfo &liCubeRunInfo);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gL1Offset,
                                      const LICommon::RunInfo &liCubeRunInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, uint64_t liS2GmOffset,
                                    const LICommon::RunInfo &liCubeRunInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t liS2GmOffset,
                                         const LICommon::RunInfo &liCubeRunInfo);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;

    TBuf<TPosition::A1> bufQL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<Q_T> queryL1_;
    LocalTensor<K_T> keyL1_;

    TBuf<TPosition::A2> bufQL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<Q_T> queryL0_;
    LocalTensor<K_T> keyL0_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<float> cL0_;

    TBuf<TPosition::VECCALC> bufUB_;
    LocalTensor<float> mm1ResUB_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t liQueryMte2BufferIndex = 0;
    uint64_t liQueryMte1BufferIndex = 0;
    uint64_t l0BufIdx_ = 0;
    uint64_t kl0BufIdx_ = 0;

    ConstInfo liCubeConstInfo;

private:
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
};

template <typename LIT>
__aicore__ inline void LightningIndexerServiceCube<LIT>::InitParams(const ConstInfo &constInfo)
{
    liCubeConstInfo = constInfo;
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceCube<LIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufUB_, 2 * CeilDiv(liCubeConstInfo.mBaseSize, 2) * liCubeConstInfo.s2BaseSize *
                                 sizeof(float)); // 大小：2(开dB) * 2 * 64 * 128 * 4 = 128KB
    mm1ResUB_ = bufUB_.Get<float>();
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceCube<LIT>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm,
                                                                             const GlobalTensor<K_T> &keyGm,
                                                                             const GlobalTensor<Q_T> &queryGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceCube<LIT>::ComputeMm1(const LICommon::RunInfo &liCubeRunInfo)
{
    CrossCoreWaitFlag<LICommon::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LICommon::ConstInfo::CROSS_VC_EVENT +
                                                                    liCubeRunInfo.loop % 2);
    CrossCoreWaitFlag<LICommon::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(
        LICommon::ConstInfo::CROSS_VC_EVENT + liCubeRunInfo.loop % 2 + LICommon::ConstInfo::AIV0_AIV1_OFFSET);
    uint64_t s2GmBaseOffset = liCubeRunInfo.s2Idx * liCubeConstInfo.s2BaseSize;
    uint64_t s1gProcessSize = liCubeRunInfo.actMBaseSize;
    uint64_t liS2ProcessSize = liCubeRunInfo.actualSingleProcessSInnerSize;
    for (uint64_t liS2GmOffset = 0; liS2GmOffset < liS2ProcessSize; liS2GmOffset += S2_BASIC_BLOCK) {
        WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        uint64_t liS2L1RealSize =
            liS2GmOffset + S2_BASIC_BLOCK > liS2ProcessSize ? liS2ProcessSize - liS2GmOffset : S2_BASIC_BLOCK;
        if (PAGE_ATTENTION) {
            KeyNd2NzForPA(liS2L1RealSize, s2GmBaseOffset + liS2GmOffset, liCubeRunInfo);
        } else {
            KeyNd2Nz(liS2L1RealSize, liS2GmOffset, liCubeRunInfo);
        }

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        // s1gProcessSize当前必定不会超过2倍的s1g basic block
        for (uint64_t liS1GmOffset = 0; liS1GmOffset < s1gProcessSize; liS1GmOffset += liCubeConstInfo.mBaseSize) {
            uint64_t s1gL1RealSize = liS1GmOffset + liCubeConstInfo.mBaseSize > s1gProcessSize ?
                                         s1gProcessSize - liS1GmOffset :
                                         liCubeConstInfo.mBaseSize;
            uint64_t liS1gL1SizeAlign2G = CeilAlign(s1gL1RealSize, 2 * liCubeConstInfo.gSize);
            if (liCubeRunInfo.isFirstS2InnerLoop && liS2GmOffset == 0) {
                liQueryMte2BufferIndex++;
                liQueryMte1BufferIndex = liQueryMte2BufferIndex;
                WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + liQueryMte2BufferIndex % QUERY_BUF_NUM);
                QueryNd2Nz(s1gL1RealSize, liS1GmOffset, liCubeRunInfo);
                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            } else {
                liQueryMte1BufferIndex = liQueryMte2BufferIndex -
                                         (CeilDiv(s1gProcessSize, liCubeConstInfo.mBaseSize) - 1 - (liS1GmOffset > 0));
            }
            for (uint64_t s2L1Offset = 0; s2L1Offset < liS2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                uint64_t liS2L0RealSize =
                    s2L1Offset + S2_BASIC_BLOCK_L0 > liS2L1RealSize ? liS2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;

                uint64_t l0Stride = liCubeConstInfo.mBaseSize;
                if (liCubeConstInfo.splitMFlag) {
                    l0Stride /= 2;
                }

                for (uint64_t s1gL1Offset = 0; s1gL1Offset < liS1gL1SizeAlign2G; s1gL1Offset += l0Stride) {
                    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    uint64_t s1gL0RealSize = s1gL1Offset + liCubeConstInfo.mBaseSize > liS1gL1SizeAlign2G ?
                                                 liS1gL1SizeAlign2G - s1gL1Offset :
                                                 liCubeConstInfo.mBaseSize;
                    if (liCubeConstInfo.splitMFlag) {
                        s1gL0RealSize = 128; // g=64, topK=2k时固定m=128
                    }
                    LoadQueryToL0a(liS1GmOffset, s1gL1Offset, liS1gL1SizeAlign2G, s1gL0RealSize, liCubeRunInfo);
                    if (s1gL1Offset == 0) {
                        LoadKeyToL0b(s2L1Offset, liS2L1RealSize, liS2L0RealSize, liCubeRunInfo);
                    }

                    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    ComputeL0c(s1gL0RealSize, liS2L0RealSize, liCubeRunInfo);

                    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

                    bool lastIter = s1gL1Offset + l0Stride >= liS1gL1SizeAlign2G;
                    if (lastIter) {
                        kl0BufIdx_++;
                    }

                    Fixp(liS1GmOffset + s1gL1Offset, liS2GmOffset + s2L1Offset, s1gL0RealSize, liS2L0RealSize,
                         liCubeRunInfo);
                    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    l0BufIdx_++;
                }
            }
            if (liS2GmOffset + S2_BASIC_BLOCK >= liS2ProcessSize && liCubeRunInfo.isLastS2InnerLoop) {
                SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + liQueryMte1BufferIndex % QUERY_BUF_NUM);
            }
        }
        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        keyL1BufIdx_++;
    }
    CrossCoreSetFlag<LICommon::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LICommon::ConstInfo::CROSS_CV_EVENT +
                                                                   liCubeRunInfo.loop % 2);
    CrossCoreSetFlag<LICommon::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(
        LICommon::ConstInfo::CROSS_CV_EVENT + liCubeRunInfo.loop % 2 + LICommon::ConstInfo::AIV0_AIV1_OFFSET);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceCube<LIT>::KeyNd2Nz(uint64_t s2L1RealSize, uint64_t liS2GmOffset,
                                                                  const LICommon::RunInfo &liCubeRunInfo)
{
    Nd2NzParams liNd2nzPara;
    liNd2nzPara.ndNum = 1;
    liNd2nzPara.nValue = s2L1RealSize; // 行数
    liNd2nzPara.dValue = liCubeConstInfo.headDim;
    liNd2nzPara.srcDValue = liCubeConstInfo.headDim;
    liNd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    liNd2nzPara.srcNdMatrixStride = 0;
    liNd2nzPara.dstNzNStride = 1;
    liNd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET],
             keyGm_[liCubeRunInfo.tensorKeyOffset + liS2GmOffset * liCubeConstInfo.headDim], liNd2nzPara);
}

// blkNum, blkSize, N2, D
template <typename LIT>
__aicore__ inline void LightningIndexerServiceCube<LIT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t liS2GmOffset,
                                                                       const LICommon::RunInfo &liCubeRunInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + liS2GmOffset) / liCubeConstInfo.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + liS2GmOffset) % liCubeConstInfo.kCacheBlockSize;
        uint64_t keyGmOffset =
            blkTableGm_.GetValue(liCubeRunInfo.bIdx * liCubeConstInfo.maxBlockNumPerBatch + s2BlkId) *
                liCubeConstInfo.keyStride0 +
            s2BlkOffset * liCubeConstInfo.headDim;

        uint64_t liS2Mte2Size = s2L1RealSize - s2L1Offset;
        liS2Mte2Size = s2BlkOffset + liS2Mte2Size >= liCubeConstInfo.kCacheBlockSize ?
                           liCubeConstInfo.kCacheBlockSize - s2BlkOffset :
                           liS2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = liS2Mte2Size; // 行数
        nd2nzPara.dValue = liCubeConstInfo.headDim;
        nd2nzPara.srcDValue = liCubeConstInfo.headDim;
        nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + s2L1Offset * FP16_BLOCK_CUBE],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += liS2Mte2Size;
    }
}

// batch, s1, n2, g, d
template <typename LIT>
__aicore__ inline void LightningIndexerServiceCube<LIT>::QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t liS1GmOffset,
                                                                    const LICommon::RunInfo &liCubeRunInfo)
{
    uint64_t dstNzC0Stride = CeilAlign(s1gL1RealSize, 2 * liCubeConstInfo.gSize);
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize; // 行数
    nd2nzPara.srcDValue = liCubeConstInfo.headDim;
    nd2nzPara.dValue = liCubeConstInfo.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(dstNzC0Stride, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(queryL1_[(liQueryMte2BufferIndex % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
             queryGm_[liCubeRunInfo.tensorQueryOffset + liS1GmOffset * liCubeConstInfo.headDim], nd2nzPara);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceCube<LIT>::LoadQueryToL0a(uint64_t liS1GmOffset, uint64_t s1gL1Offset,
                                                                        uint64_t s1gL1RealSize, uint64_t s1gL0RealSize,
                                                                        const LICommon::RunInfo &liCubeRunInfo)
{
    LoadData2DParamsV2 loadData2DParamsV2;
    if (liCubeConstInfo.splitMFlag && liCubeRunInfo.actMBaseSize > 128) { // 非尾块，切M
        uint64_t dstOffset = 0;
        loadData2DParamsV2.kStartPosition = 0;
        loadData2DParamsV2.mStep = CeilDiv(64, BLOCK_CUBE);
        loadData2DParamsV2.kStep = CeilDiv(liCubeConstInfo.headDim, FP16_BLOCK_CUBE);
        loadData2DParamsV2.srcStride = CeilDiv(s1gL1RealSize, BLOCK_CUBE);
        loadData2DParamsV2.dstStride = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
        loadData2DParamsV2.ifTranspose = false;
        for (int i = 0; i < 2; i++) {
            loadData2DParamsV2.mStartPosition = CeilDiv((s1gL1Offset / 2) + i * 128, BLOCK_CUBE);
            dstOffset = i * 64 * 16;

            LoadData(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET + dstOffset],
                     queryL1_[(liQueryMte1BufferIndex % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET], loadData2DParamsV2);
        }
    } else {
        loadData2DParamsV2.mStartPosition = CeilDiv(s1gL1Offset, BLOCK_CUBE);
        loadData2DParamsV2.kStartPosition = 0;
        loadData2DParamsV2.mStep = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
        loadData2DParamsV2.kStep = CeilDiv(liCubeConstInfo.headDim, FP16_BLOCK_CUBE);
        loadData2DParamsV2.srcStride = CeilDiv(s1gL1RealSize, BLOCK_CUBE);
        loadData2DParamsV2.dstStride = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
        loadData2DParamsV2.ifTranspose = false;

        LoadData(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
                 queryL1_[(liQueryMte1BufferIndex % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET], loadData2DParamsV2);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceCube<LIT>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize,
                                                                      uint64_t s2L0RealSize,
                                                                      const LICommon::RunInfo &liCubeRunInfo)
{
    LoadData2DParamsV2 loadData2DParamsV2;
    loadData2DParamsV2.mStartPosition = CeilDiv(s2L1Offset, BLOCK_CUBE);
    loadData2DParamsV2.kStartPosition = 0;
    loadData2DParamsV2.mStep = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.kStep = CeilDiv(liCubeConstInfo.headDim, FP16_BLOCK_CUBE);
    loadData2DParamsV2.srcStride = CeilDiv(s2L1RealSize, BLOCK_CUBE);
    loadData2DParamsV2.dstStride = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.ifTranspose = false;

    LoadData(keyL0_[(kl0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET], loadData2DParamsV2);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceCube<LIT>::ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                                    const LICommon::RunInfo &liCubeRunInfo)
{
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = liCubeConstInfo.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
         keyL0_[(kl0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceCube<LIT>::Fixp(uint64_t liS1GmOffset, uint64_t liS2GmOffset,
                                                              uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                              const LICommon::RunInfo &liCubeRunInfo)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);

    if constexpr (std::is_same<SCORE_T, uint32_t>::value) {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐
        fixpipeParams.nSize = (s2L0RealSize + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (s1gL0RealSize + 1) >> 1 << 1;
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dstStride = liCubeConstInfo.s2BaseSize;
        // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<float, float, LI_CFG_ROW_MAJOR_UB>(
            mm1ResUB_[(liCubeRunInfo.loop % 2) * CeilDiv(liCubeConstInfo.mBaseSize, 2) * liCubeConstInfo.s2BaseSize +
                      CeilDiv(liS1GmOffset, 2) * fixpipeParams.dstStride + liS2GmOffset],
            cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], fixpipeParams);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceCube<LIT>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);

    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceCube<LIT>::FreeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}
} // namespace LIKernel
#endif
