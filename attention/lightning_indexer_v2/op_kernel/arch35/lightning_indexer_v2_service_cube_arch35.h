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
 * \file lightning_indexer_v2_service_cube_arch35.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef LIGHTNING_INDEXER_V2_SERVICE_CUBE_ARCH35_H
#define LIGHTNING_INDEXER_V2_SERVICE_CUBE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_v2_common_arch35.h"

namespace LIV2Kernel {
using namespace LIV2Common;
template <typename LIT>
class LightningIndexerV2ServiceCube {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using QK_T = typename LIT::queryKeyType;
    using SCORE_T = typename LIT::scoreType;

    __aicore__ inline LightningIndexerV2ServiceCube(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                               const GlobalTensor<Q_T> &queryGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const LIV2Common::RunInfo &liV2CubeRunInfo);

    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_NUM = 2;

    // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;
    static constexpr uint32_t LIV2_QUERY_MTE1_MTE2_EVENT = EVENT_ID5;
    static constexpr uint32_t LIV2_KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t LIV2_M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;
    static constexpr uint32_t LIV2_FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID3;

    static constexpr uint64_t M_BASIC_BLOCK = 128;
    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S2_BASIC_BLOCK = 128;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t BLOCK_CUBE = 16;
    static constexpr uint64_t FP16_BLOCK_CUBE = 16;
    // ROW_MAJOR: 使能NZ2ND，输出数据格式为ND格式; true: 用于用户指定目的地址的位置是否是UB
    static constexpr FixpipeConfig LI_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK * D_BASIC_BLOCK;

protected:
    __aicore__ inline void Fixp(uint64_t liV2S1GmOffset, uint64_t liV2S2GmOffset, uint64_t s1gL0RealSize,
                                uint64_t s2L0RealSize, uint64_t s1gSizeAlign2G,
                                const LIV2Common::RunInfo &liV2CubeRunInfo);
    __aicore__ inline void ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                      const LIV2Common::RunInfo &liV2CubeRunInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                        const LIV2Common::RunInfo &liV2CubeRunInfo);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL0Offset, uint64_t s1gL1RealSize,
                                          uint64_t s1gL0RealSize, const LIV2Common::RunInfo &liV2CubeRunInfo);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gL1Offset,
                                      const LIV2Common::RunInfo &liV2CubeRunInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, uint64_t liV2S2GmOffset,
                                    const LIV2Common::RunInfo &liV2CubeRunInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t liV2S2GmOffset,
                                         const LIV2Common::RunInfo &liV2CubeRunInfo);
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
    uint64_t liV2QueryMte2BufferIndex = 0;
    uint64_t liV2QueryMte1BufferIndex = 0;
    uint64_t l0BufIdx_ = 0;

    ConstInfo liV2CubeConstInfo;

    uint64_t queryBufferOffset_ = 0;
    uint64_t l0abBufferOffset_ = 0;
    uint64_t l0cBufferOffset_ = 0;

private:
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
};

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::InitParams(const ConstInfo &constInfo)
{
    liV2CubeConstInfo = constInfo;
    queryBufferOffset_ = M_BASIC_BLOCK * D_BASIC_BLOCK;
    l0abBufferOffset_ = M_BASIC_BLOCK * D_BASIC_BLOCK_L0;
    l0cBufferOffset_ = M_BASIC_BLOCK * S2_BASIC_BLOCK_L0;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::InitBuffers(TPipe *pipe)
{
    // 大小：2(开dB) * 2 * 64 * 128 * 4 = 128KB
    pipe->InitBuffer(bufUB_, 2 * CeilDiv(M_BASIC_BLOCK, 2) * liV2CubeConstInfo.s2BaseSize * sizeof(QK_T));
    mm1ResUB_ = bufUB_.Get<QK_T>();
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * M_BASIC_BLOCK * D_BASIC_BLOCK_L0 * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * M_BASIC_BLOCK * S2_BASIC_BLOCK_L0 * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm,
                                                                               const GlobalTensor<K_T> &keyGm,
                                                                               const GlobalTensor<Q_T> &queryGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::ComputeMm1(const LIV2Common::RunInfo &liV2CubeRunInfo)
{
    CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LIV2Common::ConstInfo::CROSS_VC_EVENT +
                                                                      liV2CubeRunInfo.loop % 2);
    CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(
        LIV2Common::ConstInfo::CROSS_VC_EVENT + liV2CubeRunInfo.loop % 2 + LIV2Common::ConstInfo::AIV0_AIV1_OFFSET);
    uint64_t s2GmBaseOffset = liV2CubeRunInfo.s2Idx * liV2CubeConstInfo.s2BaseSize;
    uint64_t s1gProcessSize = liV2CubeRunInfo.actMBaseSize;
    uint64_t s2ProcessSize = liV2CubeRunInfo.actualSingleProcessSInnerSize;
    for (uint64_t liV2S2GmOffset = 0; liV2S2GmOffset < s2ProcessSize; liV2S2GmOffset += S2_BASIC_BLOCK) {
        WaitFlag<HardEvent::MTE1_MTE2>(LIV2_KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        uint64_t s2L1RealSize =
            liV2S2GmOffset + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - liV2S2GmOffset : S2_BASIC_BLOCK;
        if (PAGE_ATTENTION) {
            KeyNd2NzForPA(s2L1RealSize, s2GmBaseOffset + liV2S2GmOffset, liV2CubeRunInfo);
        } else {
            KeyNd2Nz(s2L1RealSize, liV2S2GmOffset, liV2CubeRunInfo);
        }

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        // s1gProcessSize当前必定不会超过2倍的s1g basic block
        for (uint64_t liV2S1GmOffset = 0; liV2S1GmOffset < s1gProcessSize; liV2S1GmOffset += M_BASIC_BLOCK) {
            uint64_t s1gL1RealSize =
                liV2S1GmOffset + M_BASIC_BLOCK > s1gProcessSize ? s1gProcessSize - liV2S1GmOffset : M_BASIC_BLOCK;
            uint64_t s1gL1SizeAlign2G = CeilAlign(s1gL1RealSize, 2 * liV2CubeConstInfo.gSize);
            uint64_t s1gL1SizeAlign = CeilAlign(s1gL1SizeAlign2G, BLOCK_CUBE);
            if (liV2CubeRunInfo.isFirstS2InnerLoop && liV2S2GmOffset == 0) {
                liV2QueryMte2BufferIndex++;
                liV2QueryMte1BufferIndex = liV2QueryMte2BufferIndex;
                WaitFlag<HardEvent::MTE1_MTE2>(LIV2_QUERY_MTE1_MTE2_EVENT + liV2QueryMte2BufferIndex % QUERY_BUF_NUM);
                QueryNd2Nz(s1gL1RealSize, liV2S1GmOffset, liV2CubeRunInfo);
                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            } else {
                liV2QueryMte1BufferIndex =
                    liV2QueryMte2BufferIndex - (CeilDiv(s1gProcessSize, M_BASIC_BLOCK) - 1 - (liV2S1GmOffset > 0));
            }
            for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                uint64_t s2L0RealSize =
                    s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
                for (uint64_t s1gL1Offset = 0; s1gL1Offset < s1gL1SizeAlign; s1gL1Offset += M_BASIC_BLOCK) {
                    WaitFlag<HardEvent::M_MTE1>(LIV2_M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    uint64_t liV2S1gL0RealSize =
                        s1gL1Offset + M_BASIC_BLOCK > s1gL1SizeAlign ? s1gL1SizeAlign - s1gL1Offset : M_BASIC_BLOCK;
                    LoadQueryToL0a(liV2S1GmOffset, s1gL1Offset, s1gL1SizeAlign, liV2S1gL0RealSize, liV2CubeRunInfo);
                    LoadKeyToL0b(s2L1Offset, s2L1RealSize, s2L0RealSize, liV2CubeRunInfo);

                    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                    WaitFlag<HardEvent::FIX_M>(LIV2_FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    ComputeL0c(liV2S1gL0RealSize, s2L0RealSize, liV2CubeRunInfo);

                    SetFlag<HardEvent::M_MTE1>(LIV2_M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

                    Fixp(liV2S1GmOffset + s1gL1Offset, liV2S2GmOffset + s2L1Offset, liV2S1gL0RealSize, s2L0RealSize,
                         s1gL1SizeAlign2G, liV2CubeRunInfo);
                    SetFlag<HardEvent::FIX_M>(LIV2_FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    l0BufIdx_++;
                }
            }
            if (liV2S2GmOffset + S2_BASIC_BLOCK >= s2ProcessSize && liV2CubeRunInfo.isLastS2InnerLoop) {
                SetFlag<HardEvent::MTE1_MTE2>(LIV2_QUERY_MTE1_MTE2_EVENT + liV2QueryMte1BufferIndex % QUERY_BUF_NUM);
            }
        }
        SetFlag<HardEvent::MTE1_MTE2>(LIV2_KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        keyL1BufIdx_++;
    }
    CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LIV2Common::ConstInfo::CROSS_CV_EVENT +
                                                                     liV2CubeRunInfo.loop % 2);
    CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(
        LIV2Common::ConstInfo::CROSS_CV_EVENT + liV2CubeRunInfo.loop % 2 + LIV2Common::ConstInfo::AIV0_AIV1_OFFSET);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::KeyNd2Nz(uint64_t s2L1RealSize, uint64_t liV2S2GmOffset,
                                                                    const LIV2Common::RunInfo &liV2CubeRunInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s2L1RealSize; // 行数
    nd2nzPara.dValue = liV2CubeConstInfo.headDim;
    nd2nzPara.srcDValue = liV2CubeConstInfo.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET],
             keyGm_[liV2CubeRunInfo.tensorKeyOffset + liV2S2GmOffset * liV2CubeConstInfo.headDim], nd2nzPara);
}

// blkNum, blkSize, N2, D
template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t liV2S2GmOffset,
                                                                         const LIV2Common::RunInfo &liV2CubeRunInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + liV2S2GmOffset) / liV2CubeConstInfo.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + liV2S2GmOffset) % liV2CubeConstInfo.kCacheBlockSize;
        uint64_t keyGmOffset =
            blkTableGm_.GetValue(liV2CubeRunInfo.bIdx * liV2CubeConstInfo.maxBlockNumPerBatch + s2BlkId) *
                liV2CubeConstInfo.keyStride0 +
            s2BlkOffset * liV2CubeConstInfo.headDim;

        uint64_t s2Mte2Size = s2L1RealSize - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= liV2CubeConstInfo.kCacheBlockSize ?
                         liV2CubeConstInfo.kCacheBlockSize - s2BlkOffset :
                         s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size; // 行数
        nd2nzPara.dValue = liV2CubeConstInfo.headDim;
        nd2nzPara.srcDValue = liV2CubeConstInfo.headDim;
        nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + s2L1Offset * FP16_BLOCK_CUBE],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

// batch, s1, n2, g, d
template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t liV2S1GmOffset,
                                                                      const LIV2Common::RunInfo &liV2CubeRunInfo)
{
    uint64_t dstNzC0Stride = CeilAlign(s1gL1RealSize, liV2CubeConstInfo.gSize * 2);
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize; // 行数
    nd2nzPara.dValue = liV2CubeConstInfo.headDim;
    nd2nzPara.srcDValue = liV2CubeConstInfo.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(dstNzC0Stride, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(queryL1_[(liV2QueryMte2BufferIndex % QUERY_BUF_NUM) * queryBufferOffset_],
             queryGm_[liV2CubeRunInfo.tensorQueryOffset + liV2S1GmOffset * liV2CubeConstInfo.headDim], nd2nzPara);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::LoadQueryToL0a(uint64_t liV2S1GmOffset, uint64_t s1gL1Offset,
                                                                          uint64_t s1gL1RealSize,
                                                                          uint64_t s1gL0RealSize,
                                                                          const LIV2Common::RunInfo &liV2CubeRunInfo)
{
    LoadData2DParamsV2 loadData2DParamsV2;
    loadData2DParamsV2.mStartPosition = CeilDiv(s1gL1Offset, BLOCK_CUBE);
    loadData2DParamsV2.kStartPosition = 0;
    loadData2DParamsV2.mStep = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.kStep = CeilDiv(liV2CubeConstInfo.headDim, FP16_BLOCK_CUBE);
    loadData2DParamsV2.srcStride = CeilDiv(s1gL1RealSize, BLOCK_CUBE);
    loadData2DParamsV2.dstStride = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.ifTranspose = false;

    LoadData(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * l0abBufferOffset_],
             queryL1_[(liV2QueryMte1BufferIndex % QUERY_BUF_NUM) * queryBufferOffset_], loadData2DParamsV2);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize,
                                                                        uint64_t s2L0RealSize,
                                                                        const LIV2Common::RunInfo &liV2CubeRunInfo)
{
    LoadData2DParamsV2 loadData2DParamsV2;
    loadData2DParamsV2.mStartPosition = CeilDiv(s2L1Offset, BLOCK_CUBE);
    loadData2DParamsV2.kStartPosition = 0;
    loadData2DParamsV2.mStep = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.kStep = CeilDiv(liV2CubeConstInfo.headDim, FP16_BLOCK_CUBE);
    loadData2DParamsV2.srcStride = CeilDiv(s2L1RealSize, BLOCK_CUBE);
    loadData2DParamsV2.dstStride = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.ifTranspose = false;

    LoadData(keyL0_[(l0BufIdx_ % L0_BUF_NUM) * l0abBufferOffset_],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET], loadData2DParamsV2);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                                      const LIV2Common::RunInfo &liV2CubeRunInfo)
{
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = liV2CubeConstInfo.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * l0cBufferOffset_], queryL0_[(l0BufIdx_ % L0_BUF_NUM) * l0abBufferOffset_],
         keyL0_[(l0BufIdx_ % L0_BUF_NUM) * l0abBufferOffset_], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::Fixp(uint64_t liV2S1GmOffset, uint64_t liV2S2GmOffset,
                                                                uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                                uint64_t s1gSizeAlign2G,
                                                                const LIV2Common::RunInfo &liV2CubeRunInfo)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);
    uint32_t liV2NSize = (s2L0RealSize + 7) >> 3 << 3;
    // L0C->UB
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐
    fixpipeParams.nSize = liV2NSize;
    // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
    fixpipeParams.mSize = s1gSizeAlign2G;
    // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
    // //源NZ矩阵中相邻Z排布的起始地址偏移
    fixpipeParams.srcStride = (s1gL0RealSize + 1) >> 1 << 1;
    // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除
    fixpipeParams.dstStride = UB_BANK_DEPTH_STRIDE / sizeof(QK_T);
    // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
    fixpipeParams.dualDstCtl = 1;
    if (liV2NSize <= (256 / sizeof(float))) {
        // N方向小于一个bank(256B), 只需搬一个ND块, 且不用补齐
        fixpipeParams.nSize = liV2NSize;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
    } else {
        // N方向在(256B, 512B]范围, 分2个ND搬
        fixpipeParams.nSize = S2_BASIC_BLOCK_L0 / 2;
        fixpipeParams.params.ndNum = 2;
        fixpipeParams.params.srcNdStride = ((fixpipeParams.mSize + 15) / 16) * fixpipeParams.nSize;
        fixpipeParams.params.dstNdStride = liV2CubeConstInfo.s2BaseSize * M_BASIC_BLOCK / 2;
    }

    // 将matmul结果从L0C搬运到UB
    Fixpipe<float, float, LI_CFG_ROW_MAJOR_UB>(mm1ResUB_[(liV2CubeRunInfo.loop % 2) * liV2CubeConstInfo.s2BaseSize / 2],
                                               cL0_[(l0BufIdx_ % L0_BUF_NUM) * l0cBufferOffset_], fixpipeParams);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(LIV2_KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(LIV2_KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(LIV2_KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(LIV2_QUERY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(LIV2_QUERY_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(LIV2_M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(LIV2_M_MTE1_EVENT + 1);

    SetFlag<HardEvent::FIX_M>(LIV2_FIX_M_EVENT + 0);
    SetFlag<HardEvent::FIX_M>(LIV2_FIX_M_EVENT + 1);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::FreeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(LIV2_KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(LIV2_KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(LIV2_KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(LIV2_QUERY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(LIV2_QUERY_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(LIV2_M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(LIV2_M_MTE1_EVENT + 1);

    WaitFlag<HardEvent::FIX_M>(LIV2_FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(LIV2_FIX_M_EVENT + 1);
}
} // namespace LIV2Kernel
#endif
