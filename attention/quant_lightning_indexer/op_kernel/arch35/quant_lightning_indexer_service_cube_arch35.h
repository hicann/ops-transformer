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
 * \file lightning_indexer_service_cube.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef QUANT_LIGHTNING_INDEXER_SERVICE_CUBE_H
#define QUANT_LIGHTNING_INDEXER_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../quant_lightning_indexer_common.h"

namespace QLIKernel {
using namespace QLICommon;
template <typename QLIT>
class QLIMatmul {
public:
    using Q_T = typename QLIT::queryType;
    using K_T = typename QLIT::keyType;
    using QK_T = typename QLIT::qkType;

    __aicore__ inline QLIMatmul(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                               const GlobalTensor<Q_T> &queryGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const QLICommon::RunInfo &runInfo);

    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5;         // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID3;

    static constexpr uint64_t M_BASIC_BLOCK = 256;
    static constexpr uint64_t D_BASIC_BLOCK = 128;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 256;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t FP8_BLOCK_CUBE = 32;
    static constexpr FixpipeConfig QLI_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};   // ROW_MAJOR: 使能NZ2ND，输出数据格式为ND格式; true: 用于用户指定目的地址的位置是否是UB

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t L0AB_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;

protected:
    __aicore__ inline void Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                uint64_t s2L0RealSize, const QLICommon::RunInfo &runInfo);
    __aicore__ inline void ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, const QLICommon::RunInfo &runInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                        const QLICommon::RunInfo &runInfo);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL0Offset, uint64_t s1gL1RealSize,
                                          uint64_t s1gL0RealSize, const QLICommon::RunInfo &runInfo);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gL1Offset, const QLICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset, const QLICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset, const QLICommon::RunInfo &runInfo);
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
    LocalTensor<QK_T> cL0_;

    TBuf<TPosition::VECCALC> bufUB_;
    LocalTensor<QK_T> mm1ResUB_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t queryL1Mte2BufIdx_ = 0;
    uint64_t queryL1Mte1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;

    bool isKeyCacheValid_ = false; // L1中是否有可复用的数据
    uint64_t keyGmStart_ = 0; // L1数据对应的GM S2偏移
    uint64_t keyLoadedSize_ = 0; // L1中实际加载的S2元素数量
    uint64_t s2BasicBlock_ = 128;
    uint64_t keyBufferOffset_ = 16384; // 128*128

    ConstInfo constInfo_;

private:
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;
};

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
    s2BasicBlock_ = (constInfo_.tSize <= 256) ? 256 : 128;
    keyBufferOffset_ = s2BasicBlock_ * D_BASIC_BLOCK; // 128*128
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufUB_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * constInfo_.s2BaseSize * sizeof(float));  //大小：2(开dB) * 2 * 64 * 128 * 4 = 128KB
    mm1ResUB_ = bufUB_.Get<QK_T>();
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * s2BasicBlock_ * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(float));
    cL0_ = bufL0C_.Get<QK_T>();
}

template <typename QLIT>
__aicore__ inline void
QLIMatmul<QLIT>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                   const GlobalTensor<Q_T> &queryGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ComputeMm1(const QLICommon::RunInfo &runInfo)
{
    CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_VC_EVENT + runInfo.loop % 2);
    CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_VC_EVENT + runInfo.loop % 2 + QLICommon::ConstInfo::AIV0_AIV1_OFFSET);
    uint64_t s2GmBaseOffset = runInfo.s2Idx * constInfo_.s2BaseSize;
    uint64_t s1gProcessSize = runInfo.actMBaseSize;
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    if (s2BasicBlock_ == 128) {
        for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += s2BasicBlock_) {
            WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            uint64_t s2L1RealSize =
                s2GmOffset + s2BasicBlock_ > s2ProcessSize ? s2ProcessSize - s2GmOffset : s2BasicBlock_;
            if (PAGE_ATTENTION) {
                KeyNd2NzForPA(s2L1RealSize, s2GmBaseOffset + s2GmOffset, runInfo);
            }else {
                KeyNd2Nz(s2L1RealSize, s2GmOffset, runInfo);
            }

            SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            // s1gProcessSize当前必定不会超过2倍的s1g basic block
            for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += constInfo_.mBaseSize) {
                uint64_t s1gL1RealSize =
                    s1gGmOffset + constInfo_.mBaseSize > s1gProcessSize ?
                                                        s1gProcessSize - s1gGmOffset :
                                                        constInfo_.mBaseSize;
                uint64_t s1gL1SizeAlign2G = CeilAlign(s1gL1RealSize, 2 * constInfo_.gSize);
                if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                    queryL1Mte2BufIdx_++;
                    queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
                    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                    QueryNd2Nz(s1gL1RealSize, s1gGmOffset, runInfo);
                    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                } else {
                    queryL1Mte1BufIdx_ =
                        queryL1Mte2BufIdx_ - (CeilDiv(s1gProcessSize, constInfo_.mBaseSize) - 1 - (s1gGmOffset > 0));
                }
                for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                    uint64_t s2L0RealSize =
                        s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
                    for (uint64_t s1gOffset = 0; s1gOffset < s1gL1SizeAlign2G; s1gOffset += constInfo_.mBaseSize) {
                        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        uint64_t s1gL0RealSize =
                            s1gOffset + constInfo_.mBaseSize > s1gL1SizeAlign2G ?
                                                   s1gL1SizeAlign2G - s1gOffset :
                                                   constInfo_.mBaseSize;
                        LoadQueryToL0a(s1gGmOffset, s1gOffset, s1gL1SizeAlign2G, s1gL0RealSize, runInfo);
                        LoadKeyToL0b(s2L1Offset, s2L1RealSize, s2L0RealSize, runInfo);

                        SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                        WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                        
                        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        ComputeL0c(s1gL0RealSize, s2L0RealSize, runInfo);

                        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        
                        Fixp(s1gGmOffset + s1gOffset, s2GmOffset + s2L1Offset, s1gL0RealSize, s2L0RealSize, runInfo);
                        SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        l0BufIdx_++;
                    }
                }
                if (s2GmOffset + s2BasicBlock_ >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
                    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            keyL1BufIdx_++;
        }
    } else if (s2BasicBlock_ == 256) {
        // 第一个s2循环 keycache置为false
        if (runInfo.isFirstS2InnerLoop) {
            isKeyCacheValid_ = false;
        }
        for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += s2BasicBlock_) {
            // 缓存命中不需要进行key的搬运
            bool keyCacheHit = isKeyCacheValid_ && (s2GmBaseOffset >= keyGmStart_) &&
                                 (s2GmBaseOffset + s2ProcessSize <= keyGmStart_ + keyLoadedSize_);
            if (!keyCacheHit) {
                WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
                // 缓存未命中，需要从GM搬到L1 min（256, 剩余s2）
                uint64_t s2TotalRemainNum = runInfo.actS2Size - s2GmBaseOffset;
                uint64_t s2L1LoadSize = (s2TotalRemainNum < s2BasicBlock_) ? s2TotalRemainNum : s2BasicBlock_;
                if (PAGE_ATTENTION) {
                    KeyNd2NzForPA(s2L1LoadSize, s2GmBaseOffset + s2GmOffset, runInfo);
                }else {
                    KeyNd2Nz(s2L1LoadSize, s2GmOffset, runInfo);
                }

                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);

                isKeyCacheValid_ = true;
                keyGmStart_ = s2GmBaseOffset;
                keyLoadedSize_ = s2L1LoadSize;
            }
            uint64_t l1S2Offset = s2GmBaseOffset - keyGmStart_;
            uint64_t l1TotalSize = keyLoadedSize_;
            // s1gProcessSize当前必定不会超过2倍的s1g basic block
            for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += constInfo_.mBaseSize) {
                uint64_t s1gL1RealSize =
                    s1gGmOffset + constInfo_.mBaseSize > s1gProcessSize ?
                                           s1gProcessSize - s1gGmOffset :
                                           constInfo_.mBaseSize;
                uint64_t s1gL1SizeAlign2G = CeilAlign(s1gL1RealSize, 2 * constInfo_.gSize);
                if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                    queryL1Mte2BufIdx_++;
                    queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
                    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                    QueryNd2Nz(s1gL1RealSize, s1gGmOffset, runInfo);
                    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                } else {
                    queryL1Mte1BufIdx_ =
                        queryL1Mte2BufIdx_ - (CeilDiv(s1gProcessSize, constInfo_.mBaseSize) - 1 - (s1gGmOffset > 0));
                }
                uint64_t s2Boundry = l1S2Offset + s2ProcessSize;
                for (uint64_t s2L1Offset = l1S2Offset; s2L1Offset < s2Boundry; s2L1Offset += S2_BASIC_BLOCK_L0) {
                    uint64_t s2L0RealSize =
                        s2L1Offset + S2_BASIC_BLOCK_L0 > l1S2Offset + s2ProcessSize ?
                                            l1S2Offset + s2ProcessSize - s2L1Offset :
                                            S2_BASIC_BLOCK_L0;
                    for (uint64_t s1gOffset = 0; s1gOffset < s1gL1SizeAlign2G; s1gOffset += constInfo_.mBaseSize) {
                        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        uint64_t s1gL0RealSize =
                            s1gOffset + constInfo_.mBaseSize > s1gL1SizeAlign2G ?
                                                    s1gL1SizeAlign2G - s1gOffset :
                                                    constInfo_.mBaseSize;
                        LoadQueryToL0a(s1gGmOffset, s1gOffset, s1gL1SizeAlign2G, s1gL0RealSize, runInfo);
                        LoadKeyToL0b(s2L1Offset, l1TotalSize, s2L0RealSize, runInfo);

                        SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                        WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                        
                        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        ComputeL0c(s1gL0RealSize, s2L0RealSize, runInfo);

                        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        
                        Fixp(s1gGmOffset + s1gOffset, (s2L1Offset - l1S2Offset),
                                                s1gL0RealSize, s2L0RealSize, runInfo);
                        SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        l0BufIdx_++;
                    }
                }
                if (s2GmOffset + s2BasicBlock_ >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
                    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
                }
            }
            bool l1FullyUsed = (s2GmBaseOffset + s2ProcessSize >= keyGmStart_ + keyLoadedSize_);
            if (l1FullyUsed || runInfo.isLastS2InnerLoop) {
                SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
                keyL1BufIdx_++;
                isKeyCacheValid_ =false;
            }
        }
    }

    CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_CV_EVENT + runInfo.loop % 2); 
    CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(QLICommon::ConstInfo::CROSS_CV_EVENT + runInfo.loop % 2 + QLICommon::ConstInfo::AIV0_AIV1_OFFSET); 
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                    const QLICommon::RunInfo &runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s2L1RealSize; // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * keyBufferOffset_],
             keyGm_[runInfo.tensorKeyOffset + s2GmOffset * constInfo_.headDim], nd2nzPara);
}

// blkNum, blkSize, N2, D
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                    const QLICommon::RunInfo &runInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
        uint64_t keyGmOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId) *
                                constInfo_.keyStride0 + s2BlkOffset * constInfo_.headDim;

        uint64_t s2Mte2Size = s2L1RealSize - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ? constInfo_.kCacheBlockSize - s2BlkOffset
                                                                            : s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size; // 行数
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * keyBufferOffset_ + s2L1Offset * FP8_BLOCK_CUBE],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

// batch, s1, n2, g, d
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                                 const QLICommon::RunInfo &runInfo)
{
    uint64_t dstNzC0Stride = CeilAlign(s1gL1RealSize, 2 * constInfo_.gSize);
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize; // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(dstNzC0Stride, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(queryL1_[(queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
             queryGm_[runInfo.tensorQueryOffset + s1gGmOffset * constInfo_.headDim], nd2nzPara);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadQueryToL0a(uint64_t s1gGmOffset, uint64_t s1gL1Offset, uint64_t s1gL1RealSize,
                                                     uint64_t s1gL0RealSize, const QLICommon::RunInfo &runInfo)
{
    LoadData2DParamsV2 loadData2DParamsV2;
    loadData2DParamsV2.mStartPosition = CeilDiv(s1gL1Offset, BLOCK_CUBE);
    loadData2DParamsV2.kStartPosition = 0;
    loadData2DParamsV2.mStep = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.kStep = CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE);
    loadData2DParamsV2.srcStride = CeilDiv(s1gL1RealSize, BLOCK_CUBE);
    loadData2DParamsV2.dstStride = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.ifTranspose = false;
    
    LoadData(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             queryL1_[(queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET], loadData2DParamsV2);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                                   const QLICommon::RunInfo &runInfo)
{
    LoadData2DParamsV2 loadData2DParamsV2;
    loadData2DParamsV2.mStartPosition = CeilDiv(s2L1Offset, BLOCK_CUBE);
    loadData2DParamsV2.kStartPosition = 0;
    loadData2DParamsV2.mStep = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.kStep = CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE);
    loadData2DParamsV2.srcStride = CeilDiv(s2L1RealSize, BLOCK_CUBE);
    loadData2DParamsV2.dstStride = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.ifTranspose = false;
    
    LoadData(keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * keyBufferOffset_], loadData2DParamsV2);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                const QLICommon::RunInfo &runInfo)
{
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
         keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                           uint64_t s2L0RealSize, const QLICommon::RunInfo &runInfo)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);

    // s1gL0RealSize：2*gSize(128)对齐, 最大256
    // s2L0RealSize <= S2_BASIC_BLOCK_L0, 未约束
    uint32_t nSize = (s2L0RealSize + 7) >> 3 << 3; // 32B对齐
    uint32_t mSize = (s1gL0RealSize + 1) >> 1 << 1;
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    // 固定参数
    fixpipeParams.mSize = mSize;
    fixpipeParams.srcStride = mSize; // 已16对齐
    fixpipeParams.dstStride = UB_BANK_DEPTH_STRIDE / sizeof(float); // 落到同一个bank
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数

    // nSize已保证N方向32B对齐
    if (nSize <= (256 / sizeof(float))) {
        // N方向小于一个bank(256B), 只需搬一个ND块, 且不用补齐
        fixpipeParams.nSize = nSize;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
    } else {
        // N方向在(256B, 512B]范围， 直接按512B搬, 注意此时不能开unitflag
        fixpipeParams.nSize = S2_BASIC_BLOCK_L0 / 2; // 分2个ND搬, S2_BASIC_BLOCK_L0不为128会有问题
        fixpipeParams.params.ndNum = 2;
        fixpipeParams.params.srcNdStride = ((fixpipeParams.mSize + 15) / 16) * fixpipeParams.nSize;
        fixpipeParams.params.dstNdStride = constInfo_.s2BaseSize * constInfo_.mBaseSize / 2;
    }
    Fixpipe<QK_T, QK_T, QLI_CFG_ROW_MAJOR_UB>(mm1ResUB_[(runInfo.loop % 2) * constInfo_.s2BaseSize / 2],
                                                cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], fixpipeParams);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::AllocEventID()
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

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::FreeEventID()
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
} // namespace QLIKernel
#endif