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
 * \brief use multi-buffer for matmul, better pipeline
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
    static constexpr uint64_t KEY_BUF_NUM = 2;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0A_BUF_NUM = 2;
    static constexpr uint64_t L0B_BUF_NUM = 2;
    static constexpr uint64_t L0C_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID2; // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID3;

    static constexpr uint64_t M_BASIC_BLOCK = 64;
    static constexpr uint64_t S2_BASIC_BLOCK = 256;

    static constexpr uint64_t M_BASIC_BLOCK_L1 = 64;
    static constexpr uint64_t D_BASIC_BLOCK_L1 = 1024;
    static constexpr uint64_t S2_BASIC_BLOCK_L1 = 64;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 64;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 256;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 64;

    static constexpr uint64_t BF16_BLOCK_CUBE = 16;
    static constexpr uint32_t PIPE_M_BARRIER_THRESHOLD = 10U;
    // ROW_MAJOR使能NZ2ND并输出ND格式；true表示目的地址位于UB。
    static constexpr FixpipeConfig SI_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

    static constexpr int64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1;
    static constexpr int64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1;
    static constexpr int64_t L0A_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr int64_t L0B_BUFFER_OFFSET = S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr int64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK * S2_BASIC_BLOCK;

protected:
    __aicore__ inline void Fixp(uint64_t s1gRealSize, uint64_t s2RealSize, uint32_t mm1BufferIdx, uint32_t l0cSlot);
    __aicore__ inline void ComputeL0c(int64_t l0cOffset, uint32_t l0Slot, const MmadParams &mmadParams,
                                      bool needMBarrier);
    __aicore__ inline void CopyQuerySegmentNd2Nz(LocalTensor<Q_T> queryL1Base, int64_t l1RowOffset, uint64_t l1RowAlign,
                                                 int64_t gmOffset, uint64_t nValue);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, int64_t s1gL1Offset, int64_t kGmOffset,
                                      uint32_t queryL1Slot, const SICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, int64_t s2GmOffset, int64_t kGmOffset, uint32_t keyL1Slot,
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
    // 三缓冲，每槽32KB，对应每个AIV的64/2 * 256个float结果。
    pipe->InitBuffer(bufUB_, SICommon::MM1_RES_BUFFER_NUM * SICommon::MM1_RES_SLOT_BYTES);
    mm1ResUB_ = bufUB_.Get<QK_T>();
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1 * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1 * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0A_BUF_NUM * M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0B_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0C_BUF_NUM * M_BASIC_BLOCK * S2_BASIC_BLOCK * sizeof(float));
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
    const uint32_t mm1BufferIdx = runInfo.loop % SICommon::MM1_RES_BUFFER_NUM;
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_VC_EVENT + mm1BufferIdx);
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_VC_EVENT + mm1BufferIdx +
                                                         SICommon::AIV0_AIV1_OFFSET);
    const uint64_t s1gProcessSize = runInfo.actMBaseSize;
    const uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    const uint64_t kProcessSize = constInfo_.headDim;
    const uint32_t l0cSlot = l0cBufIdx_ & (L0C_BUF_NUM - 1U);
    const uint64_t mAlignSize = CeilAlign(s1gProcessSize, static_cast<uint64_t>(BLOCK_CUBE));
    const uint32_t mBlockNum = static_cast<uint32_t>(mAlignSize / BLOCK_CUBE);

    LoadData2DParamsV2 queryLoadParams;
    queryLoadParams.mStartPosition = 0U;
    queryLoadParams.kStartPosition = 0U;
    queryLoadParams.mStep = mBlockNum;
    queryLoadParams.kStep = D_BASIC_BLOCK_L0 / BF16_BLOCK_CUBE;
    queryLoadParams.srcStride = mBlockNum;
    queryLoadParams.dstStride = mBlockNum;
    queryLoadParams.ifTranspose = false;

    MmadParams mm1Params;
    mm1Params.m = mAlignSize;
    mm1Params.k = D_BASIC_BLOCK_L0;
    mm1Params.cmatrixInitVal = false;
    mm1Params.cmatrixSource = false;

    // s2轴循环
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cSlot);
    const int64_t s2ProcessSizeI64 = static_cast<int64_t>(s2ProcessSize);
    const int64_t kProcessSizeI64 = static_cast<int64_t>(kProcessSize);
    for (int64_t s2GmOffset = 0LL; s2GmOffset < s2ProcessSizeI64;
         s2GmOffset += static_cast<int64_t>(S2_BASIC_BLOCK_L1)) {
        const uint64_t s2L1RealSize =
            static_cast<uint64_t>(s2GmOffset + static_cast<int64_t>(S2_BASIC_BLOCK_L1) > s2ProcessSizeI64 ?
                                      s2ProcessSizeI64 - s2GmOffset :
                                      static_cast<int64_t>(S2_BASIC_BLOCK_L1));
        const uint32_t nBlockNum = static_cast<uint32_t>(CeilDiv(s2L1RealSize, static_cast<uint64_t>(BLOCK_CUBE)));
        LoadData2DParamsV2 keyLoadParams;
        keyLoadParams.mStartPosition = 0U;
        keyLoadParams.kStartPosition = 0U;
        keyLoadParams.mStep = nBlockNum;
        keyLoadParams.kStep = D_BASIC_BLOCK_L0 / BF16_BLOCK_CUBE;
        keyLoadParams.srcStride = nBlockNum;
        keyLoadParams.dstStride = nBlockNum;
        keyLoadParams.ifTranspose = false;

        mm1Params.n = s2L1RealSize;
        const bool needMBarrier =
            (mBlockNum * (static_cast<uint32_t>(s2L1RealSize) / BLOCK_CUBE)) < PIPE_M_BARRIER_THRESHOLD;
        const int64_t l0cOffset =
            static_cast<int64_t>(l0cSlot) * L0C_BUFFER_OFFSET + s2GmOffset * static_cast<int64_t>(mAlignSize);

        uint32_t dL1Idx = 0U;
        for (int64_t kGmOffset = 0LL; kGmOffset < kProcessSizeI64;
             kGmOffset += static_cast<int64_t>(D_BASIC_BLOCK_L1), ++dL1Idx) {
            const uint32_t keyL1Slot = keyL1BufIdx_ & (KEY_BUF_NUM - 1U);
            const uint32_t queryL1Slot = dL1Idx & (QUERY_BUF_NUM - 1U);
            WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1Slot);
            KeyNd2Nz(s2L1RealSize, s2GmOffset, kGmOffset, keyL1Slot, runInfo);

            SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0U) {
                WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Slot);
                QueryNd2Nz(s1gProcessSize, 0U, kGmOffset, queryL1Slot, runInfo);
                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            }

            for (int64_t kL1Offset = 0LL; kL1Offset < static_cast<int64_t>(D_BASIC_BLOCK_L1);
                 kL1Offset += static_cast<int64_t>(D_BASIC_BLOCK_L0)) {
                const uint32_t l0Slot = l0BufIdx_ & (L0A_BUF_NUM - 1U);
                const uint32_t kStartPosition = static_cast<uint32_t>(kL1Offset / BLOCK_CUBE);
                queryLoadParams.kStartPosition = kStartPosition;
                keyLoadParams.kStartPosition = kStartPosition;

                WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0Slot);
                LoadData(queryL0_[static_cast<int64_t>(l0Slot) * L0A_BUFFER_OFFSET],
                         queryL1_[static_cast<int64_t>(queryL1Slot) * QUERY_BUFFER_OFFSET], queryLoadParams);
                LoadData(keyL0_[static_cast<int64_t>(l0Slot) * L0B_BUFFER_OFFSET],
                         keyL1_[static_cast<int64_t>(keyL1Slot) * KEY_BUFFER_OFFSET], keyLoadParams);

                SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                mm1Params.cmatrixInitVal = (kGmOffset + kL1Offset) == 0U;
                ComputeL0c(l0cOffset, l0Slot, mm1Params, needMBarrier);
                SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0Slot);
                l0BufIdx_++;
            }
            if (s2GmOffset + static_cast<int64_t>(S2_BASIC_BLOCK_L1) >= s2ProcessSizeI64 && runInfo.isLastS2InnerLoop) {
                SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Slot);
            }
            SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1Slot);
            keyL1BufIdx_++;
        }
    }
    Fixp(s1gProcessSize, s2ProcessSize, mm1BufferIdx, l0cSlot);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cSlot);
    l0cBufIdx_++;
    CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_CV_EVENT + mm1BufferIdx);
    CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_CV_EVENT + mm1BufferIdx +
                                                        SICommon::AIV0_AIV1_OFFSET);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::KeyNd2Nz(uint64_t s2L1RealSize, int64_t s2GmOffset, int64_t kGmOffset,
                                               uint32_t keyL1Slot, const SICommon::RunInfo &runInfo)
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
    DataCopy(keyL1_[static_cast<int64_t>(keyL1Slot) * KEY_BUFFER_OFFSET],
             keyGm_[runInfo.tensorKeyOffset + s2GmOffset * static_cast<int64_t>(constInfo_.headDim) + kGmOffset],
             nd2nzPara);
}

// batch, n2, g, s1, d
template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::CopyQuerySegmentNd2Nz(LocalTensor<Q_T> queryL1Base, int64_t l1RowOffset,
                                                            uint64_t l1RowAlign, int64_t gmOffset, uint64_t nValue)
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
    DataCopy(queryL1Base[l1RowOffset * static_cast<int64_t>(BLOCK_CUBE)], queryGm_[gmOffset], nd2nzPara);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::QueryNd2Nz(uint64_t s1gL1RealSize, int64_t s1gGmOffset, int64_t kGmOffset,
                                                 uint32_t queryL1Slot, const SICommon::RunInfo &runInfo)
{
    LocalTensor<Q_T> queryL1Base = queryL1_[static_cast<int64_t>(queryL1Slot) * QUERY_BUFFER_OFFSET];
    uint64_t l1RowAlign = CeilAlign(s1gL1RealSize, (uint64_t)BLOCK_CUBE);
    int64_t logicalMStart =
        static_cast<int64_t>(runInfo.gS1Idx) * static_cast<int64_t>(constInfo_.mBaseSize) + s1gGmOffset;
    int64_t queryBaseOffset = runInfo.tensorQueryOffset;
    if (runInfo.actS1Size == constInfo_.qSeqSize) {
        CopyQuerySegmentNd2Nz(queryL1Base, 0, l1RowAlign,
                              queryBaseOffset + logicalMStart * static_cast<int64_t>(constInfo_.headDim) + kGmOffset,
                              s1gL1RealSize);
        return;
    }

    int64_t copiedRows = 0LL;
    int64_t logicalMEnd = logicalMStart + static_cast<int64_t>(s1gL1RealSize);
    const int64_t actS1Size = static_cast<int64_t>(runInfo.actS1Size);
    const int64_t qSeqSize = static_cast<int64_t>(constInfo_.qSeqSize);
    const int64_t headDim = static_cast<int64_t>(constInfo_.headDim);
    while (logicalMStart < logicalMEnd) {
        int64_t globalGIdx = logicalMStart / actS1Size;
        int64_t globalS1Idx = logicalMStart % actS1Size;
        int64_t copyRows = Min(logicalMEnd - logicalMStart, actS1Size - globalS1Idx);
        int64_t gmOffset = queryBaseOffset + (globalGIdx * qSeqSize + globalS1Idx) * headDim + kGmOffset;
        CopyQuerySegmentNd2Nz(queryL1Base, copiedRows, l1RowAlign, gmOffset, static_cast<uint64_t>(copyRows));
        logicalMStart += copyRows;
        copiedRows += copyRows;
    }
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::ComputeL0c(int64_t l0cOffset, uint32_t l0Slot, const MmadParams &mmadParams,
                                                 bool needMBarrier)
{
    Mmad(cL0_[l0cOffset], queryL0_[static_cast<int64_t>(l0Slot) * L0A_BUFFER_OFFSET],
         keyL0_[static_cast<int64_t>(l0Slot) * L0B_BUFFER_OFFSET], mmadParams);
    if (needMBarrier) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::Fixp(uint64_t s1gRealSize, uint64_t s2RealSize, uint32_t mm1BufferIdx,
                                           uint32_t l0cSlot)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0cSlot);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0cSlot);

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;

    fixpipeParams.mSize = static_cast<uint32_t>((s1gRealSize + 1U) >> 1U << 1U);
    fixpipeParams.nSize = static_cast<uint32_t>((s2RealSize + 7U) >> 3U << 3U); // 32B对齐
    fixpipeParams.srcStride = ((fixpipeParams.mSize + BLOCK_CUBE - 1U) / BLOCK_CUBE) * BLOCK_CUBE;
    fixpipeParams.dstStride = constInfo_.s2BaseSize;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    // 将matmul结果从L0C搬运到UB。
    const int64_t mm1ResOffset =
        static_cast<int64_t>(mm1BufferIdx) * static_cast<int64_t>(SICommon::MM1_RES_SLOT_BYTES / sizeof(QK_T));
    Fixpipe<QK_T, float, SI_CFG_ROW_MAJOR_UB>(mm1ResUB_[mm1ResOffset],
                                              cL0_[static_cast<int64_t>(l0cSlot) * L0C_BUFFER_OFFSET], fixpipeParams);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

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

    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}
} // namespace SIKernel
#endif
