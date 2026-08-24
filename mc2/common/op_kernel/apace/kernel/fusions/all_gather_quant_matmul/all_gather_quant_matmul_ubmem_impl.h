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
 * \file all_gather_quant_matmul_impl.h
 * \brief AllGather + QuantMatmul fusion implementation.
 *
 * Flow:
 *   AIV: AllGatherProcess (CopyLocalData → SyncAll → WriteStatus → ReadStatus → ExecuteAllGather → CrossCoreSetFlag)
 *   AIC: MatmulProcess (ComputeSelfAddrs → rewrite aGmAddr/scaleAGmAddr → baseline GemmUniversal::Run)
 *
 * The matmul part delegates entirely to Blaze's kernel_qbmm_mx.h baseline.
 * The AllGather communication logic (formerly block_prologue_all_gather.h) is inlined here.
 */

#pragma once

#include "blaze/gemm/kernel/kernel_qbmm_mx.h"
#include "blaze/gemm/utils/common_utils.h"
#include "apace/utils/comm_resource_builder.h"
#include "apace/tiling/quant_matmul_tiling_data.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tensor_api/tensor.h"

namespace Apace {

using namespace AscendC;

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class AllGatherQuantMatmulUbmemImpl {
public:
    using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<
        ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,
        AscendC::Std::enable_if_t<AscendC::Std::is_same_v<Blaze::Gemm::KernelMmadWithScaleMx,
                                                          typename BlockMmad::DispatchPolicy::ScheduleType>>>;

    using AType = typename BlockMmad::AType;

    static constexpr uint64_t UB_ALIGN_BYTES = 32U;
    static constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;
    static constexpr uint64_t MXFP_DIVISOR_SIZE = 64;
    static constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2;
    static constexpr uint64_t DEFAULT_K_SPLIT_NUM = 2;
    static constexpr uint64_t TILE_K_NUM = 512;
    static constexpr uint32_t BUFFER_NUM = 2U;
    static constexpr uint64_t X_BLOCK_BYTES = 512U;
    static constexpr uint32_t X_PER_BLOCK_NUM = 512U;
    static constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
    static constexpr uint64_t DATA_START_OFFSET = 1024UL * 1024UL;
    static constexpr uint64_t UB_AVAIL_SIZE = 248UL * 1024UL;
    static constexpr uint16_t CROSS_CORE_FLAG_NUM_NINE = 9;
    static constexpr uint16_t CROSS_CORE_INNER_CUBE_VEC_SYNC = 0x2;
    static constexpr float STATUS_FLAG_VAL = 1.0f;
    static constexpr float STATUS_FLAG_THREDSHOLD = 0.5f;

    static constexpr uint16_t EVT_ID = 1;
    static constexpr uint16_t EVT_ID_PING = 2;
    static constexpr uint16_t EVT_ID_PONG = 3;
    static constexpr uint16_t EVT_ID_SCALE_PING = 4;
    static constexpr uint16_t EVT_ID_SCALE_PONG = 5;

    using ScalesType = int8_t;

    using MakeLayoutGM =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<int8_t>>;
    using MakeLayoutUB =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<int8_t>>;
    using MakeLayoutScaleGM =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<ScalesType>>;
    using MakeLayoutScaleUB =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<ScalesType>>;
    using MakeLayoutStatusGM =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>;
    using MakeLayoutStatusUB =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>;

    using CopyGM2UB_t = decltype(AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{}));
    using CopyUB2GM_t = decltype(AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{}));

    struct Params {
        typename KernelImpl::Params matmulKernelParams;
        GM_ADDR tilingGM{nullptr};
        GM_ADDR hcclContext{nullptr};
        GM_ADDR x1Addr{nullptr};
        GM_ADDR x1ScaleAddr{nullptr};
        GM_ADDR allGatherDataOutAddr{nullptr};
        GM_ADDR allGatherScalesOutAddr{nullptr};
    };

    __aicore__ inline AllGatherQuantMatmulUbmemImpl() {}
    __aicore__ inline ~AllGatherQuantMatmulUbmemImpl() {}

    __aicore__ inline void operator()(Params &params)
    {
        if ASCEND_IS_AIV {
            AllGatherProcess(params);
        } else {
            MatmulProcess(params);
        }
    }

private:
    __aicore__ inline void AllGatherProcess(const Params &params);
    __aicore__ inline void MatmulProcess(Params &params);

    __aicore__ inline void AllGatherInit(const Params &params);
    __aicore__ inline void CopyLocalData();
    __aicore__ inline void ExecuteAllGather();
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void ReadDataBlock(uint64_t curXOffset, uint32_t mCnt, uint32_t bufIdx);

    __aicore__ inline GM_ADDR WinDataAddr(uint32_t rankId)
    {
        return (GM_ADDR)(hcclContext_->windowsIn[rankId] + DATA_START_OFFSET);
    }
    __aicore__ inline GM_ADDR WinStatusAddr(uint32_t rankId)
    {
        return (GM_ADDR)(hcclContext_->windowsIn[rankId]);
    }

private:
    __gm__ Apace::HcclOpParam *hcclContext_{nullptr};
    __gm__ QuantMatmulTilingData *tilingData_{nullptr};

    uint32_t aivId_{0};
    uint32_t aicId_{0};
    uint32_t curRankId_{0};
    uint32_t rankSize_{0};
    uint64_t winDataSize_{0};

    uint64_t M_{0};
    uint64_t K_{0};
    uint64_t xSize_{0};
    uint64_t scaleSize_{0};
    uint64_t scaleChunkSize_{0};
    uint32_t scaleTotalChunks_{0};

    uint32_t sendCoreNumPerRank_{0};
    uint32_t remoteRankId_{0};
    uint64_t tileK_{0};
    uint32_t kDim_{0};
    uint32_t mDim_{0};
    uint32_t mBlockIdx_{0};
    uint32_t kBlockIdx_{0};
    uint32_t mMteCoreM_{0};
    uint32_t coreInnerMIndex_{0};
    uint32_t kStartIndex_{0};

    uint64_t ubWriteStateOffset_{0};
    uint64_t ubReadStateOffset_{0};
    uint64_t ubScalePingOffset_{0};
    uint64_t ubScalePongOffset_{0};
    uint64_t ubXPingOffset_{0};
    uint64_t ubXPongOffset_{0};
    uint64_t xBufSize_{0};

    GM_ADDR inputXAddr_{nullptr};
    GM_ADDR inputXScaleAddr_{nullptr};
    GM_ADDR allGatherDataOutAddr_{nullptr};
    GM_ADDR allGatherScalesOutAddr_{nullptr};

    CopyGM2UB_t copyGM2UB_ = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
    CopyUB2GM_t copyUB2GM_ = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{});
};

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue,
                                                     BlockScheduler>::AllGatherProcess(const Params &params)
{
    AllGatherInit(params);
    CopyLocalData();
    SyncAll<true>();
    WriteStatusToWin();
    ReadStatus();
    ExecuteAllGather();
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue,
                                                     BlockScheduler>::AllGatherInit(const Params &params)
{
    hcclContext_ = (__gm__ Apace::HcclOpParam *)params.hcclContext;
    tilingData_ = (__gm__ QuantMatmulTilingData *)params.tilingGM;

    rankSize_ = Apace::GetRankDim(hcclContext_);
    curRankId_ = Apace::GetRankId(hcclContext_);

    inputXAddr_ = params.x1Addr;
    inputXScaleAddr_ = params.x1ScaleAddr;
    allGatherDataOutAddr_ = params.allGatherDataOutAddr;
    allGatherScalesOutAddr_ = params.allGatherScalesOutAddr;

    M_ = tilingData_->m / rankSize_;
    K_ = tilingData_->k;
    xSize_ = M_ * K_;
    scaleSize_ = M_ * Blaze::Gemm::CeilDiv(K_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    ;
    sendCoreNumPerRank_ = GetBlockNum() / rankSize_;
    tileK_ = Blaze::Gemm::CeilDiv(K_, TILE_K_NUM);

    aivId_ = GetBlockIdx();
    aicId_ = aivId_ / GetTaskRation();

    winDataSize_ = Blaze::Gemm::CeilAlign(rankSize_ * xSize_, WIN_ADDR_ALIGN);

    kDim_ = tileK_ < DEFAULT_K_SPLIT_NUM ? tileK_ : DEFAULT_K_SPLIT_NUM;
    mDim_ = sendCoreNumPerRank_ / kDim_;
    if (mDim_ == 0) {
        mDim_ = 1;
    }
    if (mDim_ > M_) {
        mDim_ = M_;
    }

    uint32_t modCoreIndex = aicId_ % sendCoreNumPerRank_;
    remoteRankId_ = aicId_ / sendCoreNumPerRank_;
    mBlockIdx_ = modCoreIndex % mDim_;
    kBlockIdx_ = modCoreIndex / mDim_;

    uint32_t mteMSplitSize = static_cast<uint32_t>(M_) / mDim_;
    if (mteMSplitSize == 0) {
        mteMSplitSize = 1;
    }
    mMteCoreM_ = mteMSplitSize;

    uint32_t mTailNum = static_cast<uint32_t>(M_) % mDim_;
    if (mBlockIdx_ < mTailNum) {
        mMteCoreM_ += 1;
        coreInnerMIndex_ = mBlockIdx_ * mMteCoreM_;
    } else {
        coreInnerMIndex_ = mBlockIdx_ * mMteCoreM_ + mTailNum;
    }
    kStartIndex_ = kBlockIdx_ * X_PER_BLOCK_NUM;

    ubWriteStateOffset_ = 0;
    ubReadStateOffset_ = UB_ALIGN_BYTES;

    uint64_t workStart = UB_ALIGN_BYTES + 2 * rankSize_ * UB_ALIGN_BYTES;
    uint64_t workAvailable = (UB_AVAIL_SIZE > workStart) ? (UB_AVAIL_SIZE - workStart) : 0;

    uint64_t scaleHalf = (workAvailable / BUFFER_NUM / UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    scaleChunkSize_ = scaleHalf;
    if (scaleChunkSize_ == 0) {
        scaleChunkSize_ = UB_ALIGN_BYTES;
    }
    scaleTotalChunks_ = (scaleSize_ + scaleChunkSize_ - 1) / scaleChunkSize_;
    ubScalePingOffset_ = workStart;
    ubScalePongOffset_ = workStart + scaleChunkSize_;

    uint64_t totalAvailable = workAvailable / X_BLOCK_BYTES * X_BLOCK_BYTES;
    uint64_t demandSpace = Blaze::Gemm::CeilAlign(xSize_, X_BLOCK_BYTES);
    uint64_t availableSpace = totalAvailable / BUFFER_NUM / X_BLOCK_BYTES * X_BLOCK_BYTES;
    xBufSize_ = (availableSpace < demandSpace) ? availableSpace : demandSpace;
    if (xBufSize_ < X_BLOCK_BYTES) {
        xBufSize_ = X_BLOCK_BYTES;
    }
    ubXPingOffset_ = workStart;
    ubXPongOffset_ = workStart + xBufSize_;
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::MatmulProcess(Params &params)
{
    __gm__ QuantMatmulTilingData *tilingData = (__gm__ QuantMatmulTilingData *)params.tilingGM;

    auto *hcclCtx = (__gm__ Apace::HcclOpParam *)params.hcclContext;
    uint32_t curRankId = Apace::GetRankId(hcclCtx);
    uint32_t rankSize = Apace::GetRankDim(hcclCtx);

    uint64_t M = tilingData->m / rankSize;
    uint64_t K = tilingData->k;

    uint64_t xTotalBytes = M * K * sizeof(AType);
    uint64_t winScaleOffset = Blaze::Gemm::CeilDiv(rankSize * xTotalBytes, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;

    GM_ADDR selfDataAddr = (GM_ADDR)(hcclCtx->windowsIn[curRankId] + DATA_START_OFFSET);
    GM_ADDR selfScaleAddr = (GM_ADDR)((uint64_t)selfDataAddr + winScaleOffset);

    params.matmulKernelParams.mmadParams.aGmAddr = selfDataAddr;
    params.matmulKernelParams.mmadParams.scaleAGmAddr = selfScaleAddr;

    KernelImpl kernel;
    kernel(params.matmulKernelParams);
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::CopyLocalData()
{
    GM_ADDR localWinGm = WinDataAddr(curRankId_);
    uint64_t selfDataOffset = curRankId_ * xSize_;

    uint32_t copyXBlockNum = GetBlockNum() / GetTaskRation() - 1;
    if (aivId_ == copyXBlockNum) {
        uint64_t winScaleBase = winDataSize_ + curRankId_ * scaleSize_;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PING);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PONG);

        for (uint32_t chunkIdx = 0; chunkIdx < scaleTotalChunks_; ++chunkIdx) {
            uint16_t evtId = (chunkIdx % 2 == 0) ? EVT_ID_SCALE_PING : EVT_ID_SCALE_PONG;
            uint64_t ubOff = (chunkIdx % 2 == 0) ? ubScalePingOffset_ : ubScalePongOffset_;
            uint64_t chunkOffset = chunkIdx * scaleChunkSize_;
            uint64_t curChunkSize =
                (chunkOffset + scaleChunkSize_ > scaleSize_) ? (scaleSize_ - chunkOffset) : scaleChunkSize_;
            uint64_t curChunkPaddedN = Blaze::Gemm::CeilAlign(curChunkSize, UB_ALIGN_BYTES) / sizeof(ScalesType);

            auto scaleSrcLayout = MakeLayoutScaleGM{}(1, curChunkSize);
            auto scaleSrcTensor = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                                              (__gm__ ScalesType *)(inputXScaleAddr_ + chunkOffset)),
                                                          scaleSrcLayout);

            auto scaleUbLayout = MakeLayoutScaleUB{}(1, curChunkPaddedN);
            auto scaleUbTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, ScalesType>(ubOff), scaleUbLayout);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
            AscendC::Te::Copy(copyGM2UB_, scaleUbTensor, scaleSrcTensor);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);

            auto scaleDstLayout = MakeLayoutScaleGM{}(1, curChunkSize);
            auto scaleDstTensor =
                AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                            (__gm__ ScalesType *)(localWinGm + winScaleBase + chunkOffset)),
                                        scaleDstLayout);
            AscendC::Te::Copy(copyUB2GM_, scaleDstTensor, scaleUbTensor);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PING);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PONG);
        return;
    }
    if (aicId_ > copyXBlockNum) {
        return;
    }
    if (copyXBlockNum == 0) {
        return;
    }
    uint32_t mPerCore = M_ / copyXBlockNum;
    uint32_t remainderNum = M_ % copyXBlockNum;
    uint32_t mStartId;
    if (aicId_ < remainderNum) {
        mPerCore += 1;
        mStartId = aicId_ * mPerCore;
    } else {
        mStartId = remainderNum * (mPerCore + 1) + (aicId_ - remainderNum) * mPerCore;
    }

    uint64_t totalElements = static_cast<uint64_t>(mPerCore) * K_;
    uint64_t startGlobalOffset = static_cast<uint64_t>(mStartId) * K_;

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);

    for (uint64_t done = 0; done < totalElements; done += xBufSize_) {
        uint32_t chunkIdx = static_cast<uint32_t>(done / xBufSize_);
        uint16_t evtId = (chunkIdx % 2 == 0) ? EVT_ID_PING : EVT_ID_PONG;
        uint64_t ubOff = (chunkIdx % 2 == 0) ? ubXPingOffset_ : ubXPongOffset_;
        uint32_t chunkElems =
            static_cast<uint32_t>((done + xBufSize_ > totalElements) ? (totalElements - done) : xBufSize_);
        uint64_t globalOffset = startGlobalOffset + done;

        auto xUbLayout = MakeLayoutUB{}(1, chunkElems);
        auto xUbTensor =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, int8_t>(ubOff), xUbLayout);

        auto xSrcLayout = MakeLayoutGM{}(1, chunkElems);
        auto xSrcTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ int8_t *)(inputXAddr_ + globalOffset)),
            xSrcLayout);

        auto xDstLayout = MakeLayoutGM{}(1, chunkElems);
        auto xDstTensor = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                                      (__gm__ int8_t *)(localWinGm + selfDataOffset + globalOffset)),
                                                  xDstLayout);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
        AscendC::Te::Copy(copyGM2UB_, xUbTensor, xSrcTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
        AscendC::Te::Copy(copyUB2GM_, xDstTensor, xUbTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::ExecuteAllGather()
{
    GM_ADDR remoteWinGm = WinDataAddr(remoteRankId_);
    GM_ADDR localWinGm = WinDataAddr(curRankId_);

    uint32_t modCoreIndex = aicId_ % sendCoreNumPerRank_;
    if (modCoreIndex == sendCoreNumPerRank_ - 1) {
        GM_ADDR remoteScaleAddr = remoteWinGm + winDataSize_ + remoteRankId_ * scaleSize_;
        GM_ADDR localScaleAddr = localWinGm + winDataSize_ + remoteRankId_ * scaleSize_;
        GM_ADDR allGatherScaleAddr = allGatherScalesOutAddr_ + remoteRankId_ * scaleSize_;

        if (remoteRankId_ != curRankId_) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PING);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PONG);

            for (uint32_t chunkIdx = 0; chunkIdx < scaleTotalChunks_; ++chunkIdx) {
                uint16_t evtId = (chunkIdx % 2 == 0) ? EVT_ID_SCALE_PING : EVT_ID_SCALE_PONG;
                uint64_t ubOff = (chunkIdx % 2 == 0) ? ubScalePingOffset_ : ubScalePongOffset_;
                uint64_t chunkOffset = chunkIdx * scaleChunkSize_;
                uint64_t curChunkSize =
                    (chunkOffset + scaleChunkSize_ > scaleSize_) ? (scaleSize_ - chunkOffset) : scaleChunkSize_;
                uint64_t curChunkPaddedN = Blaze::Gemm::CeilAlign(curChunkSize, UB_ALIGN_BYTES) / sizeof(ScalesType);

                auto scaleSrcLayout = MakeLayoutScaleGM{}(1, curChunkSize);
                auto scaleSrcTensor = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                                                  (__gm__ ScalesType *)(remoteScaleAddr + chunkOffset)),
                                                              scaleSrcLayout);

                auto scaleUbLayout = MakeLayoutScaleUB{}(1, curChunkPaddedN);
                auto scaleUbTensor = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, ScalesType>(ubOff), scaleUbLayout);

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
                AscendC::Te::Copy(copyGM2UB_, scaleUbTensor, scaleSrcTensor);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);

                auto localScaleLayout = MakeLayoutScaleGM{}(1, curChunkSize);
                auto localScaleTensor = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((
                                                                    __gm__ ScalesType *)(localScaleAddr + chunkOffset)),
                                                                localScaleLayout);
                auto allGatherScaleLayout = MakeLayoutScaleGM{}(1, curChunkSize);
                auto allGatherScaleTensor =
                    AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                                (__gm__ ScalesType *)(allGatherScaleAddr + chunkOffset)),
                                            allGatherScaleLayout);

                AscendC::Te::Copy(copyUB2GM_, localScaleTensor, scaleUbTensor);
                AscendC::Te::Copy(copyUB2GM_, allGatherScaleTensor, scaleUbTensor);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PING);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PONG);
        }
    }

    uint32_t kLoop = tileK_ / kDim_;
    if (kBlockIdx_ < tileK_ % kDim_) {
        kLoop++;
    }
    uint32_t curXOffset = coreInnerMIndex_ * K_ + kStartIndex_;

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);

    for (uint64_t curKBlock = 0; curKBlock < kLoop; ++curKBlock) {
        uint64_t innerCurXOffset = curXOffset + curKBlock * X_PER_BLOCK_NUM * 2;
        ReadDataBlock(innerCurXOffset, mMteCoreM_, curKBlock % 2);
        PipeBarrier<PIPE_MTE3>();
        SyncAll<true>();
        CrossCoreSetFlag<CROSS_CORE_INNER_CUBE_VEC_SYNC, PIPE_MTE3>(CROSS_CORE_FLAG_NUM_NINE);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);

    if (kLoop < ((tileK_ + kDim_ - 1) / kDim_)) {
        SyncAll<true>();
        CrossCoreSetFlag<CROSS_CORE_INNER_CUBE_VEC_SYNC, PIPE_MTE3>(CROSS_CORE_FLAG_NUM_NINE);
    }
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::WriteStatusToWin()
{
    uint64_t curOffset = (aivId_ % sendCoreNumPerRank_ + curRankId_ * sendCoreNumPerRank_) * FLOAT_UB_ALIGN_NUM;

    auto statusUbLayout = MakeLayoutStatusUB{}(1, FLOAT_UB_ALIGN_NUM);
    auto statusUbTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(ubWriteStateOffset_), statusUbLayout);

    statusUbTensor[AscendC::Te::MakeCoord(0, 0)] = STATUS_FLAG_VAL;

    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVT_ID);

    GM_ADDR dstAddr = WinStatusAddr(remoteRankId_) + curOffset * sizeof(float);
    auto dstLayout = MakeLayoutStatusGM{}(1, FLOAT_UB_ALIGN_NUM);
    auto dstTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ float *)(dstAddr)), dstLayout);
    AscendC::Te::Copy(copyUB2GM_, dstTensor, statusUbTensor);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVT_ID);
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::ReadStatus()
{
    uint32_t offset = aivId_ * FLOAT_UB_ALIGN_NUM;
    GM_ADDR statusAddr = WinStatusAddr(curRankId_) + offset * sizeof(float);

    auto statusGmLayout = MakeLayoutStatusGM{}(1, FLOAT_UB_ALIGN_NUM);
    auto statusGmTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ float *)(statusAddr)), statusGmLayout);

    auto statusUbLayout = MakeLayoutStatusUB{}(1, FLOAT_UB_ALIGN_NUM);
    auto statusUbTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(ubReadStateOffset_), statusUbLayout);

    float minTarget = STATUS_FLAG_VAL - STATUS_FLAG_THREDSHOLD;
    float maxTarget = STATUS_FLAG_VAL + STATUS_FLAG_THREDSHOLD;
    float flag = -1.0f;
    while ((flag < minTarget) || (flag > maxTarget)) {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVT_ID);
        AscendC::Te::Copy(copyGM2UB_, statusUbTensor, statusGmTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVT_ID);
        flag = statusUbTensor[AscendC::Te::MakeCoord(0, 0)];
    }
    statusUbTensor[AscendC::Te::MakeCoord(0, 0)] = 0.0f;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVT_ID);
    AscendC::Te::Copy(copyUB2GM_, statusGmTensor, statusUbTensor);
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue,
                                                     BlockScheduler>::ReadDataBlock(uint64_t curXOffset, uint32_t mCnt,
                                                                                    uint32_t bufIdx)
{
    if (remoteRankId_ == curRankId_) {
        return;
    }
    if (kBlockIdx_ >= kDim_ || mBlockIdx_ >= mDim_) {
        return;
    }

    uint16_t evtId = (bufIdx % 2 == 0) ? EVT_ID_PING : EVT_ID_PONG;
    uint64_t ubOff = (bufIdx % 2 == 0) ? ubXPingOffset_ : ubXPongOffset_;

    GM_ADDR remoteWinGm = WinDataAddr(remoteRankId_);
    GM_ADDR localWinGm = WinDataAddr(curRankId_);

    GM_ADDR remoteXAddr = remoteWinGm + remoteRankId_ * xSize_;
    GM_ADDR localXAddr = localWinGm + remoteRankId_ * xSize_;
    GM_ADDR allGatherXAddr = allGatherDataOutAddr_ + remoteRankId_ * xSize_;

    uint64_t rowStartBase = curXOffset / K_;
    uint64_t colStart = curXOffset % K_;
    uint32_t rowsPerSlot = static_cast<uint32_t>(xBufSize_ / X_PER_BLOCK_NUM);
    if (rowsPerSlot == 0) {
        rowsPerSlot = 1;
    }

    auto srcTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ int8_t *)(remoteXAddr)), MakeLayoutGM{}(M_, K_));
    auto localTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ int8_t *)(localXAddr)), MakeLayoutGM{}(M_, K_));
    auto agTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ int8_t *)(allGatherXAddr)), MakeLayoutGM{}(M_, K_));

    for (uint32_t rowDone = 0; rowDone < mCnt; rowDone += rowsPerSlot) {
        uint32_t curRows = (rowDone + rowsPerSlot > mCnt) ? (mCnt - rowDone) : rowsPerSlot;
        uint64_t rowStart = rowStartBase + rowDone;

        auto srcSlice = srcTensor.Slice(
            AscendC::Te::MakeCoord(rowStart, colStart),
            AscendC::Te::MakeShape(static_cast<int64_t>(curRows), static_cast<int64_t>(X_PER_BLOCK_NUM)));
        auto ubTensor = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, int8_t>(ubOff),
                                                MakeLayoutUB{}(curRows, static_cast<int64_t>(X_PER_BLOCK_NUM)));

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
        AscendC::Te::Copy(copyGM2UB_, ubTensor, srcSlice);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);

        auto localSlice = localTensor.Slice(
            AscendC::Te::MakeCoord(rowStart, colStart),
            AscendC::Te::MakeShape(static_cast<int64_t>(curRows), static_cast<int64_t>(X_PER_BLOCK_NUM)));
        auto agSlice = agTensor.Slice(
            AscendC::Te::MakeCoord(rowStart, colStart),
            AscendC::Te::MakeShape(static_cast<int64_t>(curRows), static_cast<int64_t>(X_PER_BLOCK_NUM)));

        AscendC::Te::Copy(copyUB2GM_, localSlice, ubTensor);
        AscendC::Te::Copy(copyUB2GM_, agSlice, ubTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
    }
}

} // namespace Apace
