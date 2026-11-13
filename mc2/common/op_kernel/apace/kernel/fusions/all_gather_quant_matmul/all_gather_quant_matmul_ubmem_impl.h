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
 */

#pragma once

#include "blaze/gemm/kernel/kernel_qbmm_mx.h"
#include "blaze/gemm/utils/common_utils.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "apace/tiling/matmul_tiling/quant_matmul_tiling_data.h"

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

constexpr AscendC::SyncAllConfig AG_WAVE_SYNC_CFG = {PIPE_MTE3, PIPE_MTE3};

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
    static constexpr uint64_t DATA_START_OFFSET = 4UL * 1024UL;
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
    static constexpr uint32_t PREFETCH_MAX_ROWS = 14U;

    using ScalesType = int8_t;

    using MakeLayoutGM =
        asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<int8_t>>;
    using MakeLayoutUB =
        asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<int8_t>>;
    using MakeLayoutScaleGM =
        asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<ScalesType>>;
    using MakeLayoutScaleUB =
        asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<ScalesType>>;
    using MakeLayoutStatusGM =
        asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<float>>;
    using MakeLayoutStatusUB =
        asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<float>>;

    using CopyGM2UB_t = decltype(asc::te::make_copy(asc::te::copy_gm_to_ub{}));
    using CopyUB2GM_t = decltype(asc::te::make_copy(asc::te::copy_ub_to_gm{}));

    struct Params {
        typename KernelImpl::Params matmulKernelParams;
        // 指向 kernel 入口解析出的本地 tiling 副本(仅需 m/k)
        const QuantMatmulTilingData* tilingData{nullptr};
        GM_ADDR hcclContext{nullptr};
        GM_ADDR x1Addr{nullptr};
        GM_ADDR x1ScaleAddr{nullptr};
    };

    __aicore__ inline AllGatherQuantMatmulUbmemImpl() {}
    __aicore__ inline ~AllGatherQuantMatmulUbmemImpl() {}

    __aicore__ inline void operator()(Params& params)
    {
        if ASCEND_IS_AIV {
            AllGatherProcess(params);
        } else {
            MatmulProcess(params);
        }
    }

private:
    __aicore__ inline void AllGatherProcess(const Params& params);
    __aicore__ inline void MatmulProcess(Params& params);

    __aicore__ inline void AllGatherInit(const Params& params);
    __aicore__ inline void InitUbBufferLayout();
    __aicore__ inline void CopyLocalData();
    __aicore__ inline void CopyLocalXData();
    __aicore__ inline void CopyLocalScale();
    __aicore__ inline void ExecuteAllGather();
    __aicore__ inline void AllGatherScale();
    __aicore__ inline void AllGatherXData();
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void ReadDataBlock(uint64_t curXOffset, uint32_t mCnt, uint32_t bufIdx);
    __aicore__ inline bool HasDataWork() const;
    __aicore__ inline bool IsSingleChunkWave(uint32_t mCnt) const;
    __aicore__ inline void IssueReadDataBlock(uint64_t curXOffset, uint32_t mCnt, uint32_t bufIdx);
    __aicore__ inline void WriteDataBlock(uint64_t curXOffset, uint32_t mCnt, uint32_t bufIdx);

    __aicore__ inline GM_ADDR WinDataAddr(uint32_t rankId)
    {
        return (GM_ADDR)(hcclContext_->commBufferAddrs[rankId] + DATA_START_OFFSET);
    }
    __aicore__ inline GM_ADDR WinStatusAddr(uint32_t rankId)
    {
        return (GM_ADDR)(hcclContext_->commBufferAddrs[rankId]);
    }

    // x1 每行字节数: fp4(2 元素/字节) 为 K/2, fp8 系为 K
    static __aicore__ inline uint64_t BytesPerRow(uint64_t k)
    {
        return Blaze::Gemm::IsFp4<AType>() ? ((k + 1UL) >> 1) : k;
    }

private:
    __gm__ Apace::AivComm::CommUbmemContext* hcclContext_{nullptr};

    uint32_t aivId_{0};
    uint32_t curRankId_{0};
    uint32_t rankSize_{0};
    uint64_t winDataSize_{0};

    uint64_t M_{0};
    uint64_t K_{0};
    uint64_t bytesPerRow_{0}; // x1 每行字节数: fp4=K/2(4bit 打包), fp8=K
    uint64_t xSize_{0};       // 单 rank x1 数据字节数(= M_ * bytesPerRow_)
    uint64_t scaleSize_{0};
    uint64_t scaleChunkSize_{0};
    uint32_t scaleTotalChunks_{0};

    uint32_t numRemoteRanks_{0};
    uint32_t groupSize_{0};
    uint32_t groupIdx_{0};
    uint32_t groupCoreIdx_{0};
    uint32_t remoteRankId_{0};
    bool isLeftover_{false};
    bool isScaleCore_{false};
    uint64_t tileK_{0};
    uint32_t kDim_{0};
    uint32_t mDim_{0};
    uint32_t mBlockIdx_{0};
    uint32_t kBlockIdx_{0};
    uint32_t mMteCoreM_{0};
    uint32_t coreInnerMIndex_{0};
    uint64_t kStartIndex_{0};     // 本条 K 波内起点(字节空间)
    uint32_t sliceWidthBytes_{0}; // 单 K 切片宽度(字节): fp4=256(512 元素), fp8=512

    uint64_t ubWriteStateOffset_{0};
    uint64_t ubReadStateOffset_{0};
    uint64_t ubScalePingOffset_{0};
    uint64_t ubScalePongOffset_{0};
    uint64_t ubXPingOffset_{0};
    uint64_t ubXPongOffset_{0};
    uint64_t xBufSize_{0};

    GM_ADDR inputXAddr_{nullptr};
    GM_ADDR inputXScaleAddr_{nullptr};

    CopyGM2UB_t copyGM2UB_ = asc::te::make_copy(asc::te::copy_gm_to_ub{});
    CopyUB2GM_t copyUB2GM_ = asc::te::make_copy(asc::te::copy_ub_to_gm{});
};

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue,
                                                     BlockScheduler>::AllGatherProcess(const Params& params)
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
                                                     BlockScheduler>::AllGatherInit(const Params& params)
{
    hcclContext_ = (__gm__ Apace::AivComm::CommUbmemContext*)params.hcclContext;
    rankSize_ = hcclContext_->rankSize;
    curRankId_ = hcclContext_->rankId;
    inputXAddr_ = params.x1Addr;
    inputXScaleAddr_ = params.x1ScaleAddr;
    M_ = params.tilingData->m / rankSize_;
    K_ = params.tilingData->k;

    bytesPerRow_ = BytesPerRow(K_);
    sliceWidthBytes_ = X_PER_BLOCK_NUM >> (Blaze::Gemm::IsFp4<AType>() ? 1 : 0);
    xSize_ = M_ * bytesPerRow_;
    scaleSize_ = M_ * Blaze::Gemm::CeilDiv(K_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    tileK_ = Blaze::Gemm::CeilDiv(K_, TILE_K_NUM);
    aivId_ = GetBlockIdx();

    winDataSize_ = Blaze::Gemm::CeilAlign(rankSize_ * xSize_, WIN_ADDR_ALIGN);
    numRemoteRanks_ = rankSize_ - 1U;
    groupSize_ = GetBlockNum() / numRemoteRanks_;
    groupIdx_ = aivId_ / groupSize_;
    groupCoreIdx_ = aivId_ % groupSize_;
    isLeftover_ = (groupIdx_ >= numRemoteRanks_);
    // 做分组轮转，例如rank0的组0负责rank1，组1负责rank2，组2负责rank3；rank2的组0负责rank3，组1负责rank0，组2负责rank1
    remoteRankId_ = isLeftover_ ? curRankId_ : ((curRankId_ + 1U + groupIdx_) % rankSize_);

    kDim_ = tileK_ < DEFAULT_K_SPLIT_NUM ? static_cast<uint32_t>(tileK_) :
                                           DEFAULT_K_SPLIT_NUM; // k方向按照核来切，做多切一刀，也就是最多两个核负责
    // 当ranksize比较大，且核少时，可能出现，需截
    if (kDim_ > groupSize_) {
        kDim_ = groupSize_;
    }
    mDim_ = groupSize_ / kDim_;
    if (mDim_ > M_) {
        mDim_ = static_cast<uint32_t>(M_);
    }

    mBlockIdx_ = groupCoreIdx_ % mDim_;
    kBlockIdx_ = groupCoreIdx_ / mDim_;
    isScaleCore_ = (!isLeftover_ && (groupCoreIdx_ == groupSize_ - 1U));
    mMteCoreM_ = static_cast<uint32_t>(M_) / mDim_;

    uint32_t mTailNum = static_cast<uint32_t>(M_) % mDim_;
    // 均衡逻辑
    if (mBlockIdx_ < mTailNum) {
        mMteCoreM_ += 1;
        coreInnerMIndex_ = mBlockIdx_ * mMteCoreM_;
    } else {
        coreInnerMIndex_ = mBlockIdx_ * mMteCoreM_ + mTailNum;
    }
    kStartIndex_ = static_cast<uint64_t>(kBlockIdx_) * sliceWidthBytes_;

    InitUbBufferLayout();
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::InitUbBufferLayout()
{
    // UB 前段只放两块 32B 状态缓冲: 写 flag(偏移 0) + 读 flag(UB_ALIGN_BYTES)
    ubWriteStateOffset_ = 0;
    ubReadStateOffset_ = UB_ALIGN_BYTES;
    uint64_t workStart = 2 * UB_ALIGN_BYTES;
    uint64_t workAvailable = UB_AVAIL_SIZE - workStart;

    // scale 双 buffer(分阶段与 x 的 ping/pong 复用 workStart 之后的空间)
    scaleChunkSize_ = (workAvailable / BUFFER_NUM / UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    scaleTotalChunks_ = (scaleSize_ + scaleChunkSize_ - 1) / scaleChunkSize_;
    ubScalePingOffset_ = workStart;
    ubScalePongOffset_ = workStart + scaleChunkSize_;

    // x 双 buffer: 每块不超过半区可用空间，也不超过本核实际数据量
    uint64_t totalAvailable = workAvailable / X_BLOCK_BYTES * X_BLOCK_BYTES;
    uint64_t demandSpace = Blaze::Gemm::CeilAlign(xSize_, X_BLOCK_BYTES);
    uint64_t availableSpace = totalAvailable / BUFFER_NUM / X_BLOCK_BYTES * X_BLOCK_BYTES;
    xBufSize_ = (availableSpace < demandSpace) ? availableSpace : demandSpace;
    ubXPingOffset_ = workStart;
    ubXPongOffset_ = workStart + xBufSize_;
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::MatmulProcess(Params& params)
{
    const QuantMatmulTilingData* tilingData = params.tilingData;

    auto* hcclCtx = (__gm__ Apace::AivComm::CommUbmemContext*)params.hcclContext;
    uint32_t curRankId = hcclCtx->rankId;
    uint32_t rankSize = hcclCtx->rankSize;

    uint64_t M = tilingData->m / rankSize;
    uint64_t K = tilingData->k;

    uint64_t xTotalBytes = M * BytesPerRow(K);
    uint64_t winScaleOffset = Blaze::Gemm::CeilDiv(rankSize * xTotalBytes, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;

    GM_ADDR selfDataAddr = (GM_ADDR)(hcclCtx->commBufferAddrs[curRankId] + DATA_START_OFFSET);
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
    // 最后一个 AIV 负责本 rank 的 scale 拷贝，其余 AIV 均分 M 行 x1 数据
    uint32_t scaleCoreIdx = GetBlockNum() - 1;
    if (aivId_ == scaleCoreIdx) {
        CopyLocalScale();
        return;
    }
    CopyLocalXData();
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::CopyLocalXData()
{
    GM_ADDR localWinGm = WinDataAddr(curRankId_);
    uint64_t selfDataOffset = curRankId_ * xSize_;

    uint32_t copyXBlockNum = GetBlockNum() - 1;
    uint32_t mPerCore = M_ / copyXBlockNum;
    uint32_t remainderNum = M_ % copyXBlockNum;
    uint32_t mStartId;
    if (aivId_ < remainderNum) {
        mPerCore += 1;
        mStartId = aivId_ * mPerCore;
    } else {
        mStartId = remainderNum * (mPerCore + 1) + (aivId_ - remainderNum) * mPerCore;
    }

    uint64_t totalBytes = static_cast<uint64_t>(mPerCore) * bytesPerRow_;
    uint64_t startGlobalOffset = static_cast<uint64_t>(mStartId) * bytesPerRow_;

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);

    for (uint64_t done = 0; done < totalBytes; done += xBufSize_) {
        uint32_t chunkIdx = static_cast<uint32_t>(done / xBufSize_);
        uint16_t evtId = (chunkIdx % 2 == 0) ? EVT_ID_PING : EVT_ID_PONG;
        uint64_t ubOff = (chunkIdx % 2 == 0) ? ubXPingOffset_ : ubXPongOffset_;
        uint32_t chunkBytes = static_cast<uint32_t>((done + xBufSize_ > totalBytes) ? (totalBytes - done) : xBufSize_);
        uint64_t globalOffset = startGlobalOffset + done;

        auto xUbLayout = MakeLayoutUB{}(1, chunkBytes);
        auto xUbTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubOff), xUbLayout);

        auto xSrcLayout = MakeLayoutGM{}(1, chunkBytes);
        auto xSrcTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ int8_t*)(inputXAddr_ + globalOffset)), xSrcLayout);

        auto xDstLayout = MakeLayoutGM{}(1, chunkBytes);
        auto xDstTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ int8_t*)(localWinGm + selfDataOffset + globalOffset)),
            xDstLayout);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
        asc::te::copy(copyGM2UB_, xUbTensor, xSrcTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
        asc::te::copy(copyUB2GM_, xDstTensor, xUbTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::CopyLocalScale()
{
    GM_ADDR localWinGm = WinDataAddr(curRankId_);
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
        auto scaleSrcTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ ScalesType*)(inputXScaleAddr_ + chunkOffset)),
            scaleSrcLayout);

        auto scaleUbLayout = MakeLayoutScaleUB{}(1, curChunkPaddedN);
        auto scaleUbTensor =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ScalesType>(ubOff), scaleUbLayout);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
        asc::te::copy(copyGM2UB_, scaleUbTensor, scaleSrcTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);

        auto scaleDstLayout = MakeLayoutScaleGM{}(1, curChunkSize);
        auto scaleDstTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ ScalesType*)(localWinGm + winScaleBase + chunkOffset)),
            scaleDstLayout);
        asc::te::copy(copyUB2GM_, scaleDstTensor, scaleUbTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PONG);
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::ExecuteAllGather()
{
    // scale 核心额外负责把下一个 peer 的 scale 搬进本 rank 窗口，之后和其余核一起搬 x
    if (isScaleCore_) {
        AllGatherScale();
    }
    AllGatherXData();
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::AllGatherScale()
{
    GM_ADDR remoteScaleAddr = WinDataAddr(remoteRankId_) + winDataSize_ + remoteRankId_ * scaleSize_;
    GM_ADDR localScaleAddr = WinDataAddr(curRankId_) + winDataSize_ + remoteRankId_ * scaleSize_;

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
        auto scaleSrcTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ ScalesType*)(remoteScaleAddr + chunkOffset)),
            scaleSrcLayout);

        auto scaleUbLayout = MakeLayoutScaleUB{}(1, curChunkPaddedN);
        auto scaleUbTensor =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ScalesType>(ubOff), scaleUbLayout);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
        asc::te::copy(copyGM2UB_, scaleUbTensor, scaleSrcTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);

        auto localScaleLayout = MakeLayoutScaleGM{}(1, curChunkSize);
        auto localScaleTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ ScalesType*)(localScaleAddr + chunkOffset)),
            localScaleLayout);

        asc::te::copy(copyUB2GM_, localScaleTensor, scaleUbTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_SCALE_PONG);
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::AllGatherXData()
{
    uint32_t kLoop = tileK_ / kDim_;
    // 可能出现k方向一个核多分一个block场景
    if (kBlockIdx_ < tileK_ % kDim_) {
        kLoop++;
    }
    uint64_t curXOffset = static_cast<uint64_t>(coreInnerMIndex_) * bytesPerRow_ + kStartIndex_;

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);

    const uint32_t maxRowsPerCore = (M_ + mDim_ - 1) / mDim_;
    // 单波数据装得进一个 buffer且行数少(收益场景)时, 预取下一波 K 数据与当前波写回重叠
    const bool enablePrefetch = IsSingleChunkWave(maxRowsPerCore) && (maxRowsPerCore <= PREFETCH_MAX_ROWS);
    if (enablePrefetch) {
        IssueReadDataBlock(curXOffset, mMteCoreM_, 0);
    }

    for (uint64_t curKBlock = 0; curKBlock < kLoop; ++curKBlock) {
        uint64_t innerCurXOffset = curXOffset + curKBlock * sliceWidthBytes_ * 2; // 因为k方向2核，所以需乘2
        if (enablePrefetch) {
            WriteDataBlock(innerCurXOffset, mMteCoreM_, curKBlock % 2);
            if (curKBlock + 1 < kLoop) {
                uint64_t nextXOffset = curXOffset + (curKBlock + 1) * sliceWidthBytes_ * 2;
                IssueReadDataBlock(nextXOffset, mMteCoreM_, (curKBlock + 1) % 2);
            }
        } else {
            ReadDataBlock(innerCurXOffset, mMteCoreM_, curKBlock % 2);
        }
        PipeBarrier<PIPE_MTE3>();
        if (enablePrefetch) {
            SyncAll<true, AG_WAVE_SYNC_CFG>();
        } else {
            SyncAll<true>();
        }
        // 本次 K 数据全部落盘后通知 AIC, MMAD 侧按 K 波等待并立即开始计算。
        CrossCoreSetFlag<CROSS_CORE_INNER_CUBE_VEC_SYNC, PIPE_MTE3>(CROSS_CORE_FLAG_NUM_NINE);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PING);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_ID_PONG);

    if (kLoop < ((tileK_ + kDim_ - 1) / kDim_)) {
        SyncAll<true>();
        // 补齐尾轮 flag, 保证 AIC 每核 wait 次数与 AIV set 次数一致(ceil(K/1024))。
        CrossCoreSetFlag<CROSS_CORE_INNER_CUBE_VEC_SYNC, PIPE_MTE3>(CROSS_CORE_FLAG_NUM_NINE);
    }
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::WriteStatusToWin()
{
    if (isLeftover_) {
        return;
    }
    // 每个非 leftover 核向一个 peer 的窗口写一个 flag，共覆盖 (rankSize-1) 个 peer × groupSize 槽位。
    uint32_t peerIdx = aivId_ / groupSize_;
    uint32_t statusSlot = curRankId_ * groupSize_ + (aivId_ % groupSize_);
    uint32_t peerRank = (curRankId_ + 1U + peerIdx) % rankSize_;
    uint64_t curOffset = statusSlot * FLOAT_UB_ALIGN_NUM;

    auto statusUbLayout = MakeLayoutStatusUB{}(1, FLOAT_UB_ALIGN_NUM);
    auto statusUbTensor =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(ubWriteStateOffset_), statusUbLayout);

    statusUbTensor[asc::te::make_coord(0, 0)] = STATUS_FLAG_VAL;

    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVT_ID);

    GM_ADDR dstAddr = WinStatusAddr(peerRank) + curOffset * sizeof(float);
    auto dstLayout = MakeLayoutStatusGM{}(1, FLOAT_UB_ALIGN_NUM);
    auto dstTensor =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ float*)(dstAddr)), dstLayout);
    asc::te::copy(copyUB2GM_, dstTensor, statusUbTensor);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVT_ID);
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::ReadStatus()
{
    if (isLeftover_) {
        return;
    }
    uint32_t statusSlot = remoteRankId_ * groupSize_ + groupCoreIdx_;
    uint32_t offset = statusSlot * FLOAT_UB_ALIGN_NUM;
    GM_ADDR statusAddr = WinStatusAddr(curRankId_) + offset * sizeof(float);

    auto statusGmLayout = MakeLayoutStatusGM{}(1, FLOAT_UB_ALIGN_NUM);
    auto statusGmTensor =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ float*)(statusAddr)), statusGmLayout);

    auto statusUbLayout = MakeLayoutStatusUB{}(1, FLOAT_UB_ALIGN_NUM);
    auto statusUbTensor =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(ubReadStateOffset_), statusUbLayout);

    float minTarget = STATUS_FLAG_VAL - STATUS_FLAG_THREDSHOLD;
    float maxTarget = STATUS_FLAG_VAL + STATUS_FLAG_THREDSHOLD;
    float flag = -1.0f;
    while ((flag < minTarget) || (flag > maxTarget)) {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVT_ID);
        asc::te::copy(copyGM2UB_, statusUbTensor, statusGmTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVT_ID);
        flag = statusUbTensor[asc::te::make_coord(0, 0)];
    }
    statusUbTensor[asc::te::make_coord(0, 0)] = 0.0f;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVT_ID);
    asc::te::copy(copyUB2GM_, statusGmTensor, statusUbTensor);
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

    // 字节空间: 行号 = 偏移/每行字节数, 列起点 = 偏移%每行字节数(fp4 每行 K/2 字节)
    uint64_t rowStartBase = curXOffset / bytesPerRow_;
    uint64_t colStart = curXOffset % bytesPerRow_;
    uint32_t rowsPerSlot = static_cast<uint32_t>(xBufSize_ / sliceWidthBytes_);

    // 窗口数据区按字节行宽布局(int8 视角): fp4 行宽 = K/2 字节
    auto srcTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ int8_t*)(remoteXAddr)),
                                          MakeLayoutGM{}(M_, bytesPerRow_));
    auto localTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ int8_t*)(localXAddr)),
                                            MakeLayoutGM{}(M_, bytesPerRow_));

    for (uint32_t rowDone = 0; rowDone < mCnt; rowDone += rowsPerSlot) {
        uint32_t curRows = (rowDone + rowsPerSlot > mCnt) ? (mCnt - rowDone) : rowsPerSlot;
        uint64_t rowStart = rowStartBase + rowDone;
        // K 尾波钳位(字节空间): 切片宽度不得超过该行剩余字节数
        uint32_t curCols = (colStart + sliceWidthBytes_ > bytesPerRow_) ?
                               static_cast<uint32_t>(bytesPerRow_ - colStart) :
                               sliceWidthBytes_;

        auto srcSlice =
            srcTensor.slice(asc::te::make_coord(rowStart, colStart),
                            asc::te::make_shape(static_cast<int64_t>(curRows), static_cast<int64_t>(curCols)));
        auto ubTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubOff),
                                             MakeLayoutUB{}(curRows, static_cast<int64_t>(curCols)));

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
        asc::te::copy(copyGM2UB_, ubTensor, srcSlice);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);

        auto localSlice =
            localTensor.slice(asc::te::make_coord(rowStart, colStart),
                              asc::te::make_shape(static_cast<int64_t>(curRows), static_cast<int64_t>(curCols)));

        asc::te::copy(copyUB2GM_, localSlice, ubTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
    }
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline bool
AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::HasDataWork() const
{
    return (remoteRankId_ != curRankId_) && (kBlockIdx_ < kDim_) && (mBlockIdx_ < mDim_);
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline bool AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue,
                                                     BlockScheduler>::IsSingleChunkWave(uint32_t mCnt) const
{
    // 一个 UB buffer 单次最多能放的行数; prefetch 路径整波一次搬完、不分块, 只有装得下才可用
    uint32_t rowsPerSlot = static_cast<uint32_t>(xBufSize_ / sliceWidthBytes_);
    return mCnt <= rowsPerSlot;
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue,
                                                     BlockScheduler>::IssueReadDataBlock(uint64_t curXOffset,
                                                                                         uint32_t mCnt, uint32_t bufIdx)
{
    if (!HasDataWork()) {
        return;
    }

    uint16_t evtId = (bufIdx % 2 == 0) ? EVT_ID_PING : EVT_ID_PONG;
    uint64_t ubOff = (bufIdx % 2 == 0) ? ubXPingOffset_ : ubXPongOffset_;
    GM_ADDR remoteWinGm = WinDataAddr(remoteRankId_);
    GM_ADDR remoteXAddr = remoteWinGm + remoteRankId_ * xSize_;

    uint64_t rowStart = curXOffset / bytesPerRow_;
    uint64_t colStart = curXOffset % bytesPerRow_;
    uint32_t curCols = (colStart + sliceWidthBytes_ > bytesPerRow_) ? static_cast<uint32_t>(bytesPerRow_ - colStart) :
                                                                      sliceWidthBytes_;

    auto srcTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ int8_t*)(remoteXAddr)),
                                          MakeLayoutGM{}(M_, bytesPerRow_));
    auto srcSlice = srcTensor.slice(asc::te::make_coord(rowStart, colStart),
                                    asc::te::make_shape(static_cast<int64_t>(mCnt), static_cast<int64_t>(curCols)));
    auto ubTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubOff),
                                         MakeLayoutUB{}(mCnt, static_cast<int64_t>(curCols)));

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
    asc::te::copy(copyGM2UB_, ubTensor, srcSlice);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue,
                                                     BlockScheduler>::WriteDataBlock(uint64_t curXOffset, uint32_t mCnt,
                                                                                     uint32_t bufIdx)
{
    if (!HasDataWork()) {
        return;
    }

    uint16_t evtId = (bufIdx % 2 == 0) ? EVT_ID_PING : EVT_ID_PONG;
    uint64_t ubOff = (bufIdx % 2 == 0) ? ubXPingOffset_ : ubXPongOffset_;
    GM_ADDR localWinGm = WinDataAddr(curRankId_);
    GM_ADDR localXAddr = localWinGm + remoteRankId_ * xSize_;

    uint64_t rowStart = curXOffset / bytesPerRow_;
    uint64_t colStart = curXOffset % bytesPerRow_;
    uint32_t curCols = (colStart + sliceWidthBytes_ > bytesPerRow_) ? static_cast<uint32_t>(bytesPerRow_ - colStart) :
                                                                      sliceWidthBytes_;

    auto localTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ int8_t*)(localXAddr)),
                                            MakeLayoutGM{}(M_, bytesPerRow_));
    auto localSlice = localTensor.slice(asc::te::make_coord(rowStart, colStart),
                                        asc::te::make_shape(static_cast<int64_t>(mCnt), static_cast<int64_t>(curCols)));
    auto ubTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubOff),
                                         MakeLayoutUB{}(mCnt, static_cast<int64_t>(curCols)));

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtId);
    asc::te::copy(copyUB2GM_, localSlice, ubTensor);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtId);
}

} // namespace Apace
