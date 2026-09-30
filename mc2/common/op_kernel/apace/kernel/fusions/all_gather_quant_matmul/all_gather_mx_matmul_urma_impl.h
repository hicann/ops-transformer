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
 * \file all_gather_mx_matmul_urma_impl.h
 * \brief AllGatherQuantMatmul 算子实现，基于 FragmentTensor + 通算解耦（UDMA 通信）
 */

#pragma once

#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apace/kernel/fusions/all_gather_quant_matmul/all_gather_mx_matmul_tiling_data.h"
#include "apace/kernel/matmul/quant_batch_matmul/all_gather_qbmm_mx_kernel.h"
#include "adv_api/hcomm/hcomm.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "apace/core/aiv_comm/collective_comm_api.h"
#include "apace/core/aiv_comm/barrier/barrier_ubmem.h"
#include "apace/tiling/comm_tiling_data.h"
#include "apace/basic/buffer/buffer_channel.h"
#include "include/tensor_api/tensor.h"
#include "../../../utils/op_state_dump.h"

namespace Apace {

using namespace AscendC;
using namespace Apace::AivComm;

using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::dn_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

struct UrmaBufferSync {
    __aicore__ inline void Acquire(uint32_t flagId)
    {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagId);
    }
    __aicore__ inline void Release(uint32_t flagId)
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flagId);
    }
};

template <typename AType, typename BType, typename CType>
class AllGatherMxMatmulUrmaImpl {
public:
    __aicore__ inline AllGatherMxMatmulUrmaImpl() {}
    __aicore__ inline void Init(__gm__ CommContext* hcommCtx, GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM,
                                GM_ADDR bScaleGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM,
                                const AllGatherMxMatmulUrmaTilingData* tilingData);
    __aicore__ inline void Process();

    using QuantMatmulKernelImpl = AllGatherQbmmMxKernel<AType, BType, CType, UrmaBufferSync, true>;
    using KernelParams = typename QuantMatmulKernelImpl::Params;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    using FragmentParams = typename QuantMatmulKernelImpl::FragmentParams;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
    __aicore__ inline void InitBaseParams(const AllGatherMxMatmulUrmaTilingData* td);

    __aicore__ inline void AllGatherProcess();
    __aicore__ inline void MatmulProcess();

    __aicore__ inline GM_ADDR GetWinDataRegionBase();
    __aicore__ inline GM_ADDR GetWinScaleRegionBase();

    // 通信
    using AllGatherCommData = Apace::AivComm::CollectiveComm<Apace::AivComm::CommCollectiveOp::AllGather,
                                                             Apace::AivComm::CommMode::PUT, AType, TeamBarrier>;
    using AllGatherCommScale =
        Apace::AivComm::CollectiveComm<Apace::AivComm::CommCollectiveOp::AllGather, Apace::AivComm::CommMode::PUT,
                                       AscendC::fp8_e8m0_t, TeamBarrier>;
    AllGatherCommData allGatherData_;
    AllGatherCommScale allGatherScale_;
    CommTilingData commTilingData_{};
    CommTilingData commTilingScale_{};
    TeamBarrier teamBarrier_;

    // channel 由 fusion 统一定义并注入 kernel（对齐 a2amm 模板）：AIC/AIV 各持一份成员，游标独立推进
    Apace::Basic::BufferChannel dataChannel_;
    Apace::Basic::BufferChannel scaleChannel_;

    static constexpr uint32_t kPaddingLength = 16;

    __gm__ CommContext* hcommCtx_{};
    __gm__ CommUdmaContext* udmaCtx_{};
    __gm__ CommUbmemContext* ubmemCtx_{};

    GM_ADDR aGM_{};
    GM_ADDR aScaleGM_{};
    GM_ADDR bGM_{};
    GM_ADDR bScaleGM_{};
    GM_ADDR cGM_{};
    GM_ADDR biasGM_{};

    uint32_t rankId_{};
    uint32_t rankSize_{};
    uint32_t m_{};
    uint32_t k_{};
    uint32_t n_{};
    uint32_t tileCnt_{};
    uint32_t tileM_{};
    uint32_t tailCnt_{};
    uint32_t tailM_{};
    uint32_t paddedTailM_{};
    uint32_t commTurn_{};
    uint64_t headRows_{};
    uint64_t scaleKLen_{};
    uint64_t dataBytesPerMRow_{};
    uint64_t scaleBytesPerMRow_{};
    uint64_t cBytesPerM_{};
    uint64_t scaleKGroups_{};
    uint32_t bufferCount_{0}; // data 复用槽位数（已归一化：tiling 传 0 或 >=commTurn 折算为 commTurn，即不复用）
    uint64_t scaleWinOffset_{0}; // scale 段的 winOffset（= data 段 footprint）

    const AllGatherMxMatmulUrmaTilingData* tilingData_{};
    Mc2Kernel::OpStateDump opStateDump_{};
    static constexpr uint8_t POS_COMM_BEFORE = 0U;
    static constexpr uint8_t POS_COMP_CUBE_MATMUL_BEFORE = 1U;
};

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::Init(
    __gm__ CommContext* hcommCtx, GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
    GM_ADDR biasGM, GM_ADDR workspaceGM, const AllGatherMxMatmulUrmaTilingData* tilingData)
{
    tilingData_ = tilingData;
    hcommCtx_ = hcommCtx;
    aGM_ = aGM;
    aScaleGM_ = aScaleGM;
    bGM_ = bGM;
    bScaleGM_ = bScaleGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
#if MC2_DFX_ENABLE
    opStateDump_.Init(workspaceGM, &tilingData->dumpInfo.workspaceLayout, tilingData->mmTile.usedCoreNum);
#endif

    InitBaseParams(tilingData);

    udmaCtx_ = &(hcommCtx_->udmaCtx);
    ubmemCtx_ = &(hcommCtx_->ubmemCtx);
    rankId_ = udmaCtx_->rankId;
    rankSize_ = udmaCtx_->rankSize;

    // Win buffer 布局（remote 紧凑 + buffer 复用）：
    //   data 段：bufferCount_ 个 slot，每 slot 连续放 (rankSize-1) 个 remote tile 段（跳过本卡，
    //            remoteIdx 紧凑排列），slot 随通信轮次回绕复用。
    //   scale 段：紧接 data 段之后（scaleWinOffset_ = data 段足迹），全量 commTurn_ 个 slot，不回绕。
    uint64_t maxTileM = (tileM_ > tailM_) ? static_cast<uint64_t>(tileM_) : static_cast<uint64_t>(tailM_);
    dataChannel_.Init(static_cast<uint64_t>(rankSize_ - 1) * maxTileM * dataBytesPerMRow_, bufferCount_);
    scaleChannel_.Init(static_cast<uint64_t>(rankSize_ - 1) * maxTileM * scaleBytesPerMRow_, commTurn_);
    scaleWinOffset_ = dataChannel_.GetCapacity();
    quantMatmulKernelImpl_.SetChannels(&dataChannel_, &scaleChannel_);

    // 静态 tensor 替代 tpipe buffer
    uint32_t ubOffset = 0;
    auto commBuf = asc::te::make_mem_ptr<asc::te::location::ub, uint8_t>(ubOffset);
    ubOffset += COMM_WORKSPACE_SIZE;
    auto commScaleBuf = asc::te::make_mem_ptr<asc::te::location::ub, uint8_t>(ubOffset);
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierBuf = asc::te::make_mem_ptr<asc::te::location::ub, uint8_t>(ubOffset);
    teamBarrier_.Init(barrierBuf.get(), ubmemCtx_, rankSize_, static_cast<uint32_t>(GetBlockIdx()));

    commTilingData_.splitAxisTileSize = tileM_;  // 每个 tile 搬 tileM 行
    commTilingData_.splitAxisTileCnt = tileCnt_; // head 段 tile 数量
    commTilingData_.splitAxisTailSize = tailM_;  // tail 行数
    commTilingData_.splitAxisTailCnt = tailCnt_;
    // 通信按 Dtype 元素个数计：FP4(fp4x2) 每元素 2 个逻辑值，故按字节行宽折算
    commTilingData_.nonSplitAxisSize = dataBytesPerMRow_ / sizeof(AType); // 非切分轴k
    commTilingData_.slotNum = tilingData_->commTile.slotNum;              // data 复用（透传 tiling 决策）

    commTilingScale_.splitAxisTileSize = tileM_;
    commTilingScale_.splitAxisTileCnt = tileCnt_;
    commTilingScale_.splitAxisTailSize = tailM_;
    commTilingScale_.splitAxisTailCnt = tailCnt_;
    scaleKLen_ = scaleKGroups_ * static_cast<uint64_t>(Blaze::Gemm::MXFP_MULTI_BASE_SIZE);
    commTilingScale_.nonSplitAxisSize = scaleKLen_;
    allGatherData_.template Init<BARRIER_NONE>(udmaCtx_, teamBarrier_, commTilingData_, aGM_, commBuf.get(), rankSize_,
                                               static_cast<uint32_t>(GetBlockIdx()));
    allGatherScale_.Init(udmaCtx_, teamBarrier_, commTilingScale_, aScaleGM_, commScaleBuf.get(), rankSize_,
                         static_cast<uint32_t>(GetBlockIdx()), scaleWinOffset_);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::InitBaseParams(
    const AllGatherMxMatmulUrmaTilingData* td)
{
    const auto& ct = td->commTile;
    tileCnt_ = static_cast<uint32_t>(ct.splitAxisTileCnt);
    tileM_ = static_cast<uint32_t>(ct.splitAxisTileSize);
    tailCnt_ = static_cast<uint32_t>(ct.splitAxisTailCnt);
    tailM_ = static_cast<uint32_t>(ct.splitAxisTailSize);
    k_ = td->mmTile.k;
    n_ = td->mmTile.n;
    m_ = static_cast<uint32_t>(ct.splitAxisTileSize * ct.splitAxisTileCnt + ct.splitAxisTailSize * ct.splitAxisTailCnt);
    commTurn_ = tileCnt_ + tailCnt_;
    paddedTailM_ = (tailM_ > 0) ? ((tailM_ + kPaddingLength - 1) / kPaddingLength * kPaddingLength) : 0U;
    headRows_ = static_cast<uint64_t>(tileCnt_) * tileM_;

    scaleKGroups_ = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(k_), Blaze::Gemm::MXFP_DIVISOR_SIZE);
    // FP4 (fp4x2) 1 字节装 2 个元素：每行字节数 = k/2；FP8 等 = k * sizeof(AType)
    dataBytesPerMRow_ =
        Blaze::Gemm::IsFp4<AType>() ? (static_cast<uint64_t>(k_) >> 1) : static_cast<uint64_t>(k_) * sizeof(AType);
    scaleBytesPerMRow_ =
        scaleKGroups_ * static_cast<uint64_t>(Blaze::Gemm::MXFP_MULTI_BASE_SIZE) * sizeof(AscendC::fp8_e8m0_t);
    cBytesPerM_ = static_cast<uint64_t>(n_) * sizeof(CType);

    // buffer 复用归一化：0 或 >= commTiles 视为不复用（全量），统一折算进 bufferCount_
    uint64_t commTiles = static_cast<uint64_t>(commTurn_);
    uint64_t rawBufferCount = td->commTile.slotNum;
    bufferCount_ =
        static_cast<uint32_t>((rawBufferCount > 0 && rawBufferCount < commTiles) ? rawBufferCount : commTiles);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::Process()
{
    if ASCEND_IS_AIV {
        AllGatherProcess();
    }
    if ASCEND_IS_AIC {
        MatmulProcess();
    }
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::MatmulProcess()
{
    KernelParams params;
    params.mmTile = &tilingData_->mmTile;
    params.qbmmParams = {tilingData_->mmTile.baseM, tilingData_->mmTile.baseN, tilingData_->mmTile.baseK,
                         tilingData_->mmTile.dbL0c};
    params.fragParams = {tileCnt_,
                         tileM_,
                         tailCnt_,
                         tailM_,
                         paddedTailM_,
                         commTurn_,
                         headRows_,
                         rankId_,
                         rankSize_,
                         m_,
                         static_cast<uint64_t>(k_),
                         static_cast<uint64_t>(n_),
                         scaleKLen_};
    params.aGM = aGM_;
    params.aScaleGM = aScaleGM_;
    params.bGM = bGM_;
    params.bScaleGM = bScaleGM_;
    params.cGM = cGM_;
    params.biasGM = biasGM_;
    params.isBias = (tilingData_->isBias != 0);
    params.winDataBase = GetWinDataRegionBase();
    params.winScaleBase = GetWinScaleRegionBase();
    params.dataBytesPerMRow = dataBytesPerMRow_;
    params.scaleBytesPerMRow = scaleBytesPerMRow_;
    params.cBytesPerM = cBytesPerM_;
    opStateDump_.DoDump(DUMP_FIELD_STEP, POS_COMP_CUBE_MATMUL_BEFORE, static_cast<uint8_t>(Utils::RT_PHASE_COMM_WAIT));
    quantMatmulKernelImpl_(params, opStateDump_);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline GM_ADDR AllGatherMxMatmulUrmaImpl<AType, BType, CType>::GetWinDataRegionBase()
{
    return reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[rankId_]);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline GM_ADDR AllGatherMxMatmulUrmaImpl<AType, BType, CType>::GetWinScaleRegionBase()
{
    return GetWinDataRegionBase() + scaleWinOffset_;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::AllGatherProcess()
{
    // 写端直接复用上述 fusion 成员 channel（读端 kernel 经 SetChannels 注入同一配置，游标各自独立推进）。
    //   data 槽回绕复用（bufferCount_ 归一化后），scale 槽恒全量（commTurn_）。
    //   每 slot = (rankSize-1) 个 remote tile 段，与 comm 层 DoCommit 的槽内布局一致。

    // 预触发 HEAD：本 rank 数据不经通信，flagId=0 硬编码，不走 BufferChannel。
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0);

    opStateDump_.DoDump(DUMP_FIELD_STEP, POS_COMM_BEFORE, static_cast<uint8_t>(Utils::RT_PHASE_COMM_COMMIT));
    for (uint32_t round = 0; round < commTurn_; ++round) {
        auto dataSlot = dataChannel_.GetNextSlot();
        auto scaleSlot = scaleChannel_.GetNextSlot();
        uint32_t flagId = dataSlot.slotIdx + 1; // +1 避开 HEAD 预触发的 flagId=0

        // win buffer 复用：写 slot round%dataSlotNum 会覆盖前一轮，需先等 AIC 消费完并 Release。
        if (round >= dataChannel_.GetSlotNum()) {
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(flagId);
            AscendC::SyncAll<true>();
            if (static_cast<uint32_t>(GetBlockIdx()) < rankSize_) {
                teamBarrier_.CrossDevice();
            }
        }
        if (static_cast<uint32_t>(GetBlockIdx()) < rankSize_) {
            allGatherScale_.Commit(scaleSlot.offset);
            allGatherData_.Commit(dataSlot.offset);
            opStateDump_.DoDump(DUMP_FIELD_COMMIT);
            allGatherData_.template Wait<BARRIER_DEVICE>();
            opStateDump_.DoDump(DUMP_FIELD_WAIT);
        }
        AscendC::SyncAll<true>();
        // data-ready：round 0 对应 dependTileIdx=1，AIC 侧按 round 等待对应远端数据
        CrossCoreSetFlag<0x2, PIPE_MTE3>(flagId);
    }
    allGatherScale_.Finalize();
    allGatherData_.Finalize();
    opStateDump_.DoDump(DUMP_FIELD_PHASE, 0, static_cast<uint8_t>(Utils::RT_PHASE_COMM_FINALIZE));
    // 通信接口使用了 channel 内的字段，需要刷新 GM 地址保证跨 kernel launch 的 GM 状态与 aiCore cache 一致
    // 后续通信接口会内置刷 cache 机制，届时可移除此处调用
    dcci(reinterpret_cast<__gm__ void*>(udmaCtx_->commBufferAddrs), static_cast<uint64_t>(CacheLine::ENTIRE_DATA_CACHE),
         static_cast<uint64_t>(DcciDst::CACHELINE_OUT));
}

} // namespace Apace

__global__ __aicore__ void AllGatherQuantMatmulKernel(__gm__ Apace::AivComm::CommContext* hcommCtx, GM_ADDR aGM,
                                                      GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
                                                      GM_ADDR biasGM, AllGatherMxMatmulUrmaTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllGatherMxMatmulUrmaImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t> impl;
    impl.Init(hcommCtx, aGM, aScaleGM, bGM, bScaleGM, cGM, biasGM, nullptr, &tilingData);
    impl.Process();
}
