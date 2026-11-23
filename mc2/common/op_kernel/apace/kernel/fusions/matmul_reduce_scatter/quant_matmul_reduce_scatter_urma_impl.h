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
 * \file quant_matmul_reduce_scatter_urma_impl.h
 * \brief quantMatmulReduceScatter 算子实现
 *
 * AIC 逐 tile 计算部分和, AIV 通过 AllToAll PUT 交换并归约部分和.
 * 当前 tile 的通信与上一 tile 的归约重叠执行.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "apace/utils/apace_constant.h"
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "include/tensor_api/tensor.h"

#include "adv_api/hcomm/hcomm.h"
#include "blaze/gemm/utils/common_utils.h"

#include "apace/core/aiv_comm/collective_comm_api.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "apace/tiling/tiling_data_struct_define.h"
#include "apace/core/aiv_comm/barrier/barrier_ubmem.h"
#include "apace/basic/buffer/buffer_channel.h"

#include "quant_matmul_reduce_scatter_tiling_data.h"
#include "apace/kernel/matmul/quant_batch_matmul/qbmm_reduce_scatter_urma_kernel.h"
#include "apace/core/aiv_compute/reduce_sum.h"

namespace Apace {

using namespace Apace::AivComm;
using Apace::Basic::BufferChannel;

template <typename AType, typename BType, typename CType>
class QuantMatmulReduceScatterUrmaImpl {
public:
    __aicore__ inline QuantMatmulReduceScatterUrmaImpl() {}
    __aicore__ inline ~QuantMatmulReduceScatterUrmaImpl() {}

    __aicore__ inline void Init(__gm__ CommContext* hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM,
                                GM_ADDR scaleBGM, GM_ADDR workspaceGM, GM_ADDR cGM,
                                const QuantMatmulReduceScatterTilingData* tilingData);

    __aicore__ inline void Process();

    // Layout 定义
    using LayoutA = asc::te::nd_ext_layout_ptn;
    // B 矩阵布局: urma 通路固定转置 (N, K)
    using LayoutB = asc::te::dn_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    // 组件定义
    using BlockMmadFrag = Blaze::Gemm::Block::QmmMxBlockMmadFragment<0, false, AType, LayoutA, BType, LayoutB, CType,
                                                                     LayoutC, float, LayoutC>;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, AType>;
    using QuantMatmulKernelImpl = MatmulReduceScatterQbmmMxKernel<BlockMmadFrag, BlockScheduler>;
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using FragL1Params = typename QuantMatmulKernelImpl::FragL1Params;
    using FragmentParams = typename QuantMatmulKernelImpl::FragmentParams;

private:
    // ---------------- 通信相关组件 ----------------
    __gm__ CommUbmemContext* ubmemCtx_{nullptr};
    __gm__ CommUdmaContext* udmaCtx_{nullptr};
    // PUT 模式: AIC 写 C_i 到 workspaceGM, AIV PUT 推送到其他 rank win 区
    CollectiveComm<CommCollectiveOp::AllToAll, CommMode::PUT, CType, TeamBarrier> allToAllData_;
    TeamBarrier teamBarrier_;
    BufferChannel dataChannel_;

    struct BaseParams {
        __aicore__ inline BaseParams() = default;
        __aicore__ inline BaseParams(GM_ADDR aGm_, GM_ADDR scaleAGm_, GM_ADDR bGm_, GM_ADDR scaleBGm_,
                                     GM_ADDR workspaceGm_, GM_ADDR cGm_)
            : aGm(aGm_),
              scaleAGm(scaleAGm_),
              bGm(bGm_),
              scaleBGm(scaleBGm_),
              workspaceGm(workspaceGm_),
              cGm(cGm_)
        {}

        GM_ADDR selfWinAddr{nullptr}; // 本卡 win 数据区地址, 保存远端 rank 写入的部分和
        GM_ADDR aGm{nullptr};
        GM_ADDR scaleAGm{nullptr};
        GM_ADDR bGm{nullptr};
        GM_ADDR scaleBGm{nullptr};
        GM_ADDR workspaceGm{nullptr}; // AIC 写入部分和, AIV 从此读取并发送
        GM_ADDR cGm{nullptr};

        uint32_t rankId{0};
        uint32_t commTurn{0}; // 通信轮次, 等于 tile 总数
        uint32_t rankSize{0};
        uint64_t axisN{0};
        uint64_t axisK{0};
        uint64_t headMSize{0}; // 本 rank 每个头 tile 的有效 M 行数
        uint64_t tailMSize{0}; // 本 rank 尾 tile 的有效 M 行数, 无尾块时为 0
        uint32_t headTileCnt{0};
        uint32_t chunkM{0};          // 本 rank 的总 M 行数
        uint64_t maxTileM{0};        // win 槽内每个 rank 段预留的行数, 取头尾 tile 的较大值
        uint32_t commSlotNum{0};     // win 槽数, 取 3 与 commTurn 的较小值
        uint64_t bufferSlotBytes{0}; // 单个 win 槽的字节数
        uint64_t winDataOffset{0};   // [0, winDataOffset) 为保留前缀, PUT 数据区从其后开始
        uint64_t scaleARowBytes{0};  // ScaleA 每行字节数
    } baseParams_;

    // ---------------- reduceSum 组件 ----------------
    ReduceSumImpl::ReduceSum<CType, float> reduceSum_;

    const QuantMatmulReduceScatterTilingData* tilingData_{nullptr};

    // ---------------- AIC↔AIV CrossCore flag ----------------
    // CrossCore flag 的硬件 mode 编号, AIC 完成通知和 AIV 回执共用
    static constexpr uint8_t CROSS_CORE_NOTIFY_MODE = 0x2;
    // Fragment 的最小行对齐粒度
    static constexpr uint32_t MIN_ROW_ALIGN = 16;

    // L1 排空用 event id 6: 与 mmad 流水内部占用的 {0..5} 不冲突
    static constexpr uint16_t MTE1_MTE2_EVENT_FLAG = 6;

    // AIV 核角色: 后 rankSize 核 = 通信, 前 rsCoreNum 核 = 计算
    bool isCommBlock_{false};
    bool isComputeBlock_{false};
    uint32_t rsCoreNum_{0};

    // ---------------- 私有方法 ----------------
    __aicore__ inline void InitBaseParams(const QuantMatmulReduceScatterTilingData* tilingData);
    __aicore__ inline void MatmulProcess();
    __aicore__ inline void RunOneMatmulTile(uint32_t tileIdx, const QuantMatmulTilingData& tileT);
    __aicore__ inline void ReduceScatterProcess();
    __aicore__ inline void RunOneTileTurn(uint32_t tileIdx);
    __aicore__ inline void ReduceSumProcess(uint32_t tileIdx);
    __aicore__ inline void FlushUrmaWinPrefix();
    __aicore__ inline uint32_t GetTileM(uint32_t tileIdx);
    __aicore__ inline uint64_t GetTileMOffset(uint32_t tileIdx);
};

// =================================================================================
// 公有方法实现
// =================================================================================

template <typename AType, typename BType, typename CType>
__aicore__ inline void QuantMatmulReduceScatterUrmaImpl<AType, BType, CType>::Init(
    __gm__ CommContext* hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM, GM_ADDR workspaceGM,
    GM_ADDR cGM, const QuantMatmulReduceScatterTilingData* tilingData)
{
    tilingData_ = tilingData;
    baseParams_ = BaseParams(aGM, scaleAGM, bGM, scaleBGM, workspaceGM, cGM);

    ubmemCtx_ = &(hcommCtx->ubmemCtx);
    udmaCtx_ = &(hcommCtx->udmaCtx);
    baseParams_.rankId = udmaCtx_->rankId;
    baseParams_.rankSize = udmaCtx_->rankSize;

    InitBaseParams(tilingData);

    // 按两个 rankSize×32B 区域计算 URMA win 前缀保留量, 并按 128B 对齐;
    // PUT 数据从保留区之后开始. TeamBarrier 使用独立的 UBMEM buffer, 不位于该 URMA win 内
    uint32_t rankSize = baseParams_.rankSize;
    uint64_t winDataOffset =
        (WIN_BARRIER_REGION_NUM * static_cast<uint64_t>(rankSize) * WIN_BARRIER_FLAG_SIZE + WIN_DATA_ALIGN - 1) /
        WIN_DATA_ALIGN * WIN_DATA_ALIGN;
    baseParams_.selfWinAddr = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[baseParams_.rankId]) + winDataOffset;
    baseParams_.winDataOffset = winDataOffset;

    // 当前核类型下的实际核数: AIC = blockDim, AIV = blockDim × 2 (1: 2 配比).
    // GetBlockNum() 返回 launch blockDim, 不随 mix 比例放大, 必须乘 GetTaskRatio().
    uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    uint32_t taskRatio = static_cast<uint32_t>(AscendC::GetTaskRatio());
    uint32_t coreNum = blockNum * taskRatio;

    // 后 rankSize 个核参与 AllToAll PUT, 每核负责一个目标 rank; 本 rank 跳过 PUT
    // jobIndex = coreNum - 1 - blockIdx (最后一个 block jobIndex=0)
    uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    uint32_t jobIndex = coreNum - 1 - blockIdx;

    // 归约核数 = coreNum - rankSize; host 已保证 launch blockDim > rankSize
    rsCoreNum_ = coreNum - rankSize;
    isCommBlock_ = (jobIndex < rankSize);
    isComputeBlock_ = (blockIdx < rsCoreNum_);

    uint32_t ubOffset = 0;
    auto commBuf = asc::te::make_mem_ptr<asc::te::location::ub, uint8_t>(ubOffset);
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierBuf = asc::te::make_mem_ptr<asc::te::location::ub, uint8_t>(ubOffset);
    teamBarrier_.Init(barrierBuf.get(), ubmemCtx_, rankSize, jobIndex);

    allToAllData_.template Init<BARRIER_NONE>(udmaCtx_, teamBarrier_, tilingData->commTilingData,
                                              baseParams_.workspaceGm, commBuf.get(), rankSize, jobIndex,
                                              winDataOffset);

    dataChannel_.Init(baseParams_.bufferSlotBytes, baseParams_.commSlotNum);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void QuantMatmulReduceScatterUrmaImpl<AType, BType, CType>::Process()
{
    if (tilingData_ == nullptr || udmaCtx_ == nullptr || ubmemCtx_ == nullptr) {
        return;
    }
    if ASCEND_IS_AIC {
        MatmulProcess();
    }

    if ASCEND_IS_AIV {
        ReduceScatterProcess();
    }
}

// =================================================================================
// 参数初始化
// =================================================================================

template <typename AType, typename BType, typename CType>
__aicore__ inline void QuantMatmulReduceScatterUrmaImpl<AType, BType, CType>::InitBaseParams(
    const QuantMatmulReduceScatterTilingData* tilingData)
{
    baseParams_.headMSize = tilingData->commTilingData.splitAxisTileSize;
    baseParams_.tailMSize = tilingData->commTilingData.splitAxisTailSize;
    baseParams_.headTileCnt = static_cast<uint32_t>(tilingData->commTilingData.splitAxisTileCnt);
    baseParams_.commTurn = static_cast<uint32_t>(tilingData->commTilingData.splitAxisTileCnt +
                                                 tilingData->commTilingData.splitAxisTailCnt);

    baseParams_.axisN = tilingData->tileQbmmTilingData.n;
    baseParams_.axisK = tilingData->tileQbmmTilingData.k;

    baseParams_.scaleARowBytes = Blaze::Gemm::CeilDiv<uint64_t>(baseParams_.axisK, Blaze::Gemm::MXFP_DIVISOR_SIZE) *
                                 Blaze::Gemm::MXFP_MULTI_BASE_SIZE;

    uint64_t chunkSize = tilingData->commTilingData.splitAxisTileSize * tilingData->commTilingData.splitAxisTileCnt +
                         tilingData->commTilingData.splitAxisTailSize * tilingData->commTilingData.splitAxisTailCnt;
    baseParams_.chunkM = static_cast<uint32_t>(chunkSize);

    // 槽内按 source rankId 预留 rankSize 个 maxTileM×N 段; 本 rank 段不发 PUT, 归约时从 workspace 读取.
    // 槽数取 min(3, commTurn); 回绕复用前完成通信和跨 rank 同步, 避免覆盖仍在归约的槽
    baseParams_.maxTileM =
        (baseParams_.headMSize > baseParams_.tailMSize) ? baseParams_.headMSize : baseParams_.tailMSize;
    baseParams_.commSlotNum =
        (baseParams_.commTurn < QMRS_MAX_COMM_SLOT_NUM) ? baseParams_.commTurn : QMRS_MAX_COMM_SLOT_NUM;
    baseParams_.bufferSlotBytes =
        static_cast<uint64_t>(baseParams_.rankSize) * baseParams_.maxTileM * baseParams_.axisN * sizeof(CType);
}

// =================================================================================
// AIC 侧: 逐 tile 计算并通知 AIV
// =================================================================================

// 单 tile 计算全部 rank 段: tileCnt=1, 逻辑 M=rankSize*paddedTileM
template <typename AType, typename BType, typename CType>
__aicore__ inline void QuantMatmulReduceScatterUrmaImpl<AType, BType, CType>::RunOneMatmulTile(
    uint32_t tileIdx, const QuantMatmulTilingData& tileT)
{
    const uint32_t curTileM = GetTileM(tileIdx);
    const uint64_t tileMOffset = GetTileMOffset(tileIdx);
    const uint32_t rankSize = baseParams_.rankSize;

    // 尾 tile 的布局行数按 16 对齐; realTileM 限制 FragmentSliceCopy,
    // 补齐行不参与 A/ScaleA 的 GM 读和 C 的 GM 写
    const uint32_t paddedTileM = (curTileM + MIN_ROW_ALIGN - 1) / MIN_ROW_ALIGN * MIN_ROW_ALIGN;

    FragmentParams fragParams;
    fragParams.tileCnt = 1;
    fragParams.tileM = paddedTileM;
    fragParams.realTileM = curTileM;
    fragParams.rankSize = rankSize;
    fragParams.mPerRank = baseParams_.chunkM;
    fragParams.k = baseParams_.axisK;
    fragParams.n = baseParams_.axisN;
    // 当前 ScaleType 为 E8M0, 每单元 1B, scaleKLen 与 ScaleA 行字节数数值相同
    fragParams.scaleKLen = baseParams_.scaleARowBytes;

    Params params;
    params.fragParams = fragParams;
    params.schParams = BlockSchedulerParams{tileT.baseM,
                                            tileT.baseN,
                                            tileT.mTailTile,
                                            tileT.nTailTile,
                                            tileT.mBaseTailSplitCnt,
                                            tileT.nBaseTailSplitCnt,
                                            tileT.mTailMain,
                                            tileT.nTailMain};
    params.qbmmParams = {tileT.baseM, tileT.baseN, tileT.baseK, tileT.dbL0c};
    params.l1Params = FragL1Params{static_cast<uint64_t>(tileT.stepK) * tileT.baseK, tileT.scaleKL1, tileT.nBufferNum};

    // MXFP4 每字节打包两个元素, A 的地址偏移和行字节数需要相应除以 2
    constexpr uint64_t aPackFactor = AscendC::IsSameType<AType, fp4x2_e2m1_t>::value ? 2UL : 1UL;
    GM_ADDR aGM = baseParams_.aGm + tileMOffset * baseParams_.axisK / aPackFactor;
    GM_ADDR scaleAGM = baseParams_.scaleAGm + tileMOffset * baseParams_.scaleARowBytes;
    GM_ADDR cGM = baseParams_.workspaceGm + tileMOffset * baseParams_.axisN * sizeof(CType);
    params.mmadParams = BlockMmadParams{aGM, baseParams_.bGm, cGM, nullptr, scaleAGM, baseParams_.scaleBGm};
    params.dataBytesPerMRow = baseParams_.axisK * sizeof(AType) / aPackFactor;
    params.scaleBytesPerMRow = baseParams_.scaleARowBytes;
    params.cBytesPerM = baseParams_.axisN * sizeof(CType);

    // 每个 tile 使用新的 kernel 实例, 重新创建当前 tile 的 FragmentTensor 和地址列表
    QuantMatmulKernelImpl fragKernel;
    fragKernel(params);
}

// 逐 tile 计算全部 rank 段, 并通过 flag 通知 AIV
template <typename AType, typename BType, typename CType>
__aicore__ inline void QuantMatmulReduceScatterUrmaImpl<AType, BType, CType>::MatmulProcess()
{
    const uint32_t commTurn = baseParams_.commTurn;

    const auto& tileQbmmT = tilingData_->tileQbmmTilingData;
    const auto& tailQbmmT = tilingData_->tailQbmmTilingData;

    for (uint32_t tileIdx = 0; tileIdx < commTurn; ++tileIdx) {
        bool isHeadTile = (tileIdx < baseParams_.headTileCnt);
        const QuantMatmulTilingData& qbmmT = isHeadTile ? tileQbmmT : tailQbmmT;

        // 首次进入 tail tiling 前排空 L1; 有 head 时即 head→tail 切换, 避免覆盖尚未读取的数据
        if (tileIdx == baseParams_.headTileCnt) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
        }
        RunOneMatmulTile(tileIdx, qbmmT);

        // 同一 flagId 再次复用前等待 AIV 回执; PIPE_FIX 在本核 C 写出后发出 tile 完成通知
        if (tileIdx >= BufferChannel::FLAG_ID_MODULO) {
            AscendC::CrossCoreWaitFlag(tileIdx % BufferChannel::FLAG_ID_MODULO);
        }
        AscendC::CrossCoreSetFlag<CROSS_CORE_NOTIFY_MODE, PIPE_FIX>(tileIdx % BufferChannel::FLAG_ID_MODULO);
    }
}

// =================================================================================
// AIV 侧: 逐 tile 通信并重叠归约
// =================================================================================

template <typename AType, typename BType, typename CType>
__aicore__ inline void QuantMatmulReduceScatterUrmaImpl<AType, BType, CType>::ReduceScatterProcess()
{
    for (uint32_t tileIdx = 0; tileIdx < baseParams_.commTurn; ++tileIdx) {
        RunOneTileTurn(tileIdx);
    }

    // 通信核到达最终 SyncAll 前已完成末轮 Wait<BARRIER_DEVICE>, 屏障后再归约最后一个 tile
    AscendC::SyncAll<true>();

    if (isCommBlock_) {
        allToAllData_.template Finalize<BARRIER_NONE>();
    }
    if (isComputeBlock_) {
        ReduceSumProcess(baseParams_.commTurn - 1);
    }

    FlushUrmaWinPrefix();
}

// =================================================================================
// AIV 侧辅助方法
// =================================================================================

// 等待当前 tile 就绪, 然后并行执行当前 tile 的 PUT 和上一 tile 的归约.
// win 数据槽按 commSlotNum 回绕, CrossCore flag 按 FLAG_ID_MODULO 回绕, 两套序号独立.
template <typename AType, typename BType, typename CType>
__aicore__ inline void QuantMatmulReduceScatterUrmaImpl<AType, BType, CType>::RunOneTileTurn(uint32_t tileIdx)
{
    AscendC::CrossCoreWaitFlag(tileIdx % BufferChannel::FLAG_ID_MODULO);

    // CrossCoreWaitFlag 只证明配对 AIC 已完成; SyncAll 聚合全部 AIV, 因此屏障后所有 AIC 均已写完当前 tile.
    // 通信核到达前还已完成上一轮 Wait<BARRIER_DEVICE>, 所以随后可以读取上一 tile 的 win 数据.
    AscendC::SyncAll<true>();

    // 仅当同一 flagId 后续还会复用时发送回执; 回执在 SyncAll 后通过 PIPE_MTE3 发出
    if (tileIdx + BufferChannel::FLAG_ID_MODULO < baseParams_.commTurn) {
        AscendC::CrossCoreSetFlag<CROSS_CORE_NOTIFY_MODE, PIPE_MTE3>(tileIdx % BufferChannel::FLAG_ID_MODULO);
    }

    if (isCommBlock_) {
        // Commit 按 dataSlot.offset 写当前 tile, 内部跳过本 rank
        auto dataSlot = dataChannel_.GetNextSlot();
        allToAllData_.Commit(dataSlot.offset);
        // Drain 本核 PUT, 再执行 CrossDevice 跨 rank 同步
        allToAllData_.template Wait<BARRIER_DEVICE>();
    }
    if (isComputeBlock_ && tileIdx > 0) {
        ReduceSumProcess(tileIdx - 1);
    }
}

// 计算各 rank 输入基址: 远端数据来自 win, 本 rank 数据来自 workspace.
// reduce_sum 只负责归约, 调用方保证 tileM 非零.
template <typename AType, typename BType, typename CType>
__aicore__ inline void QuantMatmulReduceScatterUrmaImpl<AType, BType, CType>::ReduceSumProcess(uint32_t tileIdx)
{
    const uint32_t tileM = GetTileM(tileIdx);
    const uint64_t tileMOffset = GetTileMOffset(tileIdx);
    const uint64_t slotOffset = (tileIdx % baseParams_.commSlotNum) * baseParams_.bufferSlotBytes;
    const uint64_t nSize = baseParams_.axisN;
    const uint64_t rowBytes = nSize * sizeof(CType);

    using ReduceSumType = ReduceSumImpl::ReduceSum<CType, float>;
    typename ReduceSumType::RsParams rsParams;
    // workspace 保存全部 tile, 本 rank 段需要叠加 tileMOffset;
    // 远端 win 已由 slotOffset 定位到当前 tile, 槽内不再叠加 tileMOffset
    for (uint32_t rankIdx = 0; rankIdx < baseParams_.rankSize; ++rankIdx) {
        if (rankIdx == baseParams_.rankId) {
            rsParams.inputSegBaseAddrs[rankIdx] = baseParams_.workspaceGm +
                                                  static_cast<uint64_t>(rankIdx) * baseParams_.chunkM * rowBytes +
                                                  tileMOffset * rowBytes;
        } else {
            rsParams.inputSegBaseAddrs[rankIdx] =
                baseParams_.selfWinAddr + slotOffset + static_cast<uint64_t>(rankIdx) * baseParams_.maxTileM * rowBytes;
        }
    }
    rsParams.outputAddr = baseParams_.cGm + tileMOffset * rowBytes;
    rsParams.inputSegNum = baseParams_.rankSize;
    rsParams.aivNum = rsCoreNum_;
    rsParams.nSize = nSize;
    // UB 低地址为 commBuf 和 barrierBuf 预留, 归约缓冲区紧随其后
    rsParams.ubReservedBytes = COMM_WORKSPACE_SIZE + BARRIER_FLAG_SIZE;
    // 最后一个 tile 在最终 SyncAll 后归约, 没有并发 PUT, 逐份预取加载
    rsParams.tailCall = (tileIdx + 1U == baseParams_.commTurn);
    reduceSum_.Init(rsParams);
    reduceSum_.Process(tileM);
}

// =================================================================================
// 辅助方法
// =================================================================================

template <typename AType, typename BType, typename CType>
__aicore__ inline uint32_t QuantMatmulReduceScatterUrmaImpl<AType, BType, CType>::GetTileM(uint32_t tileIdx)
{
    if (tileIdx < baseParams_.headTileCnt) {
        return static_cast<uint32_t>(baseParams_.headMSize);
    }
    return static_cast<uint32_t>(baseParams_.tailMSize);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline uint64_t QuantMatmulReduceScatterUrmaImpl<AType, BType, CType>::GetTileMOffset(uint32_t tileIdx)
{
    if (tileIdx < baseParams_.headTileCnt) {
        return static_cast<uint64_t>(tileIdx) * baseParams_.headMSize;
    }
    return static_cast<uint64_t>(baseParams_.headTileCnt) * baseParams_.headMSize;
}

// 与家族整体 dcci 同目的: 跨 launch 保证 URMA channel 字段的 GM 状态与 aiCore cache 一致.
// 通道状态只在 win 保留前缀, 故按 64B 行刷 [0, winDataOffset);
// PUT 数据区由 URMA 硬件直写 HBM, 下一 launch 全量重写且不经本核 cache, 无需刷新.
// 不采用 ENTIRE_DATA_CACHE 全刷, 避免额外写回与失效. TeamBarrier 在独立 UBMEM buffer.
template <typename AType, typename BType, typename CType>
__aicore__ inline void QuantMatmulReduceScatterUrmaImpl<AType, BType, CType>::FlushUrmaWinPrefix()
{
    __gm__ uint8_t* winBase = reinterpret_cast<__gm__ uint8_t*>(udmaCtx_->commBufferAddrs[baseParams_.rankId]);
    constexpr uint64_t DCCI_CACHE_LINE_BYTES = 64;
    for (uint64_t off = 0; off < baseParams_.winDataOffset; off += DCCI_CACHE_LINE_BYTES) {
        dcci(reinterpret_cast<__gm__ void*>(winBase + off),
             static_cast<uint64_t>(AscendC::CacheLine::SINGLE_CACHE_LINE),
             static_cast<uint64_t>(AscendC::DcciDst::CACHELINE_OUT));
    }
}

} // namespace Apace
