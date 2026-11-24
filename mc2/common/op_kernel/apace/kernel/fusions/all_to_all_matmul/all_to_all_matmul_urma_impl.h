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
 * \file all_to_all_matmul_urma_impl.h
 * \brief AlltoAll MatMul（量化/非量化融合 urma_impl 版）— 通信+计算融合实现
 *
 * Run():
 *   AIV: RunAllToAll()  根据通信轮次下发通信任务和通知AIC通信已完成
 *   AIC: RunMatmul()    等待通信完成后开始cube计算
 *
 * 量化/非量化差异经模板参数 QuantFlag + if constexpr 收敛：
 *  - BlockMmad/DispatchPolicy/BlockScheduler/KernelImpl 由 IS_QUANT 经 conditional_t 类型层实例化；
 *  - TilingT 经 conditional_t 分派：量化 AllToAllQuantMatmulTilingData / 非量化 AllToAllMatmulTilingData；
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "include/tensor_api/tensor.h"

#include "adv_api/hcomm/hcomm.h"
#include "../../matmul/batch_matmul/all_to_all_mm_kernel.h"
#include "../../matmul/quant_batch_matmul/all_to_all_qbmm_mx_kernel.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "all_to_all_matmul_tiling_data.h"

#include "apace/core/aiv_comm/collective_comm_api.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "apace/tiling/tiling_data_struct_define.h"
#include "apace/core/aiv_comm/barrier/barrier_ubmem.h"
#include "../../../utils/op_state_dump.h"
#include "apace/basic/buffer/buffer_channel.h"

namespace Apace {

using namespace AscendC;
using namespace Blaze::Gemm;
using asc::te::get;
using namespace Apace::AivComm;

// 量化/非量化标志：NON_QUANT (非量化)，MX_QUANT (MX量化，为后续 KC 量化等其他量化方式打样)
enum class QuantFlag : uint8_t {
    NON_QUANT = 0,
    MX_QUANT = 1,
};

/**
 * @brief 通信等待策略：BufferSyncManager 接口
 * 只负责 flag 原语 + slotNum；kernel 自行编排 GetNextSlot 与偏移计算。
 * - enableBufferReuse=true（URMA）：Release 通知写端可覆盖，Acquire 等数据就绪。
 * - enableBufferReuse=false（hcomm）：每 tile 固定 flag，仅用 Acquire(tileIdx)，Release 不会被调用。
 */
struct UrmaBufferSync {
    uint32_t slotNum{0};

    __aicore__ inline void Acquire(uint32_t slotIdx)
    {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(slotIdx);
    }
    __aicore__ inline void Release(uint32_t slotIdx)
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(slotIdx);
    }
};

// 定义问题形状：[M, N, K, Batch]
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

#define A2AMM_URMA_CLASS_TEM_PARAMS \
    template <typename AType, typename BType, typename CType, typename BiasType, bool TransA, bool TransB, \
              QuantFlag quantFlag>
#define A2AMM_URMA_FUNC_TEM_PARAMS AType, BType, CType, BiasType, TransA, TransB, quantFlag

/**
 * @brief All2All-Matmul 核心实现类 (Hcomm GET 版本，计算通信融合)
 * 实现思路：
 * 1. AIC 负责本地计算 和远程数据计算。
 * 2. AIV 负责背景通信，通过 Hcomm GET 接口从其他卡拉取 A 和 ScaleA 数据。
 * 3. 通过流水线掩盖通信开销。
 * 4. 计算与通信逻辑合并于此类，便于精细化流水线控制。
 */
A2AMM_URMA_CLASS_TEM_PARAMS
class AllToAllMatmulUrmaImpl {
public:
    static constexpr bool IS_QUANT = (quantFlag == QuantFlag::MX_QUANT);
    // 量化路径 bias 恒为 fp32；非量化支持 fp16/bf16/fp32 三种 bias dtype
    static_assert(!IS_QUANT || AscendC::Std::is_same_v<BiasType, float>, "Quant path requires float bias.");
    static_assert(IS_QUANT || AscendC::Std::is_same_v<BiasType, float> || AscendC::Std::is_same_v<BiasType, half> ||
                      AscendC::Std::is_same_v<BiasType, bfloat16_t>,
                  "Non-quant bias must be float/half/bfloat16_t.");
    // tiling 结构类型：量化 AllToAllQuantMatmulTilingData / 非量化 AllToAllMatmulTilingData
    using TilingT = AscendC::Std::conditional_t<IS_QUANT, AllToAllQuantMatmulTilingData, AllToAllMatmulTilingData>;

    __aicore__ inline AllToAllMatmulUrmaImpl(){};
    __aicore__ inline ~AllToAllMatmulUrmaImpl() {}

    /**
     * @brief 初始化算子状态和参数
     */
    __aicore__ inline void Init(__gm__ CommContext* hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM,
                                GM_ADDR scaleBGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM,
                                const TilingT* tilingData);
    /**
     * @brief 执行算子逻辑（包含 AIC/AIV 分离逻辑）
     */
    __aicore__ inline void Run();

    using TypeA = AType;
    using TypeB = BType;
    using TypeC = CType;
    using TypeScaleA = ::fp8_e8m0_t;
    using TypeScaleB = ::fp8_e8m0_t;
    static constexpr int32_t SCALE_C0 = 2;

    // Layout 定义
    using LayoutA =
        typename AscendC::Std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutB =
        typename AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using LayoutScaleA = typename asc::te::frame_layout_format<
        AscendC::Std::conditional_t<TransA, asc::te::scalea_dn_layout_ptn, asc::te::scalea_nd_layout_ptn>,
        AscendC::Std::Int<SCALE_C0>>;
    using LayoutScaleB = typename asc::te::frame_layout_format<
        AscendC::Std::conditional_t<TransB, asc::te::scaleb_dn_layout_ptn, asc::te::scaleb_nd_layout_ptn>,
        AscendC::Std::Int<SCALE_C0>>;

    // 调度器：量化 BlockSchedulerQuantBatchMatmulV3 / 非量化 BlockSchedulerMatmulBasic
    using BlockScheduler =
        typename AscendC::Std::conditional_t<IS_QUANT,
                                             Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
                                                 ProblemShape, NONE_FULL_LOAD_MODE, LayoutA, LayoutB, TypeA>,
                                             Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape>>;
    // mmad 调度策略：量化 MatmulWithScaleMx / 非量化 MatmulMultiBlockBasic
    using DispatchPolicy = typename AscendC::Std::conditional_t<
        IS_QUANT, Blaze::Gemm::MatmulWithScaleMx<NONE_FULL_LOAD_MODE, false>,
        Blaze::Gemm::MatmulMultiBlockBasic<NONE_FULL_LOAD_MODE, 0, Blaze::Gemm::KernelMmadMultiBlockBasic, 0>>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC,
                                                    BiasType, LayoutBias>;
    // kernel：量化 AllToAllQbmmMxKernel / 非量化 AllToAllMmKernel
    using KernelImpl = typename AscendC::Std::conditional_t<
        IS_QUANT, Kernel::AllToAllQbmmMxKernel<ProblemShape, BlockMmad, BlockScheduler, UrmaBufferSync, true>,
        Kernel::AllToAllMmKernel<ProblemShape, BlockMmad, BlockScheduler, UrmaBufferSync, true>>;

    // 参数类型
    using Params = typename KernelImpl::Params;
    using BlockMmadParams = typename KernelImpl::BlockMmadParams;
    using LocalParams = typename KernelImpl::LocalParams;
    using BlockSchedulerParams = typename KernelImpl::BlockSchedulerParams;
    using MatmulMode = typename KernelImpl::MatmulMode;

    KernelImpl matmulKernelImpl_;

private:
    // ---------------- 通信相关组件与参数 ----------------
    __gm__ CommUbmemContext* syncBuffer_{nullptr};
    __gm__ CommUdmaContext* udmaCtx_{nullptr};
    CollectiveComm<CommCollectiveOp::AllToAll, CommMode::PUT, AType, TeamBarrier> allToAllA_;
    CollectiveComm<CommCollectiveOp::AllToAll, CommMode::PUT, TypeScaleA, TeamBarrier> allToAllScaleA_;
    TeamBarrier teamBarrier_;

    Apace::Basic::BufferChannel dataChannel_;
    Apace::Basic::BufferChannel scaleChannel_;

    struct BaseParams {
        GM_ADDR selfWinAddr{nullptr}; // 通信窗口地址
        GM_ADDR aGm{nullptr};
        GM_ADDR scaleAGm{nullptr};
        GM_ADDR bGm{nullptr};
        GM_ADDR scaleBGm{nullptr};
        GM_ADDR cGm{nullptr};
        GM_ADDR biasGm{nullptr};

        uint32_t rankId{0};
        int32_t commTurn{0};     // 总流水步数
        uint32_t rankSize{0};    // 卡数
        uint64_t axisM{0};       // M 轴总大小
        uint64_t axisKa{0};      // K 轴大小
        uint64_t headMSize{0};   // 长块 M 大小
        uint64_t scaleKaSize{0}; // Scale 的 K 轴字节长度（量化）
        uint64_t rankDataBytes{0};
    } baseParams_;

    const TilingT* tilingData_{nullptr};
    Mc2Kernel::OpStateDump opStateDump_{};
    static constexpr uint8_t POS_COMM_BEFORE = 0U;
    static constexpr uint8_t POS_COMP_CUBE_MATMUL_BEFORE = 1U;
    static constexpr uint8_t POS_COMP_LOCAL_BEFORE = 2U;
    // ---------------- 私有方法 ----------------
    __aicore__ inline void InitBaseParams(const TilingT* tilingData);
    __aicore__ inline void SetupParams(Params& out, MatmulMode matmulMode);
    template <typename MmTilingT>
    __aicore__ inline void SetupQuantParams(const MmTilingT* mmTile, const ProblemShape& problemShape,
                                            BlockMmadParams& mmadParams, uint32_t splitKNum, MatmulMode matmulMode,
                                            Params& out);
    template <typename MmTilingT>
    __aicore__ inline void SetupNonQuantParams(const MmTilingT* mmTile, const ProblemShape& problemShape,
                                               BlockMmadParams& mmadParams, uint32_t splitKNum, MatmulMode matmulMode,
                                               Params& out);

    __aicore__ inline void RunAllToAll();                                 // AIV 通信任务
    __aicore__ inline void RunCommLoop();                                 // AIV 通信主循环
    __aicore__ inline void RunMatmul(MatmulMode matmulMode, uint8_t pos); // AIC matmul计算
};

// =================================================================================
// 公有方法实现
// =================================================================================

A2AMM_URMA_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMatmulUrmaImpl<A2AMM_URMA_FUNC_TEM_PARAMS>::Init(
    __gm__ CommContext* hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM, GM_ADDR cGM,
    GM_ADDR biasGM, GM_ADDR workspaceGM, const TilingT* tilingData)
{
    tilingData_ = tilingData;
    baseParams_.aGm = aGM;
    baseParams_.bGm = bGM;
    baseParams_.cGm = cGM;
    baseParams_.biasGm = biasGM;
    if constexpr (IS_QUANT) {
        baseParams_.scaleAGm = scaleAGM;
        baseParams_.scaleBGm = scaleBGM;
#if MC2_DFX_ENABLE
        opStateDump_.Init(workspaceGM, &tilingData->dumpInfo.workspaceLayout,
                          tilingData->tileQbmmTilingData.usedCoreNum);
#endif
    } else {
#if MC2_DFX_ENABLE
        opStateDump_.Init(workspaceGM, &tilingData->dumpInfo.workspaceLayout, tilingData->tileMmTilingData.usedCoreNum);
#endif
    }

    syncBuffer_ = &(hcommCtx->ubmemCtx);
    udmaCtx_ = &(hcommCtx->udmaCtx);
    baseParams_.rankId = udmaCtx_->rankId;
    baseParams_.rankSize = udmaCtx_->rankSize;

    InitBaseParams(tilingData);

    baseParams_.selfWinAddr = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[baseParams_.rankId]);
    uint32_t ubOffset = 0;
    auto commBuf = asc::te::make_mem_ptr<asc::te::location::ub, uint8_t>(ubOffset); // 用512B
    ubOffset += COMM_WORKSPACE_SIZE;
    auto commScaleBuf = asc::te::make_mem_ptr<asc::te::location::ub, uint8_t>(ubOffset); // 用512B
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierBuf = asc::te::make_mem_ptr<asc::te::location::ub, uint8_t>(ubOffset);
    teamBarrier_.Init(barrierBuf.get(), syncBuffer_, baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()));

    allToAllA_.template Init<BARRIER_NONE>(udmaCtx_, teamBarrier_, tilingData->commTilingData, baseParams_.aGm,
                                           commBuf.get(), baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()));
    if constexpr (IS_QUANT) {
        allToAllScaleA_.Init(udmaCtx_, teamBarrier_, tilingData->scaleCommTilingData, baseParams_.scaleAGm,
                             commScaleBuf.get(), baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()),
                             dataChannel_.GetCapacity());
    }
}

A2AMM_URMA_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMatmulUrmaImpl<A2AMM_URMA_FUNC_TEM_PARAMS>::Run()
{
    if ASCEND_IS_AIV {
        RunAllToAll(); // AIV 通信
    }

    if ASCEND_IS_AIC {
        if (tilingData_->localMatmul == 2) {
            RunMatmul(MatmulMode::LOCAL, POS_COMP_LOCAL_BEFORE); // AIC local计算前置（GM累加模式）
        }
        RunMatmul((tilingData_->localMatmul == 1) ? MatmulMode::DEFERRED_SYNC : MatmulMode::REMOTE,
                  POS_COMP_CUBE_MATMUL_BEFORE); // AIC 通信块计算（带同步等待）
    }
}

// =================================================================================
// 通信与流水线控制实现 (AIV 侧)
// =================================================================================

A2AMM_URMA_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMatmulUrmaImpl<A2AMM_URMA_FUNC_TEM_PARAMS>::RunAllToAll()
{
    opStateDump_.DoDump(DUMP_FIELD_STEP, POS_COMM_BEFORE, static_cast<uint8_t>(Utils::RT_PHASE_COMM_COMMIT));
    RunCommLoop();

    if constexpr (IS_QUANT) {
        allToAllScaleA_.Finalize();
    }
    allToAllA_.Finalize();
    opStateDump_.DoDump(DUMP_FIELD_PHASE, 0, static_cast<uint8_t>(Utils::RT_PHASE_COMM_FINALIZE));
    // 通信接口使用了 channel 内的字段，需要刷新 GM 地址保证跨 kernel launch 的 GM 状态与 aiCore cache 一致
    // 后续通信接口会内置刷 cache 机制，届时可移除此处调用
    dcci(reinterpret_cast<__gm__ void*>(udmaCtx_->commBufferAddrs), static_cast<uint64_t>(CacheLine::ENTIRE_DATA_CACHE),
         static_cast<uint64_t>(DcciDst::CACHELINE_OUT));
}

A2AMM_URMA_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMatmulUrmaImpl<A2AMM_URMA_FUNC_TEM_PARAMS>::RunCommLoop()
{
    for (uint32_t tid = 0; tid < static_cast<uint32_t>(baseParams_.commTurn); ++tid) {
        auto dataSlot = dataChannel_.GetNextSlot();

        // win buffer 复用：写 tile tid 覆盖槽位 tid%slotNum，需等本端 cube 消费完该槽位
        // flagId 直接用 slotIdx(dataSlot.slotIdx)，<14 避开 SyncAll<true> 的 14
        if (tid >= dataChannel_.GetSlotNum()) {
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(dataSlot.slotIdx);
            AscendC::SyncAll<true>();
            if (AscendC::GetBlockIdx() < baseParams_.rankSize) {
                teamBarrier_.CrossDevice();
            }
        }
        // 前 rankSize 个核参与通信
        if (AscendC::GetBlockIdx() < baseParams_.rankSize) {
            if constexpr (IS_QUANT) {
                auto scaleSlot = scaleChannel_.GetNextSlot();
                allToAllScaleA_.Commit(scaleSlot.offset);
            }
            allToAllA_.Commit(dataSlot.offset);
            opStateDump_.DoDump(DUMP_FIELD_COMMIT);
            allToAllA_.template Wait<BARRIER_DEVICE>(); // scale与a通信共用channel，只wait一次
            opStateDump_.DoDump(DUMP_FIELD_WAIT);
        }

        AscendC::SyncAll<true>();
        // data-ready 的 flagId 直接用 slotIdx
        CrossCoreSetFlag<0x2, PIPE_MTE3>(dataSlot.slotIdx);
    }
}

// =================================================================================
// 计算逻辑实现 (AIC 侧)
// =================================================================================

A2AMM_URMA_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMatmulUrmaImpl<A2AMM_URMA_FUNC_TEM_PARAMS>::InitBaseParams(const TilingT* tilingData)
{
    baseParams_.headMSize = tilingData->commTilingData.splitAxisTileSize;
    baseParams_.commTurn = tilingData->commTilingData.splitAxisTileCnt + tilingData->commTilingData.splitAxisTailCnt;

    if constexpr (IS_QUANT) {
        baseParams_.axisM = tilingData->tileQbmmTilingData.m;
        baseParams_.axisKa = tilingData->tileQbmmTilingData.k;
        // 计算 MXFP8 格式下 Scale 每一行的字节数
        baseParams_.scaleKaSize =
            CeilDiv(baseParams_.axisKa, Blaze::Gemm::MXFP_DIVISOR_SIZE) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
    } else {
        baseParams_.axisM = tilingData->tileMmTilingData.m;
        baseParams_.axisKa = tilingData->tileMmTilingData.k;
    }

    // 计算地址偏移所需的参数（static_cast<uint64_t> 提升首操作数，避免中间溢出）
    baseParams_.rankDataBytes =
        static_cast<uint64_t>(baseParams_.axisM) * static_cast<uint64_t>(baseParams_.axisKa) * sizeof(AType);
    if constexpr (IS_QUANT) {
        if constexpr (IsFp4<AType>()) {
            // MXFP4 打包存储(2 个 fp4 占 1 字节)，A 矩阵通信字节数减半
            baseParams_.rankDataBytes >>= 1;
        }
    }
    uint64_t commTiles = tilingData->commTilingData.splitAxisTileCnt + tilingData->commTilingData.splitAxisTailCnt;
    uint64_t rawSlotNum = tilingData->commTilingData.slotNum;
    uint32_t slotNum = (rawSlotNum > 0 && rawSlotNum < commTiles) ? static_cast<uint32_t>(rawSlotNum) :
                                                                    static_cast<uint32_t>(commTiles);
    uint64_t dataBytesPerMRow = tilingData->commTilingData.nonSplitAxisSize * sizeof(AType);
    uint64_t winPerRound = static_cast<uint64_t>(baseParams_.rankSize) * baseParams_.headMSize;
    dataChannel_.Init(winPerRound * dataBytesPerMRow, slotNum);

    matmulKernelImpl_.GetBufferSyncMgr().slotNum = slotNum;
    if constexpr (IS_QUANT) {
        uint64_t scaleBytesPerMRow = tilingData->scaleCommTilingData.nonSplitAxisSize * sizeof(TypeScaleA);
        scaleChannel_.Init(winPerRound * scaleBytesPerMRow, static_cast<uint32_t>(commTiles));
        matmulKernelImpl_.SetChannels(&dataChannel_, &scaleChannel_);
    } else {
        matmulKernelImpl_.SetChannels(&dataChannel_);
    }
}

A2AMM_URMA_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMatmulUrmaImpl<A2AMM_URMA_FUNC_TEM_PARAMS>::SetupParams(Params& out,
                                                                                       MatmulMode matmulMode)
{
    BlockMmadParams mmadParams;
    uint32_t splitKNum = 1UL;
    mmadParams.cGmAddr = baseParams_.cGm; // 输出基址（两路径一致）

    if (matmulMode != MatmulMode::LOCAL) {
        // 远程路径：splitKNum + 从通信窗口读 A / ScaleA（LOCAL 不需要）
        splitKNum = (tilingData_->localMatmul == 2) ? (baseParams_.rankSize - 1) : baseParams_.rankSize;
        mmadParams.aGmAddr = baseParams_.selfWinAddr;
        if constexpr (IS_QUANT) {
            mmadParams.scaleAGmAddr = baseParams_.selfWinAddr + dataChannel_.GetCapacity();
        }
    }

    if constexpr (IS_QUANT) {
        const auto* mmTile = &tilingData_->tileQbmmTilingData;
        ProblemShape problemShape{mmTile->m, mmTile->n, mmTile->k, 1UL};
        SetupQuantParams(mmTile, problemShape, mmadParams, splitKNum, matmulMode, out);
    } else {
        const auto* mmTile = &tilingData_->tileMmTilingData;
        ProblemShape problemShape{mmTile->m, mmTile->n, mmTile->k, 1UL};
        SetupNonQuantParams(mmTile, problemShape, mmadParams, splitKNum, matmulMode, out);
    }
}

A2AMM_URMA_CLASS_TEM_PARAMS
template <typename MmTilingT>
__aicore__ inline void AllToAllMatmulUrmaImpl<A2AMM_URMA_FUNC_TEM_PARAMS>::SetupQuantParams(
    const MmTilingT* mmTile, const ProblemShape& problemShape, BlockMmadParams& mmadParams, uint32_t splitKNum,
    MatmulMode matmulMode, Params& out)
{
    // 量化 LocalParams（含 localScaleAGmAddr + splitKNum）
    LocalParams localParams{baseParams_.rankId, baseParams_.rankSize, baseParams_.axisM,
                            baseParams_.aGm,    baseParams_.scaleAGm, tilingData_->localMatmul,
                            splitKNum,          matmulMode,           static_cast<uint32_t>(baseParams_.headMSize)};
    using QuantL1Params = typename KernelImpl::L1Params;
    QuantL1Params l1Params{static_cast<uint64_t>(mmTile->stepK) * mmTile->baseK, mmTile->scaleKL1, mmTile->nBufferNum};

    mmadParams.bGmAddr = baseParams_.bGm;
    mmadParams.scaleBGmAddr = baseParams_.scaleBGm;
    mmadParams.biasGmAddr = baseParams_.biasGm;

    // 调度器参数
    BlockSchedulerParams schedulerParams{mmTile->baseM,
                                         mmTile->baseN,
                                         mmTile->mTailTile,
                                         mmTile->nTailTile,
                                         mmTile->mBaseTailSplitCnt,
                                         mmTile->nBaseTailSplitCnt,
                                         mmTile->mTailMain,
                                         mmTile->nTailMain};
    // 基础 Tiling 参数
    using QbmmTiling = typename KernelImpl::QBMMTiling;
    QbmmTiling qbmmParams{mmTile->baseM, mmTile->baseN, mmTile->baseK, mmTile->dbL0c, false};

    out = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};
}

A2AMM_URMA_CLASS_TEM_PARAMS
template <typename MmTilingT>
__aicore__ inline void AllToAllMatmulUrmaImpl<A2AMM_URMA_FUNC_TEM_PARAMS>::SetupNonQuantParams(
    const MmTilingT* mmTile, const ProblemShape& problemShape, BlockMmadParams& mmadParams, uint32_t splitKNum,
    MatmulMode matmulMode, Params& out)
{
    // 非量化 LocalParams（无 localScaleAGmAddr / splitKNum）
    LocalParams localParams{baseParams_.rankId,
                            baseParams_.rankSize,
                            baseParams_.axisM,
                            baseParams_.aGm,
                            tilingData_->localMatmul,
                            matmulMode,
                            static_cast<uint32_t>(baseParams_.headMSize)};
    mmadParams.splitKNum = splitKNum;

    // 基本 mmad 的 L1/L0 分块与缓冲参数（非量化 tiling 直接给出 L1 块与缓冲数）
    mmadParams.bGmAddr = baseParams_.bGm;
    mmadParams.biasGmAddr = baseParams_.biasGm;
    mmadParams.mL1 = mmTile->mL1;
    mmadParams.nL1 = mmTile->nL1;
    mmadParams.kL1 = mmTile->kL1;
    mmadParams.mL0 = mmTile->baseM;
    mmadParams.nL0 = mmTile->baseN;
    mmadParams.kL0 = mmTile->baseK;
    mmadParams.l1Stages = mmTile->l1BufferNum;
    mmadParams.l0cStages = (mmTile->l0cDB > 1) ? 2 : 1;

    // 调度器参数（非量化 BlockSchedulerMatmulBasic）
    BlockSchedulerParams schedulerParams{mmTile->mL1,
                                         mmTile->nL1,
                                         mmTile->kL1,
                                         mmTile->baseM,
                                         mmTile->baseN,
                                         mmTile->baseK,
                                         mmTile->mTailCnt,
                                         mmTile->nTailCnt,
                                         mmTile->mBaseTailSplitCnt,
                                         mmTile->nBaseTailSplitCnt,
                                         mmTile->mTailMain,
                                         mmTile->nTailMain,
                                         mmTile->isHf32};
    // 基础 Tiling 参数
    using MMTiling = typename KernelImpl::MMTiling;
    MMTiling mmTiling{mmTile->baseM, mmTile->baseN, mmTile->baseK, mmTile->l0cDB};

    out = {problemShape, mmadParams, schedulerParams, mmTiling, localParams};
}

A2AMM_URMA_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMatmulUrmaImpl<A2AMM_URMA_FUNC_TEM_PARAMS>::RunMatmul(MatmulMode matmulMode, uint8_t pos)
{
    Params params;
    SetupParams(params, matmulMode);
    opStateDump_.DoDump(DUMP_FIELD_STEP, pos, static_cast<uint8_t>(Utils::RT_PHASE_COMM_WAIT));
    matmulKernelImpl_(params, opStateDump_);
}

#undef A2AMM_URMA_FUNC_TEM_PARAMS
#undef A2AMM_URMA_CLASS_TEM_PARAMS

} // namespace Apace
