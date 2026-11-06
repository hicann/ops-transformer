/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_matmul_all_to_all_v_hcomm_impl.h
 * \brief AIC-only apace 通路计算/通信算子实现（GMM 先算、后 AllToAllV 发送）。
 *
 * 结构参考镜像算子 allto_allv_quant_grouped_mat_mul apace 实现与 legacy
 * mc2_templates 的算子拆分方式，由三类算子组成（流水编排见
 * quant_grouped_matmul_all_to_all_v_scheduler.h）：
 *   - QuantGmmComputeOp：路由专家 GMM（直接复用算子 legacy tiling 数据，
 *     LENGTH groupList + B/scaleB 显式偏移表全核冗余写、kAL1 置 0 走 baseK
 *     兜底——409d5e04a 真机验证形态）
 *   - QuantMmSharedComputeOp：共享专家（单组 OFFSET groupList）
 *   - QuantA2avCommOp：AIC 核内 Hccl AllToAllV（通信参数与 legacy
 *     CalcA2avCommAfterParams 同款伸缩和公式）
 *
 * Blaze 生命周期纪律（真机实证约束，务必保持）：任意时刻仅一个 Blaze 实例
 * 存活——kernel 一律在 Process 内栈上 ctor/dtor 成对构造（ctor SetFlag×10
 * 预置核内事件 → Run → dtor WaitFlag×10 消费）；双实例并存（两个 ctor 连续
 * 预置、无中间消费）会失步事件协议致 Run 挂死。
 */

#ifndef APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__
#define APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__

#include "lib/hccl/hccl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "../../matmul/grouped_matmul/gmm_kernel.h"
#include "../../../../../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"
#include "../../../../../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/communication/a2av_comm_params.h"

namespace Apace {

template <typename T>
constexpr AscendC::HcclDataType GetGmmA2avYHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return AscendC::HCCL_DATA_TYPE_BFP16;
    } else {
        return AscendC::HCCL_DATA_TYPE_FP16;
    }
}

// Blaze GMMTiling 填充：L1 K 窗口置 0 走 baseK 兜底（单 L0 块窗口——多 L0
// 块窗口路径有挂死真机证据），其余取自 legacy tiling 的 TCubeTiling
template <typename GmmTiling, typename GmmTilingData>
__aicore__ inline void FillGmmA2avTiling(GmmTiling& out, const GmmTilingData& in, uint32_t groupNum, int64_t m,
                                         int64_t n, int64_t k, uint8_t groupListType)
{
    const auto& mmTiling = in.mmTilingData;
    out.groupNum = groupNum;
    out.m = m;
    out.n = n;
    out.k = k;
    out.baseM = static_cast<uint32_t>(mmTiling.baseM);
    out.baseN = static_cast<uint32_t>(mmTiling.baseN);
    out.baseK = static_cast<uint32_t>(mmTiling.baseK);
    out.kAL1 = 0U;
    out.kBL1 = 0U;
    out.scaleKL1 = 0U;
    out.isBias = static_cast<uint8_t>(mmTiling.isBias != 0 ? 1 : 0);
    out.dbL0C = static_cast<uint8_t>(mmTiling.dbL0C);
    out.l1BufferStage = Blaze::Gemm::DOUBLE_BUFFER_COUNT;
    out.groupListType = groupListType;
}

// ============================================================================
// 路由专家 GMM 计算算子
// ============================================================================
template <typename TilingDataType, typename AType, typename BType, typename CType, bool TransB>
class QuantGmmComputeOp {
public:
    __aicore__ inline QuantGmmComputeOp() {}

    __aicore__ inline void Init(const TilingDataType* tilingData, GM_ADDR xGM, GM_ADDR xScaleGM, GM_ADDR weightGM,
                                GM_ADDR weightScaleGM, GM_ADDR gmmOutputGM, GM_ADDR tableGM)
    {
        tilingData_ = tilingData;
        xGM_ = xGM;
        xScaleGM_ = xScaleGM;
        weightGM_ = weightGM;
        weightScaleGM_ = weightScaleGM;
        gmmOutputGM_ = gmmOutputGM;
        // 表区布局见 scheduler 头注释；全核冗余写（每核写读一致，无跨核可见性依赖）
        groupListGm_ = tableGM;
        cGroupOffsetTableGm_ = groupListGm_ + RankSize() * ExpertNum() * sizeof(int64_t);
        bGroupOffsetTableGm_ = cGroupOffsetTableGm_ + RankSize() * ExpertNum() * sizeof(uint64_t);
        scaleBGroupOffsetTableGm_ = bGroupOffsetTableGm_ + RankSize() * ExpertNum() * sizeof(uint64_t);
    }

    __aicore__ inline uint32_t BatchCount() const
    {
        return tilingData_->taskTilingInfo.totalLoopCount;
    }

    __aicore__ inline uint64_t GetBatchTokenCount(uint32_t batch) const
    {
        uint64_t tokens = 0;
        const uint32_t start = batch * BatchExpertNum();
        const uint32_t expertNum = GetBatchExpertNum(batch);
        for (uint32_t jj = 0U; jj < expertNum; ++jj) {
            for (uint32_t rank = 0U; rank < RankSize(); ++rank) {
                tokens += GetSendRows(start + jj, rank);
            }
        }
        return tokens;
    }

    __aicore__ inline void Process(uint32_t batch, uint64_t tokenPrefix)
    {
        const uint32_t rankSize = RankSize();
        const uint32_t expertNum = GetBatchExpertNum(batch);
        const uint32_t groupNum = expertNum * rankSize;
        const uint32_t start = batch * BatchExpertNum();
        const uint64_t h1 = tilingData_->taskTilingInfo.H1;
        const uint64_t n1 = tilingData_->taskTilingInfo.N1;
        const uint64_t scaleK = ScaleKLen();
        const bool isFp4 = AscendC::IsSameType<AType, fp4x2_e2m1_t>::value;
        const uint64_t packedShift = isFp4 ? 1U : 0U;
        const uint64_t dataAxis = isFp4 ? (h1 + 1U) / 2U : h1;

        BuildBatchTables(start, expertNum, rankSize, n1, scaleK, packedShift);

        const int64_t maxM = static_cast<int64_t>(GetMaxExpertTokens());
        typename GmmKernel::GMMTiling gmmTiling{};
        FillGmmA2avTiling(gmmTiling, tilingData_->gmmBaseTiling, groupNum, maxM, static_cast<int64_t>(n1),
                          static_cast<int64_t>(h1), tilingData_->gmmBaseTiling.gmmQuantParams.groupListType);

        typename GmmKernel::Params params{};
        params.problemShape = ProblemShape{maxM, static_cast<int64_t>(n1), static_cast<int64_t>(h1), 0};
        // A/scaleA/C 基址推进至批前全局 token 前缀；B/scaleB 基址推进至本批首个
        // expert（批内经偏移表按 expert 定位）
        params.mmadAddrs.aGmAddr = xGM_ + tokenPrefix * dataAxis;
        params.mmadAddrs.scaleAGmAddr = xScaleGM_ + tokenPrefix * scaleK;
        params.mmadAddrs.bGmAddr = weightGM_ + ((start * h1 * n1) >> packedShift);
        params.mmadAddrs.scaleBGmAddr = weightScaleGM_ + start * n1 * scaleK;
        params.mmadAddrs.cGmAddr = gmmOutputGM_ + tokenPrefix * n1 * sizeof(CType);
        params.mmadAddrs.biasGmAddr = nullptr;
        params.groupListGmAddr = groupListGm_;
        params.gmmParams = gmmTiling;
        params.groupOffsetTables = {};
        params.groupOffsetTables.cGroupAddrTable = reinterpret_cast<const __gm__ uint64_t*>(cGroupOffsetTableGm_);
        params.groupOffsetTables.bGroupAddrTable = reinterpret_cast<const __gm__ uint64_t*>(bGroupOffsetTableGm_);
        params.groupOffsetTables.scaleBGroupAddrTable =
            reinterpret_cast<const __gm__ uint64_t*>(scaleBGroupOffsetTableGm_);

        // 镜像生命周期纪律：栈上 ctor/dtor 成对（ctor SetFlag 预置 → Run →
        // dtor WaitFlag 消费），任意时刻仅一个 Blaze 实例存活
        {
            GmmKernel kernel;
            kernel(params);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr uint32_t MX_BASIC_FACTOR = 64U;
    // rank 维数组容量沿用 legacy MAX_EP_RANK_SIZE(128)：AIC 栈帧 32KB 受限，
    // 栈数组不得随意扩容
    static constexpr uint32_t GMM_A2AV_MAX_RANK_NUM = MC2KernelTemplate::MAX_EP_RANK_SIZE;

    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGroupedMmadWithScaleMx>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    float, LayoutBias>;
    using GmmKernel = Blaze::Gemm::Kernel::GroupedMatmulKernel<ProblemShape, BlockMmad, BlockScheduler,
                                                               Blaze::Gemm::Kernel::GroupedMatmulMode::MX_QUANT>;

    __aicore__ inline uint32_t RankSize() const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.epWorldSize);
    }

    __aicore__ inline uint32_t ExpertNum() const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.e);
    }

    __aicore__ inline uint32_t BatchExpertNum() const
    {
        return tilingData_->taskTilingInfo.expertNum;
    }

    __aicore__ inline uint32_t GetSendRows(uint32_t expertIdx, uint32_t peer) const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.sendCnt[peer * ExpertNum() + expertIdx]);
    }

    __aicore__ inline uint32_t ScaleKLen() const
    {
        return static_cast<uint32_t>((tilingData_->taskTilingInfo.H1 + MX_BASIC_FACTOR - 1U) / MX_BASIC_FACTOR * 2U);
    }

    __aicore__ inline uint32_t GetBatchExpertNum(uint32_t batch) const
    {
        const uint32_t start = batch * BatchExpertNum();
        return (start + BatchExpertNum() > ExpertNum()) ? ExpertNum() - start : BatchExpertNum();
    }

    __aicore__ inline uint64_t GetMaxExpertTokens() const
    {
        uint64_t mMax = 0;
        for (uint32_t expert = 0U; expert < ExpertNum(); ++expert) {
            uint64_t tokens = 0;
            for (uint32_t rank = 0U; rank < RankSize(); ++rank) {
                tokens += GetSendRows(expert, rank);
            }
            mMax = tokens > mMax ? tokens : mMax;
        }
        return mMax;
    }

    __aicore__ inline void BuildBatchTables(uint32_t start, uint32_t expertNum, uint32_t rankSize, uint64_t n1,
                                            uint64_t scaleK, uint64_t packedShift)
    {
        // 全核冗余写相同值（每核写读一致，无跨核可见性依赖）。
        // group 序 expert-major：g = jj * rankSize + rank（与 gmmX 行序一致）
        __gm__ int64_t* groupList = reinterpret_cast<__gm__ int64_t*>(groupListGm_);
        __gm__ uint64_t* cGroupOffsetTable = reinterpret_cast<__gm__ uint64_t*>(cGroupOffsetTableGm_);
        __gm__ uint64_t* bGroupOffsetTable = reinterpret_cast<__gm__ uint64_t*>(bGroupOffsetTableGm_);
        __gm__ uint64_t* scaleBGroupOffsetTable = reinterpret_cast<__gm__ uint64_t*>(scaleBGroupOffsetTableGm_);

        // groupList（LENGTH 计数）
        for (uint32_t jj = 0U; jj < expertNum; ++jj) {
            for (uint32_t rank = 0U; rank < rankSize; ++rank) {
                groupList[jj * rankSize + rank] = static_cast<int64_t>(GetSendRows(start + jj, rank));
            }
        }

        // C 偏移表：批内 rank-major 物理布局（元素偏移，相对本批 C 基址）
        uint64_t rankTotal[GMM_A2AV_MAX_RANK_NUM] = {0};
        for (uint32_t jj = 0U; jj < expertNum; ++jj) {
            for (uint32_t rank = 0U; rank < rankSize; ++rank) {
                rankTotal[rank] += GetSendRows(start + jj, rank);
            }
        }
        uint64_t expertRunning[GMM_A2AV_MAX_RANK_NUM] = {0};
        for (uint32_t rank = 1U; rank < rankSize; ++rank) {
            expertRunning[rank] = expertRunning[rank - 1] + rankTotal[rank - 1] * n1;
        }
        for (uint32_t jj = 0U; jj < expertNum; ++jj) {
            for (uint32_t rank = 0U; rank < rankSize; ++rank) {
                const uint32_t groupIdx = jj * rankSize + rank;
                cGroupOffsetTable[groupIdx] = expertRunning[rank];
                expertRunning[rank] += static_cast<uint64_t>(GetSendRows(start + jj, rank)) * n1;
            }
        }

        // B/scaleB 偏移表：批内 expert 粒度（相对本批 B/scaleB 基址，元素偏移）
        const uint64_t perExpertW = (tilingData_->taskTilingInfo.H1 * n1) >> packedShift;
        for (uint32_t jj = 0U; jj < expertNum; ++jj) {
            for (uint32_t rank = 0U; rank < rankSize; ++rank) {
                const uint32_t groupIdx = jj * rankSize + rank;
                bGroupOffsetTable[groupIdx] = static_cast<uint64_t>(jj) * perExpertW;
                scaleBGroupOffsetTable[groupIdx] = static_cast<uint64_t>(jj) * n1 * scaleK;
            }
        }
    }

    const TilingDataType* tilingData_{nullptr};
    GM_ADDR xGM_{nullptr};
    GM_ADDR xScaleGM_{nullptr};
    GM_ADDR weightGM_{nullptr};
    GM_ADDR weightScaleGM_{nullptr};
    GM_ADDR gmmOutputGM_{nullptr};
    GM_ADDR groupListGm_{nullptr};
    GM_ADDR cGroupOffsetTableGm_{nullptr};
    GM_ADDR bGroupOffsetTableGm_{nullptr};
    GM_ADDR scaleBGroupOffsetTableGm_{nullptr};
};

// ============================================================================
// 共享专家计算算子（单组，独立输出，不走通信）
// ============================================================================
template <typename TilingDataType, typename AType, typename BType, typename CType, bool TransMmB>
class QuantMmSharedComputeOp {
public:
    __aicore__ inline QuantMmSharedComputeOp() {}

    __aicore__ inline void Init(const TilingDataType* tilingData, GM_ADDR mmXGM, GM_ADDR mmXScaleGM, GM_ADDR mmWeightGM,
                                GM_ADDR mmWeightScaleGM, GM_ADDR mmYGM, GM_ADDR tableGM)
    {
        tilingData_ = tilingData;
        mmXGM_ = mmXGM;
        mmXScaleGM_ = mmXScaleGM;
        mmWeightGM_ = mmWeightGM;
        mmWeightScaleGM_ = mmWeightScaleGM;
        mmYGM_ = mmYGM;
        // mmGroupList 位于路由表区（groupList + 三张偏移表 = 4*e*R*8）之后，
        // 布局公式与 QuantGmmComputeOp::Init 保持一致（见 scheduler 头注释）
        mmGroupListGm_ = tableGM + 4UL * RankSize() * ExpertNum() * sizeof(uint64_t);
        // 共享专家单组 OFFSET 格式 groupList：[0] = BS（全核冗余写）
        *reinterpret_cast<__gm__ int64_t*>(mmGroupListGm_) = static_cast<int64_t>(tilingData_->taskTilingInfo.BS);
    }

    __aicore__ inline void Process()
    {
        // 无共享专家时 BS/H2/N2 均为 0（tiling 侧未填）
        if (tilingData_->taskTilingInfo.BS == 0U || tilingData_->taskTilingInfo.H2 == 0U ||
            tilingData_->taskTilingInfo.N2 == 0U) {
            return;
        }
        const uint64_t m = tilingData_->taskTilingInfo.BS;
        const uint64_t n = tilingData_->taskTilingInfo.N2;
        const uint64_t k = tilingData_->taskTilingInfo.H2;
        typename MmKernel::GMMTiling mmTiling{};
        FillGmmA2avTiling(mmTiling, tilingData_->sharedGmmTiling, 1U, static_cast<int64_t>(m), static_cast<int64_t>(n),
                          static_cast<int64_t>(k), GROUP_LIST_TYPE_OFFSET);

        typename MmKernel::Params params{};
        params.problemShape =
            ProblemShape{static_cast<int64_t>(m), static_cast<int64_t>(n), static_cast<int64_t>(k), 0};
        params.mmadAddrs.aGmAddr = mmXGM_;
        params.mmadAddrs.scaleAGmAddr = mmXScaleGM_;
        params.mmadAddrs.bGmAddr = mmWeightGM_;
        params.mmadAddrs.scaleBGmAddr = mmWeightScaleGM_;
        params.mmadAddrs.cGmAddr = mmYGM_;
        params.mmadAddrs.biasGmAddr = nullptr;
        params.groupListGmAddr = mmGroupListGm_;
        params.gmmParams = mmTiling;
        params.groupOffsetTables = {};

        // 镜像生命周期纪律：栈上 ctor/dtor 成对（此时路由侧最后一批的 dtor 已
        // 消费事件，本 ctor 为全新预置，无双实例并存）
        {
            MmKernel kernel;
            kernel(params);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr uint8_t GROUP_LIST_TYPE_OFFSET = 0U;

    using LayoutA = asc::te::nd_ext_layout_ptn;
    using MmLayoutB = AscendC::Std::conditional_t<TransMmB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGroupedMmadWithScaleMx>;
    using MmBlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, MmLayoutB, CType, LayoutC,
                                                      float, LayoutBias>;
    using MmKernel = Blaze::Gemm::Kernel::GroupedMatmulKernel<ProblemShape, MmBlockMmad, BlockScheduler,
                                                              Blaze::Gemm::Kernel::GroupedMatmulMode::MX_QUANT>;

    __aicore__ inline uint32_t RankSize() const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.epWorldSize);
    }

    __aicore__ inline uint32_t ExpertNum() const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.e);
    }

    const TilingDataType* tilingData_{nullptr};
    GM_ADDR mmXGM_{nullptr};
    GM_ADDR mmXScaleGM_{nullptr};
    GM_ADDR mmWeightGM_{nullptr};
    GM_ADDR mmWeightScaleGM_{nullptr};
    GM_ADDR mmYGM_{nullptr};
    GM_ADDR mmGroupListGm_{nullptr};
};

// ============================================================================
// AIC 核内 AllToAllV 通信算子（非阻塞提交 + 尾部统一等待）
// ============================================================================
template <typename TilingDataType, typename CType, AscendC::HcclServerType ServerType>
class QuantA2avCommOp {
public:
    __aicore__ inline QuantA2avCommOp() {}

    __aicore__ inline void Init(const TilingDataType* tilingData, GM_ADDR gmmOutputGM, GM_ADDR yGM)
    {
        tilingData_ = tilingData;
        gmmOutputGM_ = gmmOutputGM;
        yGM_ = yGM;
        hccl_.InitV2(AscendC::GetHcclContext<0>(), &tilingData_->hcclA2avTiling.hcclInitTiling);
        hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(TilingDataType, hcclA2avTiling)) +
                            static_cast<uint64_t>(offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling)));
    }

    // 提交本批非阻塞 AllToAllV（sendBuf=路由专家 GMM 输出，recvBuf=y）
    __aicore__ inline void Launch(uint32_t batch)
    {
        // 通信参数逐字节对齐 409d5e04a / 非 apace 通路：legacy A2avCommParams
        // （128 rank 容量、4KB 栈），公式为 CalcA2avCommAfterParams 同款伸缩和形态
        MC2KernelTemplate::A2avCommParams commParams;
        const uint32_t e = ExpertNum();
        const uint32_t rankDim = RankSize();
        const uint64_t n1 = tilingData_->taskTilingInfo.N1;
        const int32_t* sendCounts = tilingData_->taskTilingInfo.sendCnt;
        const int32_t* recvCounts = tilingData_->taskTilingInfo.recvCnt;
        const uint32_t start = batch * BatchExpertNum();
        const uint32_t expertNum = GetBatchExpertNum(batch);

        // 本批 send/recv 计数（元素单位，axis = N1）
        for (uint32_t i = 0U; i < rankDim; i++) {
            commParams.sendCnt[i] = 0UL;
            commParams.recvCnt[i] = 0UL;
            for (uint32_t expertIdx = start; expertIdx < start + expertNum; expertIdx++) {
                commParams.sendCnt[i] += static_cast<uint64_t>(sendCounts[expertIdx + i * e]) * n1;
                commParams.recvCnt[i] += static_cast<uint64_t>(recvCounts[expertIdx + i * e]) * n1;
            }
        }

        // send 偏移：GMM 输出为批间串接、批内 rank-major 布局
        // （本批起点 = 批前全部 token 累计；批内 rank 段首 = 本批起点 + 前 rank 段累计）
        uint64_t batchStartOffset = 0UL;
        for (uint32_t i = 0U; i < start; i++) {
            for (uint32_t j = 0U; j < rankDim; j++) {
                batchStartOffset += static_cast<uint64_t>(sendCounts[i + j * e]);
            }
        }
        batchStartOffset *= n1;
        commParams.sendOffset[0] = batchStartOffset;
        uint64_t rankRunningOffset = 0UL;
        for (uint32_t i = 1U; i < rankDim; i++) {
            for (uint32_t expertIdx = start; expertIdx < start + expertNum; expertIdx++) {
                rankRunningOffset += static_cast<uint64_t>(sendCounts[expertIdx + (i - 1) * e]) * n1;
            }
            commParams.sendOffset[i] = batchStartOffset + rankRunningOffset;
        }

        // recv 偏移：y 按 recvCounts 源 rank 聚合顺序（rank-major，伸缩和恒等式）
        commParams.recvOffset[0] = 0UL;
        for (uint32_t i = 0U; i < start; i++) {
            commParams.recvOffset[0] += static_cast<uint64_t>(recvCounts[i]) * n1;
        }
        for (uint32_t i = 1U; i < rankDim; i++) {
            commParams.recvOffset[i] = commParams.recvOffset[i - 1];
            for (uint32_t j = 0U; j < e; j++) {
                commParams.recvOffset[i] += static_cast<uint64_t>(recvCounts[start + (i - 1) * e + j]) * n1;
            }
        }

        a2avHandles_[submitted_] = hccl_.template AlltoAllV<true>(
            gmmOutputGM_, commParams.sendCnt, commParams.sendOffset, Y_HCCL_DATA_TYPE, yGM_, commParams.recvCnt,
            commParams.recvOffset, Y_HCCL_DATA_TYPE, static_cast<uint8_t>(1));
        submitted_++;
    }

    __aicore__ inline void WaitAll()
    {
        // 全核按提交顺序等待全部批次完成
        for (uint32_t i = 0U; i < submitted_; ++i) {
            hccl_.Wait(a2avHandles_[i]);
        }
    }

    __aicore__ inline void Finalize()
    {
        if (submitted_ > 0U) {
            hccl_.Finalize();
        }
    }

private:
    // 非阻塞 AllToAllV handle 上限：totalLoopCount <= 4，留余量取 8
    static constexpr uint32_t GMM_A2AV_MAX_BATCH_NUM = 8U;
    static constexpr AscendC::HcclDataType Y_HCCL_DATA_TYPE = GetGmmA2avYHcclDataType<CType>();

    __aicore__ inline uint32_t RankSize() const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.epWorldSize);
    }

    __aicore__ inline uint32_t ExpertNum() const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.e);
    }

    __aicore__ inline uint32_t BatchExpertNum() const
    {
        return tilingData_->taskTilingInfo.expertNum;
    }

    __aicore__ inline uint32_t GetSendRows(uint32_t expertIdx, uint32_t peer) const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.sendCnt[peer * ExpertNum() + expertIdx]);
    }

    __aicore__ inline uint32_t GetRecvRows(uint32_t expertIdx, uint32_t peer) const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.recvCnt[peer * ExpertNum() + expertIdx]);
    }

    __aicore__ inline uint32_t GetBatchExpertNum(uint32_t batch) const
    {
        const uint32_t start = batch * BatchExpertNum();
        return (start + BatchExpertNum() > ExpertNum()) ? ExpertNum() - start : BatchExpertNum();
    }

    const TilingDataType* tilingData_{nullptr};
    AscendC::Hccl<ServerType> hccl_;
    AscendC::HcclHandle a2avHandles_[GMM_A2AV_MAX_BATCH_NUM]{};
    uint32_t submitted_{0};
    GM_ADDR gmmOutputGM_{nullptr};
    GM_ADDR yGM_{nullptr};
};

} // namespace Apace

#endif // APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__
