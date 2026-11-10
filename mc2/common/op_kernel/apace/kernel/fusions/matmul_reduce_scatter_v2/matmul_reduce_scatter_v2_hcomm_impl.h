/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "lib/hccl/hccl.h"
#if MMRS_APACE_FP16_BF16_DTYPE
#include "../../matmul/batch_matmul/matmul_reduce_scatter_v2_kernel.h"
#endif

#if MMRS_APACE_MX_DTYPE
#include "../../matmul/quant_batch_matmul/matmul_reduce_scatter_v2_mx_kernel.h"
#endif
#include "matmul_reduce_scatter_v2_tiling_data.h"

namespace Apace {

template <typename AType, typename BType, typename CType, bool BTrans, typename TilingDataType,
          AscendC::HcclServerType ServerType>
class MatmulReduceScatterV2HcommImpl {
public:
    static constexpr bool IS_MX = AscendC::IsSameType<TilingDataType, hcommMatmulReduceScatterV2TilingData>::value;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ != 2201 && __NPU_ARCH__ != 3510
    static constexpr bool IS_HIF4 = AscendC::IsSameType<AType, hifloat4x2_t>::value;
    static_assert(!(AscendC::IsSameType<AType, hifloat4x2_t>::value ||
                    AscendC::IsSameType<BType, hifloat4x2_t>::value) ||
                      AscendC::IsSameType<CType, half>::value || AscendC::IsSameType<CType, bfloat16_t>::value,
                  "HiFP4 MMRS only supports float16 or bfloat16 output");
#else
    static constexpr bool IS_HIF4 = false;
#endif
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = AscendC::Std::conditional_t<BTrans, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using BiasType = AscendC::Std::conditional_t<IS_MX, float, CType>;
#if MMRS_APACE_MX_DTYPE
#if defined(__NPU_ARCH__) && __NPU_ARCH__ != 2201 && __NPU_ARCH__ != 3510
    // HiFP4 复用 MXFP4 的分块调度，矩阵乘仍使用 HiFP4 类型。
    using SchedulerAType = AscendC::Std::conditional_t<IS_HIF4, fp4x2_e2m1_t, AType>;
#else
    using SchedulerAType = AType;
#endif
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, SchedulerAType>;
#else
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<0, 0, Blaze::Gemm::KernelMmadMultiBlockBasic, 0>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape>;
#endif
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutC>;
#if MMRS_APACE_MX_DTYPE
    using Kernel = Blaze::Gemm::Kernel::MatmulReduceScatterV2QbmmMxKernel<ProblemShape, BlockMmad, BlockScheduler>;
    using MatmulTiling = typename Kernel::QBMMTiling;
#else
    using Kernel = Blaze::Gemm::Kernel::MatmulReduceScatterV2Kernel<ProblemShape, BlockMmad, BlockScheduler>;
    using MatmulTiling = typename Kernel::MatmulTiling;
#endif
    using Params = typename Kernel::Params;

    __aicore__ inline explicit MatmulReduceScatterV2HcommImpl(const TilingDataType* tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM,
                                GM_ADDR yGM, GM_ADDR contextGM, GM_ADDR workspaceGM)
    {
        aGM_ = aGM;
        bGM_ = bGM;
        x1ScaleGM_ = x1ScaleGM;
        x2ScaleGM_ = x2ScaleGM;
        yGM_ = yGM;
        workspaceGM_ = workspaceGM;
        hccl_.InitV2(contextGM, &tilingData_->mc2InitTiling);
        hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(TilingDataType, mc2CcTiling)));
        if constexpr (IS_MX) {
            biasGM_ = IS_HIF4 ? nullptr : biasGM;
            rankDim_ = hccl_.GetRankDim();
            rankN_ = static_cast<uint64_t>(tilingData_->commTilingData.nonSplitAxisSize);
            dataType_ = AscendC::IsSameType<CType, float>::value ?
                            AscendC::HCCL_DATA_TYPE_FP32 :
                            (AscendC::IsSameType<CType, bfloat16_t>::value ? AscendC::HCCL_DATA_TYPE_BFP16 :
                                                                             AscendC::HCCL_DATA_TYPE_FP16);
        } else {
            biasGM_ = tilingData_->mC2Mmv3TileTilingData.tCubeTiling.isBias ? biasGM : nullptr;
            rankDim_ = tilingData_->param.rankDim;
            rankN_ = tilingData_->param.rankN;
            dataType_ = tilingData_->dataType;
        }
    }

    __aicore__ inline void Run()
    {
        Params tileParams;
        Params tailParams;
        uint32_t tileCount;
        uint32_t tailCount;
        uint64_t tileM;
        if constexpr (IS_MX) {
            const auto& commTiling = tilingData_->commTilingData;
            tileCount = static_cast<uint32_t>(commTiling.splitAxisTileCnt);
            tailCount = static_cast<uint32_t>(commTiling.splitAxisTailCnt);
            const bool invalidRank = rankDim_ < 2 || rankDim_ > APACE_MAX_RANK_DIM || (rankDim_ & (rankDim_ - 1U)) != 0;
            const bool invalidTiling =
                invalidRank || tileCount == 0 ||
                (tileCount > 0 &&
                 tilingData_->tileQbmmTilingData.m != static_cast<uint64_t>(commTiling.splitAxisTileSize) * rankDim_) ||
                (tailCount > 0 &&
                 tilingData_->tailQbmmTilingData.m != static_cast<uint64_t>(commTiling.splitAxisTailSize) * rankDim_);
            if (invalidTiling) {
                hccl_.Finalize();
                return;
            }
            tileM = static_cast<uint64_t>(commTiling.splitAxisTileSize);
            const uint64_t tailM = static_cast<uint64_t>(commTiling.splitAxisTailSize);
            const uint64_t perRankM = tileM * tileCount + tailM * tailCount;
            SetupMatmulParams(tileParams, tilingData_->tileQbmmTilingData, tileM, perRankM);
            if (tailCount > 0) {
                SetupMatmulParams(tailParams, tilingData_->tailQbmmTilingData, tailM, perRankM);
            }
        } else {
            const auto& cfg = tilingData_->param;
            tileCount = cfg.tileCnt;
            tailCount = cfg.tailCnt;
            tileM = tilingData_->mC2Mmv3TileTilingData.tCubeTiling.M / cfg.rankDim;
            SetupMatmulParams(tileParams, tilingData_->mC2Mmv3TileTilingData, tileM, cfg.rankM / cfg.rankDim);
            if (tailCount > 0) {
                const uint64_t tailM = tilingData_->mC2Mmv3TailTilingData.tCubeTiling.M / cfg.rankDim;
                SetupMatmulParams(tailParams, tilingData_->mC2Mmv3TailTilingData, tailM, cfg.rankM / cfg.rankDim);
            }
        }
        RunSliceGroup(tileParams, tileCount, 0);
        if (tailCount > 0) {
            RunSliceGroup(tailParams, tailCount, tileM * tileCount);
        }
        WaitPendingReduceScatter();
        hccl_.Finalize();
    }

private:
    template <typename GroupTiling>
    __aicore__ inline void SetupMatmulParams(Params& params, const GroupTiling& tiling, uint64_t sliceM,
                                             uint64_t perRankM)
    {
        params.mmParams = {aGM_, bGM_, workspaceGM_, biasGM_, x1ScaleGM_, x2ScaleGM_};
        params.sliceParams = {rankDim_, perRankM, 0, sliceM};
        if constexpr (IS_MX) {
            const uint32_t isBias = biasGM_ == nullptr ? MatmulTiling::BIAS_DISABLED : MatmulTiling::BIAS_ENABLED;
            params.problemShape = ProblemShape{static_cast<int64_t>(tiling.m), static_cast<int64_t>(tiling.n),
                                               static_cast<int64_t>(tiling.k), 1L};
            params.l1Params = {static_cast<uint64_t>(tiling.stepK) * tiling.baseK, tiling.scaleKL1, tiling.nBufferNum};
            params.schedulerParams = {tiling.baseM,
                                      tiling.baseN,
                                      tiling.mTailTile,
                                      tiling.nTailTile,
                                      tiling.mBaseTailSplitCnt,
                                      tiling.nBaseTailSplitCnt,
                                      tiling.mTailMain,
                                      tiling.nTailMain};
            params.qbmmParams = {tiling.baseM, tiling.baseN, tiling.baseK, tiling.dbL0c, isBias};
        } else {
            const auto& cube = tiling.tCubeTiling;
            const uint32_t stepK = cube.stepKa < cube.stepKb ? cube.stepKa : cube.stepKb;
            params.problemShape = ProblemShape{cube.M, cube.N, cube.Ka, 1L};
            params.l1Params = {static_cast<uint64_t>(cube.baseK) * stepK, 0, DOUBLE_BUFFER_COUNT};
            params.schedulerParams.mL1 = cube.baseM;
            params.schedulerParams.nL1 = cube.baseN;
            params.schedulerParams.kL1 = params.l1Params.kL1;
            params.schedulerParams.baseM = cube.baseM;
            params.schedulerParams.baseN = cube.baseN;
            params.schedulerParams.baseK = cube.baseK;
            params.schedulerParams.mTailCnt = tiling.mTailCnt;
            params.schedulerParams.nTailCnt = tiling.nTailCnt;
            params.schedulerParams.mBaseTailSplitCnt = tiling.mBaseTailSplitCnt;
            params.schedulerParams.nBaseTailSplitCnt = tiling.nBaseTailSplitCnt;
            params.schedulerParams.mTailMain = tiling.mTailMain;
            params.schedulerParams.nTailMain = tiling.nTailMain;
            params.schedulerParams.windowLen = tiling.aswWindowLen;
            params.matmulTiling = {static_cast<uint32_t>(cube.baseM), static_cast<uint32_t>(cube.baseN),
                                   static_cast<uint32_t>(cube.baseK), cube.dbL0C > 1 ? DOUBLE_BUFFER_COUNT : 1};
        }
    }

    __aicore__ inline void RunSliceGroup(const Params& groupParams, uint32_t sliceCount, uint64_t groupStart)
    {
        const auto& sp = groupParams.sliceParams;
        if constexpr (IS_MX) {
            const uint64_t logicalM = sp.sliceM * sp.rankDim;
            if (sp.rankDim == 0 || sp.sliceM == 0 || sp.perRankM == 0 ||
                groupStart + static_cast<uint64_t>(sliceCount) * sp.sliceM > sp.perRankM ||
                static_cast<uint64_t>(asc::te::get<Blaze::Gemm::MNK_M>(groupParams.problemShape)) != logicalM) {
                return;
            }
        }
        kernel_.InitGroup(groupParams);
        Params params = groupParams;
        const uint64_t n = static_cast<uint64_t>(asc::te::get<Blaze::Gemm::MNK_N>(params.problemShape));
        for (uint32_t i = 0; i < sliceCount; ++i) {
            const uint64_t sliceStart = groupStart + static_cast<uint64_t>(i) * sp.sliceM;
            params.sliceParams.sliceStart = sliceStart;
            // 各通信片写入 workspace 的不同区域；两个槽位只保存通信句柄。
            params.mmParams.cGmAddr = workspaceGM_ + sliceStart * sp.rankDim * n * sizeof(CType);
            kernel_.ProcessSlice(params);
            SyncAicCores();
            IssueReduceScatter(sliceStart, sp.sliceM);
        }
    }

    __aicore__ inline void SyncAicCores()
    {
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(AIC_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(AIC_SYNC_EVENT_ID);
    }

    __aicore__ inline void IssueReduceScatter(uint64_t sliceStart, uint64_t sliceM)
    {
        const uint32_t slot = issuedHandleCount_ % HCOMM_PIPELINE_DEPTH;
        if (issuedHandleCount_ >= HCOMM_PIPELINE_DEPTH) {
            hccl_.Wait(pipelineHandles_[slot]);
            SyncAicCores();
        }
        GM_ADDR sendBuffer = workspaceGM_ + sliceStart * rankDim_ * rankN_ * sizeof(CType);
        GM_ADDR recvBuffer = yGM_ + sliceStart * rankN_ * sizeof(CType);
        pipelineHandles_[slot] = hccl_.template ReduceScatter<true>(sendBuffer, recvBuffer, sliceM * rankN_,
                                                                    static_cast<AscendC::HcclDataType>(dataType_),
                                                                    AscendC::HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
        ++issuedHandleCount_;
    }

    __aicore__ inline void WaitPendingReduceScatter()
    {
        const uint32_t pendingCount =
            issuedHandleCount_ < HCOMM_PIPELINE_DEPTH ? issuedHandleCount_ : HCOMM_PIPELINE_DEPTH;
        const uint32_t firstPending = issuedHandleCount_ - pendingCount;
        for (uint32_t i = 0; i < pendingCount; ++i) {
            hccl_.Wait(pipelineHandles_[(firstPending + i) % HCOMM_PIPELINE_DEPTH]);
        }
        SyncAicCores();
    }

    static constexpr uint32_t DOUBLE_BUFFER_COUNT = 2U;
    static constexpr uint32_t HCOMM_PIPELINE_DEPTH = 2U;
    static constexpr uint32_t APACE_MAX_RANK_DIM = 64U;
    static constexpr uint16_t AIC_SYNC_EVENT_ID = 3U;
    const TilingDataType* tilingData_;
    Kernel kernel_;
    AscendC::Hccl<ServerType> hccl_;
    AscendC::HcclHandle pipelineHandles_[HCOMM_PIPELINE_DEPTH]{};
    uint32_t issuedHandleCount_{0};
    uint32_t rankDim_{0};
    uint64_t rankN_{0};
    uint64_t dataType_{0};
    GM_ADDR aGM_{nullptr};
    GM_ADDR bGM_{nullptr};
    GM_ADDR biasGM_{nullptr};
    GM_ADDR x1ScaleGM_{nullptr};
    GM_ADDR x2ScaleGM_{nullptr};
    GM_ADDR yGM_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
};

} // namespace Apace
