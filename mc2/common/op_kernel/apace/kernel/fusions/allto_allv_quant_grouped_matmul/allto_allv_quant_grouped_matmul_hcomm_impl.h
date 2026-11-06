/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_quant_grouped_matmul_hcomm_impl.h
 * \brief AIC-only HCCL AllToAllV + Blaze MX GroupedMatMul implementation.
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "../../matmul/grouped_matmul/gmm_kernel.h"

namespace Apace {

template <typename T>
constexpr AscendC::HcclDataType GetA2avGmmHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, float8_e5m2_t>::value) {
        return AscendC::HCCL_DATA_TYPE_FP8E5M2;
    } else if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) {
        return AscendC::HCCL_DATA_TYPE_UINT8;
    } else {
        return AscendC::HCCL_DATA_TYPE_FP8E4M3;
    }
}

template <typename TilingDataType, typename AType, typename BType, typename CType, bool TransB, bool TransMmB,
          AscendC::HcclServerType ServerType>
class AllToAllvQuantGmmHcommImpl {
public:
    explicit __aicore__ inline AllToAllvQuantGmmHcommImpl(const TilingDataType* tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR gmmX, GM_ADDR gmmXScale, GM_ADDR gmmWeight, GM_ADDR gmmWeightScale, GM_ADDR mmX,
                                GM_ADDR mmXScale, GM_ADDR mmWeight, GM_ADDR mmWeightScale, GM_ADDR gmmY, GM_ADDR mmY,
                                GM_ADDR permuteOut, GM_ADDR workspace)
    {
        xGM_ = gmmX;
        xScaleGM_ = gmmXScale;
        weightGM_ = gmmWeight;
        weightScaleGM_ = gmmWeightScale;
        yGM_ = gmmY;
        mmXGM_ = mmX;
        mmXScaleGM_ = mmXScale;
        mmWeightGM_ = mmWeight;
        mmWeightScaleGM_ = mmWeightScale;
        mmYGM_ = mmY;
        dataRecvGM_ = tilingData_->isPermuteOut ? permuteOut : workspace;
        scaleRecvGM_ = workspace + tilingData_->workspaceInfo.wsDataRecvSize;
        sparseGroupListGM_ = scaleRecvGM_ + tilingData_->workspaceInfo.wsScaleRecvSize;
        aGroupOffsetTableGM_ = sparseGroupListGM_ + RankSize() * ExpertNum() * 2U * sizeof(int64_t);
        scaleAGroupOffsetTableGM_ = aGroupOffsetTableGM_ + RankSize() * ExpertNum() * sizeof(uint64_t);
        mmGroupListGM_ = scaleAGroupOffsetTableGM_ + RankSize() * ExpertNum() * sizeof(uint64_t);
        BuildGroupLists();

        hccl_.InitV2(AscendC::GetHcclContext<0>(), &tilingData_->hcclA2avTilingInfo.hcclInitTiling);
        hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(TilingDataType, hcclA2avTilingInfo)) +
                            static_cast<uint64_t>(offsetof(decltype(tilingData_->hcclA2avTilingInfo), a2avCcTiling)));
    }

    __aicore__ inline void Run()
    {
        const uint32_t batchExpertNum = BatchExpertNum();
        SubmitBatch(0U, MinU32(batchExpertNum, ExpertNum()));
        if (tilingData_->isNeedMM) {
            RunLocalMatmul();
        }

        for (uint32_t startExpertIdx = 0U; startExpertIdx < ExpertNum(); startExpertIdx += batchExpertNum) {
            const uint32_t actualExpertNum = MinU32(batchExpertNum, ExpertNum() - startExpertIdx);
            WaitBatch();
            const uint32_t nextExpertIdx = startExpertIdx + actualExpertNum;
            if (nextExpertIdx < ExpertNum()) {
                SubmitBatch(nextExpertIdx, MinU32(batchExpertNum, ExpertNum() - nextExpertIdx));
            }
            RunExpertMatmul(startExpertIdx, actualExpertNum);
        }

        AscendC::SyncAll();
        hccl_.Finalize();
    }

private:
    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    static constexpr uint32_t A2AV_GMM_MAX_RANK_NUM = 256U;
    static constexpr uint32_t MX_BASIC_FACTOR = 64U;
    static constexpr uint8_t GROUP_LIST_TYPE_OFFSET = 0U;
    static constexpr uint8_t GROUP_LIST_TYPE_SPARSE = 2U;
    static constexpr AscendC::HcclDataType X_HCCL_DATA_TYPE = GetA2avGmmHcclDataType<AType>();
    static constexpr AscendC::HcclDataType SCALE_HCCL_DATA_TYPE = AscendC::HCCL_DATA_TYPE_FP8E8M0;

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

    using MmLayoutB = AscendC::Std::conditional_t<TransMmB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
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

    __aicore__ inline uint32_t BatchExpertNum() const
    {
        if (tilingData_->isPermuteOut) {
            return 1U;
        }
        const uint32_t batchExpertNum = static_cast<uint32_t>(tilingData_->taskTilingInfo.expertNum);
        return batchExpertNum == 0U ? 1U : batchExpertNum;
    }

    __aicore__ inline uint32_t GetSendRows(uint32_t expertIdx, uint32_t peer) const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.sendCnt[peer * ExpertNum() + expertIdx]);
    }

    __aicore__ inline uint32_t GetRecvRows(uint32_t expertIdx, uint32_t peer) const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.recvCnt[peer * ExpertNum() + expertIdx]);
    }

    __aicore__ inline uint32_t ScaleKLen() const
    {
        return static_cast<uint32_t>((tilingData_->taskTilingInfo.H1 + MX_BASIC_FACTOR - 1U) / MX_BASIC_FACTOR * 2U);
    }

    __aicore__ inline void BuildGroupLists()
    {
        __gm__ int64_t* sparseGroupList = reinterpret_cast<__gm__ int64_t*>(sparseGroupListGM_);
        for (uint32_t expert = 0U; expert < ExpertNum(); ++expert) {
            for (uint32_t peer = 0U; peer < RankSize(); ++peer) {
                const uint32_t segment = expert * RankSize() + peer;
                sparseGroupList[segment * 2U] = static_cast<int64_t>(expert);
                sparseGroupList[segment * 2U + 1U] = static_cast<int64_t>(GetRecvRows(expert, peer));
            }
        }
        BuildGroupOffsetTables();
        *reinterpret_cast<__gm__ int64_t*>(mmGroupListGM_) = static_cast<int64_t>(tilingData_->taskTilingInfo.BS);
        AscendC::SyncAll();
    }

    __aicore__ inline void BuildGroupOffsetTables()
    {
        __gm__ uint64_t* aOffsets = reinterpret_cast<__gm__ uint64_t*>(aGroupOffsetTableGM_);
        __gm__ uint64_t* scaleAOffsets = reinterpret_cast<__gm__ uint64_t*>(scaleAGroupOffsetTableGM_);
        const uint64_t k = tilingData_->taskTilingInfo.H1;
        const uint64_t dataAxis = AscendC::IsSameType<AType, fp4x2_e2m1_t>::value ? (k + 1U) / 2U : k;
        const uint64_t scaleAxis = ScaleKLen();
        const uint32_t batchExpertNum = BatchExpertNum();

        for (uint32_t start = 0U; start < ExpertNum(); start += batchExpertNum) {
            const uint32_t actualExpertNum = MinU32(batchExpertNum, ExpertNum() - start);
            uint64_t rankBaseRow = GetExpertStartRow(start);
            for (uint32_t peer = 0U; peer < RankSize(); ++peer) {
                uint64_t expertRowOffset = rankBaseRow;
                for (uint32_t e = 0U; e < actualExpertNum; ++e) {
                    const uint32_t expert = start + e;
                    const uint32_t segment = expert * RankSize() + peer;
                    aOffsets[segment] = expertRowOffset * dataAxis;
                    scaleAOffsets[segment] = expertRowOffset * scaleAxis;
                    expertRowOffset += GetRecvRows(expert, peer);
                }
                rankBaseRow = expertRowOffset;
            }
        }
    }

    __aicore__ inline uint64_t GetExpertStartRow(uint32_t expertIdx) const
    {
        uint64_t rows = 0U;
        for (uint32_t previous = 0U; previous < expertIdx; ++previous) {
            for (uint32_t peer = 0U; peer < RankSize(); ++peer) {
                rows += GetRecvRows(previous, peer);
            }
        }
        return rows;
    }

    __aicore__ inline void PrepareBatchCommunication(uint32_t startExpertIdx, uint32_t expertNum)
    {
        const uint64_t k = tilingData_->taskTilingInfo.H1;
        const uint64_t dataAxis = AscendC::IsSameType<AType, fp4x2_e2m1_t>::value ? (k + 1U) / 2U : k;
        const uint64_t scaleAxis = ScaleKLen();
        uint64_t sendPeerBaseRow = 0U;
        uint64_t recvRowOffset = GetExpertStartRow(startExpertIdx);

        for (uint32_t peer = 0U; peer < RankSize(); ++peer) {
            uint64_t sendExpertOffset = 0U;
            uint64_t sendPeerRows = 0U;
            for (uint32_t e = 0U; e < ExpertNum(); ++e) {
                const uint64_t rows = GetSendRows(e, peer);
                if (e < startExpertIdx) {
                    sendExpertOffset += rows;
                }
                sendPeerRows += rows;
            }

            uint64_t sendRows = 0U;
            uint64_t recvRows = 0U;
            for (uint32_t e = 0U; e < expertNum; ++e) {
                sendRows += GetSendRows(startExpertIdx + e, peer);
                recvRows += GetRecvRows(startExpertIdx + e, peer);
            }
            dataSendCount_[peer] = sendRows * dataAxis;
            dataSendOffset_[peer] = (sendPeerBaseRow + sendExpertOffset) * dataAxis;
            dataRecvCount_[peer] = recvRows * dataAxis;
            dataRecvOffset_[peer] = recvRowOffset * dataAxis;
            scaleSendCount_[peer] = sendRows * scaleAxis;
            scaleSendOffset_[peer] = (sendPeerBaseRow + sendExpertOffset) * scaleAxis;
            scaleRecvCount_[peer] = recvRows * scaleAxis;
            scaleRecvOffset_[peer] = recvRowOffset * scaleAxis;

            sendPeerBaseRow += sendPeerRows;
            recvRowOffset += recvRows;
        }
    }

    __aicore__ inline void SubmitBatch(uint32_t startExpertIdx, uint32_t expertNum)
    {
        PrepareBatchCommunication(startExpertIdx, expertNum);
        scaleHandle_ =
            hccl_.template AlltoAllV<true>(xScaleGM_, scaleSendCount_, scaleSendOffset_, SCALE_HCCL_DATA_TYPE,
                                           scaleRecvGM_, scaleRecvCount_, scaleRecvOffset_, SCALE_HCCL_DATA_TYPE);
        dataHandle_ = hccl_.template AlltoAllV<true>(xGM_, dataSendCount_, dataSendOffset_, X_HCCL_DATA_TYPE,
                                                     dataRecvGM_, dataRecvCount_, dataRecvOffset_, X_HCCL_DATA_TYPE);
    }

    __aicore__ inline void WaitBatch()
    {
        hccl_.Wait(scaleHandle_);
        hccl_.Wait(dataHandle_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RunExpertMatmul(uint32_t startExpertIdx, uint32_t expertNum)
    {
        int64_t expertRows = 0;
        for (uint32_t e = 0U; e < expertNum; ++e) {
            for (uint32_t peer = 0U; peer < RankSize(); ++peer) {
                expertRows += static_cast<int64_t>(GetRecvRows(startExpertIdx + e, peer));
            }
        }
        if (expertRows == 0) {
            return;
        }

        const uint64_t expertStartRow = GetExpertStartRow(startExpertIdx);
        const uint64_t k = tilingData_->taskTilingInfo.H1;
        const uint64_t n = tilingData_->taskTilingInfo.N1;
        typename GmmKernel::GMMTiling gmmTiling{};
        FillGmmTiling(gmmTiling, tilingData_->gmmQuantTilingData, expertNum * RankSize(), expertRows, n, k,
                      GROUP_LIST_TYPE_SPARSE);

        typename GmmKernel::Params params{};
        params.problemShape = ProblemShape{expertRows, static_cast<int64_t>(n), static_cast<int64_t>(k), 0};
        params.mmadAddrs.aGmAddr = dataRecvGM_;
        params.mmadAddrs.scaleAGmAddr = scaleRecvGM_;
        params.mmadAddrs.bGmAddr = weightGM_;
        params.mmadAddrs.scaleBGmAddr = weightScaleGM_;
        params.mmadAddrs.cGmAddr = yGM_ + expertStartRow * n * sizeof(CType);
        params.mmadAddrs.biasGmAddr = nullptr;
        params.groupListGmAddr = sparseGroupListGM_ + startExpertIdx * RankSize() * 2U * sizeof(int64_t);
        params.gmmParams = gmmTiling;
        params.groupOffsetTables.aGroupAddrTable =
            reinterpret_cast<const __gm__ uint64_t*>(aGroupOffsetTableGM_) + startExpertIdx * RankSize();
        params.groupOffsetTables.scaleAGroupAddrTable =
            reinterpret_cast<const __gm__ uint64_t*>(scaleAGroupOffsetTableGM_) + startExpertIdx * RankSize();

        {
            GmmKernel kernel;
            kernel(params);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RunLocalMatmul()
    {
        typename MmKernel::GMMTiling mmTiling{};
        const uint64_t m = tilingData_->taskTilingInfo.BS;
        const uint64_t n = tilingData_->taskTilingInfo.N2;
        const uint64_t k = tilingData_->taskTilingInfo.H2;
        FillGmmTiling(mmTiling, tilingData_->mmQuantTilingData, 1U, m, n, k, GROUP_LIST_TYPE_OFFSET);

        typename MmKernel::Params params{};
        params.problemShape =
            ProblemShape{static_cast<int64_t>(m), static_cast<int64_t>(n), static_cast<int64_t>(k), 0};
        params.mmadAddrs.aGmAddr = mmXGM_;
        params.mmadAddrs.scaleAGmAddr = mmXScaleGM_;
        params.mmadAddrs.bGmAddr = mmWeightGM_;
        params.mmadAddrs.scaleBGmAddr = mmWeightScaleGM_;
        params.mmadAddrs.cGmAddr = mmYGM_;
        params.mmadAddrs.biasGmAddr = nullptr;
        params.groupListGmAddr = mmGroupListGM_;
        params.gmmParams = mmTiling;
        params.groupOffsetTables = {};

        {
            MmKernel kernel;
            kernel(params);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <typename GmmTiling, typename GmmTilingData>
    __aicore__ inline void FillGmmTiling(GmmTiling& out, const GmmTilingData& in, uint32_t groupNum, int64_t m,
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
        out.isBias = 0U;
        out.dbL0C = static_cast<uint8_t>(mmTiling.dbL0C);
        out.l1BufferStage = 2U;
        out.groupListType = groupListType;
    }

private:
    const TilingDataType* tilingData_{nullptr};
    AscendC::Hccl<ServerType> hccl_;
    AscendC::HcclHandle scaleHandle_{0};
    AscendC::HcclHandle dataHandle_{0};

    GM_ADDR xGM_{nullptr};
    GM_ADDR xScaleGM_{nullptr};
    GM_ADDR weightGM_{nullptr};
    GM_ADDR weightScaleGM_{nullptr};
    GM_ADDR yGM_{nullptr};
    GM_ADDR mmXGM_{nullptr};
    GM_ADDR mmXScaleGM_{nullptr};
    GM_ADDR mmWeightGM_{nullptr};
    GM_ADDR mmWeightScaleGM_{nullptr};
    GM_ADDR mmYGM_{nullptr};
    GM_ADDR dataRecvGM_{nullptr};
    GM_ADDR scaleRecvGM_{nullptr};
    GM_ADDR sparseGroupListGM_{nullptr};
    GM_ADDR aGroupOffsetTableGM_{nullptr};
    GM_ADDR scaleAGroupOffsetTableGM_{nullptr};
    GM_ADDR mmGroupListGM_{nullptr};

    uint64_t dataSendCount_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t dataSendOffset_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t dataRecvCount_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t dataRecvOffset_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t scaleSendCount_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t scaleSendOffset_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t scaleRecvCount_[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint64_t scaleRecvOffset_[A2AV_GMM_MAX_RANK_NUM] = {0};
};

} // namespace Apace
