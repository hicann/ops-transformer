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
 * \file mega_moe_wave_a16w16.h
 * \brief MegaMoe A16W16 输入通信与专家计算
 */

#ifndef MEGA_MOE_WAVE_A16W16_H
#define MEGA_MOE_WAVE_A16W16_H

#include "../arch35_92/common/mega_moe_arch_policy.h"

#ifdef MEGA_MOE_ARCH35_DEVICE
#include "mega_moe_wave_a8w8.h"
#include "stage/mega_moe_gmm_bf16.h"

namespace MegaMoeImpl {

template <>
struct InputPrepareSelector<bfloat16_t> {
    using Config = Bf16InputConfig;
    using Scratch = Bf16InputScratch;
};

template <>
struct GmmBlockContextSelector<bfloat16_t, BF16_PASSTHROUGH> {
    using Type = Bf16GmmBlockContext;
};

#define TemplateMegaMoeA16W16WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename MoeWeightType, int32_t MoeQuantMode, \
        typename SharedWeightType, int32_t SharedQuantMode, int32_t MoeWeight1Format, int32_t MoeWeight2Format, \
        int32_t SharedWeight1Format, int32_t SharedWeight2Format
#define TemplateMegaMoeA16W16WaveTypeFunc \
    XType, OutputType, TopkWeightsType, MoeWeightType, MoeQuantMode, SharedWeightType, SharedQuantMode, \
        MoeWeight1Format, MoeWeight2Format, SharedWeight1Format, SharedWeight2Format

template <TemplateMegaMoeA16W16WaveTypeClass>
class MegaMoeA16W16Wave
    : public MegaMoeA8W8Wave<TemplateMegaMoeA16W16WaveTypeFunc, MegaMoeA16W16Wave<TemplateMegaMoeA16W16WaveTypeFunc>> {
private:
    using Self = MegaMoeA16W16Wave<TemplateMegaMoeA16W16WaveTypeFunc>;
    using WaveBase = MegaMoeA8W8Wave<TemplateMegaMoeA16W16WaveTypeFunc, Self>;
    using RuntimeBase = MegaMoe<TemplateMegaMoeA16W16WaveTypeFunc>;
    friend WaveBase;
    friend RuntimeBase;
    Bf16GmmBlockContext moeBlockContext_{};

protected:
    static constexpr uint16_t SHARED_ACTIVATION_SUB_BLOCK_IDX = 0U;

    __aicore__ inline void InitPathConfigs(const MegaMoeTilingData* tilingData)
    {
        const auto& compute = tilingData->compute;
        auto& inputConfig = this->quantProcessConfig_;
        inputConfig.layout = CalcTokenScaleLayout(compute.h, compute.dispatchInputStorage, compute.topK,
                                                  tilingData->comm.topkWeightsPrefetch);
        inputConfig.stageBytes = static_cast<uint32_t>(
            CalcBf16InputStageBytes(inputConfig.layout, compute.topK, Std::IsSame<TopkWeightsType, bfloat16_t>::value));
        const uint32_t sendCoreIdx = this->sharedExpertNum_ > 0U ? this->blockIdx_ : this->aivCoreIdx_;
        this->sendMaskConfig_ = CreateSendMaskConfig(this->params_, sendCoreIdx);
        this->tokenDispatchConfig_ = {
            .maxOutputSize = static_cast<uint64_t>(compute.maxOutputSize),
            .bufferConfig = tilingData->dispatchBufferConfig,
            .routeIndexAlignSize = static_cast<uint64_t>(CalcDispatchRouteIndexAlignSize(compute)),
            .quantWinOffset = static_cast<uint64_t>(this->params_.communication.layout.quantTokenScale.offsetBytes),
            .quantTokenAlignBytes = static_cast<uint32_t>(inputConfig.layout.dataBytes),
            .quantScaleAlignBytes = 0U,
            .quantTokenScaleAlignBytes = static_cast<uint32_t>(inputConfig.layout.tokenScaleBytes),
            .dedup = CreateDedupConfig(this->params_)};
        const int32_t flagSlots =
            static_cast<int32_t>(Ops::Base::CeilDiv(static_cast<int64_t>(compute.maxOutputSize),
                                                    static_cast<int64_t>(RuntimeBase::GMM1_TILE_M))) *
            INT_CACHELINE;
        this->InitSyncWorkspaceConfigs(flagSlots, flagSlots);
        this->InitGmmConfigs();
        this->InitQuantTokenBufferConfig();
        this->InitTokenUnpermuteConfig();
        this->epilogueOp_.Init(
            {.yGmAddr = this->params_.workspace.layout.routed.gmm2Input.data.get_ptr(this->params_.workspace.base),
             .clampLimit = tilingData->clampLimit,
             .actMode = tilingData->actMode,
             .actSubMode = tilingData->actSubMode,
             .activationAlpha = tilingData->activationAlpha,
             .activationBeta = tilingData->activationBeta,
             .topkWeightsPrefetch = tilingData->isTopkWeightsPrefetch});
    }

    __aicore__ inline uint32_t InitInputScratchTensors(uint32_t scratchAddr)
    {
        const uint32_t stageBytes = this->quantProcessConfig_.stageBytes;
        for (uint32_t slot = 0U; slot < DOUBLE_BUFFER; ++slot) {
            this->quantScratch_.stages[slot] = reinterpret_cast<__ubuf__ uint8_t*>(scratchAddr + slot * stageBytes);
        }
        // 有效 token 和权重每次覆盖；记录中的对齐填充在本次 launch 保持为零。
        FillUbVF<int16_t>(reinterpret_cast<__ubuf__ int16_t*>(scratchAddr), int16_t{0},
                          DOUBLE_BUFFER * stageBytes / sizeof(int16_t));
        PipeFence<PIPE_V, PIPE_MTE2, SYNC_EVENT_ID2>();
        return scratchAddr + DOUBLE_BUFFER * stageBytes;
    }

    __aicore__ inline void RunGmm1ActivationSlice(const ExpertTokenPosition& gmm1Position,
                                                  ExpertLoopState& gmm1ExpertState, GMMAddrInfo& gmm1AddrInfo,
                                                  const ProblemShape& gmm1WaveProblemShape,
                                                  GmmRuntimeState& runtimeState, uint32_t waveEndTokenIndexInExpert,
                                                  uint64_t expertRowCount)
    {
        const auto& workspace = this->params_.workspace.layout;
        const auto base = this->params_.workspace.base;
        const auto& ring = this->gmmExecutionConfig_.ring;
        const uint64_t logicalRow =
            static_cast<uint64_t>(gmm1ExpertState.globalTokenStartIndex) + gmm1Position.tokenIndexInExpert;
        const uint64_t paddedRow = gmm1ExpertState.paddedTokenStartIndex + gmm1Position.tokenIndexInExpert;
        const uint64_t dispatchRow = ring.dispatch ? CalcMteRingPhysicalRow(paddedRow, ring.rows) : logicalRow;
        const uint64_t activationRow = ring.activation ? CalcMteRingPhysicalRow(paddedRow, ring.rows) : logicalRow;
        const uint64_t n = this->commonConfig_.gmm1OutputDim, k = this->commonConfig_.tokenHiddenDim;
        const uint64_t group = gmm1Position.tokenIndexInExpert / MegaMoeBf16GmmConfig::TILE_M;
        gmm1AddrInfo.aGlobal = workspace.routed.gmm1Input.data.get_ptr(base) + dispatchRow * k * sizeof(bfloat16_t);
        gmm1AddrInfo.bGlobal = GetExpertWeightAddr<bfloat16_t>(
            this->moeWeightTensorListAddrs_.weight1, this->gmmExecutionConfig_.isPerExpertWeightTensor,
            gmm1Position.expertIdx, static_cast<uint64_t>(gmm1Position.expertIdx) * n * k);
        gmm1AddrInfo.aScaleGlobal = nullptr;
        gmm1AddrInfo.bScaleGlobal = nullptr;
        gmm1AddrInfo.metaInfoGlobal =
            workspace.routing.receivedTokenMetadata.get_ptr(base) + logicalRow * META_INFO_SIZE * sizeof(int32_t);
        gmm1AddrInfo.dispatchToGmm1Flag =
            workspace.sync.dispatchReady.template get_ptr<int32_t>(base) +
            static_cast<uint64_t>(gmm1Position.expertIdx) * this->syncWorkspaceLayout_.dispatchFlagSlotCountPerExpert +
            group * INT_CACHELINE;
        gmm1AddrInfo.activationToGmm2Flag = workspace.sync.activationReady.template get_ptr<int32_t>(base) +
                                            static_cast<uint64_t>(gmm1Position.expertIdx) *
                                                this->syncWorkspaceLayout_.activationFlagSlotCountPerExpert +
                                            group * INT_CACHELINE;
        this->epilogueOp_.SetOutputRowOffset(activationRow);
        Bf16Gmm::RunGmm1<MoeWeight1Format != FORMAT_ND, false>(
            moeBlockContext_, this->epilogueOp_, gmm1WaveProblemShape, gmm1AddrInfo, this->gmmExecutionConfig_.blockJob,
            runtimeState, this->commonConfig_.activationOutputDivisor);
        (void)waveEndTokenIndexInExpert;
        (void)expertRowCount;
    }

    __aicore__ inline void RunGmm2Slice(const ExpertTokenPosition& gmm2Position, uint32_t waveEndTokenIndexInExpert,
                                        const ExpertLoopState& gmm2ExpertState, GMMAddrInfo& gmm2AddrInfo,
                                        uint32_t& startBlockIdx)
    {
        const auto& workspace = this->params_.workspace.layout;
        const auto base = this->params_.workspace.base;
        const auto& ring = this->gmmExecutionConfig_.ring;
        const uint64_t logicalRow =
            static_cast<uint64_t>(gmm2ExpertState.globalTokenStartIndex) + gmm2Position.tokenIndexInExpert;
        const uint64_t paddedRow = gmm2ExpertState.paddedTokenStartIndex + gmm2Position.tokenIndexInExpert;
        const uint64_t activationRow = ring.activation ? CalcMteRingPhysicalRow(paddedRow, ring.rows) : logicalRow;
        const uint64_t resultRow = ring.gmm2 ? CalcMteRingPhysicalRow(paddedRow, ring.rows) : logicalRow;
        const uint64_t n = this->commonConfig_.tokenHiddenDim;
        const uint64_t k = this->commonConfig_.gmm1OutputDim / this->commonConfig_.activationOutputDivisor;
        const uint64_t group = gmm2Position.tokenIndexInExpert / MegaMoeBf16GmmConfig::TILE_M;
        gmm2AddrInfo.aGlobal = workspace.routed.gmm2Input.data.get_ptr(base) + activationRow * k * sizeof(bfloat16_t);
        gmm2AddrInfo.bGlobal = GetExpertWeightAddr<bfloat16_t>(
            this->moeWeightTensorListAddrs_.weight2, this->gmmExecutionConfig_.isPerExpertWeightTensor,
            gmm2Position.expertIdx, static_cast<uint64_t>(gmm2Position.expertIdx) * n * k);
        gmm2AddrInfo.aScaleGlobal = nullptr;
        gmm2AddrInfo.bScaleGlobal = nullptr;
        gmm2AddrInfo.gmm2OutGlobal = workspace.routed.gmm2Output.get_ptr(base) + resultRow * n * sizeof(bfloat16_t);
        gmm2AddrInfo.metaInfoGlobal =
            workspace.routing.receivedTokenMetadata.get_ptr(base) + logicalRow * META_INFO_SIZE * sizeof(int32_t);
        gmm2AddrInfo.activationToGmm2Flag = workspace.sync.activationReady.template get_ptr<int32_t>(base) +
                                            static_cast<uint64_t>(gmm2Position.expertIdx) *
                                                this->syncWorkspaceLayout_.activationFlagSlotCountPerExpert +
                                            group * INT_CACHELINE;
        ProblemShape waveShape = gmm2ExpertState.problemShape;
        Get<M_VALUE>(waveShape) = waveEndTokenIndexInExpert - gmm2Position.tokenIndexInExpert;
        if (IsCombineDedupOn(this->params_.tilingData->compute.dedupMode)) {
            Bf16Gmm::RunGmm2<MoeWeight2Format != FORMAT_ND, false, false>(
                moeBlockContext_, waveShape, gmm2AddrInfo, this->params_, this->gmmExecutionConfig_.blockJob,
                startBlockIdx, this->commonConfig_.activationOutputDivisor);
        } else {
            Bf16Gmm::RunGmm2<MoeWeight2Format != FORMAT_ND, false, true>(
                moeBlockContext_, waveShape, gmm2AddrInfo, this->params_, this->gmmExecutionConfig_.blockJob,
                startBlockIdx, this->commonConfig_.activationOutputDivisor);
        }
    }

    __aicore__ inline void ConfigureSharedGmm1Input(GMMAddrInfo& gmmAddrInfo, GmmExecutionConfig&,
                                                    Gmm1ActivationSync&) const
    {
        gmmAddrInfo.aGlobal = this->params_.aGmAddr;
        gmmAddrInfo.aScaleGlobal = nullptr;
    }

    __aicore__ inline void RunSharedExpertGmm1Activation(GMMAddrInfo& gmmAddrInfo, const ProblemShape& problemShape,
                                                         const GmmExecutionConfig& sharedGmmConfig,
                                                         GmmRuntimeState& runtimeState, uint32_t sharedExpertIdx)
    {
        const auto& workspace = this->params_.workspace.layout;
        const auto base = this->params_.workspace.base;
        const uint64_t rows = this->commonConfig_.tokenNum, n = this->commonConfig_.gmm1OutputDim;
        const uint64_t k = this->commonConfig_.tokenHiddenDim;
        gmmAddrInfo.bGlobal = GetExpertWeightAddr<bfloat16_t>(this->sharedWeightTensorListAddrs_.weight1,
                                                              sharedGmmConfig.isPerExpertWeightTensor, sharedExpertIdx,
                                                              sharedExpertIdx * n * k);
        gmmAddrInfo.bScaleGlobal = nullptr;
        gmmAddrInfo.dispatchToGmm1Flag = nullptr;
        gmmAddrInfo.metaInfoGlobal = nullptr;
        gmmAddrInfo.activationToGmm2Flag =
            workspace.mte.sync.sharedActivationReady.template get_ptr<int32_t>(base) +
            static_cast<uint64_t>(sharedExpertIdx) * CalcSharedActivationFlagElementsPerExpert(rows);
        this->sharedEpilogueOp_.SetOutputRowOffset(sharedExpertIdx * rows);
        Bf16Gmm::RunGmm1<SharedWeight1Format != FORMAT_ND, true>(
            this->sharedBlockContext_, this->sharedEpilogueOp_, problemShape, gmmAddrInfo, sharedGmmConfig.blockJob,
            runtimeState, this->commonConfig_.activationOutputDivisor);
    }

    __aicore__ inline void RunSharedExpertGmm2(GMMAddrInfo& gmmAddrInfo, const ProblemShape& problemShape,
                                               uint32_t sharedExpertIdx)
    {
        const auto& workspace = this->params_.workspace.layout;
        const auto base = this->params_.workspace.base;
        const uint64_t rows = this->commonConfig_.tokenNum, n = this->commonConfig_.tokenHiddenDim;
        const uint64_t k = this->commonConfig_.gmm1OutputDim / this->commonConfig_.activationOutputDivisor;
        const uint64_t firstRow = sharedExpertIdx * rows;
        gmmAddrInfo.aGlobal = workspace.shared.gmm2Input.data.get_ptr(base) + firstRow * k * sizeof(bfloat16_t);
        gmmAddrInfo.bGlobal = GetExpertWeightAddr<bfloat16_t>(this->sharedWeightTensorListAddrs_.weight2,
                                                              this->gmmExecutionConfig_.isPerExpertWeightTensor,
                                                              sharedExpertIdx, sharedExpertIdx * n * k);
        gmmAddrInfo.aScaleGlobal = nullptr;
        gmmAddrInfo.bScaleGlobal = nullptr;
        gmmAddrInfo.gmm2OutGlobal = workspace.shared.gmm2Output.get_ptr(base) + firstRow * n * sizeof(bfloat16_t);
        gmmAddrInfo.activationToGmm2Flag =
            workspace.mte.sync.sharedActivationReady.template get_ptr<int32_t>(base) +
            static_cast<uint64_t>(sharedExpertIdx) * CalcSharedActivationFlagElementsPerExpert(rows);
        gmmAddrInfo.sharedExpertGmm2TileCounter = workspace.mte.sync.sharedGmm2Tiles.template get_ptr<int32_t>(base) +
                                                  static_cast<uint64_t>(sharedExpertIdx) *
                                                      Ops::Base::CeilDiv(rows, uint64_t{MegaMoeBf16GmmConfig::TILE_M}) *
                                                      INT_CACHELINE;
        Bf16Gmm::RunGmm2<SharedWeight2Format != FORMAT_ND, true, false>(
            this->sharedBlockContext_, problemShape, gmmAddrInfo, this->params_, this->gmmExecutionConfig_.blockJob,
            this->startBlockIdx_, this->commonConfig_.activationOutputDivisor);
    }

    __aicore__ inline SharedGmmStartCondition GetSharedGmmStartCondition() const
    {
        return this->sharedExpertNum_ == 0U ? SharedGmmStartCondition::NoSharedExpert :
                                              SharedGmmStartCondition::InputDirectly;
    }
};

#undef TemplateMegaMoeA16W16WaveTypeClass
#undef TemplateMegaMoeA16W16WaveTypeFunc
} // namespace MegaMoeImpl
#endif
#endif
