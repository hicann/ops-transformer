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

#define TemplateMegaMoeA16W16WaveTypeClass \
    typename TopkWeightsType, int32_t MoeWeight1Format, int32_t MoeWeight2Format, int32_t SharedWeight1Format, \
        int32_t SharedWeight2Format
#define TemplateMegaMoeA16W16WaveTypeFunc \
    TopkWeightsType, MoeWeight1Format, MoeWeight2Format, SharedWeight1Format, SharedWeight2Format

template <TemplateMegaMoeA16W16WaveTypeClass>
class MegaMoeA16W16Wave
    : public MegaMoeA8W8Wave<bfloat16_t, bfloat16_t, TopkWeightsType, bfloat16_t, BF16_PASSTHROUGH, bfloat16_t,
                             BF16_PASSTHROUGH, MoeWeight1Format, MoeWeight2Format, SharedWeight1Format,
                             SharedWeight2Format, MegaMoeA16W16Wave<TemplateMegaMoeA16W16WaveTypeFunc>> {
private:
    using Self = MegaMoeA16W16Wave<TemplateMegaMoeA16W16WaveTypeFunc>;
    using WaveBase = MegaMoeA8W8Wave<bfloat16_t, bfloat16_t, TopkWeightsType, bfloat16_t, BF16_PASSTHROUGH, bfloat16_t,
                                     BF16_PASSTHROUGH, MoeWeight1Format, MoeWeight2Format, SharedWeight1Format,
                                     SharedWeight2Format, Self>;
    using RuntimeBase =
        MegaMoe<bfloat16_t, bfloat16_t, TopkWeightsType, bfloat16_t, BF16_PASSTHROUGH, bfloat16_t, BF16_PASSTHROUGH,
                MoeWeight1Format, MoeWeight2Format, SharedWeight1Format, SharedWeight2Format>;
    friend WaveBase;
    friend RuntimeBase;
    Bf16GmmBlockContext moeBlockContext_{};

protected:
    __aicore__ inline void InitStageConfigs(const MegaMoeTilingData* tilingData)
    {
        const auto& compute = tilingData->compute;
        auto& inputConfig = this->quantProcessConfig_;
        const auto layout = CalcTokenScaleLayout(compute.h, compute.dispatchInputStorage, compute.topK,
                                                 tilingData->comm.topkWeightsPrefetch);
        inputConfig = {.dataValidBytes = static_cast<uint32_t>(layout.dataValidBytes),
                       .dataBytes = static_cast<uint32_t>(layout.dataBytes),
                       .tokenScaleBytes = static_cast<uint32_t>(layout.tokenScaleBytes),
                       .stageBytes = static_cast<uint32_t>(CalcBf16InputStageBytes(
                           layout, compute.topK, Std::IsSame<TopkWeightsType, bfloat16_t>::value))};
        this->tokenDispatchConfig_ = {
            .maxOutputSize = static_cast<uint64_t>(compute.maxOutputSize),
            .bufferConfig = tilingData->dispatchBufferConfig,
            .routeIndexAlignSize = static_cast<uint64_t>(CalcDispatchRouteIndexAlignSize(compute)),
            .quantWinOffset = static_cast<uint64_t>(this->params_.communication.layout.quantTokenScale.offsetBytes),
            .quantTokenAlignBytes = inputConfig.dataBytes,
            .quantScaleAlignBytes = 0U,
            .quantTokenScaleAlignBytes = inputConfig.tokenScaleBytes,
            .dedup = CreateDedupConfig(this->params_)};
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
        this->quantScratch_.stage0 = reinterpret_cast<__ubuf__ uint8_t*>(scratchAddr);
        this->quantScratch_.stage1 = reinterpret_cast<__ubuf__ uint8_t*>(scratchAddr + stageBytes);
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
        UpdateMoeExpertGmm1GlobalBuffer<bfloat16_t, bfloat16_t, bfloat16_t, void, 1U, false>(
            this->params_.tilingData->isTopkWeightsPrefetch, this->gmmExecutionConfig_, this->syncWorkspaceLayout_,
            this->params_.workspace.layout, this->params_.workspace.base, this->moeWeightTensorListAddrs_,
            this->epilogueOp_, gmm1AddrInfo, gmm1ExpertState,
            this->commonConfig_.gmm1OutputDim / this->commonConfig_.activationOutputDivisor,
            gmm1Position.tokenIndexInExpert);
        Bf16Gmm::RunGmm1<MoeWeight1Format != FORMAT_ND, false>(
            moeBlockContext_, this->epilogueOp_, gmm1WaveProblemShape, gmm1AddrInfo, this->gmmExecutionConfig_.blockJob,
            runtimeState, this->commonConfig_.activationOutputDivisor, this->params_.tilingData->isTopkWeightsPrefetch);
        (void)waveEndTokenIndexInExpert;
        (void)expertRowCount;
    }

    __aicore__ inline void RunGmm2Slice(const ExpertTokenPosition& gmm2Position, uint32_t waveEndTokenIndexInExpert,
                                        const ExpertLoopState& gmm2ExpertState, GMMAddrInfo& gmm2AddrInfo,
                                        uint32_t& startBlockIdx)
    {
        UpdateMoeExpertGmm2GlobalBuffer<bfloat16_t, bfloat16_t, void>(
            this->gmmExecutionConfig_, this->syncWorkspaceLayout_, this->params_.workspace.layout,
            this->params_.workspace.base, this->moeWeightTensorListAddrs_, gmm2AddrInfo, gmm2ExpertState,
            this->commonConfig_.gmm1OutputDim / this->commonConfig_.activationOutputDivisor,
            gmm2Position.tokenIndexInExpert);
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

    __aicore__ inline void RunSharedExpertGmm1Activation(GMMAddrInfo& gmmAddrInfo, const ProblemShape& problemShape,
                                                         const GmmExecutionConfig& sharedGmmConfig,
                                                         GmmRuntimeState& runtimeState, uint32_t sharedExpertIdx)
    {
        UpdateSharedExpertGmm1GlobalBuffer<bfloat16_t, bfloat16_t, void, false>(
            this->commonConfig_, sharedGmmConfig, this->params_.workspace.layout, this->params_.workspace.base,
            this->sharedWeightTensorListAddrs_, this->sharedEpilogueOp_, gmmAddrInfo, sharedExpertIdx);
        Bf16Gmm::RunGmm1<SharedWeight1Format != FORMAT_ND, true>(
            this->sharedBlockContext_, this->sharedEpilogueOp_, problemShape, gmmAddrInfo, sharedGmmConfig.blockJob,
            runtimeState, this->commonConfig_.activationOutputDivisor, false);
    }

    __aicore__ inline void RunSharedExpertGmm2(GMMAddrInfo& gmmAddrInfo, const ProblemShape& problemShape,
                                               uint32_t sharedExpertIdx)
    {
        UpdateSharedExpertGmm2GlobalBuffer<bfloat16_t, bfloat16_t, void, MegaMoeBf16GmmConfig::TILE_M>(
            this->commonConfig_, this->gmmExecutionConfig_, this->params_.workspace.layout,
            this->params_.workspace.base, this->sharedWeightTensorListAddrs_, gmmAddrInfo, sharedExpertIdx);
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
