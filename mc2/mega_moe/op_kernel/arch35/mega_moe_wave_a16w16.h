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
 * \brief MegaMoe A16W16 类型与派生流程框架
 */

#ifndef MEGA_MOE_WAVE_A16W16_H
#define MEGA_MOE_WAVE_A16W16_H

#ifndef MEGA_MOE_ARCH92_DEVICE
#include "mega_moe_wave_a8w8.h"

namespace MegaMoeImpl {

// TODO 只允许参与类型和接口检查；任何实际调用都必须先补齐实现。
template <typename Impl>
__aicore__ inline void A16W16ImplementationTodo()
{
    static_assert(!Std::IsSame<Impl, Impl>::value, "MegaMoe A16W16 TODO implementation is not ready.");
}

struct Bf16InputConfig {
    // TODO-F1：描述 BF16 token 与可选 FP32 topK 路由权重的 TokenScale 字节布局。
};

struct Bf16InputScratch {
    // TODO-F2：补齐 BF16 双缓冲及路由权重转换视图，不分配 MX 量化临时区。
};

struct Bf16GmmBlockContext {
    // TODO-F3：补齐无 scale 的 BF16 GMM 上下文，实际分块与 Host 预算对齐。
};

template <uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch>
struct Bf16ActivationEpilogue {
    // TODO-F4：补齐激活计算、BF16 GM 输出及可选路由权重处理接口。
};

template <>
struct InputConfigSelector<bfloat16_t> {
    using Type = Bf16InputConfig;
};

template <>
struct InputScratchSelector<bfloat16_t> {
    using Type = Bf16InputScratch;
};

template <uint32_t TileM, uint32_t TileN, bool Prefetch>
struct ActivationEpilogueSelector<bfloat16_t, bfloat16_t, TileM, TileN, Prefetch> {
    using Type = Bf16ActivationEpilogue<TileM, TileN, Prefetch>;
};

template <>
struct GmmBlockContextSelector<bfloat16_t, BF16_PASSTHROUGH> {
    using Type = Bf16GmmBlockContext;
};

#define TemplateMegaMoeA16W16WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename MoeWeightType, int32_t MoeQuantMode, \
        typename SharedWeightType, int32_t SharedQuantMode, int32_t MoeWeight1Format, int32_t MoeWeight2Format, \
        int32_t SharedWeight1Format, int32_t SharedWeight2Format, int32_t CombineQuantMode, bool TopkWeightsPrefetch, \
        typename TopkIndexType
#define TemplateMegaMoeA16W16WaveTypeFunc \
    XType, OutputType, TopkWeightsType, MoeWeightType, MoeQuantMode, SharedWeightType, SharedQuantMode, \
        MoeWeight1Format, MoeWeight2Format, SharedWeight1Format, SharedWeight2Format, CombineQuantMode, \
        TopkWeightsPrefetch, TopkIndexType

template <TemplateMegaMoeA16W16WaveTypeClass>
class MegaMoeA16W16Wave
    : public MegaMoeA8W8Wave<TemplateMegaMoeA16W16WaveTypeFunc,
                             MegaMoeA16W16Wave<TemplateMegaMoeA16W16WaveTypeFunc>> {
private:
    using Self = MegaMoeA16W16Wave<TemplateMegaMoeA16W16WaveTypeFunc>;
    using WaveBase = MegaMoeA8W8Wave<TemplateMegaMoeA16W16WaveTypeFunc, Self>;
    using RuntimeBase = MegaMoe<TemplateMegaMoeA16W16WaveTypeFunc>;
    friend WaveBase;
    friend RuntimeBase;

    static_assert(Std::IsSame<XType, bfloat16_t>::value && Std::IsSame<OutputType, bfloat16_t>::value &&
                      Std::IsSame<MoeWeightType, bfloat16_t>::value &&
                      Std::IsSame<SharedWeightType, bfloat16_t>::value,
                  "MegaMoe A16W16 requires BF16 input, output and weights.");
    static_assert(MoeQuantMode == BF16_PASSTHROUGH && SharedQuantMode == BF16_PASSTHROUGH,
                  "MegaMoe A16W16 requires BF16 expert input types.");
    static_assert(CombineQuantMode == COMBINE_NO_QUANT, "MegaMoe A16W16 requires unquantized Combine.");

public:
    // TODO 和同步契约完成前，继承的 Process 入口保持编译期禁用。
    static constexpr bool kImplementationReady = false;

protected:
    __aicore__ inline void InitPathConfigs(const MegaMoeTilingData* tilingData)
    {
        // TODO-F5：配置 BF16 TokenScale、Dispatch、GM 临时区及公共阶段描述。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline uint32_t InitInputScratchTensors(uint32_t scratchAddr)
    {
        // TODO-F6：按字节预算布置 BF16 暂存区，返回后续公共暂存区的起始地址。
        A16W16ImplementationTodo<Self>();
        return 0U;
    }

    __aicore__ inline void PrepareInputTokenRange(const WorkRange& tokenRange)
    {
        // TODO-F7：直接打包 BF16 token 和可选 FP32 路由权重，不生成 scale。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void DispatchBuffInit()
    {
        // TODO-F8：建立 BF16 Dispatch 的字节地址视图和接收记录描述。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void DispatchTokenRangeForWave(const ExpertTokenRange& dispatchRange)
    {
        // TODO-F8：复用 Wave 范围规划，填充普通和去重 BF16 搬运与 ready 发布。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void InitMoeGmmTileCounts()
    {
        // TODO-F9：按实际 BF16 GMM 分块初始化任务数，保持 Wave 规划口径一致。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void InitMoeGmmSync(
        GMMAddrInfo& gmm1AddrInfo, GMMAddrInfo& gmm2AddrInfo, Gmm1ActivationSync& gmm1ActivationSync,
        Gmm2CombineSync& gmm2CombineSync)
    {
        // TODO-F9/F10：关联 BF16 GM 激活交接及 GMM2/Combine 的同步对象。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void RunGmm1ActivationSlice(
        const ExpertTokenPosition& gmm1Position, ExpertLoopState& gmm1ExpertState, GMMAddrInfo& gmm1AddrInfo,
        const ProblemShape& gmm1WaveProblemShape, GmmRuntimeState& runtimeState,
        uint32_t waveEndTokenIndexInExpert, uint64_t expertRowCount)
    {
        // TODO-F9：绑定无 scale 地址，执行 BF16 GMM1、GM 落地和无量化激活。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void RunGmm2Slice(
        const ExpertTokenPosition& gmm2Position, uint32_t waveEndTokenIndexInExpert,
        const ExpertLoopState& gmm2ExpertState, GMMAddrInfo& gmm2AddrInfo, uint32_t& startBlockIdx)
    {
        // TODO-F10：绑定 BF16 激活和权重，执行 GMM2 并发布结果。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void InitSharedGmm1Epilogue()
    {
        // TODO-F11/F12：建立共享专家无 scale 的激活配置。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void ConfigureSharedGmm1Input(
        GMMAddrInfo& gmmAddrInfo, GmmExecutionConfig& gmmConfig,
        Gmm1ActivationSync& sharedGmm1ActivationSync) const
    {
        // TODO-F11：确定共享 BF16 输入来源、行跨度和数据就绪依赖。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void RunSharedExpertGmm1Activation(
        GMMAddrInfo& gmmAddrInfo, const ProblemShape& problemShape, const GmmExecutionConfig& sharedGmmConfig,
        GmmRuntimeState& runtimeState, uint32_t sharedExpertIdx)
    {
        // TODO-F12：绑定当前共享专家地址，执行 BF16 GMM1 和无量化激活。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void RunSharedExpertGmm2(
        GMMAddrInfo& gmmAddrInfo, const ProblemShape& problemShape, uint32_t sharedExpertIdx)
    {
        // TODO-F13：绑定当前共享专家地址，执行 BF16 GMM2。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void FinishMoeActivationSync(int32_t vecSetSyncCom)
    {
        // TODO-F14：按 BF16 GM 交接协议排空 MoE 激活同步。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void FinishSharedActivationSync(GmmRuntimeState& runtimeState)
    {
        // TODO-F15：按 BF16 GM 交接协议排空共享激活同步。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline SharedGmmStartCondition GetSharedGmmStartCondition() const
    {
        // TODO-F5/F11：根据共享 BF16 输入来源选择启动条件。
        A16W16ImplementationTodo<Self>();
        return SharedGmmStartCondition::NoSharedExpert;
    }

    __aicore__ inline void WaitForInputPreparation(SharedGmmStartCondition sharedGmmStartCondition)
    {
        // TODO-F5/F11：按 BF16 输入就绪和 GM 输出需求等待，不沿用 MX UB 输出假设。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void SyncBeforeSharedExpertGmm1()
    {
        // TODO-F11/F12：完成共享 GMM1 所需的输入和暂存区交接。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline void SyncBeforeMoeExpertStages()
    {
        // TODO-F9：完成共享计算到 MoE 计算之间的 BF16 暂存区交接。
        A16W16ImplementationTodo<Self>();
    }

    __aicore__ inline uint16_t GetMoeActivationSubBlockIdx() const
    {
        // TODO-F9/F14：按 BF16 GM 激活实现确定同步角色。
        A16W16ImplementationTodo<Self>();
        return 0U;
    }

    __aicore__ inline uint16_t GetSharedActivationSubBlockIdx() const
    {
        // TODO-F12/F15：按共享 BF16 GM 激活实现确定同步角色。
        A16W16ImplementationTodo<Self>();
        return 0U;
    }
};

#undef TemplateMegaMoeA16W16WaveTypeClass
#undef TemplateMegaMoeA16W16WaveTypeFunc
} // namespace MegaMoeImpl
#endif
#endif
