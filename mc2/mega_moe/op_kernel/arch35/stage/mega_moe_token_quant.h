/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_TOKEN_QUANT_H
#define MEGA_MOE_TOKEN_QUANT_H

#include "../common/mega_moe_utils.h"
#if __has_include("../../../common/quantize_functions.h")
#include "../../../common/quantize_functions.h"
#else
#include "../../../../common/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

using namespace AscendC;

struct QuantProcessConfig {
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    uint32_t quantScaleValidCountPerToken;
};

/*
 * 计算量化通信记录的对齐布局（普通与 wave 编排模板共用）。
 * 记录布局 = Align256(token 数据) + Align32(scale) + prefetch 时附加 Align32(topk 权重)，
 * 与 host CalcDispatchBufferConfig 的 copyBufferBytes 契约恒相等。
 */
template <typename ActivationType, typename QuantScaleOutType, bool TopkWeightsPrefetch, uint32_t AElemsPerByte,
          bool DedupWeightsEligible = true>
__aicore__ inline QuantProcessConfig CreateQuantProcessConfig(uint32_t tokenHiddenDim, const Params& params)
{
    uint32_t quantScaleValidCountPerToken = Ops::Base::CeilDiv(tokenHiddenDim, static_cast<uint32_t>(ALIGN_32));
    uint32_t quantTokenAlignBytes =
        Ops::Base::CeilAlign(tokenHiddenDim / AElemsPerByte, static_cast<uint32_t>(ALIGN_256)) * sizeof(ActivationType);
    uint32_t quantScaleAlignBytes =
        Ops::Base::CeilAlign(quantScaleValidCountPerToken * static_cast<uint32_t>(sizeof(QuantScaleOutType)),
                             static_cast<uint32_t>(ALIGN_32));
    uint32_t quantTokenScaleAlignBytes = quantTokenAlignBytes + quantScaleAlignBytes;
    // 布局与 host 同式：prefetch 或 tkw=0 的 combine 去重（发送端乘权）都携带权重段。
    // 权重段只属于 MoE 通信记录：共享专家独立量化的记录不参与 combine，DedupWeightsEligible=false
    // 时与上游同为编译期判定，host 也只按无权重段给它分配（InitializeMteSharedExpertInput）。
    if (TopkWeightsPrefetch || (DedupWeightsEligible && (params.tilingData->topkWeightsPrefetch == 1 ||
                                                         IsCombineDedupOn(params.tilingData->dedupMode)))) {
        uint32_t weightAlignBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(params.tilingData->topK * sizeof(float)),
                                                         static_cast<uint32_t>(ALIGN_32));
        quantTokenScaleAlignBytes += weightAlignBytes;
    }
    return {quantTokenAlignBytes, quantScaleAlignBytes, quantTokenScaleAlignBytes, quantScaleValidCountPerToken};
}

// 记录是否带 topk 权重段：布局由 CreateQuantProcessConfig 唯一决定，消费方按布局判定，不各自重算条件。
__aicore__ inline bool QuantRecordHasWeights(const QuantProcessConfig& config)
{
    return config.quantTokenScaleAlignBytes > config.quantTokenAlignBytes + config.quantScaleAlignBytes;
}

template <typename ActivationType>
struct QuantProcessScratch {
    GlobalTensor<bfloat16_t> inputGm;
    GlobalTensor<uint8_t> outputGm;
    LocalTensor<bfloat16_t> xInTensor0;
    LocalTensor<bfloat16_t> xInTensor1;
    LocalTensor<ActivationType> xOutTensor0;
    LocalTensor<ActivationType> xOutTensor1;
    LocalTensor<uint16_t> mxTempTensor;
};

struct PreQuantScratch {
    LocalTensor<uint8_t> preQuantStage0;
    LocalTensor<uint8_t> preQuantStage1;
};

template <typename TopkWeightsType, typename ActivationType, bool NeedMte2VSync = true>
__aicore__ inline void PrefetchTopkWeights(GM_ADDR tokenTopkWeightsAddr, uint32_t topK,
                                           const QuantProcessConfig& config,
                                           QuantProcessScratch<ActivationType>& scratch,
                                           const LocalTensor<ActivationType>& xOutTensor, TEventID event)
{
    GlobalTensor<TopkWeightsType> weightGm;
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ TopkWeightsType*>(tokenTopkWeightsAddr));
    uint32_t weightOffsetInUb = config.quantTokenAlignBytes + config.quantScaleAlignBytes;
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        // 前两段保存 maxExp 和 halfScale，下一 token 的预取不能覆盖当前 token 的量化中间结果。
        // 权重暂存与输入使用同一双缓冲槽；H<=8192 时前两段最多 1024B，两个权重槽共 128B。
        uint32_t weightTempOffset =
            2U * Ops::Base::CeilAlign(config.quantScaleValidCountPerToken, static_cast<uint32_t>(ALIGN_32)) +
            static_cast<uint32_t>(event) * ALIGN_32;
        LocalTensor<TopkWeightsType> weightBf16Tmp =
            scratch.mxTempTensor[weightTempOffset].template ReinterpretCast<TopkWeightsType>();
        DataCopyPad(weightBf16Tmp, weightGm, {1U, static_cast<uint32_t>(topK * sizeof(TopkWeightsType)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        LocalTensor<float> weightFp32Ub = xOutTensor[weightOffsetInUb].template ReinterpretCast<float>();
        Cast(weightFp32Ub, weightBf16Tmp, AscendC::RoundMode::CAST_NONE, topK);
        PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<TopkWeightsType> weightUb =
            xOutTensor[weightOffsetInUb].template ReinterpretCast<TopkWeightsType>();
        DataCopyPad(weightUb, weightGm, {1U, static_cast<uint32_t>(topK * sizeof(TopkWeightsType)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        if constexpr (NeedMte2VSync) {
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        }
    }
}

// MoE 和共享专家使用相同的参数结构，量化类型在编译期确定。
template <int32_t Mode, typename OutType, typename StorageType>
struct TokenQuantParams {
    static constexpr int32_t QUANT_MODE = Mode;
    using QuantOutType = OutType;
    const QuantProcessConfig& config;
    QuantProcessScratch<StorageType>& scratch;
};

// maxExp 已由公共流程计算；这里只按目标格式计算 scale 和量化数据。
template <int32_t QuantMode, typename QuantOutType, typename ActivationType>
__aicore__ inline void QuantizeTokenInUb(const LocalTensor<bfloat16_t>& input,
                                         const LocalTensor<ActivationType>& xOutTensor,
                                         const LocalTensor<uint16_t>& mxTemp, const QuantProcessConfig& config,
                                         uint32_t hiddenDim)
{
    auto* maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(mxTemp.GetPhyAddr());
    auto* halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t*>(
        mxTemp[Ops::Base::CeilAlign(config.quantScaleValidCountPerToken, static_cast<uint32_t>(ALIGN_32))]
            .GetPhyAddr());
    auto* srcAddr = reinterpret_cast<__ubuf__ bfloat16_t*>(input.GetPhyAddr());
    auto* outDataAddr = reinterpret_cast<__ubuf__ int8_t*>(xOutTensor.GetPhyAddr());
    auto* mxScaleAddr = reinterpret_cast<__ubuf__ uint16_t*>(xOutTensor[config.quantTokenAlignBytes].GetPhyAddr());

    Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, config.quantScaleValidCountPerToken);
    if constexpr (QuantMode == E2M1_QUANT) {
        Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
    } else {
        Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
    }
}

// 装载一个 token 的输入到 xInTensor；记录带权重段（prefetch 或 tkw=0 combine 去重）时随后
// 预取整 token 的 topk 权重进 xOutTensor 尾段（tkw=1 实例编译期常量折叠，零开销）。
template <typename TopkWeightsType, bool TopkWeightsPrefetch, typename ActivationType, typename ScratchType>
__aicore__ inline void LoadTokenInputAndWeights(const MoeStageCommonConfig& common, const QuantProcessConfig& config,
                                                ScratchType& scratch, GM_ADDR topkWeightsAddr, uint32_t tokenIndex,
                                                const LocalTensor<bfloat16_t>& xInTensor,
                                                const LocalTensor<ActivationType>& xOutTensor, TEventID event)
{
    const uint32_t hiddenDim = common.tokenHiddenDim;
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    DataCopyPad(xInTensor, scratch.inputGm[static_cast<uint64_t>(tokenIndex) * hiddenDim],
                {1U, static_cast<uint16_t>(hiddenDim * sizeof(bfloat16_t)), 0U, 0U}, {true, 0, 0, 0});
    if (TopkWeightsPrefetch || QuantRecordHasWeights(config)) {
        GM_ADDR tokenTopkWeightsAddr =
            topkWeightsAddr + static_cast<uint64_t>(tokenIndex) * common.topK * sizeof(TopkWeightsType);
        PrefetchTopkWeights<TopkWeightsType>(tokenTopkWeightsAddr, common.topK, config, scratch, xOutTensor, event);
    } else {
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    }
}

// 仅由 AIV 调用，tokenRange 非空；量化指定范围的本卡 token，按逐 token 交织布局写入数据与 scale。
template <typename TopkWeightsType, bool TopkWeightsPrefetch, typename MoeQuantParams, typename... SharedQuantParams>
__aicore__ inline void QuantizeLocalTokens(const WorkRange& tokenRange, const MoeStageCommonConfig& common,
                                           GM_ADDR topkWeightsAddr, const MoeQuantParams& moeQuant,
                                           const SharedQuantParams&... sharedQuant)
{
    auto& scratch = moeQuant.scratch;
    const auto& config = moeQuant.config;
    uint32_t hiddenDim = common.tokenHiddenDim;
    // 量化 scratch（mxTemp/xOut0/xOut1/xIn0/xIn1）的跨 launch 残留清零由各编排的
    // SendAndQuantBuffInit 在分配处一次性完成（span 清零，与布局同源），见其注释。
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t index = 0; index < tokenRange.count; ++index) {
        bool useFirstBuffer = index % DOUBLE_BUFFER == 0;
        auto event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = useFirstBuffer ? scratch.xInTensor0 : scratch.xInTensor1;
        auto xOutTensor = useFirstBuffer ? scratch.xOutTensor0 : scratch.xOutTensor1;
        uint32_t tokenIndex = tokenRange.start + index;
        LoadTokenInputAndWeights<TopkWeightsType, TopkWeightsPrefetch>(common, config, scratch, topkWeightsAddr,
                                                                       tokenIndex, xInTensor, xOutTensor, event);
        // 同一 token 的输入与分组方式相同，最大指数只统计一次。
        Quant::ComputeMaxExp(reinterpret_cast<__ubuf__ bfloat16_t*>(xInTensor.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ uint16_t*>(scratch.mxTempTensor.GetPhyAddr()), hiddenDim);
        // 每个目标使用相同的量化和搬出流程；空参数包不生成共享量化代码。
        auto processQuantOutput = [&](auto quant) __attribute__((cce_aicore))
        {
            using QuantParams = decltype(quant);
            const auto& outputConfig = quant.config;
            auto outputTensor = useFirstBuffer ? quant.scratch.xOutTensor0 : quant.scratch.xOutTensor1;
            QuantizeTokenInUb<QuantParams::QUANT_MODE, typename QuantParams::QuantOutType>(
                xInTensor, outputTensor, scratch.mxTempTensor, outputConfig, hiddenDim);
            SetFlag<HardEvent::V_MTE3>(event);
            WaitFlag<HardEvent::V_MTE3>(event);
            auto outputBytes = outputTensor.template ReinterpretCast<uint8_t>();
            DataCopyPad(
                quant.scratch.outputGm[static_cast<uint64_t>(tokenIndex) * outputConfig.quantTokenScaleAlignBytes],
                outputBytes, {1U, outputConfig.quantTokenScaleAlignBytes, 0U, 0U, 0U});
        };
        processQuantOutput(moeQuant);
        // sizeof... 统计参数包中的参数个数：不传共享量化参数时为 0，传入一份时为 1。
        // if constexpr 在编译期判断，无共享参数时不生成下面的代码。
        if constexpr (sizeof...(SharedQuantParams) > 0) {
            processQuantOutput(sharedQuant...);
        }
        // 有独立共享量化时，槽的释放排在共享输出写回之后。
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// 聚合预量化搬运的 GM 视图和参数，不负责申请 GM 或 UB 空间。
struct PreQuantizedCopyContext {
    uint32_t dataValidBytes;
    uint32_t scaleValidBytes;
    GlobalTensor<uint8_t> dataSrcGlobalTensor;
    GlobalTensor<uint8_t> scaleSrcGlobalTensor;
    GlobalTensor<uint8_t> tokenScaleDstGlobalTensor;
};

// GM -> UB：每个 token 占一行。对齐时使用 DataCopyParams，非对齐时由 DataCopyPad 补齐到 32B。
__aicore__ inline void PreQuantCopyIn(LocalTensor<uint8_t> dst, GlobalTensor<uint8_t> src, uint32_t tokenCount,
                                      uint32_t validBytes, uint32_t rowBytes)
{
    if (tokenCount == 0U || validBytes == 0U) {
        return;
    }
    constexpr uint32_t blockBytes = static_cast<uint32_t>(ALIGN_32);
    uint32_t alignedBytes = Ops::Base::CeilAlign(validBytes, blockBytes);
    const bool srcAligned = (reinterpret_cast<uint64_t>(src.GetPhyAddr()) & (blockBytes - 1U)) == 0U;
    const bool dstAligned = (reinterpret_cast<uint64_t>(dst.GetPhyAddr()) & (blockBytes - 1U)) == 0U;
    if (validBytes % blockBytes == 0U && srcAligned && dstAligned) {
        /*
         * arch35 的 DataCopyParams 中 blockLen 和两个 gap 均以 32B 为单位。
         * 当 stage 中的各行紧密排列时，将整个 tile 合并为一次连续突发搬运。
         */
        if (rowBytes == validBytes) {
            DataCopyParams params{1U, static_cast<uint16_t>(tokenCount * validBytes / blockBytes), 0U, 0U};
            DataCopy(dst, src, params);
        } else {
            DataCopyParams params{static_cast<uint16_t>(tokenCount), static_cast<uint16_t>(validBytes / blockBytes), 0U,
                                  static_cast<uint16_t>((rowBytes - validBytes) / blockBytes)};
            DataCopy(dst, src, params);
        }
    } else {
        DataCopyExtParams params{static_cast<uint16_t>(tokenCount), validBytes, 0U,
                                 static_cast<uint32_t>((rowBytes - alignedBytes) / blockBytes), 0U};
        /*
         * 下游不会读取目标行的 padding，无需额外填充；
         * 非对齐路径只保留一次带 padding 的 MTE2 搬运。
         */
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
        DataCopyPad(dst, src, params, padParams);
    }
}

// UB -> GM：stage 中已经包含完整记录，因此使用一次 MTE3 写出整个 tile。
__aicore__ inline void PreQuantCopyOutRecords(GlobalTensor<uint8_t> dst, LocalTensor<uint8_t> src, uint32_t tokenCount,
                                              uint32_t recordBytes)
{
    if (tokenCount == 0U || recordBytes == 0U) {
        return;
    }
    // 完整记录按 32B 对齐，UB stage 和 GM 输出记录的起始地址也按 32B 对齐，直接按块搬出整个 tile。
    constexpr uint32_t blockBytes = static_cast<uint32_t>(ALIGN_32);
    DataCopyParams params{static_cast<uint16_t>(tokenCount), static_cast<uint16_t>(recordBytes / blockBytes), 0U, 0U};
    DataCopy(dst, src, params);
}

__aicore__ inline void PreQuantCastBf16Weights(LocalTensor<uint8_t> stage, uint32_t tileCount, uint32_t topK,
                                               uint32_t weightSrcRowBytes, uint32_t weightSrcOffset,
                                               uint32_t recordBytes, uint32_t weightDstOffset)
{
    LocalTensor<bfloat16_t> weightSrc = stage[weightSrcOffset].template ReinterpretCast<bfloat16_t>();
    for (uint32_t row = 0U; row < tileCount; ++row) {
        LocalTensor<float> weightDst = stage[row * recordBytes + weightDstOffset].template ReinterpretCast<float>();
        Cast(weightDst, weightSrc[row * (weightSrcRowBytes / sizeof(bfloat16_t))], AscendC::RoundMode::CAST_NONE, topK);
    }
}

template <typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PreQuantWriteTile(LocalTensor<uint8_t> stage, TEventID event, uint32_t tileCount, uint32_t topK,
                                         uint32_t recordBytes, uint32_t weightSrcOffset, uint32_t weightSrcRowBytes,
                                         uint32_t weightDstOffset, GlobalTensor<uint8_t> output)
{
    if constexpr (TopkWeightsPrefetch && Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        PreQuantCastBf16Weights(stage, tileCount, topK, weightSrcRowBytes, weightSrcOffset, recordBytes,
                                weightDstOffset);
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
    } else {
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
    }
    PreQuantCopyOutRecords(output, stage, tileCount, recordBytes);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
}

// 每批 token、scale 和可选 weight 搬入同一个记录缓冲区，再一次性写出。
template <typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PackPreQuantizedTokenRange(const WorkRange& tokenRange, const MoeStageCommonConfig& common,
                                                  const Params& params, const QuantProcessConfig& config,
                                                  const PreQuantizedCopyContext& copyContext, PreQuantScratch& scratch)
{
    const uint32_t tokensPerCopyBatch = params.tilingData->preQuantTileTokens;
    const uint32_t recordBytes = config.quantTokenScaleAlignBytes;
    const uint32_t weightOffset = config.quantTokenAlignBytes + config.quantScaleAlignBytes;
    const uint32_t weightValidBytes = common.topK * sizeof(TopkWeightsType);
    const uint32_t weightSrcRowBytes = Ops::Base::CeilAlign(weightValidBytes, static_cast<uint32_t>(ALIGN_32));
    const uint32_t weightSrcOffset = tokensPerCopyBatch * recordBytes;

    if (tokenRange.count == 0U) {
        return;
    }
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    for (uint32_t batchIndex = 0U, batchTokenOffset = 0U; batchTokenOffset < tokenRange.count;
         ++batchIndex, batchTokenOffset += tokensPerCopyBatch) {
        uint32_t batchTokenCount = tokensPerCopyBatch < tokenRange.count - batchTokenOffset ?
                                       tokensPerCopyBatch :
                                       tokenRange.count - batchTokenOffset;
        TEventID event = (batchIndex % DOUBLE_BUFFER) == 0U ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<uint8_t> stage = (event == EVENT_ID0) ? scratch.preQuantStage0 : scratch.preQuantStage1;
        uint64_t dataOffset = static_cast<uint64_t>(batchTokenOffset) * copyContext.dataValidBytes;
        uint64_t scaleOffset = static_cast<uint64_t>(batchTokenOffset) * copyContext.scaleValidBytes;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        PreQuantCopyIn(stage, copyContext.dataSrcGlobalTensor[dataOffset], batchTokenCount, copyContext.dataValidBytes,
                       recordBytes);
        PreQuantCopyIn(stage[config.quantTokenAlignBytes], copyContext.scaleSrcGlobalTensor[scaleOffset],
                       batchTokenCount, copyContext.scaleValidBytes, recordBytes);

        if constexpr (TopkWeightsPrefetch) {
            GlobalTensor<uint8_t> weightGm;
            uint64_t weightOffsetInGm =
                static_cast<uint64_t>(tokenRange.start + batchTokenOffset) * common.topK * sizeof(TopkWeightsType);
            weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.probsGmAddr) + weightOffsetInGm);
            if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
                PreQuantCopyIn(stage[weightSrcOffset], weightGm, batchTokenCount, weightValidBytes, weightSrcRowBytes);
                SetFlag<AscendC::HardEvent::MTE2_V>(event);
            } else {
                PreQuantCopyIn(stage[weightOffset], weightGm, batchTokenCount, weightValidBytes, recordBytes);
                SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            }
        } else {
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        }

        if (batchIndex > 0U) {
            TEventID previousEvent = ((batchIndex - 1U) % DOUBLE_BUFFER) == 0U ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<uint8_t> previousStage =
                (previousEvent == EVENT_ID0) ? scratch.preQuantStage0 : scratch.preQuantStage1;
            uint64_t previousOutputOffset =
                static_cast<uint64_t>(batchTokenOffset - tokensPerCopyBatch) * static_cast<uint64_t>(recordBytes);
            PreQuantWriteTile<TopkWeightsType, TopkWeightsPrefetch>(
                previousStage, previousEvent, tokensPerCopyBatch, common.topK, recordBytes, weightSrcOffset,
                weightSrcRowBytes, weightOffset, copyContext.tokenScaleDstGlobalTensor[previousOutputOffset]);
        }
    }

    const uint32_t lastBatchIndex = (tokenRange.count - 1U) / tokensPerCopyBatch;
    const uint32_t lastTokenOffset = lastBatchIndex * tokensPerCopyBatch;
    TEventID lastEvent = (lastBatchIndex % DOUBLE_BUFFER) == 0U ? EVENT_ID0 : EVENT_ID1;
    LocalTensor<uint8_t> lastStage = (lastEvent == EVENT_ID0) ? scratch.preQuantStage0 : scratch.preQuantStage1;
    uint64_t lastOutputOffset = static_cast<uint64_t>(lastTokenOffset) * static_cast<uint64_t>(recordBytes);
    PreQuantWriteTile<TopkWeightsType, TopkWeightsPrefetch>(
        lastStage, lastEvent, tokenRange.count - lastTokenOffset, common.topK, recordBytes, weightSrcOffset,
        weightSrcRowBytes, weightOffset, copyContext.tokenScaleDstGlobalTensor[lastOutputOffset]);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// 仅由 AIV 调用，tokenRange 非空；将已量化的 x 和逐 32 元素 scale 按 dispatch 布局拼接，按需附加 topK 权重。
// 不对 x/scales 做数值转换。
template <typename XType, typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PackPreQuantizedLocalTokens(const WorkRange& tokenRange, const MoeStageCommonConfig& common,
                                                   const Params& params, const QuantProcessConfig& config,
                                                   GM_ADDR outputAddr, PreQuantScratch& scratch)
{
    constexpr uint32_t X_ELEMS_PER_BYTE = PackedElementTraits<XType>::ELEMENTS_PER_BYTE;
    PreQuantizedCopyContext copyContext;
    copyContext.dataValidBytes = common.tokenHiddenDim / X_ELEMS_PER_BYTE;
    copyContext.scaleValidBytes = config.quantScaleValidCountPerToken * static_cast<uint32_t>(sizeof(fp8_e8m0_t));
    uint64_t dataRangeOffset =
        static_cast<uint64_t>(tokenRange.start) * static_cast<uint64_t>(copyContext.dataValidBytes);
    uint64_t scaleRangeOffset =
        static_cast<uint64_t>(tokenRange.start) * static_cast<uint64_t>(copyContext.scaleValidBytes);
    uint64_t tokenScaleRangeOffset =
        static_cast<uint64_t>(tokenRange.start) * static_cast<uint64_t>(config.quantTokenScaleAlignBytes);
    copyContext.dataSrcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.aGmAddr) +
                                                    dataRangeOffset);
    copyContext.scaleSrcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.xScaleGmAddr) +
                                                     scaleRangeOffset);
    copyContext.tokenScaleDstGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(outputAddr) +
                                                          tokenScaleRangeOffset);

    PackPreQuantizedTokenRange<TopkWeightsType, TopkWeightsPrefetch>(tokenRange, common, params, config, copyContext,
                                                                     scratch);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_QUANT_H
