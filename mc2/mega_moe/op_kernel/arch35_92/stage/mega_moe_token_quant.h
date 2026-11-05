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

#include "../common/mega_moe_sync.h"
#include "../common/mega_moe_utils.h"
#include "../common/mega_moe_copy.h"
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
                                                         IsCombineDedupOn(params.tilingData->compute.dedupMode)))) {
        uint32_t weightAlignBytes = Ops::Base::CeilAlign(
            static_cast<uint32_t>(params.tilingData->compute.topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
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
    __gm__ bfloat16_t* inputGm = nullptr;
    __gm__ uint8_t* outputGm = nullptr;
    __ubuf__ bfloat16_t* xInTensor0 = nullptr;
    __ubuf__ bfloat16_t* xInTensor1 = nullptr;
    __ubuf__ ActivationType* xOutTensor0 = nullptr;
    __ubuf__ ActivationType* xOutTensor1 = nullptr;
    __ubuf__ uint16_t* mxTempTensor = nullptr;
};

struct PreQuantScratch {
    __ubuf__ uint8_t* preQuantStage0 = nullptr;
    __ubuf__ uint8_t* preQuantStage1 = nullptr;
};

template <typename TopkWeightsType, typename ActivationType, bool NeedMte2VSync = true>
__aicore__ inline void PrefetchTopkWeights(GM_ADDR tokenTopkWeightsAddr, uint32_t topK,
                                           const QuantProcessConfig& config,
                                           QuantProcessScratch<ActivationType>& scratch,
                                           __ubuf__ ActivationType* xOutTensor, TEventID event)
{
    __gm__ TopkWeightsType* weightGm = reinterpret_cast<__gm__ TopkWeightsType*>(tokenTopkWeightsAddr);
    uint32_t weightOffsetInUb = config.quantTokenAlignBytes + config.quantScaleAlignBytes;
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        // 前两段保存 maxExp 和 halfScale，下一 token 的预取不能覆盖当前 token 的量化中间结果。
        // 权重暂存与输入使用同一双缓冲槽；H<=8192 时前两段最多 1024B，两个权重槽共 128B。
        uint32_t weightTempOffset =
            2U * Ops::Base::CeilAlign(config.quantScaleValidCountPerToken, static_cast<uint32_t>(ALIGN_32)) +
            static_cast<uint32_t>(event) * ALIGN_32;
        __ubuf__ TopkWeightsType* weightBf16Tmp =
            reinterpret_cast<__ubuf__ TopkWeightsType*>(scratch.mxTempTensor + weightTempOffset);
        CopyGmToUb(weightBf16Tmp, weightGm, static_cast<uint32_t>(topK * sizeof(TopkWeightsType)));
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
        __ubuf__ float* weightFp32Ub = reinterpret_cast<__ubuf__ float*>(xOutTensor + weightOffsetInUb);
#if defined(MEGA_MOE_ARCH92_DEVICE)
        {
            LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(weightFp32Ub), topK);
            LocalTensor<bfloat16_t> src(TPosition::VECCALC, reinterpret_cast<uint64_t>(weightBf16Tmp), topK);
            Cast(dst, src, RoundMode::CAST_NONE, topK);
        }
#else
        CastBf16ToFp32VF(weightFp32Ub, weightBf16Tmp, topK);
#endif

    } else {
        __ubuf__ TopkWeightsType* weightUb = reinterpret_cast<__ubuf__ TopkWeightsType*>(xOutTensor + weightOffsetInUb);
        CopyGmToUb(weightUb, weightGm, static_cast<uint32_t>(topK * sizeof(TopkWeightsType)));
        if constexpr (NeedMte2VSync) {
            PipeNotify<PIPE_MTE2, PIPE_V>(event);
            PipeWait<PIPE_MTE2, PIPE_V>(event);
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
__aicore__ inline void QuantizeTokenInUb(__ubuf__ bfloat16_t* input, __ubuf__ ActivationType* xOutTensor,
                                         __ubuf__ uint16_t* mxTemp, const QuantProcessConfig& config,
                                         uint32_t hiddenDim)
{
    auto* halfScaleAddr =
        mxTemp + Ops::Base::CeilAlign(config.quantScaleValidCountPerToken, static_cast<uint32_t>(ALIGN_32));
    auto* outDataAddr = reinterpret_cast<__ubuf__ int8_t*>(xOutTensor);
    auto* mxScaleAddr = reinterpret_cast<__ubuf__ uint16_t*>(xOutTensor + config.quantTokenAlignBytes);

    Quant::ComputeScale<QuantOutType>(mxTemp, mxScaleAddr, halfScaleAddr, config.quantScaleValidCountPerToken);
    if constexpr (QuantMode == E2M1_QUANT) {
        Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            input, halfScaleAddr, outDataAddr, hiddenDim);
    } else {
        Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            input, halfScaleAddr, outDataAddr, hiddenDim);
    }
}

// 装载一个 token 的输入到 xInTensor；记录带权重段（prefetch 或 tkw=0 combine 去重）时随后
// 预取整 token 的 topk 权重进 xOutTensor 尾段（tkw=1 实例编译期常量折叠，零开销）。
template <typename TopkWeightsType, bool TopkWeightsPrefetch, typename ActivationType, typename ScratchType>
__aicore__ inline void LoadTokenInputAndWeights(const MoeStageCommonConfig& common, const QuantProcessConfig& config,
                                                ScratchType& scratch, GM_ADDR topkWeightsAddr, uint32_t tokenIndex,
                                                __ubuf__ bfloat16_t* xInTensor, __ubuf__ ActivationType* xOutTensor,
                                                TEventID event)
{
    const uint32_t hiddenDim = common.tokenHiddenDim;
    PipeWait<PIPE_MTE3, PIPE_MTE2>(event);
    CopyGmToUb(xInTensor, (scratch.inputGm + (static_cast<uint64_t>(tokenIndex) * hiddenDim)),
               static_cast<uint16_t>(hiddenDim * sizeof(bfloat16_t)), GmLoadCacheMode::NORMAL_FIRST_VICTIM, true);
    if (TopkWeightsPrefetch || QuantRecordHasWeights(config)) {
        GM_ADDR tokenTopkWeightsAddr =
            topkWeightsAddr + static_cast<uint64_t>(tokenIndex) * common.topK * sizeof(TopkWeightsType);
        PrefetchTopkWeights<TopkWeightsType>(tokenTopkWeightsAddr, common.topK, config, scratch, xOutTensor, event);
    } else {
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
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
    PipeNotify<PIPE_MTE3, PIPE_MTE2>(EVENT_ID0);
    PipeNotify<PIPE_MTE3, PIPE_MTE2>(EVENT_ID1);
    for (uint32_t index = 0; index < tokenRange.count; ++index) {
        bool useFirstBuffer = index % DOUBLE_BUFFER == 0;
        auto event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = useFirstBuffer ? scratch.xInTensor0 : scratch.xInTensor1;
        auto xOutTensor = useFirstBuffer ? scratch.xOutTensor0 : scratch.xOutTensor1;
        uint32_t tokenIndex = tokenRange.start + index;
        LoadTokenInputAndWeights<TopkWeightsType, TopkWeightsPrefetch>(common, config, scratch, topkWeightsAddr,
                                                                       tokenIndex, xInTensor, xOutTensor, event);
        // 同一 token 的输入与分组方式相同，最大指数只统计一次。
        Quant::ComputeMaxExp(reinterpret_cast<__ubuf__ bfloat16_t*>(xInTensor),
                             reinterpret_cast<__ubuf__ uint16_t*>(scratch.mxTempTensor), hiddenDim);
        // 每个目标使用相同的量化和搬出流程；空参数包不生成共享量化代码。
        auto processQuantOutput = [&](auto quant) __attribute__((cce_aicore))
        {
            using QuantParams = decltype(quant);
            const auto& outputConfig = quant.config;
            auto outputTensor = useFirstBuffer ? quant.scratch.xOutTensor0 : quant.scratch.xOutTensor1;
            QuantizeTokenInUb<QuantParams::QUANT_MODE, typename QuantParams::QuantOutType>(
                xInTensor, outputTensor, scratch.mxTempTensor, outputConfig, hiddenDim);
            PipeNotify<PIPE_V, PIPE_MTE3>(event);
            PipeWait<PIPE_V, PIPE_MTE3>(event);
            auto outputBytes = reinterpret_cast<__ubuf__ uint8_t*>(outputTensor);
            CopyUbToGm(
                (quant.scratch.outputGm + (static_cast<uint64_t>(tokenIndex) * outputConfig.quantTokenScaleAlignBytes)),
                outputBytes, outputConfig.quantTokenScaleAlignBytes);
        };
        processQuantOutput(moeQuant);
        // sizeof... 统计参数包中的参数个数：不传共享量化参数时为 0，传入一份时为 1。
        // if constexpr 在编译期判断，无共享参数时不生成下面的代码。
        if constexpr (sizeof...(SharedQuantParams) > 0) {
            processQuantOutput(sharedQuant...);
        }
        // 有独立共享量化时，槽的释放排在共享输出写回之后。
        PipeNotify<PIPE_MTE3, PIPE_MTE2>(event);
    }
    PipeWait<PIPE_MTE3, PIPE_MTE2>(EVENT_ID0);
    PipeWait<PIPE_MTE3, PIPE_MTE2>(EVENT_ID1);
}

// 聚合预量化搬运的 GM 视图和参数，不负责申请 GM 或 UB 空间。
struct PreQuantizedCopyContext {
    uint32_t dataValidBytes;
    uint32_t scaleValidBytes;
    __gm__ uint8_t* dataGm = nullptr;
    __gm__ uint8_t* scaleGm = nullptr;
    __gm__ uint8_t* recordsGm = nullptr;
};

// GM rows are compact; UB rows include data/scale/weight record space.
__aicore__ inline void PreQuantCopyIn(__ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t tokenCount,
                                      uint32_t validBytes, uint32_t rowBytes)
{
    if (tokenCount == 0U || validBytes == 0U) {
        return;
    }
    if (rowBytes == validBytes && validBytes % ALIGN_32 == 0U) {
        CopyGmToUb(dst, src, tokenCount * validBytes);
    } else {
        CopyGmToUb(dst, src, CopyRows{static_cast<uint16_t>(tokenCount), validBytes, validBytes, rowBytes});
    }
}

// Records already contain the complete wire layout, including padding.
__aicore__ inline void PreQuantCopyOutRecords(__gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint32_t tokenCount,
                                              uint32_t recordBytes)
{
    if (tokenCount == 0U || recordBytes == 0U) {
        return;
    }
    CopyUbToGm(dst, src, CopyRows{static_cast<uint16_t>(tokenCount), recordBytes, recordBytes, recordBytes});
}

#if defined(MEGA_MOE_ARCH92_DEVICE)
__aicore__ inline void PreQuantCastBf16Weights(__ubuf__ uint8_t* stage, uint32_t tileCount, uint32_t topK,
                                               uint32_t weightSrcRowBytes, uint32_t weightSrcOffset,
                                               uint32_t recordBytes, uint32_t weightDstOffset)
{
    __ubuf__ bfloat16_t* weightSrc = reinterpret_cast<__ubuf__ bfloat16_t*>((stage + (weightSrcOffset)));
    for (uint32_t row = 0U; row < tileCount; ++row) {
        __ubuf__ float* weightDst = reinterpret_cast<__ubuf__ float*>((stage + (row * recordBytes + weightDstOffset)));
        {
            LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(weightDst), topK);
            LocalTensor<bfloat16_t> src(
                TPosition::VECCALC,
                reinterpret_cast<uint64_t>((weightSrc + (row * (weightSrcRowBytes / sizeof(bfloat16_t))))), topK);
            Cast(dst, src, RoundMode::CAST_NONE, topK);
        }
    }
}
#else
__simd_vf__ inline void PreQuantCastBf16Weights(__ubuf__ uint8_t* stage, uint32_t tileCount, uint32_t topK,
                                                uint32_t weightSrcRowBytes, uint32_t weightSrcOffset,
                                                uint32_t recordBytes, uint32_t weightDstOffset)
{
    auto* weightSrc = reinterpret_cast<__ubuf__ bfloat16_t*>(stage + weightSrcOffset);
    vector_int32_t indexReg;
    asc_arange(indexReg, 0);
    uint32_t maskCount = topK;
    vector_bool active = asc_update_mask_b32(maskCount);
    vector_bfloat16_t inputReg;
    vector_float outputReg;
    // 源权重位于全部record之后，行间读写不重叠；整个tile只启动一次VF。
    // Host限定TopK<=32，各行复用同一索引和掩码，只读取有效权重。
    for (uint32_t row = 0U; row < tileCount; ++row) {
        auto* weightDst = reinterpret_cast<__ubuf__ float*>(stage + row * recordBytes + weightDstOffset);
        asc_gather(inputReg, weightSrc + row * (weightSrcRowBytes / sizeof(bfloat16_t)),
                   reinterpret_cast<vector_uint32_t&>(indexReg), active);
        asc_bfloat162float(outputReg, inputReg, active, ASC_POSITION_EVEN);
        asc_storealign(weightDst, outputReg, active);
    }
}
#endif

template <typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PreQuantWriteTile(__ubuf__ uint8_t* stage, TEventID event, uint32_t tileCount, uint32_t topK,
                                         uint32_t recordBytes, uint32_t weightSrcOffset, uint32_t weightSrcRowBytes,
                                         uint32_t weightDstOffset, __gm__ uint8_t* output)
{
    if constexpr (TopkWeightsPrefetch && Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        PipeWait<PIPE_MTE2, PIPE_V>(event);
        PreQuantCastBf16Weights(stage, tileCount, topK, weightSrcRowBytes, weightSrcOffset, recordBytes,
                                weightDstOffset);
        PipeNotify<PIPE_V, PIPE_MTE3>(event);
        PipeWait<PIPE_V, PIPE_MTE3>(event);
    } else {
        PipeWait<PIPE_MTE2, PIPE_MTE3>(event);
    }
    PreQuantCopyOutRecords(output, stage, tileCount, recordBytes);
    PipeNotify<PIPE_MTE3, PIPE_MTE2>(event);
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
    PipeNotify<PIPE_MTE3, PIPE_MTE2>(EVENT_ID0);
    PipeNotify<PIPE_MTE3, PIPE_MTE2>(EVENT_ID1);

    for (uint32_t batchIndex = 0U, batchTokenOffset = 0U; batchTokenOffset < tokenRange.count;
         ++batchIndex, batchTokenOffset += tokensPerCopyBatch) {
        uint32_t batchTokenCount = tokensPerCopyBatch < tokenRange.count - batchTokenOffset ?
                                       tokensPerCopyBatch :
                                       tokenRange.count - batchTokenOffset;
        TEventID event = (batchIndex % DOUBLE_BUFFER) == 0U ? EVENT_ID0 : EVENT_ID1;
        __ubuf__ uint8_t* stage = (event == EVENT_ID0) ? scratch.preQuantStage0 : scratch.preQuantStage1;
        uint64_t dataOffset = static_cast<uint64_t>(batchTokenOffset) * copyContext.dataValidBytes;
        uint64_t scaleOffset = static_cast<uint64_t>(batchTokenOffset) * copyContext.scaleValidBytes;

        PipeWait<PIPE_MTE3, PIPE_MTE2>(event);
        PreQuantCopyIn(stage, copyContext.dataGm + dataOffset, batchTokenCount, copyContext.dataValidBytes,
                       recordBytes);
        PreQuantCopyIn((stage + (config.quantTokenAlignBytes)), copyContext.scaleGm + scaleOffset, batchTokenCount,
                       copyContext.scaleValidBytes, recordBytes);

        if constexpr (TopkWeightsPrefetch) {
            __gm__ uint8_t* weightGm = nullptr;
            uint64_t weightOffsetInGm =
                static_cast<uint64_t>(tokenRange.start + batchTokenOffset) * common.topK * sizeof(TopkWeightsType);
            weightGm = reinterpret_cast<__gm__ uint8_t*>(params.probsGmAddr) + weightOffsetInGm;
            if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
                PreQuantCopyIn((stage + (weightSrcOffset)), weightGm, batchTokenCount, weightValidBytes,
                               weightSrcRowBytes);
                PipeNotify<PIPE_MTE2, PIPE_V>(event);
            } else {
                PreQuantCopyIn((stage + (weightOffset)), weightGm, batchTokenCount, weightValidBytes, recordBytes);
                PipeNotify<PIPE_MTE2, PIPE_MTE3>(event);
            }
        } else {
            PipeNotify<PIPE_MTE2, PIPE_MTE3>(event);
        }

        if (batchIndex > 0U) {
            TEventID previousEvent = ((batchIndex - 1U) % DOUBLE_BUFFER) == 0U ? EVENT_ID0 : EVENT_ID1;
            __ubuf__ uint8_t* previousStage =
                (previousEvent == EVENT_ID0) ? scratch.preQuantStage0 : scratch.preQuantStage1;
            uint64_t previousOutputOffset =
                static_cast<uint64_t>(batchTokenOffset - tokensPerCopyBatch) * static_cast<uint64_t>(recordBytes);
            PreQuantWriteTile<TopkWeightsType, TopkWeightsPrefetch>(
                previousStage, previousEvent, tokensPerCopyBatch, common.topK, recordBytes, weightSrcOffset,
                weightSrcRowBytes, weightOffset, copyContext.recordsGm + previousOutputOffset);
        }
    }

    const uint32_t lastBatchIndex = (tokenRange.count - 1U) / tokensPerCopyBatch;
    const uint32_t lastTokenOffset = lastBatchIndex * tokensPerCopyBatch;
    TEventID lastEvent = (lastBatchIndex % DOUBLE_BUFFER) == 0U ? EVENT_ID0 : EVENT_ID1;
    __ubuf__ uint8_t* lastStage = (lastEvent == EVENT_ID0) ? scratch.preQuantStage0 : scratch.preQuantStage1;
    uint64_t lastOutputOffset = static_cast<uint64_t>(lastTokenOffset) * static_cast<uint64_t>(recordBytes);
    PreQuantWriteTile<TopkWeightsType, TopkWeightsPrefetch>(
        lastStage, lastEvent, tokenRange.count - lastTokenOffset, common.topK, recordBytes, weightSrcOffset,
        weightSrcRowBytes, weightOffset, copyContext.recordsGm + lastOutputOffset);
    PipeWait<PIPE_MTE3, PIPE_MTE2>(EVENT_ID0);
    PipeWait<PIPE_MTE3, PIPE_MTE2>(EVENT_ID1);
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
    copyContext.dataGm = reinterpret_cast<__gm__ uint8_t*>(params.aGmAddr) + dataRangeOffset;
    copyContext.scaleGm = reinterpret_cast<__gm__ uint8_t*>(params.xScaleGmAddr) + scaleRangeOffset;
    copyContext.recordsGm = reinterpret_cast<__gm__ uint8_t*>(outputAddr) + tokenScaleRangeOffset;

    PackPreQuantizedTokenRange<TopkWeightsType, TopkWeightsPrefetch>(tokenRange, common, params, config, copyContext,
                                                                     scratch);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_QUANT_H
