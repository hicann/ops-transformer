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
template <typename ActivationType, typename QuantScaleOutType, bool TopkWeightsPrefetch, uint32_t AElemsPerByte>
__aicore__ inline QuantProcessConfig CreateQuantProcessConfig(uint32_t tokenHiddenDim, const Params &params)
{
    uint32_t quantScaleValidCountPerToken = Ops::Base::CeilDiv(tokenHiddenDim, static_cast<uint32_t>(ALIGN_32));
    uint32_t quantTokenAlignBytes =
        Ops::Base::CeilAlign(tokenHiddenDim / AElemsPerByte, static_cast<uint32_t>(ALIGN_256)) * sizeof(ActivationType);
    uint32_t quantScaleAlignBytes =
        Ops::Base::CeilAlign(quantScaleValidCountPerToken * static_cast<uint32_t>(sizeof(QuantScaleOutType)),
                             static_cast<uint32_t>(ALIGN_32));
    uint32_t quantTokenScaleAlignBytes = quantTokenAlignBytes + quantScaleAlignBytes;
    if constexpr (TopkWeightsPrefetch) {
        uint32_t weightAlignBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(params.tilingData->topK * sizeof(float)),
                                                         static_cast<uint32_t>(ALIGN_32));
        quantTokenScaleAlignBytes += weightAlignBytes;
    }
    return {quantTokenAlignBytes, quantScaleAlignBytes, quantTokenScaleAlignBytes, quantScaleValidCountPerToken};
}

template <typename ActivationType>
struct QuantProcessScratch {
    LocalTensor<bfloat16_t> xInTensor0;
    LocalTensor<bfloat16_t> xInTensor1;
    LocalTensor<ActivationType> xOutTensor0;
    LocalTensor<ActivationType> xOutTensor1;
    LocalTensor<uint16_t> mxTempTensor;
};

template <typename TopkWeightsType, typename ActivationType>
__aicore__ inline void PrefetchTopkWeights(GM_ADDR tokenTopkWeightsAddr, uint32_t topK,
                                           const QuantProcessConfig &config,
                                           QuantProcessScratch<ActivationType> &scratch,
                                           const LocalTensor<ActivationType> &xOutTensor, TEventID event)
{
    GlobalTensor<TopkWeightsType> weightGm;
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ TopkWeightsType *>(tokenTopkWeightsAddr));
    uint32_t weightOffsetInUb = config.quantTokenAlignBytes + config.quantScaleAlignBytes;
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        LocalTensor<TopkWeightsType> weightBf16Tmp = scratch.mxTempTensor.template ReinterpretCast<TopkWeightsType>();
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
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    }
}

template <int32_t QuantMode, typename QuantOutType, typename ActivationType>
__aicore__ inline void QuantizeTokenInUb(const LocalTensor<bfloat16_t> &xInTensor,
                                         const LocalTensor<ActivationType> &xOutTensor,
                                         const LocalTensor<uint16_t> &mxTempTensor, const QuantProcessConfig &config,
                                         uint32_t hiddenDim)
{
    __ubuf__ uint16_t *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(mxTempTensor.GetPhyAddr());
    __ubuf__ uint16_t *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
        mxTempTensor[Ops::Base::CeilAlign(config.quantScaleValidCountPerToken, static_cast<uint32_t>(ALIGN_32))]
            .GetPhyAddr());
    __ubuf__ bfloat16_t *srcAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(xInTensor.GetPhyAddr());
    __ubuf__ int8_t *outDataAddr = reinterpret_cast<__ubuf__ int8_t *>(xOutTensor.GetPhyAddr());
    __ubuf__ uint16_t *mxScaleAddr =
        reinterpret_cast<__ubuf__ uint16_t *>(xOutTensor[config.quantTokenAlignBytes].GetPhyAddr());

    Quant::ComputeMaxExp(srcAddr, maxExpAddr, hiddenDim);
    Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, config.quantScaleValidCountPerToken);
    if constexpr (QuantMode == E2M1_QUANT) {
        Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
    } else {
        Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
    }
}

// 量化一个逻辑 AIV 任务负责的本卡 token，并按逐 token 交织布局写入量化数据与 scale。
template <int32_t QuantMode, typename QuantOutType, typename ActivationType, typename TopkWeightsType,
          bool TopkWeightsPrefetch>
__aicore__ inline void QuantizeLocalTokens(const AivJobContext &job, const MoeStageCommonConfig &common,
                                           const Params &params, const QuantProcessConfig &config, GM_ADDR outputAddr,
                                           QuantProcessScratch<ActivationType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    WorkRange tokenRange = TilingByJobContext(common.tokenNum, job.jobIndex, job.totalJobs, 1U);
    if (tokenRange.count == 0U) {
        return;
    }
    GlobalTensor<uint8_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(outputAddr));
    uint32_t hiddenDim = common.tokenHiddenDim;
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.aGmAddr) +
                                    static_cast<uint64_t>(tokenRange.start) * hiddenDim);
    // 量化 scratch（mxTemp/xOut0/xOut1/xIn0/xIn1）的跨 launch 残留清零由各编排的
    // SendAndQuantBuffInit 在分配处一次性完成（span 清零，与布局同源），见其注释。
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t index = 0; index < tokenRange.count; ++index) {
        bool useFirstBuffer = index % DOUBLE_BUFFER == 0;
        auto event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = useFirstBuffer ? scratch.xInTensor0 : scratch.xInTensor1;
        auto xOutTensor = useFirstBuffer ? scratch.xOutTensor0 : scratch.xOutTensor1;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor[static_cast<uint64_t>(index) * hiddenDim],
                    {1U, static_cast<uint16_t>(hiddenDim * sizeof(bfloat16_t)), 0U, 0U}, {true, 0, 0, 0});
        uint32_t tokenIndex = tokenRange.start + index;
        if constexpr (TopkWeightsPrefetch) {
            GM_ADDR tokenTopkWeightsAddr =
                params.probsGmAddr + static_cast<uint64_t>(tokenIndex) * common.topK * sizeof(TopkWeightsType);
            PrefetchTopkWeights<TopkWeightsType>(tokenTopkWeightsAddr, common.topK, config, scratch, xOutTensor, event);
        } else {
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        }
        QuantizeTokenInUb<QuantMode, QuantOutType>(xInTensor, xOutTensor, scratch.mxTempTensor, config, hiddenDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(output[static_cast<uint64_t>(tokenIndex) * config.quantTokenScaleAlignBytes], xOutBytesTensor,
                    {1U, config.quantTokenScaleAlignBytes, 0U, 0U, 0U});
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
    DataCopyExtParams dataCopyInParams;
    DataCopyExtParams scaleCopyInParams;
    DataCopyPadExtParams<uint8_t> copyInPadParams;
    DataCopyExtParams tokenScaleCopyOutParams;
};

// 执行一个任务范围内的双 buffer 搬运，flag 初始化、复用同步和收尾等待集中在此处。
template <typename ActivationType, typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PackPreQuantizedTokenRange(const WorkRange &tokenRange, const MoeStageCommonConfig &common,
                                                  const Params &params, const QuantProcessConfig &config,
                                                  const PreQuantizedCopyContext &copyContext,
                                                  QuantProcessScratch<ActivationType> &scratch)
{
    // xOut 双缓冲已经由 SendAndQuantBuffInit 整段清零；这里只覆盖有效 data/scale/weight，
    // data、scale 以及 optional weight 的各自对齐 padding 因此保持为 0。
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t index = 0U; index < tokenRange.count; ++index) {
        bool useFirstBuffer = index % DOUBLE_BUFFER == 0U;
        TEventID event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<ActivationType> tokenScaleTensor = useFirstBuffer ? scratch.xOutTensor0 : scratch.xOutTensor1;
        LocalTensor<uint8_t> tokenScaleBytesTensor = tokenScaleTensor.template ReinterpretCast<uint8_t>();
        uint64_t dataOffset = static_cast<uint64_t>(index) * static_cast<uint64_t>(copyContext.dataValidBytes);
        uint64_t scaleOffset = static_cast<uint64_t>(index) * static_cast<uint64_t>(copyContext.scaleValidBytes);
        uint64_t tokenScaleOffset =
            static_cast<uint64_t>(index) * static_cast<uint64_t>(config.quantTokenScaleAlignBytes);

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(tokenScaleBytesTensor, copyContext.dataSrcGlobalTensor[dataOffset], copyContext.dataCopyInParams,
                    copyContext.copyInPadParams);
        DataCopyPad(tokenScaleBytesTensor[config.quantTokenAlignBytes], copyContext.scaleSrcGlobalTensor[scaleOffset],
                    copyContext.scaleCopyInParams, copyContext.copyInPadParams);
        if constexpr (TopkWeightsPrefetch) {
            uint32_t tokenIndex = tokenRange.start + index;
            GM_ADDR tokenTopkWeightsAddr =
                params.probsGmAddr + static_cast<uint64_t>(tokenIndex) * common.topK * sizeof(TopkWeightsType);
            PrefetchTopkWeights<TopkWeightsType>(tokenTopkWeightsAddr, common.topK, config, scratch, tokenScaleTensor,
                                                 event);
            if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
                // BF16 topK 权重需要由 Vector 转为 FP32 后写入拼接数据，Vector 是最终生产者。
                SetFlag<AscendC::HardEvent::V_MTE3>(event);
                WaitFlag<AscendC::HardEvent::V_MTE3>(event);
            }
        }
        if constexpr (!TopkWeightsPrefetch || !Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
            // FP32 预取和不预取场景均由 MTE2 直接搬入，同步后再由 MTE3 搬出。
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        }
        DataCopyPad(copyContext.tokenScaleDstGlobalTensor[tokenScaleOffset], tokenScaleBytesTensor,
                    copyContext.tokenScaleCopyOutParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// 将调用方已量化的 x 和逐 32 元素 scale 按 dispatch 布局拼接，按需附加 topK 权重。
// 不对 x/scales 做数值转换。
template <typename XType, typename ActivationType, typename TopkWeightsType, bool TopkWeightsPrefetch>
__aicore__ inline void PackPreQuantizedLocalTokens(const AivJobContext &job, const MoeStageCommonConfig &common,
                                                   const Params &params, const QuantProcessConfig &config,
                                                   GM_ADDR outputAddr, QuantProcessScratch<ActivationType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    WorkRange tokenRange = TilingByJobContext(common.tokenNum, job.jobIndex, job.totalJobs, 1U);
    if (tokenRange.count == 0U) {
        return;
    }

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
    copyContext.dataSrcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.aGmAddr) +
                                                    dataRangeOffset);
    copyContext.scaleSrcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.xScaleGmAddr) +
                                                     scaleRangeOffset);
    copyContext.tokenScaleDstGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(outputAddr) +
                                                          tokenScaleRangeOffset);

    copyContext.dataCopyInParams = {1U, copyContext.dataValidBytes, 0U, 0U, 0U};
    copyContext.scaleCopyInParams = {1U, copyContext.scaleValidBytes, 0U, 0U, 0U};
    copyContext.copyInPadParams = {true, 0U, 0U, 0U};
    copyContext.tokenScaleCopyOutParams = {1U, config.quantTokenScaleAlignBytes, 0U, 0U, 0U};
    PackPreQuantizedTokenRange<ActivationType, TopkWeightsType, TopkWeightsPrefetch>(tokenRange, common, params, config,
                                                                                     copyContext, scratch);
}

// 编译期根据 x 类型选择本卡 token/scale 数据准备方式：BF16 沿用动态量化，
// FP8/FP4 只打包调用方提供的 x/scales，不实例化输入量化分支。
template <typename XType, int32_t QuantMode, typename QuantOutType, typename ActivationType, typename TopkWeightsType,
          bool TopkWeightsPrefetch>
__aicore__ inline void PrepareLocalTokens(const AivJobContext &job, const MoeStageCommonConfig &common,
                                          const Params &params, const QuantProcessConfig &config, GM_ADDR outputAddr,
                                          QuantProcessScratch<ActivationType> &scratch)
{
    if constexpr (Std::IsSame<XType, bfloat16_t>::value) {
        QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
            job, common, params, config, outputAddr, scratch);
    } else {
        PackPreQuantizedLocalTokens<XType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
            job, common, params, config, outputAddr, scratch);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_QUANT_H
