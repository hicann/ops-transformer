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
 * \file block_epilogue_activation_mx_quant.h
 * \brief
 */

#ifndef BLOCK_EPILOGUE_ACTIVATION_MX_QUANT_H
#define BLOCK_EPILOGUE_ACTIVATION_MX_QUANT_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../common/mega_moe_constants.h"
#include "../../common/mega_moe_utils.h"
#include "activation/activation_common.h"
#include "activation/situglu_activation.h"
#include "activation/swiglu_activation.h"
#include "activation/swigluoai_activation.h"
#include "block_epilogue_ub_layout.h"
#include "quant/mx_quant_common.h"
#include "quant/mx_quant_compute_scale.h"
#include "quant/mxfp4_quant.h"
#include "quant/mxfp8_quant.h"

namespace ActivationQuantMsg {
using ActMode = MegaMoeImpl::MegaMoeActMode;
using ActSubMode = MegaMoeImpl::MegaMoeActSubMode;

constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
} // namespace ActivationQuantMsg

namespace MegaMoeImpl {

using namespace AscendC;
using namespace ActivationQuantMsg;

template <typename DataTypeOut_, typename DataTypeIn_, uint32_t TileM = 256, uint32_t TileN = 256,
          bool TopkWeightsPrefetch = false, bool IsInterleaved_ = false>
class BlockEpilogueActivationMxQuant {
public:
    __aicore__ inline BlockEpilogueActivationMxQuant() {}

    static constexpr uint32_t MAX_SINGLE_MN = TileM * TileN;

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        float clampLimit{0.0f};
        uint8_t actMode{static_cast<uint8_t>(ActMode::SWIGLU)};
        uint8_t actSubMode{static_cast<uint8_t>(ActSubMode::DEFAULT)};
        float activationAlpha{1.0f};
        float activationBeta{1.0f};
        Arguments() = default;
    };

    // params
    using Params = Arguments;

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;

    // shape
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline auto GetTopkWeightTensor();
    // 计算并写回一个激活/量化 tile，不包含流水通知。
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord,
                                      uint16_t pingpongIdx = 0);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);

private:
    using UbPointers = Detail::ActivationMxQuantUbPointers<DataTypeIn>;

    __aicore__ inline void RunMxQuantTile(const UbPointers &ubPointers, uint16_t validRowCount,
                                          uint32_t outputRowStrideElements);

    __aicore__ inline void RunGatedActivationTile(__ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                                  __ubuf__ bfloat16_t *gluResAddr, uint16_t validRowCount,
                                                  uint16_t validColumnCount, uint32_t outputRowStrideElements);

    __aicore__ inline void StoreQuantOutput(AscendC::GlobalTensor<int8_t> &dst, AscendC::LocalTensor<int8_t> &src,
                                            uint64_t blockCount, uint64_t offset, uint64_t n, uint64_t singleN);

    __aicore__ inline void StoreQuantScaleCompact(AscendC::GlobalTensor<int8_t> &dst, AscendC::LocalTensor<int8_t> &src,
                                                  uint64_t blockCount, uint64_t offset, uint64_t scaleN,
                                                  uint64_t singleN);

    // GM ADDR
    GlobalTensor<int8_t> quantOutputGlobal_;
    GlobalTensor<int8_t> quantScaleGlobal_;
    GM_ADDR yGmAddr_{nullptr};
    GM_ADDR yScaleGmAddr_{nullptr};

    // UB ADDR
    static constexpr uint32_t kUbFirstOffset =
        Detail::BuildActivationMxQuantUbOffsets<DataTypeIn, TileM, TileN, IsInterleaved_>().firstInputOffsetBytes;
    static constexpr uint32_t kUbInputElementCapacity =
        Detail::BuildActivationMxQuantUbOffsets<DataTypeIn, TileM, TileN, IsInterleaved_>().firstInputElementCapacity;
    static constexpr uint32_t kUbSecondOffset =
        Detail::BuildActivationMxQuantUbOffsets<DataTypeIn, TileM, TileN, IsInterleaved_>().secondInputOffsetBytes;
    LocalTensor<DataTypeIn> l0cOutUbFirst_{TPosition::VECIN, kUbFirstOffset, kUbInputElementCapacity};
    LocalTensor<DataTypeIn> l0cOutUbSecond_{TPosition::VECIN, kUbSecondOffset, kUbInputElementCapacity};
    LocalTensor<int8_t> quantOutput_;
    LocalTensor<int8_t> quantScaleOutput_;
    LocalTensor<bfloat16_t> gluRes_;
    LocalTensor<uint16_t> maxExp_;
    LocalTensor<uint16_t> halfScale_;
    LocalTensor<float> weightUb_;

    int64_t intermediateHiddenSize_;
    int64_t intermediateHiddenScaleElements_;

    float clampLimit_{0.0f};
    uint8_t actMode_{static_cast<uint8_t>(ActMode::SWIGLU)};
    uint8_t actSubMode_{static_cast<uint8_t>(ActSubMode::DEFAULT)};
    float activationAlpha_{1.0f};
    float activationBeta_{1.0f};
};

template <typename DataTypeOut_, typename DataTypeIn_, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved_>
__aicore__ inline void
BlockEpilogueActivationMxQuant<DataTypeOut_, DataTypeIn_, TileM, TileN, TopkWeightsPrefetch,
                               IsInterleaved_>::StoreQuantOutput(AscendC::GlobalTensor<int8_t> &dst,
                                                                 AscendC::LocalTensor<int8_t> &src, uint64_t blockCount,
                                                                 uint64_t offset, uint64_t n, uint64_t singleN)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount; // 128
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ub2GmParams.blockLen = singleN >> 1;
        ub2GmParams.dstStride = (n - singleN) >> 1;
        offset = offset >> 1;
    } else {
        uint64_t nDstUbAligned =
            Ops::Base::CeilAlign(static_cast<uint64_t>(singleN), static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
        ub2GmParams.blockLen = singleN; // 256
        ub2GmParams.srcStride = (nDstUbAligned - singleN) / AscendC::ONE_BLK_SIZE;
        ub2GmParams.dstStride = n - singleN;
    }
    AscendC::DataCopyPad(dst[offset], src, ub2GmParams);
}

template <typename DataTypeOut_, typename DataTypeIn_, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved_>
__aicore__ inline void
BlockEpilogueActivationMxQuant<DataTypeOut_, DataTypeIn_, TileM, TileN, TopkWeightsPrefetch,
                               IsInterleaved_>::StoreQuantScaleCompact(AscendC::GlobalTensor<int8_t> &dst,
                                                                       AscendC::LocalTensor<int8_t> &src,
                                                                       uint64_t blockCount, uint64_t offset,
                                                                       uint64_t scaleN, uint64_t singleN)
{
    AscendC::DataCopyExtParams ub2GmParams{0, 0, 0, 0, 0};
    auto blockScaleN = Ops::Base::CeilDiv(static_cast<uint64_t>(singleN), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                       MXFP_MULTI_BASE_SIZE;
    // scale layout in UB is already compact: (mSize, blockScaleN). Compact copy avoids (mSize*8)->(mSize,32).
    ub2GmParams.blockCount = blockCount; // 128
    ub2GmParams.blockLen = blockScaleN;  // 8
    ub2GmParams.srcStride = 0;
    ub2GmParams.dstStride = scaleN - blockScaleN;
    AscendC::DataCopyPad<int8_t, AscendC::PaddingMode::Compact>(dst[offset], src, ub2GmParams);
}

template <typename DataTypeOut_, typename DataTypeIn_, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved_>
__aicore__ inline void BlockEpilogueActivationMxQuant<DataTypeOut_, DataTypeIn_, TileM, TileN, TopkWeightsPrefetch,
                                                      IsInterleaved_>::Init(Params const &params)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // Params 的生命周期可能短于 Epilogue 对象，因此保存输出 GM 地址。
    yGmAddr_ = params.yGmAddr;
    yScaleGmAddr_ = params.yScaleGmAddr;
    clampLimit_ = params.clampLimit;
    actMode_ = params.actMode;
    actSubMode_ = params.actSubMode;
    activationAlpha_ = params.activationAlpha;
    activationBeta_ = params.activationBeta;

    constexpr auto ubOffsets = Detail::BuildActivationMxQuantUbOffsets<DataTypeIn, TileM, TileN, IsInterleaved_>();
    gluRes_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, ubOffsets.activationOutputOffsetBytes,
                                      ubOffsets.activationQuantElementCapacity);
    quantOutput_ = LocalTensor<int8_t>(TPosition::VECOUT, ubOffsets.quantOutputOffsetBytes,
                                       ubOffsets.activationQuantElementCapacity);
    quantScaleOutput_ =
        LocalTensor<int8_t>(TPosition::VECOUT, ubOffsets.quantScaleOffsetBytes, ubOffsets.scaleElementCapacity);
    maxExp_ = LocalTensor<uint16_t>(TPosition::VECCALC, ubOffsets.maxExpOffsetBytes, ubOffsets.scaleElementCapacity);
    halfScale_ =
        LocalTensor<uint16_t>(TPosition::VECCALC, ubOffsets.reciprocalScaleOffsetBytes, ubOffsets.scaleElementCapacity);

    // weight UB: 放在 VECIN 区之后的安全区域，仅 mode=1 分配
    if constexpr (TopkWeightsPrefetch) {
        weightUb_ = LocalTensor<float>(TPosition::VECCALC, ubOffsets.topkWeightOffsetBytes,
                                       ubOffsets.topkWeightElementCapacity);
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved_>
__aicore__ inline void BlockEpilogueActivationMxQuant<DataTypeOut_, DataTypeIn_, TileM, TileN, TopkWeightsPrefetch,
                                                      IsInterleaved_>::UpdateGlobalAddr(const BlockCoord &baseOffset)
{
    if constexpr (g_coreType == AIV) {
        quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t *)yGmAddr_ + Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer((__gm__ int8_t *)yScaleGmAddr_ + Get<Y_SCALE_IDX>(baseOffset));
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved_>
__aicore__ inline void
BlockEpilogueActivationMxQuant<DataTypeOut_, DataTypeIn_, TileM, TileN, TopkWeightsPrefetch,
                               IsInterleaved_>::UpdateNextProblem(const ProblemShape &problemShape)
{
    intermediateHiddenSize_ = Get<N_VALUE>(problemShape);
    intermediateHiddenScaleElements_ =
        Ops::Base::CeilDiv(static_cast<uint64_t>(intermediateHiddenSize_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
}

template <typename DataTypeOut_, typename DataTypeIn_, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved_>
__aicore__ inline void
BlockEpilogueActivationMxQuant<DataTypeOut_, DataTypeIn_, TileM, TileN, TopkWeightsPrefetch,
                               IsInterleaved_>::RunGatedActivationTile(__ubuf__ DataTypeIn *firstSrc,
                                                                       __ubuf__ DataTypeIn *secondSrc,
                                                                       __ubuf__ bfloat16_t *gluResAddr,
                                                                       uint16_t validRowCount,
                                                                       uint16_t validColumnCount,
                                                                       uint32_t outputRowStrideElements)
{
    uint32_t inputRowStrideElements;
    if constexpr (IsInterleaved_) {
        // interleaved源布局为[x1, x2]连续存放在同一行，下一行stride是2*validColumnCount。
        // 注意：2*validColumnCount==实际行距(TileN) 仅在满 tile 成立；tiling 已约束 hiddenDim%256==0，
        // 交织调度宽度为完整 hiddenDim，故交织路径不会出现尾 tile；若未来放宽须改用 TileN。
        inputRowStrideElements = static_cast<uint32_t>(validColumnCount) * 2U;
    } else {
        // 非交织源是两块独立 UB tile，生产端按固定行距 TileN 写入（MakeLayoutC(tileM, L1_TILE_N)）；
        // 尾块 validColumnCount < TileN 时行距不随有效宽度收缩，否则第 1 行起整行错位。
        inputRowStrideElements = TileN;
    }

    // 当前 tiling 下非交织列宽为满块 256 或尾块 128，交织激活列宽为 128 且无尾 tile；以下仍保留
    // 原实现对更广 validColumnCount 的主循环、尾计算和两段补零能力，便于后续放宽对齐约束。
    uint16_t rowLoopCount = validRowCount;
    uint16_t fullVectorLoopCount = validColumnCount / Activation::VECTOR_LENGTH_FP32;
    uint32_t tailElementCountPerRow = validColumnCount % Activation::VECTOR_LENGTH_FP32;
    uint16_t needTailVectorCompute = 0U;
    uint16_t needAdditionalPaddingStore = 0U;
    uint32_t tailComputeMaskElementCount = 0U;
    uint32_t tailStoreMaskElementCount = 0U;
    uint32_t additionalPaddingStoreMaskElementCount = 0U;
    __ubuf__ DataTypeIn *gateTail = firstSrc;
    __ubuf__ DataTypeIn *upTail = secondSrc;
    __ubuf__ bfloat16_t *outputTail = gluResAddr;
    __ubuf__ bfloat16_t *additionalPaddingOutput = gluResAddr;
    if (tailElementCountPerRow > 0U) {
        tailComputeMaskElementCount = tailElementCountPerRow;
        needTailVectorCompute = 1U;
        const uint32_t tailAndPaddingElementCount =
            outputRowStrideElements - fullVectorLoopCount * Activation::VECTOR_LENGTH_FP32;
        if (tailAndPaddingElementCount <= Activation::VECTOR_LENGTH_FP32) {
            tailStoreMaskElementCount = tailAndPaddingElementCount;
        } else {
            needAdditionalPaddingStore = 1U;
            tailStoreMaskElementCount = Activation::VECTOR_LENGTH_FP32;
            additionalPaddingStoreMaskElementCount = tailAndPaddingElementCount - Activation::VECTOR_LENGTH_FP32;
        }
        const uint32_t tailColumnOffsetElements = fullVectorLoopCount * Activation::VECTOR_LENGTH_FP32;
        gateTail = firstSrc + tailColumnOffsetElements;
        upTail = secondSrc + tailColumnOffsetElements;
        outputTail = gluResAddr + tailColumnOffsetElements;
        additionalPaddingOutput = outputTail + needTailVectorCompute * Activation::VECTOR_LENGTH_FP32;
    }

    Activation::GatedActivationTileContext<DataTypeIn> gatedActivationContext;
    gatedActivationContext.gate = firstSrc;
    gatedActivationContext.up = secondSrc;
    gatedActivationContext.output = gluResAddr;
    if constexpr (TopkWeightsPrefetch) {
        gatedActivationContext.topkWeights = (__ubuf__ float *)weightUb_.GetPhyAddr();
    } else {
        gatedActivationContext.topkWeights = nullptr;
    }
    gatedActivationContext.gateTail = gateTail;
    gatedActivationContext.upTail = upTail;
    gatedActivationContext.outputTail = outputTail;
    gatedActivationContext.additionalPaddingOutput = additionalPaddingOutput;
    gatedActivationContext.inputRowStrideElements = inputRowStrideElements;
    gatedActivationContext.outputRowStrideElements = outputRowStrideElements;
    gatedActivationContext.rowLoopCount = rowLoopCount;
    gatedActivationContext.fullVectorLoopCount = fullVectorLoopCount;
    gatedActivationContext.needTailVectorCompute = needTailVectorCompute;
    gatedActivationContext.needAdditionalPaddingStore = needAdditionalPaddingStore;
    gatedActivationContext.tailComputeMaskElementCount = tailComputeMaskElementCount;
    gatedActivationContext.tailStoreMaskElementCount = tailStoreMaskElementCount;
    gatedActivationContext.additionalPaddingStoreMaskElementCount = additionalPaddingStoreMaskElementCount;

    const ActMode actMode = static_cast<ActMode>(actMode_);
    const ActSubMode actSubMode = static_cast<ActSubMode>(actSubMode_);
    if (actMode == ActMode::SITU) {
        const float invBeta = activationBeta_ != 0.0f ? 1.0f / activationBeta_ : 1.0f;
        if (actSubMode == ActSubMode::LINEAR) {
            const float invAlpha = activationAlpha_ != 0.0f ? 1.0f / activationAlpha_ : 1.0f;
            const Activation::SituGluParams situGluParams{clampLimit_, activationBeta_, invBeta, activationAlpha_,
                                                          invAlpha};
            Activation::RunSiTUGLU<DataTypeIn, TopkWeightsPrefetch, true>(gatedActivationContext, situGluParams);
        } else {
            const Activation::SituGluParams situGluParams{clampLimit_, activationBeta_, invBeta, 1.0f, 1.0f};
            Activation::RunSiTUGLU<DataTypeIn, TopkWeightsPrefetch, false>(gatedActivationContext, situGluParams);
        }
    } else if (actMode == ActMode::SWIGLU_STEP) {
        const Activation::SwiGluParams swiGluParams{clampLimit_};
        Activation::RunSwiGLU<DataTypeIn, TopkWeightsPrefetch, true>(gatedActivationContext, swiGluParams);
    } else if (actMode == ActMode::SWIGLU_OAI) {
        const Activation::SwiGluOaiParams swiGluOaiParams{clampLimit_, activationAlpha_, activationBeta_};
        Activation::RunSwiGLUOAI<DataTypeIn, TopkWeightsPrefetch>(gatedActivationContext, swiGluOaiParams);
    } else {
        const Activation::SwiGluParams swiGluParams{clampLimit_};
        Activation::RunSwiGLU<DataTypeIn, TopkWeightsPrefetch, false>(gatedActivationContext, swiGluParams);
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved_>
__aicore__ inline void BlockEpilogueActivationMxQuant<DataTypeOut_, DataTypeIn_, TileM, TileN, TopkWeightsPrefetch,
                                                      IsInterleaved_>::RunMxQuantTile(const UbPointers &ubPointers,
                                                                                      uint16_t validRowCount,
                                                                                      uint32_t outputRowStrideElements)
{
    const uint32_t dataCount = validRowCount * outputRowStrideElements;
    const uint32_t scaleCount = dataCount / ONE_BLK_SIZE;
    const uint16_t dataLoopCount = static_cast<uint16_t>((dataCount + MxQuant::DATA_ELEMENT_COUNT_PER_LOOP - 1) /
                                                         MxQuant::DATA_ELEMENT_COUNT_PER_LOOP);
    const uint16_t scaleLoopCount = static_cast<uint16_t>((scaleCount + MxQuant::SCALE_ELEMENT_COUNT_PER_VECTOR - 1) /
                                                          MxQuant::SCALE_ELEMENT_COUNT_PER_VECTOR);

    MxQuant::ComputeGroupMaxExp(ubPointers.activationOutput, ubPointers.maxExp, dataCount, dataLoopCount);
    MxQuant::ComputeMxScale<DataTypeOut>(ubPointers.maxExp, ubPointers.quantScale, ubPointers.reciprocalScale,
                                         scaleCount, scaleLoopCount);
    if constexpr (IsSameType<DataTypeOut, fp8_e4m3fn_t>::value || IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        MxQuant::QuantizeMxFp8Data<DataTypeOut>(ubPointers.activationOutput, ubPointers.reciprocalScale,
                                                ubPointers.quantOutput, dataCount, dataLoopCount);
    }
    if constexpr (IsSameType<DataTypeOut, fp4x2_e2m1_t>::value || IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        MxQuant::QuantizeMxFp4Data<DataTypeOut>(ubPointers.activationOutput, ubPointers.reciprocalScale,
                                                ubPointers.quantOutput, dataCount, dataLoopCount);
    }
}

template <typename DataTypeOut_, typename DataTypeIn_, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved_>
__aicore__ inline auto BlockEpilogueActivationMxQuant<DataTypeOut_, DataTypeIn_, TileM, TileN, TopkWeightsPrefetch,
                                                      IsInterleaved_>::GetTopkWeightTensor()
{
    return weightUb_;
}

template <typename DataTypeOut_, typename DataTypeIn_, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved_>
__aicore__ inline void BlockEpilogueActivationMxQuant<DataTypeOut_, DataTypeIn_, TileM, TileN, TopkWeightsPrefetch,
                                                      IsInterleaved_>::operator()(const BlockShape &blockShape,
                                                                                  const BlockCoord &blockCoord,
                                                                                  uint16_t pingpongIdx)
{
    uint32_t tileRowCount = Get<M_VALUE>(blockShape);    // 128
    uint32_t tileColumnCount = Get<N_VALUE>(blockShape); // 256

    if (tileRowCount == 0) {
        return;
    }

    uint64_t yOffset = Get<Y_IDX>(blockCoord);
    uint64_t yScaleOffset = Get<Y_SCALE_IDX>(blockCoord);
    __ubuf__ DataTypeIn *secondInputBase = nullptr;
    if constexpr (!IsInterleaved_) {
        secondInputBase = (__ubuf__ DataTypeIn *)l0cOutUbSecond_.GetPhyAddr();
    }
    auto ubPointers = Detail::ResolveActivationMxQuantUbPointers<DataTypeIn, MAX_SINGLE_MN, IsInterleaved_>(
        (__ubuf__ DataTypeIn *)l0cOutUbFirst_.GetPhyAddr(), secondInputBase,
        (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr(), (__ubuf__ int8_t *)quantOutput_.GetPhyAddr(),
        (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr(), (__ubuf__ uint16_t *)maxExp_.GetPhyAddr(),
        (__ubuf__ uint16_t *)halfScale_.GetPhyAddr(), tileColumnCount, pingpongIdx);
    uint32_t outputRowStrideElements = Ops::Base::CeilAlign(tileColumnCount, static_cast<uint32_t>(ONE_BLK_SIZE));
    RunGatedActivationTile(ubPointers.firstInput, ubPointers.secondInput, ubPointers.activationOutput,
                           static_cast<uint16_t>(tileRowCount), static_cast<uint16_t>(tileColumnCount),
                           outputRowStrideElements);
    RunMxQuantTile(ubPointers, static_cast<uint16_t>(tileRowCount), outputRowStrideElements);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    // scale已按compact布局生成，直接copy到GM，省掉原先TransMxScaleLayout重排scale。
    if constexpr (IsInterleaved_) {
        if (pingpongIdx == 1U) {
            LocalTensor<int8_t> quantOutputPong = quantOutput_[ubPointers.selectedInt8BufferOffsetElements];
            LocalTensor<int8_t> quantScalePong = quantScaleOutput_[ubPointers.selectedInt8BufferOffsetElements];
            StoreQuantOutput(quantOutputGlobal_, quantOutputPong, tileRowCount, yOffset, intermediateHiddenSize_,
                             tileColumnCount);
            StoreQuantScaleCompact(quantScaleGlobal_, quantScalePong, tileRowCount, yScaleOffset,
                                   intermediateHiddenScaleElements_, tileColumnCount);
        } else {
            StoreQuantOutput(quantOutputGlobal_, quantOutput_, tileRowCount, yOffset, intermediateHiddenSize_,
                             tileColumnCount);
            StoreQuantScaleCompact(quantScaleGlobal_, quantScaleOutput_, tileRowCount, yScaleOffset,
                                   intermediateHiddenScaleElements_, tileColumnCount);
        }
    } else {
        StoreQuantOutput(quantOutputGlobal_, quantOutput_, tileRowCount, yOffset, intermediateHiddenSize_,
                         tileColumnCount);
        StoreQuantScaleCompact(quantScaleGlobal_, quantScaleOutput_, tileRowCount, yScaleOffset,
                               intermediateHiddenScaleElements_, tileColumnCount);
    }
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
}

} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // BLOCK_EPILOGUE_ACTIVATION_MX_QUANT_H
