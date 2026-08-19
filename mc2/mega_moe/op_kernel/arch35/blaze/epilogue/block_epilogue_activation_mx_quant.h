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
#include "gated_activation.h"

namespace ActivationQuantMsg {
enum class QuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
};

enum class QuantDtype : uint8_t {
    DEFAULT = 0x0U,
    FP8_E4M3FN = 0x1U,
    FP8_E5M2 = 0x1U << 1,
};

using ActMode = MegaMoeImpl::MegaMoeActMode;
using ActSubMode = MegaMoeImpl::MegaMoeActSubMode;

constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr uint32_t GROUP_FLAG_IDX = 2;
constexpr uint32_t M_LOC_IDX = 4;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr uint32_t FLAG_VALUE_ONE = 1;
constexpr int64_t QUANT_ONCE_NUM = 256;
constexpr int64_t QUANT_ONCE_NUM_FP4 = 128;
constexpr int64_t SCALE_ONCE_NUM = 8;
constexpr int64_t MX_SCALE_PACK_COUNT = 64;

constexpr AscendC::MicroAPI::CastTrait ctInt322Fp32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait ctFp322Half = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32Zero = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32One = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

static constexpr AscendC::MicroAPI::DivSpecificMode DIV_MODE = {
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    true,
};
static constexpr AscendC::MicroAPI::CastTrait CAST_ONE = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_80 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_81 = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_82 = {
    AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_83 = {
    AscendC::MicroAPI::RegLayout::THREE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
} // namespace ActivationQuantMsg

namespace MegaMoeImpl {

using namespace AscendC;
using namespace ActivationQuantMsg;

#define BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS \
    template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_, typename DataTypeX1Scale_, \
              bool IsTensorList_, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch, bool IsInterleaved_>
#define BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS \
    DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, IsTensorList_, TileM, TileN, TopkWeightsPrefetch, \
        IsInterleaved_

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_, typename DataTypeX1Scale_,
          bool IsTensorList_, uint32_t TileM = 256, uint32_t TileN = 256, bool TopkWeightsPrefetch = false,
          bool IsInterleaved_ = false>
class BlockEpilogueActivationMxQuant {
public:
    __aicore__ inline BlockEpilogueActivationMxQuant() {}

    static constexpr uint32_t MAX_TILE_M = TileM;
    static constexpr uint32_t MAX_SINGLE_MN = TileM * TileN;

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
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
    using DataTypeX1Scale = DataTypeX1Scale_;
    using DataTypeX2Scale = DataTypeX2Scale_;

    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BaseOffset = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // y, yScale, x2Scale, x1Scale, bias
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline auto GetFirstL0c2UbTensor();
    __aicore__ inline auto GetSecondL0c2UbTensor();
    __aicore__ inline auto GetTopkWeightTensor();
    // 计算并写回一个激活/量化 tile，不包含流水通知。
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord,
                                      uint16_t pingpongIdx = 0);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);

private:
    __aicore__ inline void VFDoActivationForMX(uint16_t mSize, uint16_t pingpongIdx = 0, uint64_t mLoc = 0);

    __aicore__ inline void VFDoActivationAndQuantForMX(__ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
                                                       __ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                                       __ubuf__ bfloat16_t *gluResAddr, __ubuf__ uint16_t *maxExpAddr,
                                                       __ubuf__ uint16_t *halfScaleLocalAddr, uint16_t mSize,
                                                       uint16_t nSize, uint64_t mLoc = 0);

    __aicore__ inline void VFDoActivation(__ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                          __ubuf__ bfloat16_t *gluResAddr, uint16_t mSize, uint16_t nSize,
                                          uint32_t nDstUbAligned);

    __aicore__ inline void ComputeScale(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                        __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB,
                                        uint16_t loopNumScale);

    __aicore__ inline void ComputeMaxExp(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                         uint32_t totalCountInUB, uint16_t loopNum);

    __aicore__ inline void ComputeDataForQuantTargetFp8(__ubuf__ bfloat16_t *srcAddr,
                                                        __ubuf__ uint16_t *halfScaleLocalAddr,
                                                        __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);

    __aicore__ inline void ComputeDataForQuantTargetFp4(__ubuf__ bfloat16_t *srcAddr,
                                                        __ubuf__ uint16_t *halfScaleLocalAddr,
                                                        __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);

    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src);

    __aicore__ inline void CopyScaleFromUb2GmCompact(uint64_t blockCount, uint64_t offset,
                                                     AscendC::LocalTensor<int8_t> &src);
    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<int8_t> quantScaleGlobal_;
    GM_ADDR yGmAddr_{nullptr};
    GM_ADDR yScaleGmAddr_{nullptr};

    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    static constexpr uint32_t kUbSecondOffset =
        (MAX_SINGLE_MN * sizeof(DataTypeIn) * 2U <= 256U * 1024U) ? (MAX_SINGLE_MN * sizeof(DataTypeIn)) : 0U;
    AscendC::LocalTensor<DataTypeIn> l0cOutUbSecond_{AscendC::TPosition::VECIN, kUbSecondOffset, MAX_SINGLE_MN};
    AscendC::LocalTensor<int8_t> quantOutput_;
    AscendC::LocalTensor<int8_t> quantScaleOutput_;
    AscendC::LocalTensor<bfloat16_t> gluRes_;
    AscendC::LocalTensor<uint16_t> maxExp_;
    AscendC::LocalTensor<uint16_t> halfScale_;
    AscendC::LocalTensor<float> weightUb_;

    int64_t n_;
    int64_t scaleN_;
    uint32_t subBlockIdx_ = AscendC::GetSubBlockIdx();
    uint32_t singleM_; // cur singleShapeM
    uint32_t singleN_;
    bool isBiasEpilogue_ = false;
    int64_t UBBlockSize_ = 0;
    uint32_t vlForHalfNumber_ = 0;
    uint16_t elementAfterReduce_ = 0;
    uint16_t fpEmax_ = 0;

    BlockCoord blockCoord_{0, 0, 0, 0, 0, 0};
    float clampLimit_{0.0f};
    uint8_t actMode_{static_cast<uint8_t>(ActMode::SWIGLU)};
    uint8_t actSubMode_{static_cast<uint8_t>(ActSubMode::DEFAULT)};
    float activationAlpha_{1.0f};
    float activationBeta_{1.0f};
};

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(
    Params const &params)
{
    if constexpr (g_coreType == AscendC::AIC) {
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
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        fpEmax_ = FP4_E2M1_MAX_EXP;
    } else {
        fpEmax_ = FP4_E1M2_MAX_EXP;
    }

    // 非交织布局使用完整 tile，交织布局使用带 ping-pong 偏移的半 tile。
    constexpr uint32_t MAX_SINGLE_MN_ALIAS = IsInterleaved_ ? MAX_SINGLE_MN / ACTIVATION_N_HALF : MAX_SINGLE_MN;
    constexpr uint32_t gluResOffset = 0;
    gluRes_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, gluResOffset, MAX_SINGLE_MN_ALIAS);
    constexpr uint32_t quantOutputOffset = gluResOffset + MAX_SINGLE_MN_ALIAS * sizeof(bfloat16_t);
    quantOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, quantOutputOffset, MAX_SINGLE_MN_ALIAS);
    constexpr uint32_t quantScaleOffset = quantOutputOffset + MAX_SINGLE_MN_ALIAS * sizeof(int8_t);
    quantScaleOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, quantScaleOffset,
                                                     MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t maxExpOffset = quantScaleOffset + MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE * sizeof(int8_t);
    maxExp_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, maxExpOffset,
                                             MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t halfScaleOffset = maxExpOffset + MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    halfScale_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, halfScaleOffset,
                                                MAX_SINGLE_MN_ALIAS / AscendC::ONE_BLK_SIZE);

    // weight UB: 放在 VECIN 区之后的安全区域，仅 mode=1 分配
    if constexpr (TopkWeightsPrefetch) {
        constexpr uint32_t vecinEnd = MAX_SINGLE_MN * sizeof(DataTypeIn) * 2U;
        weightUb_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, vecinEnd, MAX_TILE_M * INT32_PER_256B);
    }
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(
    const BlockCoord &baseOffset)
{
    if constexpr (g_coreType == AscendC::AIV) {
        quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t *)yGmAddr_ + Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer((__gm__ int8_t *)yScaleGmAddr_ + Get<Y_SCALE_IDX>(baseOffset));
    }
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape &problemShape)
{
    n_ = Get<N_VALUE>(problemShape); // n/2
    scaleN_ =
        Ops::Base::CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyOutputFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount; // 128
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ub2GmParams.blockLen = singleN_ >> 1;
        ub2GmParams.dstStride = (n_ - singleN_) >> 1;
        offset = offset >> 1;
    } else {
        uint64_t nDstUbAligned =
            Ops::Base::CeilAlign(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
        ub2GmParams.blockLen = singleN_; // 256
        ub2GmParams.srcStride = (nDstUbAligned - singleN_) / AscendC::ONE_BLK_SIZE;
        ub2GmParams.dstStride = n_ - singleN_;
    }
    AscendC::DataCopyPad(quantOutputGlobal_[offset], src, ub2GmParams);
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyScaleFromUb2GmCompact(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{0, 0, 0, 0, 0};
    auto blockScaleN = Ops::Base::CeilDiv(static_cast<uint64_t>(singleN_),
                                          static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                       MXFP_MULTI_BASE_SIZE; // 256 / 32 = 8
    // scale layout in UB is already compact: (mSize, blockScaleN). Compact copy avoids (mSize*8)->(mSize,32).
    ub2GmParams.blockCount = blockCount; // 128
    ub2GmParams.blockLen = blockScaleN;  // 8
    ub2GmParams.srcStride = 0;
    ub2GmParams.dstStride = scaleN_ - blockScaleN;
    AscendC::DataCopyPad<int8_t, AscendC::PaddingMode::Compact>(quantScaleGlobal_[offset], src, ub2GmParams);
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxExp(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    int64_t onceNum = QUANT_ONCE_NUM;
    int64_t scaleNum = SCALE_ONCE_NUM;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0, vdExpExtract1;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16, vdMaxExp;
        AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::MaskReg scaleMask1, scaleMask2;
        AscendC::MicroAPI::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, onceNum); // copy two chunks from srcAddr to regbase
            AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, expMaskBF16,
                                   scaleMask1);
            AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, expMaskBF16,
                                   scaleMask1);
            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, scaleNum);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeScale(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale) // 128*8  8
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask, vdMaxExp;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16); // MAX_EXP_FOR_BF16表示bf16正无穷 大小：128
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, sharedExp, scaleValue, scaleBias, halfScale;
        AscendC::MicroAPI::Duplicate(maxExpValue, fpEmax_);     // 0x0780 大小：128 对应bf16指数位后四位
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS); // 0x7f00 大小：128
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor, zeroRegTensor, nanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8); // 0x00ff 大小：128
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);                 // 0 大小：128
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);  // 0x7f81 大小：128
        AscendC::MicroAPI::MaskReg invalidDataMask, specialDataMask;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);   // 0x0040 大小：128
        for (uint16_t i = 0; i < loopNumScale; i++) {                               // 8
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB); // 128*8
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, QUANT_ONCE_NUM_FP4); // 每次搬运128个数到vdMaxExp
            // 得到不等于INF的结果掩码 cmpResult
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            // 得到不等于0的结果掩码 zeroMask
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            // 得到小于或等于0x0780的结果掩码 invalidDataMask
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue,
                                                                       preMaskScale);
            // 将vdMaxExp中小于或等于0x0780的结果替换成0x0780
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale); // sharedExp = vdMaxExp - 0x0780
            // 逻辑右移7位 当前指数位在减去0x0780后，已移至最低位
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            // 将scaleValue中INF的结果替换成0x00ff
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            // 将scaleValue中原来是0的结果替换成0
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);
            // 将scaleValue中数取低半部分，搬运到mxScaleLocalAddr uint16--int8
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                                     MX_SCALE_PACK_COUNT, preMaskScale);
            // 得到sharedExp等于0x7f00的结果掩码 specialDataMask
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias,
                                                                       preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale); // halfScale = 0x7f00 - sharedExp
            // 将halfScale中原等于INF的数值替换成0x7f81
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            // 将halfScale中原等于0的数值替换成0
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            // 将halfScale中原等于0x7f00的数值替换成0x0040
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
            // 将128个数搬运到halfScaleLocalAddr uint16--uint16
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, QUANT_ONCE_NUM_FP4, preMaskScale);
        }
    }
    return;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp8(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    using T = bfloat16_t;
    using U = DataTypeOut;
    (void)totalCountInUB;
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One;
        AscendC::MicroAPI::RegTensor<float> vdExp1FP32Zero, vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP8Zero, vdExp0FP8One;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP8Zero, vdExp1FP8One;
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB8 =
            AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopNum; i++) {
            // DIST_DINTLV_B16:双搬入模式，读取2*VL长度数据，将偶数索引的元素存入dst0，奇数索引的元素存入dst1
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                                      onceXNum);
            // 将halfScale中的8个数uint16广播到halfScaleForMul中，halfScale[0]*16 halfScale[1]*16...
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce);
            // vdExp0/vdExp1乘以广播后的halfScale，得到量化前缩放值
            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, maskAll);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, maskAll);
            AscendC::MicroAPI::Cast<float, T, ActivationImpl::CAST_ZERO>(vdExp0FP32Zero, vdExp0, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp0FP32One, vdExp0, maskAll);
            AscendC::MicroAPI::Cast<float, T, ActivationImpl::CAST_ZERO>(vdExp1FP32Zero, vdExp1, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp1FP32One, vdExp1, maskAll);
            // CAST_32_TO_80/82/81/83把4路fp32 lane cast到fp8 lane，后续按uint8合并成连续fp8输出
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_80>(vdExp0FP8Zero, vdExp0FP32Zero, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_82>(vdExp0FP8One, vdExp0FP32One, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_81>(vdExp1FP8Zero, vdExp1FP32Zero, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_83>(vdExp1FP8One, vdExp1FP32One, maskAll);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8One, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp1FP8Zero, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp1FP8One, maskAllB8);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                // 将src中有效元素的低8bit数据连续存储于dst中
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8Zero, onceXNum, maskAllB8);
        }
    }
    return;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp4(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    using T = bfloat16_t;
    using U = DataTypeOut;
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    int64_t onceYNum = OUT_ELE_NUM_ONE_BLK;
    static constexpr AscendC::MicroAPI::CastTrait castTrait = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP4, vdExp1FP4;
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                                      onceXNum);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce);
            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
            AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask1);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP4, onceYNum, dataMask1);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP4, onceYNum, dataMask1);
        }
    }
    return;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoActivation(
    __ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc, __ubuf__ bfloat16_t *gluResAddr, uint16_t mSize,
    uint16_t nSize, uint32_t nDstUbAligned)
{
    uint32_t nSrcUbAligned;
    if constexpr (IsInterleaved_) {
        // interleaved源布局为[x1, x2]连续存放在同一行，下一行stride是2*nSize。
        // 注意：2*nSize==实际行距(TileN) 仅在满 tile 成立；tiling 已约束 hiddenDim%256==0，
        // 交织调度宽度为完整 hiddenDim，故交织路径不会出现尾 tile；若未来放宽须改用 TileN。
        nSrcUbAligned = static_cast<uint32_t>(nSize) * 2U;
    } else {
        // 非交织源是两块独立 UB tile，生产端按固定行距 TileN 写入（MakeLayoutC(tileM, L1_TILE_N)）；
        // 尾块 nSize < TileN 时行距不随 nSize 收缩，否则第 1 行起整行错位。
        nSrcUbAligned = TileN;
    }
    uint16_t dim0VfTimes = mSize;
    uint16_t dim1VfTimes = nSize / ActivationImpl::VF_LEN_FP32;
    uint32_t dim1Tail = nSize % ActivationImpl::VF_LEN_FP32;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
    __ubuf__ DataTypeIn *firstTailAddr = firstSrc;
    __ubuf__ DataTypeIn *secondTailAddr = secondSrc;
    __ubuf__ bfloat16_t *activationTailAddr1 = gluResAddr;
    __ubuf__ bfloat16_t *activationTailAddr2 = gluResAddr;
    if (dim1Tail > 0) {
        mask1Num = dim1Tail;
        dim1TailTimes = 1;
        uint32_t padNum = nDstUbAligned - dim1VfTimes * ActivationImpl::VF_LEN_FP32;
        if (padNum <= ActivationImpl::VF_LEN_FP32) {
            mask2Num = padNum;
        } else {
            dim1Tail2 = 1;
            mask2Num = ActivationImpl::VF_LEN_FP32;
            mask3Num = padNum - ActivationImpl::VF_LEN_FP32;
        }
        uint32_t offsetAlign = dim1VfTimes * ActivationImpl::VF_LEN_FP32;
        firstTailAddr = firstSrc + offsetAlign;
        secondTailAddr = secondSrc + offsetAlign;
        activationTailAddr1 = gluResAddr + offsetAlign;
        activationTailAddr2 = gluResAddr + offsetAlign + dim1TailTimes * ActivationImpl::VF_LEN_FP32;
    }

    ActivationImpl::ActivationContext<DataTypeIn> ctx;
    ctx.firstSrc = firstSrc;
    ctx.secondSrc = secondSrc;
    ctx.gluResAddr = gluResAddr;
    ctx.firstTailAddr = firstTailAddr;
    ctx.secondTailAddr = secondTailAddr;
    ctx.activationTailAddr1 = activationTailAddr1;
    ctx.activationTailAddr2 = activationTailAddr2;
    if constexpr (TopkWeightsPrefetch) {
        ctx.weightUbAddr = (__ubuf__ float *)weightUb_.GetPhyAddr();
    } else {
        ctx.weightUbAddr = nullptr;
    }

    ctx.nSrcUbAligned = nSrcUbAligned;
    ctx.nDstUbAligned = nDstUbAligned;
    ctx.dim0VfTimes = dim0VfTimes;
    ctx.dim1VfTimes = dim1VfTimes;
    ctx.dim1Tail = dim1Tail;
    ctx.dim1TailTimes = dim1TailTimes;
    ctx.dim1Tail2 = dim1Tail2;
    ctx.mask1Num = mask1Num;
    ctx.mask2Num = mask2Num;
    ctx.mask3Num = mask3Num;

    const auto activation = static_cast<ActMode>(actMode_);
    if (activation == ActMode::SITU) {
        const float invBeta = (activationBeta_ != 0.0f) ? (1.0f / activationBeta_) : 1.0f;
        if (actSubMode_ == static_cast<uint8_t>(ActSubMode::LINEAR)) {
            const float invAlpha = (activationAlpha_ != 0.0f) ? (1.0f / activationAlpha_) : 1.0f;
            const ActivationImpl::SiTUParams situParams{clampLimit_, activationBeta_, invBeta, activationAlpha_,
                                                        invAlpha};
            ActivationImpl::RunSiTU<DataTypeIn, TopkWeightsPrefetch, true, IsInterleaved_>(ctx, situParams);
        } else {
            const ActivationImpl::SiTUParams situParams{clampLimit_, activationBeta_, invBeta, 1.0f, 1.0f};
            ActivationImpl::RunSiTU<DataTypeIn, TopkWeightsPrefetch, false, IsInterleaved_>(ctx, situParams);
        }
    } else if (activation == ActMode::SWIGLU_STEP) {
        const ActivationImpl::SwiGLUParams swiGluParams{clampLimit_};
        ActivationImpl::RunSwiGLU<DataTypeIn, TopkWeightsPrefetch, true, IsInterleaved_>(ctx, swiGluParams);
    } else if (activation == ActMode::SWIGLU_OAI) {
        const ActivationImpl::SwiGLUOaiParams swiGluOaiParams{clampLimit_, activationAlpha_, activationBeta_};
        ActivationImpl::RunSwiGLUOai<DataTypeIn, TopkWeightsPrefetch, IsInterleaved_>(ctx, swiGluOaiParams);
    } else {
        const ActivationImpl::SwiGLUParams swiGluParams{clampLimit_};
        ActivationImpl::RunSwiGLU<DataTypeIn, TopkWeightsPrefetch, false, IsInterleaved_>(ctx, swiGluParams);
    }
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoActivationAndQuantForMX(
    __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst, __ubuf__ DataTypeIn *firstSrc,
    __ubuf__ DataTypeIn *secondSrc, __ubuf__ bfloat16_t *gluResAddr, __ubuf__ uint16_t *maxExpAddr,
    __ubuf__ uint16_t *halfScaleLocalAddr, uint16_t mSize, uint16_t nSize, uint64_t mLoc)
{
    uint32_t nDstUbAligned =
        Ops::Base::CeilAlign(static_cast<uint32_t>(nSize), static_cast<uint32_t>(AscendC::ONE_BLK_SIZE));
    VFDoActivation(firstSrc, secondSrc, gluResAddr, mSize, nSize, nDstUbAligned);

    uint32_t totalDataInUb = mSize * nDstUbAligned;
    uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE;
    uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * 2 - 1) / (vlForHalfNumber_ * 2); // 128
    uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;         // 8
    ComputeMaxExp(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum);                          // 获取最大值
    ComputeScale(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum); // 计算scale和halfScale
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        ComputeDataForQuantTargetFp8(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb,
                                     loopDataNum); // 计算量化后的值
    }
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ComputeDataForQuantTargetFp4(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    return;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoActivationForMX(
    uint16_t mSize, uint16_t pingpongIdx, uint64_t mLoc)
{
    constexpr uint32_t pongElemOf_DataTypeIn = MAX_SINGLE_MN;
    constexpr uint32_t pongElemOf_bf16 = MAX_SINGLE_MN * sizeof(DataTypeIn) / sizeof(bfloat16_t);
    constexpr uint32_t pongElemOf_int8 = MAX_SINGLE_MN * sizeof(DataTypeIn);
    constexpr uint32_t pongElemOf_uint16 = MAX_SINGLE_MN * sizeof(DataTypeIn) / sizeof(uint16_t);
    // 为交织 ping-pong 布局选择当前使用的半 tile 缓冲区。
    const uint32_t pongMul = (IsInterleaved_ && pingpongIdx == 1U) ? 1U : 0U;

    __ubuf__ DataTypeIn *l0cOutUbBase =
        (__ubuf__ DataTypeIn *)l0cOutUbFirst_.GetPhyAddr() + pongMul * pongElemOf_DataTypeIn;
    __ubuf__ bfloat16_t *gluResAddr = (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr() + pongMul * pongElemOf_bf16;
    __ubuf__ int8_t *quantOutputInUbAddr = (__ubuf__ int8_t *)quantOutput_.GetPhyAddr() + pongMul * pongElemOf_int8;
    __ubuf__ uint16_t *quantScaleOutputInUbAddr =
        (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr() + pongMul * pongElemOf_uint16;
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxExp_.GetPhyAddr() + pongMul * pongElemOf_uint16;
    __ubuf__ uint16_t *halfScaleAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr() + pongMul * pongElemOf_uint16;

    __ubuf__ DataTypeIn *l0cOutUbSecondAddr;
    if constexpr (IsInterleaved_) {
        l0cOutUbSecondAddr = l0cOutUbBase + singleN_;
    } else {
        l0cOutUbSecondAddr = (__ubuf__ DataTypeIn *)l0cOutUbSecond_.GetPhyAddr();
    }
    VFDoActivationAndQuantForMX(quantOutputInUbAddr, quantScaleOutputInUbAddr, l0cOutUbBase, l0cOutUbSecondAddr,
                                gluResAddr, maxExpAddr, halfScaleAddr, mSize, singleN_, mLoc);
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetFirstL0c2UbTensor()
{
    return l0cOutUbFirst_;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetSecondL0c2UbTensor()
{
    return l0cOutUbSecond_;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetTopkWeightTensor()
{
    return weightUb_;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(
    const BlockShape &blockShape, const BlockCoord &blockCoord, uint16_t pingpongIdx)
{
    singleM_ = Get<M_VALUE>(blockShape); // 128
    singleN_ = Get<N_VALUE>(blockShape); // 256
    blockCoord_ = blockCoord;

    if (singleM_ == 0) {
        return;
    }

    vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t); // 256 / 2 = 128
    UBBlockSize_ = BLOCK_SIZE;                                         // 32
    elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / UBBlockSize_;    // 256 / 32 = 8

    uint64_t yOffset = Get<Y_IDX>(blockCoord);
    uint64_t yScaleOffset = Get<Y_SCALE_IDX>(blockCoord);
    uint64_t mLoc = Get<M_LOC_IDX>(blockCoord);
    VFDoActivationForMX(singleM_, pingpongIdx, mLoc); // switch(x)*y 计算quant quantScale
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    // scale已按compact布局生成，直接copy到GM，省掉原先TransMxScaleLayout重排scale。
    if constexpr (IsInterleaved_) {
        constexpr uint32_t PONG_INT8_ELEMS = MAX_SINGLE_MN * sizeof(DataTypeIn);
        if (pingpongIdx == 1U) {
            AscendC::LocalTensor<int8_t> quantOutputPong = quantOutput_[PONG_INT8_ELEMS];
            AscendC::LocalTensor<int8_t> quantScalePong = quantScaleOutput_[PONG_INT8_ELEMS];
            CopyOutputFromUb2Gm(singleM_, yOffset, quantOutputPong);
            CopyScaleFromUb2GmCompact(singleM_, yScaleOffset, quantScalePong);
        } else {
            CopyOutputFromUb2Gm(singleM_, yOffset, quantOutput_);
            CopyScaleFromUb2GmCompact(singleM_, yScaleOffset, quantScaleOutput_);
        }
    } else {
        CopyOutputFromUb2Gm(singleM_, yOffset, quantOutput_);
        CopyScaleFromUb2GmCompact(singleM_, yScaleOffset, quantScaleOutput_);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
}

} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // BLOCK_EPILOGUE_ACTIVATION_MX_QUANT_H
