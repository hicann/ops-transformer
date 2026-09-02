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

#ifndef EPILOGUE_BLOCK_EPILOGUE_ACTIVATION_MX_QUANT_H
#define EPILOGUE_BLOCK_EPILOGUE_ACTIVATION_MX_QUANT_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../utils/cgmct_common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tensor_utils.h"
#include "../tile/tile_copy_policy.h"

namespace Cgmct {
namespace Gemm {
namespace Block {

namespace Gmmsg {
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
} // namespace Gmmsg

namespace {
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_SINGLE_MN = 128 * 256;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint32_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t FP8_E5M2_MAX = 0x37924925;
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925;
constexpr uint32_t NUMBER_ZERO = 0x00000000;
constexpr uint32_t NUMBER_TWO_FIVE_FOUR = 0x000000fe;
constexpr uint32_t NUMBER_HALF = 0x00400000;
constexpr float GELU_TANH_NEG_SQRT_EIGHT_OVER_PI = -1.595769121f * 0.044715f;
constexpr float GELU_TANH_APPROX_FACTOR = 1.0f / 0.044715f;
constexpr uint32_t INTERLEAVED_REG_FACTOR = 2;
constexpr uint32_t HALF_REG_FACTOR = 2;
constexpr uint32_t DOUBLE_SPLIT = 2;
constexpr uint32_t SCALE_ALG_OCP = 0;
constexpr float SCALAR_ONE = 1.0f;
} // namespace

static constexpr AscendC::MicroAPI::DivSpecificMode DIV_MODE = {
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    true,
};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
#define QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS \
    template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_, \
              typename DataTypeX1Scale_, bool IsTensorList_>
#define QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS \
    L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, IsTensorList_

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
class BlockEpilogueActivationQuant {
public:
    __aicore__ inline BlockEpilogueActivationQuant() {}

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
        uint32_t roundMode;
        uint32_t scaleAlg;
        float dstTypeMax;
        uint32_t activationType;
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
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline auto GetL0c2UbTensor();
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);

private:
    __aicore__ inline void VFDoActivationForMX(uint16_t mSize);
    __aicore__ inline void TransMxScaleLayout(uint16_t mSize, uint16_t scaleBlockN);

    template <Gmmsg::QuantMode quantMode>
    __aicore__ inline void VFDoActivationAndQuantForMX(__ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
                                                       __ubuf__ DataTypeIn *src, uint16_t mSize, uint16_t nSize);

    __aicore__ inline void ComputeScale(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                        __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB,
                                        uint16_t loopNumScale);

    __aicore__ inline void ComputeMaxExp(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                         uint32_t totalCountInUB, uint16_t loopNum);

    __aicore__ inline void ComputeMaxExpcuBLAS(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                               uint32_t totalCountInUB, uint16_t loopNum);

    __aicore__ inline void ComputeScalecuBLAS(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                              __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB,
                                              uint16_t loopNumScale);

    __aicore__ inline void ComputeDataForQuantTargetFp8(__ubuf__ bfloat16_t *srcAddr,
                                                        __ubuf__ uint16_t *halfScaleLocalAddr,
                                                        __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);

    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src);
    __aicore__ inline void CopyOutputCompactFromUb2Gm(uint64_t blockCount, uint64_t offset,
                                                      AscendC::LocalTensor<int8_t> &src);
    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src);
    __aicore__ inline void CopyScaleCompactFromUb2Gm(uint64_t blockCount, uint64_t offset,
                                                     AscendC::LocalTensor<int8_t> &src);
    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<int8_t> quantScaleGlobal_;

    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> l0cOutUb_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    AscendC::LocalTensor<int8_t> quantOutput_;
    AscendC::LocalTensor<int8_t> quantScaleOutput_;
    AscendC::LocalTensor<int8_t> quantScaleBlockOutput_;
    AscendC::LocalTensor<bfloat16_t> activationRes_;
    AscendC::LocalTensor<uint16_t> maxExp_;
    AscendC::LocalTensor<uint16_t> halfScale_;

    const Params *params_;

    int64_t m_;
    int64_t n_;
    int64_t scaleN_;
    int64_t scaleBlockN_;
    uint32_t subBlockIdx_ = AscendC::GetSubBlockIdx();
    uint32_t singleM_; // cur singleShapeM
    uint32_t singleN_;
    bool isBiasEpilogue_ = false;

    int64_t UBBlockSize_ = 0;
    uint32_t vlForHalfNumber_ = 0;
    uint16_t elementAfterReduce_ = 0;
    uint16_t fpEmax_ = 0;
    uint32_t dtypeMax_ = FP8_E4M3_MAX;

    BlockCoord blockCoord_{0, 0, 0, 0, 0};
};

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(
    Params const &params)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = &params;
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
        dtypeMax_ = FP8_E4M3_MAX;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP;
        dtypeMax_ = FP8_E5M2_MAX;
    } else {
        fpEmax_ = FP8_E5M2_MAX_EXP;
        dtypeMax_ = FP8_E5M2_MAX;
    }

    // out
    constexpr uint32_t afterIn = MAX_SINGLE_MN * sizeof(DataTypeIn);
    quantOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, afterIn, MAX_SINGLE_MN);
    quantScaleOutput_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, afterIn + MAX_SINGLE_MN * sizeof(int8_t), MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t afterIO =
        afterIn + MAX_SINGLE_MN * sizeof(int8_t) + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(int8_t);
    // Activation res
    activationRes_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, afterIO, MAX_SINGLE_MN);
    constexpr uint32_t afterIOAndGlu = afterIO + MAX_SINGLE_MN * sizeof(bfloat16_t);
    // sharedExp
    maxExp_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, afterIOAndGlu,
                                             MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t afterIOAndGluExp = afterIOAndGlu + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    halfScale_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, afterIOAndGluExp,
                                                MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t realScaleBlockOffset =
        afterIOAndGluExp + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    auto mPerVec = CeilDiv(static_cast<uint64_t>(params_->baseM), static_cast<uint64_t>(DOUBLE_SPLIT));
    quantScaleBlockOutput_ =
        AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, realScaleBlockOffset, mPerVec * AscendC::ONE_BLK_SIZE);
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(
    const BlockCoord &baseOffset)
{
    if ASCEND_IS_AIV {
        quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t *)params_->yGmAddr + Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer((__gm__ int8_t *)params_->yScaleGmAddr + Get<Y_SCALE_IDX>(baseOffset));
    }
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape &problemShape)
{
    m_ = Get<MNK_M>(problemShape);
    n_ = Get<MNK_N>(problemShape);
    scaleN_ = CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyOutputFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.blockLen = singleN_ * sizeof(int8_t);
    ub2GmParams.dstStride = (n_ - singleN_) * sizeof(int8_t);
    AscendC::DataCopyPad(quantOutputGlobal_[offset], src, ub2GmParams);
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyOutputCompactFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopy(quantOutputGlobal_[offset], src, blockCount * singleN_);
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyScaleFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    auto blockScaleN =
        CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    ub2GmParams.blockLen = blockScaleN * sizeof(int8_t);
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.dstStride = (scaleN_ - blockScaleN) * sizeof(int8_t);
    AscendC::DataCopyPad(quantScaleGlobal_[offset], src, ub2GmParams);
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyScaleCompactFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t> &src)
{
    AscendC::DataCopy(quantScaleGlobal_[offset], src, blockCount * scaleBlockN_);
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxExp(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract1;

        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg scaleMask1;
        AscendC::MicroAPI::MaskReg scaleMask2;
        AscendC::MicroAPI::UnalignReg u1;
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber_ * INTERLEAVED_REG_FACTOR);
            AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, expMaskBF16,
                                   scaleMask1);
            AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, expMaskBF16,
                                   scaleMask1);

            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, elementAfterReduce_);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeScale(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(maxExpValue, fpEmax_);
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::MaskReg invalidDataMask, specialDataMask;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, vlForHalfNumber_);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask,
                                                                       preMaskScale); // INF/NAN
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue,
                                                                       preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vlForHalfNumber_ / HALF_REG_FACTOR, preMaskScale);

            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, sharedExp, zeroRegTensor,
                                                                       preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias,
                                                                       preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, vlForHalfNumber_, preMaskScale);
        }
    }
    return;
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxExpcuBLAS(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    (void)totalCountInUB;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1;
        AscendC::MicroAPI::RegTensor<uint16_t> absMask;
        AscendC::MicroAPI::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg mask =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignReg ureg;

        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber_ * INTERLEAVED_REG_FACTOR);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0,
                                   (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, absMask, mask);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1,
                                   (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, absMask, mask);
            AscendC::MicroAPI::Max(vdMaxExp, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0,
                                   (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, mask);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, mask);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, ureg, elementAfterReduce_);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeScalecuBLAS(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    (void)totalScaleInUB;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> max16;
        AscendC::MicroAPI::RegTensor<uint32_t> max32;
        AscendC::MicroAPI::RegTensor<uint32_t> exp32;
        AscendC::MicroAPI::RegTensor<uint32_t> man32;
        AscendC::MicroAPI::RegTensor<uint32_t> expAddOne32;
        AscendC::MicroAPI::RegTensor<uint32_t> extractExp;
        AscendC::MicroAPI::RegTensor<uint16_t> expOut;
        AscendC::MicroAPI::RegTensor<uint32_t> halfScale;
        AscendC::MicroAPI::RegTensor<uint16_t> recExpOut;

        AscendC::MicroAPI::RegTensor<uint32_t> invMax;
        AscendC::MicroAPI::Duplicate(invMax, dtypeMax_);
        AscendC::MicroAPI::RegTensor<uint32_t> manMaskFP32;
        AscendC::MicroAPI::Duplicate(manMaskFP32, MAN_MASK_FLOAT);
        AscendC::MicroAPI::RegTensor<uint32_t> expMask;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_FP32);
        AscendC::MicroAPI::RegTensor<uint32_t> zeroRegTensor32;
        AscendC::MicroAPI::Duplicate(zeroRegTensor32, 0);
        AscendC::MicroAPI::RegTensor<uint32_t> scaleBias;
        AscendC::MicroAPI::Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
        AscendC::MicroAPI::RegTensor<uint32_t> nanRegTensor;
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION_PACK);
        AscendC::MicroAPI::RegTensor<uint32_t> fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);

        AscendC::MicroAPI::MaskReg cmpResult;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg p0;
        AscendC::MicroAPI::MaskReg p1;
        AscendC::MicroAPI::MaskReg p2;
        uint32_t halfMaskElemNum = vlForHalfNumber_ / HALF_REG_FACTOR;
        AscendC::MicroAPI::MaskReg dataMaskB16Half = AscendC::MicroAPI::UpdateMask<uint16_t>(halfMaskElemNum);
        AscendC::MicroAPI::MaskReg maskFloat = AscendC::MicroAPI::CreateMask<uint32_t>();

        static constexpr AscendC::MicroAPI::CastTrait castTraitBf162Float = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        for (uint16_t i = 0; i < loopNumScale; i++) {
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                max16, maxExpAddr, vlForHalfNumber_ / HALF_REG_FACTOR);

            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitBf162Float>(
                (AscendC::MicroAPI::RegTensor<float> &)max32, (AscendC::MicroAPI::RegTensor<bfloat16_t> &)max16,
                maskFloat);
            AscendC::MicroAPI::Compare<uint32_t, AscendC::CMPMODE::LT>(cmpResult, max32, expMask, maskFloat);
            AscendC::MicroAPI::Compare<uint32_t, AscendC::CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, maskFloat);

            AscendC::MicroAPI::Mul((AscendC::MicroAPI::RegTensor<float> &)max32,
                                   (AscendC::MicroAPI::RegTensor<float> &)max32,
                                   (AscendC::MicroAPI::RegTensor<float> &)invMax, maskFloat);
            AscendC::MicroAPI::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, maskFloat);
            AscendC::MicroAPI::And(man32, max32, manMaskFP32, maskFloat);

            AscendC::MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(p0, exp32, NUMBER_ZERO, maskFloat);
            AscendC::MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::LT>(p1, exp32, NUMBER_TWO_FIVE_FOUR,
                                                                             maskFloat);
            AscendC::MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(p2, man32, NUMBER_ZERO, maskFloat);
            AscendC::MicroAPI::MaskAnd(p0, p0, p1, maskFloat);
            AscendC::MicroAPI::MaskAnd(p0, p0, p2, maskFloat);
            AscendC::MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::EQ>(p1, exp32, NUMBER_ZERO, maskFloat);
            AscendC::MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(p2, man32, NUMBER_HALF, maskFloat);
            AscendC::MicroAPI::MaskAnd(p1, p1, p2, maskFloat);
            AscendC::MicroAPI::MaskOr(p0, p0, p1, maskFloat);

            AscendC::MicroAPI::Adds(expAddOne32, exp32, 1, maskFloat);
            AscendC::MicroAPI::Select(extractExp, expAddOne32, exp32, p0);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(expOut, extractExp);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr + i * vlForHalfNumber_ / HALF_REG_FACTOR / HALF_REG_FACTOR, expOut, dataMaskB16Half);

            AscendC::MicroAPI::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskFloat);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, extractExp, maskFloat);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(recExpOut, halfScale);
            AscendC::MicroAPI::DataCopy<uint16_t>(halfScaleLocalAddr + i * vlForHalfNumber_ / HALF_REG_FACTOR,
                                                  recExpOut, dataMaskB16Half);
        }
    }
    return;
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp8(
    __ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    (void)totalCountInUB;
    using T = bfloat16_t;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1 = AscendC::MicroAPI::CreateMask<T>();
        AscendC::MicroAPI::MaskReg dataMask2 = AscendC::MicroAPI::CreateMask<T>();
        AscendC::MicroAPI::MaskReg dataMask3 = AscendC::MicroAPI::CreateMask<T>();
        AscendC::MicroAPI::MaskReg dataMask4 = AscendC::MicroAPI::CreateMask<T>();
        AscendC::MicroAPI::MaskReg dataMask5 = AscendC::MicroAPI::CreateMask<DataTypeOut>();
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<DataTypeOut> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;

        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to80 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to81 = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to82 = {
            AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTrait32to83 = {
            AscendC::MicroAPI::RegLayout::THREE, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber_ * INTERLEAVED_REG_FACTOR);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce_);

            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
            AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
            AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
            AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
            AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);

            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to80>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to82>(vdExp0FP8One, vdExp0FP32One, dataMask3);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to81>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
            AscendC::MicroAPI::Cast<DataTypeOut, float, castTrait32to83>(vdExp1FP8One, vdExp1FP32One, dataMask4);

            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8One, dataMask5);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp1FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp1FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp1FP8One, dataMask5);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t> &)vdExp1FP8Zero, dataMask5);

            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP8Zero,
                vlForHalfNumber_ * INTERLEAVED_REG_FACTOR, dataMask5);
        }
    }
    return;
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
template <Gmmsg::QuantMode quantMode>
__aicore__ inline void
BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoActivationAndQuantForMX(
    __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst, __ubuf__ DataTypeIn *src, uint16_t mSize, uint16_t nSize)
{
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    uint16_t OneRowRepeatTimes = CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(sizePerRepeat));
    uint32_t nSrcUbAligned = CeilAlign(nSize, AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn));
    uint32_t nDstUbAligned = CeilAlign(nSize, AscendC::ONE_BLK_SIZE);
    __ubuf__ bfloat16_t *activationResAddr = (__ubuf__ bfloat16_t *)activationRes_.GetPhyAddr();

    // GELU_TANH
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<DataTypeIn>(elementNum);
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
                AscendC::MicroAPI::RegTensor<bfloat16_t> castOut;
                AscendC::MicroAPI::RegTensor<float> input;
                AscendC::MicroAPI::RegTensor<float> inputSqr;
                AscendC::MicroAPI::RegTensor<float> inputCub;
                AscendC::MicroAPI::RegTensor<float> actOut;

                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * sizePerRepeat;
                AscendC::MicroAPI::DataCopy(input, src + l0cOutOffset);
                AscendC::MicroAPI::Mul(inputSqr, input, input, mask);
                AscendC::MicroAPI::Mul(inputCub, inputSqr, input, mask);
                AscendC::MicroAPI::Axpy(inputCub, input, GELU_TANH_APPROX_FACTOR, mask);
                AscendC::MicroAPI::Muls(inputCub, inputCub, GELU_TANH_NEG_SQRT_EIGHT_OVER_PI, mask);
                AscendC::MicroAPI::Exp(inputCub, inputCub, mask);
                AscendC::MicroAPI::Adds(inputCub, inputCub, SCALAR_ONE, mask);
                AscendC::MicroAPI::Div<float, &DIV_MODE>(actOut, input, inputCub, mask);
                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(castOut, actOut, mask);
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * sizePerRepeat;
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    activationResAddr + dstUbOffset, castOut, mask);
            }
        }
    }

    // quant
    uint32_t totalDataInUb = mSize * nDstUbAligned;
    uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE;
    uint16_t loopDataNum =
        (totalDataInUb + vlForHalfNumber_ * INTERLEAVED_REG_FACTOR - 1) / (vlForHalfNumber_ * INTERLEAVED_REG_FACTOR);
    uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - 1) / vlForHalfNumber_;
    uint16_t loopScaleCublasNum =
        (totalScaleInUb + (vlForHalfNumber_ / HALF_REG_FACTOR) - 1) / (vlForHalfNumber_ / HALF_REG_FACTOR);
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)maxExp_.GetPhyAddr();
    __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)halfScale_.GetPhyAddr();
    if (params_->scaleAlg == SCALE_ALG_OCP) {
        ComputeMaxExp(activationResAddr, maxExpAddr, totalDataInUb, loopDataNum);
        ComputeScale(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);
    } else {
        ComputeMaxExpcuBLAS(activationResAddr, maxExpAddr, totalDataInUb, loopDataNum);
        ComputeScalecuBLAS(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleCublasNum);
    }
    ComputeDataForQuantTargetFp8(activationResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoActivationForMX(
    uint16_t mSize)
{
    __ubuf__ int8_t *quantOutputInUbAddr = (__ubuf__ int8_t *)quantOutput_.GetPhyAddr();
    __ubuf__ uint16_t *quantScaleOutputInUbAddr = (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ DataTypeIn *l0cOutUbAddr = (__ubuf__ DataTypeIn *)l0cOutUb_.GetPhyAddr();
    VFDoActivationAndQuantForMX<Gmmsg::QuantMode::MX_PERGROUP_MODE>(quantOutputInUbAddr, quantScaleOutputInUbAddr,
                                                                    l0cOutUbAddr, mSize, singleN_);
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::TransMxScaleLayout(
    uint16_t mSize, uint16_t scaleBlockN)
{
    __ubuf__ int8_t *quantScaleOutputInUbAddr = (__ubuf__ int8_t *)quantScaleOutput_.GetPhyAddr();
    __ubuf__ int8_t *quantScaleBlockOutputInUbAddr = (__ubuf__ int8_t *)quantScaleBlockOutput_.GetPhyAddr();
    AscendC::Duplicate<int8_t>(quantScaleBlockOutput_, 0, mSize * AscendC::ONE_BLK_SIZE);
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            uint32_t elemNum = scaleBlockN;
            AscendC::MicroAPI::MaskReg maskScaleN = AscendC::MicroAPI::UpdateMask<int8_t>(elemNum);
            AscendC::MicroAPI::RegTensor<int8_t> vreg0;
            AscendC::MicroAPI::UnalignReg u0, u1;
            auto srcUb = quantScaleOutputInUbAddr + mIdx * scaleBlockN;
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, srcUb);
            AscendC::MicroAPI::DataCopyUnAlign(vreg0, u0, srcUb);
            auto dstUb = quantScaleBlockOutputInUbAddr + mIdx * AscendC::ONE_BLK_SIZE;
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(dstUb, vreg0, maskScaleN);
        }
    }
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetL0c2UbTensor()
{
    return l0cOutUb_;
}

QMM_BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(
    const BlockShape &blockShape, const BlockCoord &blockCoord)
{
    singleM_ = Get<MNK_M>(blockShape);
    singleN_ = Get<MNK_N>(blockShape);
    scaleBlockN_ =
        CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    blockCoord_ = blockCoord;
    auto singleMPerVec = CeilDiv(static_cast<uint64_t>(singleM_), static_cast<uint64_t>(AscendC::GetTaskRation()));
    uint64_t mOffset = subBlockIdx_ * singleMPerVec;
    if (mOffset >= singleM_) {
        return;
    }
    uint64_t singleMInVec = Min(static_cast<uint64_t>(singleM_) - mOffset, singleMPerVec);
    vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    UBBlockSize_ = BLOCK_SIZE;
    elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / UBBlockSize_;

    VFDoActivationForMX(singleMInVec);
    uint64_t yOffset = Get<Y_IDX>(blockCoord) + mOffset * n_;
    uint64_t yScaleOffset = Get<Y_SCALE_IDX>(blockCoord) + mOffset * scaleN_;
    TransMxScaleLayout(singleMInVec, scaleBlockN_);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(singleMInVec, yOffset, quantOutput_);
    CopyScaleFromUb2Gm(singleMInVec, yScaleOffset, quantScaleBlockOutput_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}
} // namespace Block
} // namespace Gemm
} // namespace Cgmct

#endif // EPILOGUE_BLOCK_EPILOGUE_ACTIVATION_MX_QUANT_H
#endif
