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
 * \file block_epilogue_swiglu_mx_quant.h
 * \brief
 */

#ifndef EPILOGUE_BLOCK_EPILOGUE_SWIGLU_MX_QUANT_H
#define EPILOGUE_BLOCK_EPILOGUE_SWIGLU_MX_QUANT_H
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
#include "mx_quant_reduce.h"

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
constexpr float DEFAULT_CLAMP_LIMIT = 7.0F;
constexpr float DEFAULT_GLU_ALPHA = 1.702F;
constexpr float DEFAULT_GLU_BIAS = 1.0F;
constexpr uint16_t SINGLE_DMA_BLOCK = 1;
constexpr uint16_t EXPONENT_ROUND_INCREMENT = 1;
constexpr uint32_t VECTOR_CEIL_ADJUSTMENT = 1;
constexpr uint32_t PACKED_HALF_LANE_SHIFT = 1;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint64_t MX_QUANT_COMPUTE_ALIGN = 64UL; // MX量化计算的对齐约束: N轴需按64对齐处理
constexpr uint64_t DATA_BLOCK_SIZE = 32UL;        // UB上一个DataBlock的大小，即32字节。
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_SINGLE_MN = 64 * 256;
constexpr uint32_t CGMCT_INTERLEAVED_REG_FACTOR = 2;
constexpr uint32_t PADDED_QUANT_SCALE_SIZE = 64 * 32;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t BF16_ABSOLUTE_VALUE_MASK = 0x7fff;
constexpr uint16_t BF16_MANTISSA_MASK = 0x007f;
constexpr uint16_t E4M3FN_MAX_MANTISSA_THRESHOLD = 0x0060;
constexpr uint16_t E4M3FN_MAX_EXPONENT = 8;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
} // namespace

constexpr AscendC::Reg::CastTrait ctInt322Fp32 = {AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN,
                                                  AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait ctFp322Half = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                 AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait ctHalf2Fp32Zero = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait ctHalf2Fp32One = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
                                                    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

static constexpr AscendC::Reg::DivSpecificMode DIV_MODE = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    true,
};
static constexpr AscendC::Reg::CastTrait CAST_BF16_FP16_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait CAST_FP32_TO_BF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
#define QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS \
    template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_, \
              typename DataTypeX1Scale_, bool IsTensorList_>
#define QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS \
    L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, IsTensorList_

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
class BlockEpilogueSwigluQuant {
public:
    __aicore__ inline BlockEpilogueSwigluQuant() {}

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        // V3 SwiGLU controls.  The defaults preserve the V2 standard SwiGLU path.
        int64_t swigluMode{0};
        float clampLimit{DEFAULT_CLAMP_LIMIT};
        float gluAlpha{DEFAULT_GLU_ALPHA};
        float gluBias{DEFAULT_GLU_BIAS};
        uint32_t scaleAlg{0};
        float dstTypeMax{0.0F};
        Arguments() = default;
    };

    // params
    using Params = Arguments;

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeX1Scale = DataTypeX1Scale_;
    using DataTypeX2Scale = DataTypeX2Scale_;

    // 输出为FP4类型(e2m1/e1m2)的编译期标记
    static constexpr bool IS_FP4_OUT =
        AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value || AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value;

    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BaseOffset = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t>; // y, yScale, x2Scale, x1Scale, bias
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const& params);
    __aicore__ inline auto GetFirstL0c2UbTensor();
    __aicore__ inline auto GetSecondL0c2UbTensor();
    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape);

private:
    __aicore__ inline void VFDoSwigluForMX(uint16_t mSize);
    __aicore__ inline void TransMxScaleLayout(uint16_t mSize, uint16_t scaleBlockN);

    template <Gmmsg::QuantMode quantMode>
    __aicore__ inline void VFDoSwigluAndQuantForMX(__ubuf__ int8_t* outputDst, __ubuf__ uint16_t* scaleDst,
                                                   __ubuf__ DataTypeIn* firstSrc, __ubuf__ DataTypeIn* secondSrc,
                                                   uint16_t mSize, uint16_t nSize);

    __aicore__ inline void ComputeScale(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                        __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB,
                                        uint16_t loopNumScale);

    __aicore__ inline void ComputeMaxExp(__ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* maxExpAddr,
                                         uint32_t totalCountInUB, uint16_t loopNum);

    __aicore__ inline void ComputeMaxExpCublas(__ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* maxExpAddr,
                                               uint32_t totalCountInUB, uint16_t loopNum);

    __aicore__ inline void ComputeScaleCublasFromBf16(__ubuf__ uint16_t* maxValueAddr,
                                                      __ubuf__ uint16_t* mxScaleLocalAddr,
                                                      __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB,
                                                      uint16_t loopNumScale);

    __aicore__ inline void ComputeDataForQuantTargetFp8(__ubuf__ bfloat16_t* srcAddr,
                                                        __ubuf__ uint16_t* halfScaleLocalAddr,
                                                        __ubuf__ int8_t* outLocalAddr, uint16_t loopNum);

    __aicore__ inline void ComputeDataForQuantTargetFp4(__ubuf__ bfloat16_t* srcAddr,
                                                        __ubuf__ uint16_t* halfScaleLocalAddr,
                                                        __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB,
                                                        uint16_t loopNum);

    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t>& src);

    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, uint64_t offset,
                                              AscendC::LocalTensor<AscendC::fp8_e8m0_t>& src);
    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> quantScaleGlobal_;

    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    AscendC::LocalTensor<DataTypeIn> l0cOutUbSecond_{AscendC::TPosition::VECIN, MAX_SINGLE_MN * sizeof(DataTypeIn),
                                                     MAX_SINGLE_MN};
    AscendC::LocalTensor<int8_t> quantOutput_;
    AscendC::LocalTensor<int8_t> quantScaleOutput_;
    AscendC::LocalTensor<AscendC::fp8_e8m0_t> quantScaleBlockOutput_;
    AscendC::LocalTensor<bfloat16_t> gluRes_;
    AscendC::LocalTensor<uint16_t> maxExp_;
    AscendC::LocalTensor<uint16_t> halfScale_;

    const Params* params_;

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

    int64_t swigluMode_ = 0;
    float clampLimit_ = DEFAULT_CLAMP_LIMIT;
    float gluAlpha_ = DEFAULT_GLU_ALPHA;
    float gluBias_ = DEFAULT_GLU_BIAS;

    BlockCoord blockCoord_{0, 0, 0, 0, 0};
};

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(
    Params const& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = &params;
    swigluMode_ = params.swigluMode;
    clampLimit_ = params.clampLimit;
    gluAlpha_ = params.gluAlpha;
    gluBias_ = params.gluBias;
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value) {
        fpEmax_ = FP8_E4M3_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        fpEmax_ = FP8_E5M2_MAX_EXP;
    } else if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value) {
        fpEmax_ = FP4_E2M1_MAX_EXP;
    } else {
        fpEmax_ = FP4_E1M2_MAX_EXP;
    }

    // out
    constexpr uint32_t afterIn = 2 * MAX_SINGLE_MN * sizeof(DataTypeIn); // 2: 2 block in
    quantOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, afterIn, MAX_SINGLE_MN);
    quantScaleOutput_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECOUT, afterIn + MAX_SINGLE_MN * sizeof(int8_t), MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t afterIO =
        afterIn + MAX_SINGLE_MN * sizeof(int8_t) + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(int8_t);
    // swiglu res
    gluRes_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, afterIO, MAX_SINGLE_MN);
    constexpr uint32_t afterIOAndGlu = afterIO + MAX_SINGLE_MN * sizeof(bfloat16_t);
    // sharedExp
    maxExp_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, afterIOAndGlu,
                                             MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t afterIOAndGluExp = afterIOAndGlu + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    halfScale_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, afterIOAndGluExp,
                                                MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t realScaleBlockOffset =
        afterIOAndGluExp + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(uint16_t);
    quantScaleBlockOutput_ =
        AscendC::LocalTensor<AscendC::fp8_e8m0_t>(AscendC::TPosition::VECOUT, realScaleBlockOffset,
                                                  params_->baseM / AscendC::GetTaskRation() * AscendC::ONE_BLK_SIZE);
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(
    const BlockCoord& baseOffset)
{
    if ASCEND_IS_AIV {
        quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t*)params_->yGmAddr + Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer((__gm__ AscendC::fp8_e8m0_t*)params_->yScaleGmAddr +
                                          Get<Y_SCALE_IDX>(baseOffset));
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape& problemShape)
{
    n_ = Get<MNK_N>(problemShape);
    scaleN_ = CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyOutputFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t>& src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.blockLen = singleN_ * sizeof(int8_t);
    ub2GmParams.dstStride = (n_ - singleN_) * sizeof(int8_t);

    if (unlikely(singleN_ % MX_QUANT_COMPUTE_ALIGN != 0)) {
        uint64_t actualDataBlockNum = CeilDiv(static_cast<uint64_t>(singleN_), DATA_BLOCK_SIZE);
        uint64_t alignedDataBlockNum = CeilDiv(Align64(static_cast<uint64_t>(singleN_)), DATA_BLOCK_SIZE);
        // UB中每行按64对齐存放，srcStride用于跳过行尾补齐的DataBlock: 对齐后块数 - 原始singleN块数
        ub2GmParams.srcStride = (alignedDataBlockNum - actualDataBlockNum);
        if constexpr (IS_FP4_OUT) {
            ub2GmParams.srcStride = ub2GmParams.srcStride >> 1;
        }
    }

    if constexpr (IS_FP4_OUT) {
        ub2GmParams.blockLen = ub2GmParams.blockLen >> 1;
        ub2GmParams.dstStride = ub2GmParams.dstStride >> 1;
        offset = offset >> 1;
    }
    AscendC::DataCopyPad(quantOutputGlobal_[offset], src, ub2GmParams);
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyScaleFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<AscendC::fp8_e8m0_t>& src)
{
    // scaleBlockN_ is the UB source row pitch (it is based on Align64(singleN)
    // because ComputeMxQuant also computes the aligned tail).  It is not the
    // number of valid scale bytes in the public yScale row.  Copy only the
    // valid groups; otherwise a 32-value tile would copy its padding byte and
    // the following tile would be addressed past the compact yScale row.
    constexpr uint64_t scaleGroupSize = MXFP_DIVISOR_SIZE / MXFP_MULTI_BASE_SIZE;
    auto validScaleN = CeilDiv(static_cast<uint64_t>(singleN_), scaleGroupSize);
    // quantScaleBlockOutput_ uses a 32-byte UB row for every logical M row,
    // whereas yScale is compact (validScaleN bytes per row).  A multi-row
    // DataCopyPad with a sub-32-byte blockLen expresses its source stride in
    // hardware blocks, so it cannot represent this 32-byte-to-compact-row
    // conversion without rounding ambiguity.  Copy each row from its explicit
    // 32-byte UB address instead.  This keeps the public yScale rows compact
    // for all N tiles, including the common N=64 output tile.
    AscendC::DataCopyExtParams params{SINGLE_DMA_BLOCK, 0, 0, 0, 0};
    params.blockCount = SINGLE_DMA_BLOCK;
    params.blockLen = static_cast<uint32_t>(validScaleN * sizeof(AscendC::fp8_e8m0_t));
    for (uint64_t row = 0; row < blockCount; ++row) {
        AscendC::DataCopyPad(quantScaleGlobal_[offset + row * scaleN_], src[row * AscendC::ONE_BLK_SIZE], params);
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxExp(
    __ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    MxQuantDetail::ReduceBf16MaxExponent(srcAddr, maxExpAddr, totalCountInUB, loopNum, vlForHalfNumber_,
                                         elementAfterReduce_);
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeScale(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp0, vdExp1;
        AscendC::Reg::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
        AscendC::Reg::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
        AscendC::Reg::Duplicate(maxExpValue, fpEmax_);
        AscendC::Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::Reg::Duplicate(zeroRegTensor, 0);
        AscendC::Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::Reg::MaskReg invalidDataMask, specialDataMask;
        AscendC::Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr,
                                                                                          vlForHalfNumber_);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask,
                                                                  preMaskScale); // INF/NAN
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                           vlForHalfNumber_ >> 1, preMaskScale);

            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, vlForHalfNumber_, preMaskScale);
        }
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxExpCublas(
    __ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<bfloat16_t> value0;
        AscendC::Reg::RegTensor<bfloat16_t> value1;
        AscendC::Reg::RegTensor<uint16_t> absMask;
        AscendC::Reg::RegTensor<uint16_t> maxValue;
        AscendC::Reg::MaskReg mask0;
        AscendC::Reg::MaskReg mask1;
        AscendC::Reg::MaskReg maskEven;
        AscendC::Reg::MaskReg maskOdd;
        AscendC::Reg::UnalignRegForStore unalign;
        AscendC::Reg::Duplicate(absMask, BF16_ABSOLUTE_VALUE_MASK);
        for (uint16_t i = 0; i < loopNum; ++i) {
            mask0 = AscendC::Reg::UpdateMask<bfloat16_t>(totalCountInUB);
            mask1 = AscendC::Reg::UpdateMask<bfloat16_t>(totalCountInUB);
            AscendC::Reg::MaskDeInterleave<bfloat16_t>(maskEven, maskOdd, mask0, mask1);
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                    AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                value0, value1, srcAddr, vlForHalfNumber_ * CGMCT_INTERLEAVED_REG_FACTOR);
            AscendC::Reg::And(reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value0),
                              reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value0), absMask, maskEven);
            AscendC::Reg::And(reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value1),
                              reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value1), absMask, maskOdd);
            AscendC::Reg::Max(maxValue, reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value0),
                              reinterpret_cast<AscendC::Reg::RegTensor<uint16_t>&>(value1), mask0);
            AscendC::Reg::ReduceMaxWithDataBlock(maxValue, maxValue, mask0);
            AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, maxValue, unalign, elementAfterReduce_);
        }
        AscendC::Reg::StoreUnAlignPost(maxExpAddr, unalign, 0);
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp8(
    __ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr,
    uint16_t loopNum)
{
    using T = bfloat16_t;
    using U = DataTypeOut;
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg maskAllB8 = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
        AscendC::Reg::RegTensor<T> vdExp0, vdExp1;
        AscendC::Reg::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
        AscendC::Reg::RegTensor<U> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;
        static constexpr AscendC::Reg::CastTrait castTraitZero = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::Reg::CastTrait castTraitOne = {
            AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::Reg::CastTrait castFp32ToFp8LayoutZero = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::Reg::CastTrait castFp32ToFp8LayoutOne = {
            AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::Reg::CastTrait castFp32ToFp8LayoutTwo = {
            AscendC::Reg::RegLayout::TWO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::Reg::CastTrait castFp32ToFp8LayoutThree = {
            AscendC::Reg::RegLayout::THREE, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::Reg::DataCopy<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber_ * CGMCT_INTERLEAVED_REG_FACTOR);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                         elementAfterReduce_);

            AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, maskAll);
            AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, maskAll);
            AscendC::Reg::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, maskAll);
            AscendC::Reg::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, maskAll);
            AscendC::Reg::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, maskAll);
            AscendC::Reg::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, maskAll);
            AscendC::Reg::Cast<U, float, castFp32ToFp8LayoutZero>(vdExp0FP8Zero, vdExp0FP32Zero, maskAll);
            AscendC::Reg::Cast<U, float, castFp32ToFp8LayoutTwo>(vdExp0FP8One, vdExp0FP32One, maskAll);
            AscendC::Reg::Cast<U, float, castFp32ToFp8LayoutOne>(vdExp1FP8Zero, vdExp1FP32Zero, maskAll);
            AscendC::Reg::Cast<U, float, castFp32ToFp8LayoutThree>(vdExp1FP8One, vdExp1FP32One, maskAll);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8One, maskAllB8);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp1FP8Zero, maskAllB8);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp0FP8Zero,
                              (AscendC::Reg::RegTensor<uint8_t>&)vdExp1FP8One, maskAllB8);
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_NORM_B8>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP8Zero,
                vlForHalfNumber_ * CGMCT_INTERLEAVED_REG_FACTOR, maskAllB8);
        }
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp4(
    __ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr,
    uint32_t totalCountInUB, uint16_t loopNum)
{
    using T = bfloat16_t;
    using U = DataTypeOut;
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg dataMask1;
        AscendC::Reg::MaskReg dataMask2;
        AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
        AscendC::Reg::RegTensor<T> vdExp0;
        AscendC::Reg::RegTensor<T> vdExp1;
        AscendC::Reg::RegTensor<T> vdExp0Convert;
        AscendC::Reg::RegTensor<T> vdExp1Convert;

        AscendC::Reg::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::Reg::RegTensor<bfloat16_t> vdExp1BF16;

        AscendC::Reg::RegTensor<U> vdExp0FP4;
        AscendC::Reg::RegTensor<U> vdExp1FP4;

        static constexpr AscendC::Reg::CastTrait castTrait = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            dataMask2 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
            AscendC::Reg::DataCopy<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr,
                vlForHalfNumber_ * 2); // copy two chunks from srcAddr to regbase
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                         elementAfterReduce_);

            AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
            AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
            AscendC::Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::Reg::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
            AscendC::Reg::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask2);

            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::Reg::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMask2);
        }
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
template <Gmmsg::QuantMode quantMode>
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluAndQuantForMX(
    __ubuf__ int8_t* outputDst, __ubuf__ uint16_t* scaleDst, __ubuf__ DataTypeIn* firstSrc,
    __ubuf__ DataTypeIn* secondSrc, uint16_t mSize, uint16_t nSize)
{
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    uint16_t OneRowRepeatTimes = CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(sizePerRepeat));
    uint32_t nSrcUbAligned = CeilAlign(nSize, AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn));
    uint32_t nDstUbAligned64 = Align64(nSize); // 64 对齐

    const float scalarOne = 1.0;
    __ubuf__ bfloat16_t* gluResAddr = (__ubuf__ bfloat16_t*)gluRes_.GetPhyAddr();

    if (unlikely(nSize % MX_QUANT_COMPUTE_ALIGN != 0)) {
        AscendC::Duplicate<bfloat16_t>(gluRes_, 0, mSize * nDstUbAligned64);
    }

    // swiglu
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) { // 需要计算m次
            AscendC::Reg::MaskReg mask;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) { // 每次计算m=1, n=64的数据大小
                uint32_t outputOffset = static_cast<uint32_t>(vfBlockIdx) * sizePerRepeat;
                uint32_t remaining = nSize - outputOffset;
                uint32_t elementNum = remaining < sizePerRepeat ? remaining : sizePerRepeat;
                // The vector computation is performed in the MMAD accumulator
                // type (FP32); keep this mask in the accumulator lane domain
                // so the PACK_B32 store uses the same register layout as the
                // existing MX quantization epilogues.
                mask = AscendC::Reg::UpdateMask<DataTypeIn>(elementNum);
                AscendC::Reg::RegTensor<bfloat16_t> verg7;
                AscendC::Reg::RegTensor<float> swishInput, gateInput;
                AscendC::Reg::RegTensor<float> verg1, verg2, verg3, verg4, verg6, swishOutput;

                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + outputOffset;
                AscendC::Reg::DataCopy(swishInput, firstSrc + l0cOutOffset);
                AscendC::Reg::DataCopy(gateInput, secondSrc + l0cOutOffset);

                if (swigluMode_ == 0) {
                    AscendC::Reg::Muls(verg2, swishInput, -(scalarOne), mask);
                    AscendC::Reg::Exp(verg3, verg2, mask);
                    AscendC::Reg::Adds(verg4, verg3, scalarOne, mask);
                    AscendC::Reg::Div<float, &DIV_MODE>(swishOutput, swishInput, verg4, mask);
                } else {
                    AscendC::Reg::Mins(verg1, swishInput, clampLimit_, mask);
                    AscendC::Reg::Muls(verg2, verg1, -gluAlpha_, mask);
                    AscendC::Reg::Exp(verg3, verg2, mask);
                    AscendC::Reg::Adds(verg4, verg3, scalarOne, mask);
                    AscendC::Reg::Div<float, &DIV_MODE>(swishOutput, verg1, verg4, mask);
                    AscendC::Reg::Mins(gateInput, gateInput, clampLimit_, mask);
                    AscendC::Reg::Maxs(gateInput, gateInput, -clampLimit_, mask);
                    AscendC::Reg::Adds(gateInput, gateInput, gluBias_, mask);
                }

                AscendC::Reg::Mul(verg6, swishOutput, gateInput, mask);

                AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(verg7, verg6, mask);
                uint32_t dstUbOffset = mIdx * nDstUbAligned64 + vfBlockIdx * sizePerRepeat;
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(gluResAddr + dstUbOffset,
                                                                                           verg7, mask);
            }
        }
    }

    // quant
    uint32_t totalDataInUb = mSize * nDstUbAligned64; // nDstUbAligned64 aligned by 64
    uint32_t totalScaleInUb = totalDataInUb / AscendC::ONE_BLK_SIZE;
    uint16_t loopDataNum = (totalDataInUb + vlForHalfNumber_ * 2 - 1) / (vlForHalfNumber_ * 2);
    uint16_t loopScaleNum = (totalScaleInUb + vlForHalfNumber_ - VECTOR_CEIL_ADJUSTMENT) / vlForHalfNumber_;
    __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)maxExp_.GetPhyAddr();
    __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)halfScale_.GetPhyAddr();
    if (params_->scaleAlg == 0) {
        ComputeMaxExp(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum);
        ComputeScale(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, loopScaleNum);
    } else {
        ComputeMaxExpCublas(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum);
        const uint16_t cublasLoopScaleNum =
            static_cast<uint16_t>((totalScaleInUb + vlForHalfNumber_ - VECTOR_CEIL_ADJUSTMENT) / vlForHalfNumber_);
        ComputeScaleCublasFromBf16(maxExpAddr, scaleDst, halfScaleLocalAddr, totalScaleInUb, cublasLoopScaleNum);
    }
    if constexpr (AscendC::IsSameType<DataTypeOut, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        ComputeDataForQuantTargetFp8(gluResAddr, halfScaleLocalAddr, outputDst, loopDataNum);
    }
    if constexpr (IS_FP4_OUT) {
        ComputeDataForQuantTargetFp4(gluResAddr, halfScaleLocalAddr, outputDst, totalDataInUb, loopDataNum);
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeScaleCublasFromBf16(
    __ubuf__ uint16_t* maxValueAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> maxValue, exponent, mantissa, roundedExponent;
        AscendC::Reg::RegTensor<uint16_t> scaleValue, shiftedScale, reciprocalValue;
        AscendC::Reg::RegTensor<uint16_t> mantissaMask, roundThreshold, exponentOffset;
        AscendC::Reg::RegTensor<uint16_t> reciprocalBias, zero, fp8Nan, nan;
        AscendC::Reg::MaskReg nonzeroMask, roundMask, normalScaleMask, infNanMask, mask;
        AscendC::Reg::Duplicate(mantissaMask, BF16_MANTISSA_MASK);
        AscendC::Reg::Duplicate(roundThreshold, E4M3FN_MAX_MANTISSA_THRESHOLD);
        AscendC::Reg::Duplicate(exponentOffset, E4M3FN_MAX_EXPONENT);
        AscendC::Reg::Duplicate(reciprocalBias, BF16_EXP_BIAS);
        AscendC::Reg::Duplicate(zero, static_cast<uint16_t>(0));
        AscendC::Reg::Duplicate(fp8Nan, MAX_EXP_FOR_FP8);
        AscendC::Reg::Duplicate(nan, NAN_CUSTOMIZATION);

        for (uint16_t i = 0; i < loopNumScale; ++i) {
            mask = AscendC::Reg::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(maxValue, maxValueAddr,
                                                                                          vlForHalfNumber_);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(nonzeroMask, maxValue, zero, mask);
            AscendC::Reg::ShiftRights(exponent, maxValue, static_cast<int16_t>(SHR_NUM_FOR_BF16), mask);
            // Classify the original BF16 exponent before finite values can round up to 0xff.
            AscendC::Reg::CompareScalar<uint16_t, AscendC::CMPMODE::EQ>(infNanMask, exponent, MAX_EXP_FOR_FP8, mask);
            AscendC::Reg::And(mantissa, maxValue, mantissaMask, mask);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::GT>(roundMask, mantissa, roundThreshold, mask);
            AscendC::Reg::Adds(roundedExponent, exponent, EXPONENT_ROUND_INCREMENT, mask);
            AscendC::Reg::Select<uint16_t>(exponent, roundedExponent, exponent, roundMask);
            AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::GE>(normalScaleMask, exponent, exponentOffset, mask);
            AscendC::Reg::Sub(scaleValue, exponent, exponentOffset, mask);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zero, normalScaleMask);
            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zero, nonzeroMask);
            AscendC::Reg::Select<uint16_t>(scaleValue, fp8Nan, scaleValue, infNanMask);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vlForHalfNumber_ >> PACKED_HALF_LANE_SHIFT, mask);
            AscendC::Reg::ShiftLefts(shiftedScale, scaleValue, static_cast<int16_t>(SHR_NUM_FOR_BF16), mask);
            AscendC::Reg::Sub(reciprocalValue, reciprocalBias, shiftedScale, mask);
            AscendC::Reg::Select<uint16_t>(reciprocalValue, reciprocalValue, zero, nonzeroMask);
            AscendC::Reg::Select<uint16_t>(reciprocalValue, nan, reciprocalValue, infNanMask);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, reciprocalValue, vlForHalfNumber_, mask);
        }
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluForMX(
    uint16_t mSize)
{
    __ubuf__ int8_t* quantOutputInUbAddr = (__ubuf__ int8_t*)quantOutput_.GetPhyAddr();
    __ubuf__ uint16_t* quantScaleOutputInUbAddr = (__ubuf__ uint16_t*)quantScaleOutput_.GetPhyAddr();
    __ubuf__ DataTypeIn* l0cOutUbFirstAddr = (__ubuf__ DataTypeIn*)l0cOutUbFirst_.GetPhyAddr();
    __ubuf__ DataTypeIn* l0cOutUbSecondAddr = (__ubuf__ DataTypeIn*)l0cOutUbSecond_.GetPhyAddr();
    VFDoSwigluAndQuantForMX<Gmmsg::QuantMode::MX_PERGROUP_MODE>(quantOutputInUbAddr, quantScaleOutputInUbAddr,
                                                                l0cOutUbFirstAddr, l0cOutUbSecondAddr, mSize, singleN_);
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::TransMxScaleLayout(
    uint16_t mSize, uint16_t scaleBlockN)
{
    __ubuf__ int8_t* quantScaleOutputInUbAddr = (__ubuf__ int8_t*)quantScaleOutput_.GetPhyAddr();
    __ubuf__ int8_t* quantScaleBlockOutputInUbAddr = (__ubuf__ int8_t*)quantScaleBlockOutput_.GetPhyAddr();
    // scale layout: (mSize*8) -> (mSize,32)
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            constexpr uint64_t scaleGroupSize = MXFP_DIVISOR_SIZE / MXFP_MULTI_BASE_SIZE;
            uint32_t elemNum = CeilDiv(static_cast<uint64_t>(singleN_), scaleGroupSize);
            AscendC::Reg::MaskReg maskScaleN = AscendC::Reg::UpdateMask<int8_t>(elemNum);
            AscendC::Reg::RegTensor<int8_t> vreg0;
            AscendC::Reg::UnalignReg u0, u1;
            auto srcUb = quantScaleOutputInUbAddr + mIdx * scaleBlockN;
            AscendC::Reg::DataCopyUnAlignPre(u0, srcUb);
            AscendC::Reg::DataCopyUnAlign(vreg0, u0, srcUb);
            auto dstUb = quantScaleBlockOutputInUbAddr + mIdx * AscendC::ONE_BLK_SIZE;
            AscendC::Reg::DataCopy<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(dstUb, vreg0, maskScaleN);
        }
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetFirstL0c2UbTensor()
{
    return l0cOutUbFirst_;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetSecondL0c2UbTensor()
{
    return l0cOutUbSecond_;
}

QMM_BLOCK_EPILOGUE_SWIGLU_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluQuant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(
    const BlockShape& blockShape, const BlockCoord& blockCoord)
{
    singleM_ = Get<MNK_M>(blockShape);
    singleN_ = Get<MNK_N>(blockShape);
    // ComputeMxQuant uses Align64(singleN_) as its UB row pitch and emits one
    // packed scale byte for each 32 values. Keep the source row pitch
    // separate from the number of valid output scales copied to GM.
    // ComputeScale emits one byte per 32 values. The public MX scale row
    // packs two such bytes for every 64 output values, so the UB source row
    // pitch is ceil(Align64(N), 64) * 2 rather than ceil(N, 64).
    scaleBlockN_ = CeilDiv(Align64(static_cast<uint64_t>(singleN_)), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                   MXFP_MULTI_BASE_SIZE;
    blockCoord_ = blockCoord;
    auto halfSingleM = CeilDiv(static_cast<uint64_t>(singleM_), static_cast<uint64_t>(AscendC::GetTaskRation()));
    uint64_t singleMInVec = subBlockIdx_ == 1 ? singleM_ - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = subBlockIdx_ * halfSingleM;

    vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    UBBlockSize_ = BLOCK_SIZE;
    elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / UBBlockSize_;

    VFDoSwigluForMX(singleMInVec);
    uint64_t yOffset = Get<Y_IDX>(blockCoord) + subBlockIdx_ * halfSingleM * n_;
    uint64_t yScaleOffset = Get<Y_SCALE_IDX>(blockCoord) + subBlockIdx_ * halfSingleM * scaleN_;
    TransMxScaleLayout(singleMInVec, scaleBlockN_);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    CopyOutputFromUb2Gm(singleMInVec, yOffset, quantOutput_);
    CopyScaleFromUb2Gm(singleMInVec, yScaleOffset, quantScaleBlockOutput_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    return;
}
} // namespace Block
} // namespace Gemm
} // namespace Cgmct

#endif // EPILOGUE_BLOCK_EPILOGUE_SWIGLU_QUANT_H
#endif
