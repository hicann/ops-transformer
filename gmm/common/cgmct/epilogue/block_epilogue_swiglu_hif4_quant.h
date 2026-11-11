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
 * \file block_epilogue_swiglu_hif4_quant.h
 * \brief
 */

#ifndef EPILOGUE_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_H
#define EPILOGUE_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../utils/cgmct_common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/tensor_utils.h"
#include "../tile/tile_copy_policy.h"

namespace Cgmct {
namespace Gemm {
namespace Block {

namespace {
constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
constexpr int64_t DIGIT_TWO = 2L;
constexpr uint32_t MAX_SINGLE_MN = 128 * 256;
constexpr uint32_t ELEMS_PER_QUANT_GROUP = 64;        // 每组64个bf16元素
constexpr uint32_t GROUP_COUNT_ALIGN = 64;            // 组数对齐粒度
constexpr uint16_t PER4_MAX_GROUPS_PER_ITER = 8;      // ComputeMaxPer4每次处理8组
constexpr uint16_t PER8_MAX_BUF_BYTES_PER_ITER = 512; // ComputeMaxPer8每次处理的uint32字节数
constexpr uint16_t SCALE_BUF_BYTES_PER_ITER = 512;    // ComputeScale每次处理的float字节数
constexpr uint16_t FP4_QUANT_ELEMS_PER_ITER = 128;    // QuantFp4每次处理128个bf16
constexpr uint32_t MAX_ALIGN_GROUP_COUNT = 256;

constexpr uint32_t E8_COUNT_PER_GROUP = 2;
constexpr uint32_t E1P8_COUNT_PER_GROUP = 4;
constexpr uint32_t E1P16_COUNT_PER_GROUP = 2;
constexpr uint32_t MAXPER4_COUNT_PER_GROUP = 32;
constexpr uint32_t MAXPER8_COUNT_PER_GROUP = 16;
constexpr uint32_t TMPSCALE_COUNT_PER_GROUP = 64;

constexpr int64_t DINTLV_BF16_STEP = 256L;
constexpr int64_t FP4_PACK_STORE_STEP = 16L;
constexpr int64_t SCALE_B16_LOAD_STEP = 64L;
constexpr int64_t NORM_BF16_LOAD_STEP = 128L;
constexpr int64_t E8_COPY_STEP = 512L;
constexpr int64_t OTHER_BUF_COPY_STEP = 256L;
} // namespace

static constexpr AscendC::MicroAPI::DivSpecificMode DIV_MODE = {
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    false,
};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait castTraitReciprocalCast = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
#define QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS \
    template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_, \
              typename DataTypeX1Scale_, bool IsTensorList_>
#define QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS \
    L0TileShape_, DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, IsTensorList_

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
class BlockEpilogueSwigluHif4Quant {
public:
    __aicore__ inline BlockEpilogueSwigluHif4Quant() {}

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM;
        uint32_t baseN;
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
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t>; // y, yScale, x2Scale, x1Scale, bias
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const& params);
    __aicore__ inline auto GetFirstL0c2UbTensor();
    __aicore__ inline auto GetSecondL0c2UbTensor();
    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& blockCoord);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape);

private:
    __aicore__ inline void VFDoSwigluForHif4(uint16_t mSize);
    __aicore__ inline void TransHif4ScaleLayout(uint16_t mSize, uint16_t scaleBlockN);

    __aicore__ inline void VFDoSwigluAndQuantForHIF4(__ubuf__ int32_t* outputDst, __ubuf__ uint16_t* scaleDst,
                                                     __ubuf__ DataTypeIn* firstSrc, __ubuf__ DataTypeIn* secondSrc,
                                                     uint16_t mSize, uint16_t nSize);
    __aicore__ inline void ComputeMaxPer4(uint16_t regLoop, __ubuf__ bfloat16_t* xAddr1,
                                          __ubuf__ uint16_t* maxPer4BufAddr, uint32_t totalCountInUB);
    __aicore__ inline void ComputeMaxPer8(uint16_t maxPer8RegLoop, __ubuf__ uint16_t* maxPer4BufReadAddr,
                                          __ubuf__ uint16_t* e8BufAddr, __ubuf__ uint16_t* maxPer8BufAddr);
    __aicore__ inline void ComputeTmpScale(uint16_t regLoop, __ubuf__ uint16_t* maxPer4BufRead2Addr,
                                           __ubuf__ uint16_t* maxPer8BufReadAddr, __ubuf__ uint16_t* e8BufRead1Addr,
                                           __ubuf__ uint32_t* e1p8BufAddr, __ubuf__ uint8_t* e1p16BufAddr,
                                           __ubuf__ uint16_t* tmpScaleAddr);
    __aicore__ inline void ComputeScale(uint16_t scaleRegLoop, __ubuf__ uint8_t* e8BufRead2Addr,
                                        __ubuf__ uint8_t* e1p8BufReadAddr, __ubuf__ uint8_t* e1p16BufReadAddr,
                                        __ubuf__ uint16_t* scaleAddr);
    __aicore__ inline void ComputeDataForQuantTargetFp4(__ubuf__ bfloat16_t* srcAddr,
                                                        __ubuf__ uint16_t* halfScaleLocalAddr,
                                                        __ubuf__ int32_t* outLocalAddr, uint16_t outRegLoop);

    __aicore__ inline void CopyOutputFromUb2Gm(uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t>& src);

    __aicore__ inline void CopyScaleFromUb2Gm(uint64_t blockCount, uint64_t offset,
                                              AscendC::LocalTensor<uint32_t>& src);
    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<uint32_t> quantScaleGlobal_;

    // UB ADDR
    AscendC::LocalTensor<DataTypeIn> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    AscendC::LocalTensor<DataTypeIn> l0cOutUbSecond_{AscendC::TPosition::VECIN, MAX_SINGLE_MN * sizeof(DataTypeIn),
                                                     MAX_SINGLE_MN};
    AscendC::LocalTensor<int8_t> quantOutput_;
    AscendC::LocalTensor<AscendC::hif4_scale> quantScaleOutput_;
    AscendC::LocalTensor<uint32_t> quantScaleBlockOutput_;
    AscendC::LocalTensor<bfloat16_t> gluRes_;
    AscendC::LocalTensor<uint16_t> maxExp_;
    AscendC::LocalTensor<uint16_t> halfScale_;

    AscendC::LocalTensor<uint16_t> e8buf_;
    AscendC::LocalTensor<uint8_t> e1p8buf_;
    AscendC::LocalTensor<uint8_t> e1p16buf_;
    AscendC::LocalTensor<uint16_t> tmpscale_;
    AscendC::LocalTensor<uint16_t> maxper4_;
    AscendC::LocalTensor<uint16_t> maxper8_;

    const Params* params_;

    int64_t n_;
    int64_t scaleN_;
    int64_t scaleBlockN_;
    uint32_t singleM_; // cur singleShapeM
    uint32_t singleN_;
    int64_t gmCopyInRegStep_ = 0;
    int64_t gmCopyOutRegStep_ = 0;
    int64_t gmCopyOutScaleRegStep_ = 0;
    int64_t gmCopyInRegStep2_ = 0;
    int64_t gmCopyTmpBufE8Step_ = 0;
    int64_t gmCopyTmpBufOtherStep_ = 0;

    constexpr static AscendC::MicroAPI::CastTrait castTrait_ = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
};

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(
    Params const& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    params_ = &params;
    gmCopyInRegStep_ = DINTLV_BF16_STEP;
    gmCopyOutRegStep_ = FP4_PACK_STORE_STEP;
    gmCopyOutScaleRegStep_ = SCALE_B16_LOAD_STEP;
    gmCopyInRegStep2_ = NORM_BF16_LOAD_STEP;
    gmCopyTmpBufE8Step_ = E8_COPY_STEP;
    gmCopyTmpBufOtherStep_ = OTHER_BUF_COPY_STEP;

    // VECOUT: 2 = quantOutput + quantScaleOutput
    constexpr uint32_t afterIn = 2 * MAX_SINGLE_MN * sizeof(DataTypeIn);
    quantOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, afterIn, MAX_SINGLE_MN);
    quantScaleOutput_ = AscendC::LocalTensor<AscendC::hif4_scale>(
        AscendC::TPosition::VECOUT, afterIn + MAX_SINGLE_MN * sizeof(int8_t), MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE);
    constexpr uint32_t afterIO =
        afterIn + MAX_SINGLE_MN * sizeof(int8_t) + MAX_SINGLE_MN / AscendC::ONE_BLK_SIZE * sizeof(AscendC::hif4_scale);

    // VECCALC: gluRes + hif4 temp buffers + quantScaleBlock
    gluRes_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, 0, MAX_SINGLE_MN);
    uint32_t offset = afterIO;

    e8buf_ =
        AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, offset, MAX_ALIGN_GROUP_COUNT * E8_COUNT_PER_GROUP);
    offset += MAX_ALIGN_GROUP_COUNT * E8_COUNT_PER_GROUP * sizeof(uint16_t);

    e1p8buf_ = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECCALC, offset,
                                             MAX_ALIGN_GROUP_COUNT * E1P8_COUNT_PER_GROUP);
    offset += MAX_ALIGN_GROUP_COUNT * E1P8_COUNT_PER_GROUP * sizeof(uint8_t);

    e1p16buf_ = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECCALC, offset,
                                              MAX_ALIGN_GROUP_COUNT * E1P16_COUNT_PER_GROUP);
    offset += MAX_ALIGN_GROUP_COUNT * E1P16_COUNT_PER_GROUP * sizeof(uint8_t);

    maxper4_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, offset,
                                              MAX_ALIGN_GROUP_COUNT * MAXPER4_COUNT_PER_GROUP);
    offset += MAX_ALIGN_GROUP_COUNT * MAXPER4_COUNT_PER_GROUP * sizeof(uint16_t);

    maxper8_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, offset,
                                              MAX_ALIGN_GROUP_COUNT * MAXPER8_COUNT_PER_GROUP);
    offset += MAX_ALIGN_GROUP_COUNT * MAXPER8_COUNT_PER_GROUP * sizeof(uint16_t);

    tmpscale_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, offset,
                                               MAX_ALIGN_GROUP_COUNT * TMPSCALE_COUNT_PER_GROUP);
    offset += MAX_ALIGN_GROUP_COUNT * TMPSCALE_COUNT_PER_GROUP * sizeof(uint16_t);

    quantScaleBlockOutput_ =
        AscendC::LocalTensor<uint32_t>(AscendC::TPosition::VECOUT, offset, params_->baseM * AscendC::ONE_BLK_SIZE);
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(
    const BlockCoord& baseOffset)
{
    if ASCEND_IS_AIV {
        quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t*)params_->yGmAddr + Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer((__gm__ uint32_t*)params_->yScaleGmAddr + Get<Y_SCALE_IDX>(baseOffset));
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape& problemShape)
{
    n_ = Get<MNK_N>(problemShape);
    scaleN_ = CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE));
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyOutputFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<int8_t>& src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount;
    // 2：每字节打包两个 4-bit 输出元素。
    ub2GmParams.blockLen = singleN_ / 2;
    ub2GmParams.dstStride = (n_ - singleN_) >> 1;
    offset = offset >> 1;
    AscendC::DataCopyPad(quantOutputGlobal_[offset], src, ub2GmParams);
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::CopyScaleFromUb2Gm(
    uint64_t blockCount, uint64_t offset, AscendC::LocalTensor<uint32_t>& src)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    auto blockScaleN = CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE));
    ub2GmParams.blockLen = blockScaleN * sizeof(uint32_t);
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.dstStride = (scaleN_ - blockScaleN) * sizeof(uint32_t);
    AscendC::DataCopyPad(quantScaleGlobal_[offset], src, ub2GmParams);
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeDataForQuantTargetFp4(
    __ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* tmpScaleReadAddr, __ubuf__ int32_t* outLocalAddr,
    uint16_t outRegLoop)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vregOut;
        AscendC::MicroAPI::RegTensor<uint16_t> vregScaleTmp;
        AscendC::MicroAPI::RegTensor<fp4x2_e1m2_t> vbOut;
        AscendC::MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<uint8_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < outRegLoop; loopIdx++) {
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_NORM>(vregOut, srcAddr, gmCopyInRegStep2_);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_US_B16>(vregScaleTmp, tmpScaleReadAddr,
                                                                                  gmCopyOutScaleRegStep_);
            AscendC::MicroAPI::Mul((AscendC::MicroAPI::RegTensor<bfloat16_t>&)vregOut,
                                   (AscendC::MicroAPI::RegTensor<bfloat16_t>&)vregOut,
                                   (AscendC::MicroAPI::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
            AscendC::MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut, vregOut, maskAll);
            AscendC::MicroAPI::DataCopy<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int32_t>&)vbOut, gmCopyOutRegStep_, maskAll);
        }
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxPer4(
    uint16_t regLoop, __ubuf__ bfloat16_t* srcAddr, __ubuf__ uint16_t* maxPer4BufAddr, uint32_t totalCountInUB)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vregIn1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vregIn2;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vregIn3;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vregIn4;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vregMax1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vregMax2;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vregOut;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        // 0x7fff：清除 BF16 符号位，保留绝对值位模式。
        AscendC::MicroAPI::Duplicate(expMaskBF16, 0x7fff);
        AscendC::MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<uint8_t>();
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                (AscendC::MicroAPI::RegTensor<bfloat16_t>&)vregIn1, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)vregIn2,
                srcAddr, gmCopyInRegStep_);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1, expMaskBF16, maskAll);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2, expMaskBF16, maskAll);
            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                (AscendC::MicroAPI::RegTensor<bfloat16_t>&)vregIn3, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)vregIn4,
                srcAddr, gmCopyInRegStep_);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3, expMaskBF16, maskAll);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4, expMaskBF16, maskAll);

            AscendC::MicroAPI::Max(vregMax1, vregIn1, vregIn2, maskAll);
            AscendC::MicroAPI::Max(vregMax2, vregIn3, vregIn4, maskAll);
            AscendC::MicroAPI::DeInterleave(vregMax1, vregMax2, vregMax1, vregMax2);
            AscendC::MicroAPI::Max(vregOut, vregMax1, vregMax2, maskAll);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxPer4BufAddr, (AscendC::MicroAPI::RegTensor<uint16_t>&)vregOut, gmCopyInRegStep2_, maskAll);
        }
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeMaxPer8(
    uint16_t maxPer8RegLoop, __ubuf__ uint16_t* maxPer4BufReadAddr, __ubuf__ uint16_t* e8BufAddr,
    __ubuf__ uint16_t* maxPer8BufAddr)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vreg0Maxper4;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vreg1Maxper4;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vreg0Maxper8;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vreg0Maxper64;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vreg1Maxper64;
        AscendC::MicroAPI::RegTensor<uint16_t> vreg0E6m2;
        AscendC::MicroAPI::RegTensor<uint16_t> vreg1E6m2;
        AscendC::MicroAPI::RegTensor<uint16_t> vregBf16E6M2ScaleBias;
        // 0x3e12：转 E6M2 前所乘缩放系数的 BF16 位模式。
        AscendC::MicroAPI::Duplicate(vregBf16E6M2ScaleBias, 0x3e12);
        AscendC::MicroAPI::UnalignReg ureg1;
        AscendC::MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<uint8_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitE6M2 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < maxPer8RegLoop; loopIdx++) {
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4,
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg1Maxper4, maxPer4BufReadAddr,
                gmCopyInRegStep2_ * DIGIT_TWO);
            AscendC::MicroAPI::Max(vreg0Maxper8, vreg0Maxper4, vreg1Maxper4, maskAll);
            AscendC::MicroAPI::Interleave(vreg0Maxper4, vreg1Maxper4, vreg0Maxper4, vreg1Maxper4);

            AscendC::MicroAPI::ReduceMaxWithDataBlock<uint16_t>((AscendC::MicroAPI::RegTensor<uint16_t>&)vreg0Maxper64,
                                                                (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4,
                                                                maskAll);
            AscendC::MicroAPI::ReduceMaxWithDataBlock<uint16_t>((AscendC::MicroAPI::RegTensor<uint16_t>&)vreg1Maxper64,
                                                                (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg1Maxper4,
                                                                maskAll);

            AscendC::MicroAPI::Mul(vreg0Maxper64, vreg0Maxper64,
                                   (AscendC::MicroAPI::RegTensor<bfloat16_t>&)vregBf16E6M2ScaleBias, maskAll);
            AscendC::MicroAPI::Mul(vreg1Maxper64, vreg1Maxper64,
                                   (AscendC::MicroAPI::RegTensor<bfloat16_t>&)vregBf16E6M2ScaleBias, maskAll);

            AscendC::MicroAPI::Cast<float8_e6m2_t, bfloat16_t, castTraitE6M2>(
                (AscendC::MicroAPI::RegTensor<float8_e6m2_t>&)vreg0E6m2, vreg0Maxper64,
                maskAll); // 只有前8位才是有效值
            AscendC::MicroAPI::Cast<float8_e6m2_t, bfloat16_t, castTraitE6M2>(
                (AscendC::MicroAPI::RegTensor<float8_e6m2_t>&)vreg1E6m2, vreg1Maxper64,
                maskAll); // 只有前8位才是有效值
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxPer8BufAddr, (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg0Maxper8, gmCopyInRegStep2_, maskAll);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                // 8：每次写出 8 个有效 scale 搬运元素。
                e8BufAddr, vreg0E6m2, ureg1, 8);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                // 8：每次写出 8 个有效 scale 搬运元素。
                e8BufAddr, vreg1E6m2, ureg1, 8);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(e8BufAddr, ureg1, 0);
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeTmpScale(
    uint16_t regLoop, __ubuf__ uint16_t* maxPer4BufRead2Addr, __ubuf__ uint16_t* maxPer8BufReadAddr,
    __ubuf__ uint16_t* e8BufRead1Addr, __ubuf__ uint32_t* e1p8BufAddr, __ubuf__ uint8_t* e1p16BufAddr,
    __ubuf__ uint16_t* tmpScaleAddr)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vreg0Maxper4;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vreg0Maxper8;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vreg0Maxper64;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vb16_Ea_rec_bf16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vreg0E6m2;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vregTmpScale;
        float number1 = 1.0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vregNum1;
        AscendC::MicroAPI::Duplicate(vregNum1, number1);
        // 0.5：超过层级阈值时将 scale 乘数减半。
        float numberHalf = 0.5;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vregNumhalf;
        AscendC::MicroAPI::Duplicate(vregNumhalf, static_cast<bfloat16_t>(numberHalf));
        AscendC::MicroAPI::MaskReg vb1E8Comp;
        AscendC::MicroAPI::MaskReg vb1E16Comp;
        AscendC::MicroAPI::UnalignReg ureg1;
        AscendC::MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<uint8_t>();
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                // 8：E8 scale 广播加载的地址更新步长。
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg0Maxper64, e8BufRead1Addr, 8);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_US_B16>(
                // 64：每轮加载 64 个 per-8 最大值。
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg0Maxper8, maxPer8BufReadAddr, 64);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_NORM>(
                // 128：每轮加载 128 个 per-4 最大值。
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, maxPer4BufRead2Addr, 128);
            AscendC::MicroAPI::ReciprocalCast<bfloat16_t, float8_e6m2_t, castTraitReciprocalCast>(
                (AscendC::MicroAPI::RegTensor<bfloat16_t>&)vb16_Ea_rec_bf16,
                (AscendC::MicroAPI::RegTensor<float8_e6m2_t>&)vreg0Maxper64, maskAll);
            AscendC::MicroAPI::Mul(vreg0Maxper8, vreg0Maxper8, vb16_Ea_rec_bf16, maskAll);
            // 4：per-8 归一化最大值的 scale 减半阈值。
            AscendC::MicroAPI::CompareScalar<bfloat16_t, AscendC::CMPMODE::GE>(vb1E8Comp, vreg0Maxper8, 4, maskAll);
            AscendC::MicroAPI::Select(vreg0Maxper8, vregNumhalf, vregNum1, vb1E8Comp);
            AscendC::MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);
            AscendC::MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
            // 2：per-4 归一化最大值的 scale 减半阈值。
            AscendC::MicroAPI::CompareScalar<bfloat16_t, AscendC::CMPMODE::GE>(vb1E16Comp, vreg0Maxper4, 2, maskAll);
            AscendC::MicroAPI::Select(vreg0Maxper4, vregNumhalf, vregNum1, vb1E16Comp);
            AscendC::MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
            AscendC::MicroAPI::Mul(vregTmpScale, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::StoreDist::DIST_INTLV_B16>(
                tmpScaleAddr + loopIdx * gmCopyTmpBufOtherStep_, (AscendC::MicroAPI::RegTensor<uint16_t>&)vregTmpScale,
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vregTmpScale, maskAll);
            AscendC::MicroAPI::DataCopyUnAlign<uint32_t>(e1p8BufAddr, vb1E8Comp, ureg1);
            AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        // 16：打包 mask 存储后地址前进 16 字节。
                                        AscendC::MicroAPI::MaskDist::DIST_PACK>(e1p16BufAddr, vb1E16Comp, 16);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost<uint32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(e1p8BufAddr,
                                                                                                           ureg1, 0);
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ComputeScale(
    uint16_t scaleRegLoop, __ubuf__ uint8_t* e8BufRead2Addr, __ubuf__ uint8_t* e1p8BufReadAddr,
    __ubuf__ uint8_t* e1p16BufReadAddr, __ubuf__ uint16_t* scaleAddr)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint8_t> vregE8;
        AscendC::MicroAPI::RegTensor<uint8_t> vreg1E8;
        AscendC::MicroAPI::RegTensor<uint8_t> vreg1E16First;
        AscendC::MicroAPI::RegTensor<uint8_t> vreg1E16Second;

        AscendC::MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<uint8_t>();
        for (uint16_t loopIdx = 0; loopIdx < scaleRegLoop; loopIdx++) {
            AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DS_B8>(vregE8, e8BufRead2Addr,
                                                                                 gmCopyTmpBufE8Step_); // 512
            AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg1E8, e1p8BufReadAddr,
                                                                                gmCopyTmpBufOtherStep_); // 256
            AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg1E16First, e1p16BufReadAddr,
                                                                                gmCopyTmpBufOtherStep_); // 256
            AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg1E16Second, e1p16BufReadAddr,
                                                                                gmCopyTmpBufOtherStep_); // 256

            AscendC::MicroAPI::Interleave<uint8_t>(vregE8, vreg1E8, vregE8, vreg1E8);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::StoreDist::DIST_INTLV_B16>(
                scaleAddr + loopIdx * gmCopyTmpBufE8Step_, (AscendC::MicroAPI::RegTensor<uint16_t>&)vregE8,
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg1E16First, maskAll);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::StoreDist::DIST_INTLV_B16>(
                scaleAddr + loopIdx * gmCopyTmpBufE8Step_ + gmCopyTmpBufOtherStep_,
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg1E8,
                (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg1E16Second, maskAll);
        }
    }
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluAndQuantForHIF4(
    __ubuf__ int32_t* outputDst, __ubuf__ uint16_t* scaleDst, __ubuf__ DataTypeIn* firstSrc,
    __ubuf__ DataTypeIn* secondSrc, uint16_t mSize, uint16_t nSize)
{
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(DataTypeIn);
    uint16_t OneRowRepeatTimes =
        CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(sizePerRepeat)); // 一行需要循环的次数，等于n/64
    uint32_t nSrcUbAligned = CeilAlign(nSize, AscendC::ONE_BLK_SIZE / sizeof(DataTypeIn));
    uint32_t nDstUbAligned = CeilAlign(nSize, AscendC::ONE_BLK_SIZE); // out 1B
    const float swigluIdentity = 1.0;
    __ubuf__ bfloat16_t* gluResAddr = (__ubuf__ bfloat16_t*)gluRes_.GetPhyAddr();
    // swiglu
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<uint8_t>();
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {                                   // 需要计算m次
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) { // 每次计算m=1, n=64的数据大小
                AscendC::MicroAPI::RegTensor<bfloat16_t> verg7;
                AscendC::MicroAPI::RegTensor<float> swishInput, gateInput;
                AscendC::MicroAPI::RegTensor<float> verg1, verg2, verg3, verg4, verg5, verg6, swishOutput;

                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * sizePerRepeat;
                AscendC::MicroAPI::DataCopy(swishInput, firstSrc + l0cOutOffset);
                // // swish
                AscendC::MicroAPI::Muls(verg2, swishInput, -(swigluIdentity), mask);
                AscendC::MicroAPI::Exp(verg3, verg2, mask);
                AscendC::MicroAPI::Adds(verg4, verg3, swigluIdentity, mask);
                AscendC::MicroAPI::Div<float, &DIV_MODE>(swishOutput, swishInput, verg4, mask);

                // load gate data to regbase
                AscendC::MicroAPI::DataCopy(gateInput, secondSrc + l0cOutOffset);

                // swish * gate
                AscendC::MicroAPI::Mul(verg6, swishOutput, gateInput, mask);

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(verg7, verg6, mask);
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * sizePerRepeat;
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    gluResAddr + dstUbOffset, verg7, mask);
            }
        }
    }

    // quant
    uint32_t totalDataInUb = mSize * nDstUbAligned;
    uint32_t totalGroupNum = CeilDiv(totalDataInUb, ELEMS_PER_QUANT_GROUP);
    uint32_t alignGroupNum = CeilAlign(totalGroupNum, GROUP_COUNT_ALIGN);
    uint16_t regLoop = CeilDiv(totalGroupNum, PER4_MAX_GROUPS_PER_ITER);
    uint16_t maxPer8RegLoop =
        CeilDiv(totalDataInUb * sizeof(bfloat16_t) / sizeof(uint32_t), PER8_MAX_BUF_BYTES_PER_ITER);
    uint16_t scaleRegLoop = CeilDiv(alignGroupNum * sizeof(float), SCALE_BUF_BYTES_PER_ITER);
    uint16_t outRegLoop = CeilDiv(totalDataInUb, FP4_QUANT_ELEMS_PER_ITER);
    __ubuf__ uint16_t* maxPer4BufAddr = (__ubuf__ uint16_t*)maxper4_.GetPhyAddr();
    __ubuf__ uint16_t* maxPer8BufAddr = (__ubuf__ uint16_t*)maxper8_.GetPhyAddr();
    __ubuf__ uint16_t* e8BufAddr = (__ubuf__ uint16_t*)e8buf_.GetPhyAddr();
    __ubuf__ uint32_t* e1p8BufAddr = (__ubuf__ uint32_t*)e1p8buf_.GetPhyAddr();
    __ubuf__ uint8_t* e1p16BufAddr = (__ubuf__ uint8_t*)e1p16buf_.GetPhyAddr();
    __ubuf__ uint16_t* tmpScaleAddr = (__ubuf__ uint16_t*)tmpscale_.GetPhyAddr();
    __ubuf__ uint8_t* e8BufAddr2 = (__ubuf__ uint8_t*)e8buf_.GetPhyAddr();
    __ubuf__ uint8_t* e1p8BufReadAddr = (__ubuf__ uint8_t*)e1p8buf_.GetPhyAddr();
    auto gluResReadAddr = gluResAddr;
    auto maxPer4BufReadAddr = maxPer4BufAddr;
    auto maxPer4BufRead2Addr = maxPer4BufAddr;
    auto e1p16BufReadAddr = e1p16BufAddr;
    auto e8BufRead1Addr = e8BufAddr;
    auto e8BufRead2Addr = e8BufAddr2;
    ComputeMaxPer4(regLoop, gluResReadAddr, maxPer4BufAddr, totalDataInUb);
    ComputeMaxPer8(maxPer8RegLoop, maxPer4BufReadAddr, e8BufAddr, maxPer8BufAddr);
    ComputeTmpScale(regLoop, maxPer4BufRead2Addr, maxPer8BufAddr, e8BufRead1Addr, e1p8BufAddr, e1p16BufAddr,
                    tmpScaleAddr);
    ComputeScale(scaleRegLoop, e8BufRead2Addr, e1p8BufReadAddr, e1p16BufReadAddr, scaleDst);
    __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)tmpscale_.GetPhyAddr();
    ComputeDataForQuantTargetFp4(gluResAddr, halfScaleLocalAddr, outputDst, outRegLoop);
    return;
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoSwigluForHif4(
    uint16_t mSize)
{
    __ubuf__ int32_t* quantOutputInUbAddr = (__ubuf__ int32_t*)quantOutput_.GetPhyAddr();
    __ubuf__ uint16_t* quantScaleOutputInUbAddr = (__ubuf__ uint16_t*)quantScaleOutput_.GetPhyAddr();
    __ubuf__ DataTypeIn* l0cOutUbFirstAddr = (__ubuf__ DataTypeIn*)l0cOutUbFirst_.GetPhyAddr();
    __ubuf__ DataTypeIn* l0cOutUbSecondAddr = (__ubuf__ DataTypeIn*)l0cOutUbSecond_.GetPhyAddr();
    VFDoSwigluAndQuantForHIF4(quantOutputInUbAddr, quantScaleOutputInUbAddr, l0cOutUbFirstAddr, l0cOutUbSecondAddr,
                              mSize, singleN_);
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::TransHif4ScaleLayout(
    uint16_t mSize, uint16_t scaleBlockN)
{
    __ubuf__ uint32_t* srcAddr = (__ubuf__ uint32_t*)quantScaleOutput_.GetPhyAddr();
    __ubuf__ uint32_t* dstAddr = (__ubuf__ uint32_t*)quantScaleBlockOutput_.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::UnalignRegForLoad ureg0;
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            uint32_t scaleCountPerRow = scaleBlockN;
            AscendC::MicroAPI::MaskReg maskScaleN = AscendC::MicroAPI::UpdateMask<uint32_t>(scaleCountPerRow);
            AscendC::MicroAPI::RegTensor<uint32_t> vreg0;
            auto srcUb = srcAddr + mIdx * scaleBlockN;
            AscendC::MicroAPI::LoadUnAlignPre(ureg0, srcUb);
            AscendC::MicroAPI::LoadUnAlign(vreg0, ureg0, srcUb);
            auto dstUb = dstAddr + mIdx * (AscendC::ONE_BLK_SIZE / sizeof(uint32_t));
            AscendC::MicroAPI::DataCopy<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dstUb, vreg0,
                                                                                               maskScaleN);
        }
    }
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto
BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetFirstL0c2UbTensor()
{
    return l0cOutUbFirst_;
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto
BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetSecondL0c2UbTensor()
{
    return l0cOutUbSecond_;
}

QMM_BLOCK_EPILOGUE_SWIGLU_HIF4_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueSwigluHif4Quant<QMM_BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(
    const BlockShape& blockShape, const BlockCoord& blockCoord)
{
    singleN_ = Get<MNK_N>(blockShape);
    singleM_ = Get<MNK_M>(blockShape);
    scaleBlockN_ = CeilDiv(static_cast<uint64_t>(singleN_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE));
    auto halfSingleM = singleM_;
    uint64_t singleMInVec = halfSingleM;
    if (singleMInVec == 0) {
        return;
    }

    VFDoSwigluForHif4(singleMInVec);
    uint64_t yOffset = Get<Y_IDX>(blockCoord);
    uint64_t yScaleOffset = Get<Y_SCALE_IDX>(blockCoord);
    TransHif4ScaleLayout(singleMInVec, scaleBlockN_);
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
