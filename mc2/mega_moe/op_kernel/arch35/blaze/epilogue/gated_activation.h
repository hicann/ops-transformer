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
 * \file gated_activation.h
 * \brief A5（arch35）BlockEpilogueActivationMxQuant 门控激活实现。
 *
 */

#ifndef MEGA_MOE_ARCH35_GATED_ACTIVATION_H
#define MEGA_MOE_ARCH35_GATED_ACTIVATION_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../common/mega_moe_utils.h"

namespace MegaMoeImpl {
namespace ActivationImpl {

constexpr uint32_t VF_LEN_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);
static constexpr AscendC::MicroAPI::CastTrait CAST_ZERO = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP16_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename DataTypeInT>
struct ActivationContext {
    __ubuf__ DataTypeInT *firstSrc = nullptr;
    __ubuf__ DataTypeInT *secondSrc = nullptr;
    __ubuf__ bfloat16_t *gluResAddr = nullptr;
    __ubuf__ DataTypeInT *firstTailAddr = nullptr;
    __ubuf__ DataTypeInT *secondTailAddr = nullptr;
    __ubuf__ bfloat16_t *activationTailAddr1 = nullptr;
    __ubuf__ bfloat16_t *activationTailAddr2 = nullptr;
    __ubuf__ float *weightUbAddr = nullptr;

    uint32_t nSrcUbAligned = 0;
    uint32_t nDstUbAligned = 0;

    uint16_t dim0VfTimes = 0;
    uint16_t dim1VfTimes = 0;
    uint32_t dim1Tail = 0;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;

    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
};

struct SwiGLUParams {
    float clampLimit;
};

struct SiTUParams {
    float clampLimit;
    float beta;
    float invBeta;
    float alpha;
    float invAlpha;
};

struct SwiGLUOaiParams {
    float clampLimit;
    float alpha;
    float beta;
};

/*
 * ACTIVATION_IMPL_COMPUTE_TANH_TWOPATH：两段式 tanh 宏（多项式路径 + sigmoid 分解路径）
 *   多项式路径（|x|<0.6）：x*(1 + c1*x^2 + c2*x^4 + c3*x^6 + c4*x^8)，使用霍纳法和 FusedMulDstAdd
 *   sigmoid 路径（|x|>=0.6）：2/(1+e^{-2x}) - 1，天然保号且无相消
 *   系数来源：tanh_dag.h / situ_mx_quant_common.h
 *   c1=-0.333327681, c2=0.133152977, c3=-0.0523039624, c4=0.0157396831
 * 用法：ACTIVATION_IMPL_COMPUTE_TANH_TWOPATH(result, zReg, msk, oneReg, absReg, sqrReg,
 *                           polyReg, tmpReg, expReg, c1Reg, c2Reg, cmpMask);
 * 注意：宏在函数内展开，所用寄存器须在调用点函数作用域内声明；
 *       宏会覆盖 sqrReg/polyReg/tmpReg/expReg/absReg/cmpMask，结果写入 result
 */
#define ACTIVATION_IMPL_COMPUTE_TANH_TWOPATH(result, zReg, msk, oneReg, absReg, sqrReg, polyReg, tmpReg, expReg, \
                                             c1Reg, c2Reg, cmpMask) \
    do { \
        AscendC::MicroAPI::Mul(sqrReg, zReg, zReg, msk); \
        AscendC::MicroAPI::Muls(polyReg, sqrReg, 0.0157396831f, msk); \
        AscendC::MicroAPI::Adds(polyReg, polyReg, -0.0523039624f, msk); \
        AscendC::MicroAPI::FusedMulDstAdd(polyReg, sqrReg, c2Reg, msk); \
        AscendC::MicroAPI::FusedMulDstAdd(polyReg, sqrReg, c1Reg, msk); \
        AscendC::MicroAPI::Mul(polyReg, polyReg, sqrReg, msk); \
        AscendC::MicroAPI::FusedMulDstAdd(polyReg, zReg, zReg, msk); \
        AscendC::MicroAPI::Muls(expReg, zReg, -2.0f, msk); \
        AscendC::MicroAPI::Exp(expReg, expReg, msk); \
        AscendC::MicroAPI::Adds(expReg, expReg, 1.0f, msk); \
        AscendC::MicroAPI::Div(tmpReg, oneReg, expReg, msk); \
        AscendC::MicroAPI::Muls(tmpReg, tmpReg, 2.0f, msk); \
        AscendC::MicroAPI::Adds(tmpReg, tmpReg, -1.0f, msk); \
        AscendC::MicroAPI::Abs(absReg, zReg, msk); \
        AscendC::MicroAPI::CompareScalar<float, AscendC::CMPMODE::GE>(cmpMask, absReg, 0.60000002384185791016f, msk); \
        AscendC::MicroAPI::Select(result, tmpReg, polyReg, cmpMask); \
    } while (0)

/*
 * RunSwiGLU：SwiGLU 和 SwiGLU-Step 激活计算
 *   SwiGLU：先裁剪 x1，再计算 SiLU，并与对称裁剪后的 x2 相乘
 *   SwiGLU-Step：先计算 x1 的 SiLU 并裁剪，再与对称裁剪后的 x2 相乘
 *   包含主循环、尾循环和补零处理，所有寄存器均在本函数内声明
 */
template <typename DataTypeIn, bool TopkWeightsPrefetch, bool IsStep, bool IsInterleaved>
__aicore__ inline void RunSwiGLU(const ActivationContext<DataTypeIn> &ctx, const SwiGLUParams &swiGluParams)
{
    const float scalarOne = 1.0f;
    const float negScalarOne = -1.0f;
    bfloat16_t numZero = 0;
    uint32_t mask1Num = ctx.mask1Num;
    uint32_t mask2Num = ctx.mask2Num;
    uint32_t mask3Num = ctx.mask3Num;
    uint16_t dim0VfTimes = ctx.dim0VfTimes;
    uint16_t dim1VfTimes = ctx.dim1VfTimes;
    uint16_t dim1TailTimes = ctx.dim1TailTimes;
    uint16_t dim1Tail2 = ctx.dim1Tail2;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX1;
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<float> weightReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> zeroReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(mask3Num);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            if constexpr (TopkWeightsPrefetch) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    weightReg, ctx.weightUbAddr + dim0vfLoopIdx * INT32_PER_256B + WEIGHT_INDEX);
            }
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(
                    dim0vfLoopIdx, ctx.nSrcUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstSrc, srcIdxOffset);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondSrc, srcIdxOffset);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask);

                if constexpr (!IsStep) {
                    AscendC::MicroAPI::Mins(vregX1F, vregX1F, swiGluParams.clampLimit, mask);
                }
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, swiGluParams.clampLimit, mask);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -swiGluParams.clampLimit, mask);

                // 计算 SiLU(x1) * x2。
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask); // x 取负
                AscendC::MicroAPI::Exp(expReg, negReg, mask);                 // 计算 exp(-x)
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);    // 计算 exp(-x) + 1
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask);   // 计算 SiLU(x)
                if constexpr (IsStep) {
                    AscendC::MicroAPI::Mins(sigmoidReg, sigmoidReg, swiGluParams.clampLimit, mask);
                }
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask); // 计算 SiLU(x) * y
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(
                    dim0vfLoopIdx, ctx.nDstUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.gluResAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 =
                AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, ctx.nSrcUbAligned);
            AscendC::MicroAPI::AddrReg outOffset1 =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, ctx.nDstUbAligned);
            for (uint16_t tailIdx = 0; tailIdx < dim1TailTimes; tailIdx++) {
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask1);

                if constexpr (!IsStep) {
                    AscendC::MicroAPI::Mins(vregX1F, vregX1F, swiGluParams.clampLimit, mask1);
                }
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, swiGluParams.clampLimit, mask1);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -swiGluParams.clampLimit, mask1);

                // 计算尾块的 SiLU(x1) * x2。
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask1);
                if constexpr (IsStep) {
                    AscendC::MicroAPI::Mins(sigmoidReg, sigmoidReg, swiGluParams.clampLimit, mask1);
                }
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask1);
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask1);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.activationTailAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t padIdx = 0; padIdx < dim1Tail2; padIdx++) {
                AscendC::MicroAPI::Duplicate(zeroReg, numZero);
                AscendC::MicroAPI::DataCopy<bfloat16_t>(ctx.activationTailAddr2, zeroReg, outOffset1, mask3);
            }
        }
    }
}

/*
 * RunSiTU：SiTU 激活完整计算流程
 *   situ_a = beta * tanh(gate/beta) * sigmoid(gate)
 *   LINEAR 子模式：out = situ_a * (alpha * tanh(up/alpha))
 *   DEFAULT 子模式：out = situ_a * up
 *   tanh 采用两段式（多项式路 + sigmoid 分解路），保号无相消
 *   包含主循环、尾循环和补零处理。
 */
template <typename DataTypeIn, bool TopkWeightsPrefetch, bool IsLinear, bool IsInterleaved>
__aicore__ inline void RunSiTU(const ActivationContext<DataTypeIn> &ctx, const SiTUParams &situParams)
{
    // tanh 多项式近似的奇次项系数（8 阶 Padé 近似，仅取奇次项）
    // 参见 ACTIVATION_IMPL_COMPUTE_TANH_TWOPATH 宏注释，c1/c2 对应 x^3/x^5 的系数
    const float tanhC1 = -0.333327681f; // c1: tanh 多项式 x^3 项系数，约 -1/3
    const float tanhC2 = 0.133152977f;  // c2: tanh 多项式 x^5 项系数，约 2/15
    const float scalarOne = 1.0f;
    const float negScalarOne = -1.0f;
    bfloat16_t numZero = 0;
    uint32_t mask1Num = ctx.mask1Num;
    uint32_t mask2Num = ctx.mask2Num;
    uint32_t mask3Num = ctx.mask3Num;
    uint16_t dim0VfTimes = ctx.dim0VfTimes;
    uint16_t dim1VfTimes = ctx.dim1VfTimes;
    uint16_t dim1TailTimes = ctx.dim1TailTimes;
    uint16_t dim1Tail2 = ctx.dim1Tail2;

    __VEC_SCOPE__
    {
        // 通用寄存器（搬入、裁剪、写回）
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX1;
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<float> weightReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> zeroReg;

        // SiTU 专用寄存器
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> zReg;
        AscendC::MicroAPI::RegTensor<float> betaReg;
        AscendC::MicroAPI::RegTensor<float> oneReg;

        // tanh 两段式专用寄存器
        AscendC::MicroAPI::RegTensor<float> absReg;
        AscendC::MicroAPI::RegTensor<float> sqrReg;
        AscendC::MicroAPI::RegTensor<float> polyReg;
        AscendC::MicroAPI::RegTensor<float> tanhTmpReg;
        AscendC::MicroAPI::RegTensor<float> c1Reg;
        AscendC::MicroAPI::RegTensor<float> c2Reg;
        AscendC::MicroAPI::MaskReg cmpMaskReg;

        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(mask3Num);

        // 循环前初始化常量（全量广播，不依赖掩码）
        AscendC::MicroAPI::Duplicate(oneReg, scalarOne);
        AscendC::MicroAPI::Duplicate(c1Reg, tanhC1);
        AscendC::MicroAPI::Duplicate(c2Reg, tanhC2);

        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            if constexpr (TopkWeightsPrefetch) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    weightReg, ctx.weightUbAddr + dim0vfLoopIdx * INT32_PER_256B + WEIGHT_INDEX);
            }
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(
                    dim0vfLoopIdx, ctx.nSrcUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstSrc, srcIdxOffset);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondSrc, srcIdxOffset);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask);

                AscendC::MicroAPI::Mins(vregX1F, vregX1F, situParams.clampLimit, mask);
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, situParams.clampLimit, mask);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -situParams.clampLimit, mask);

                // SiTU 激活主循环。
                // 门控路径：z = gate / beta
                AscendC::MicroAPI::Muls(zReg, vregX1F, situParams.invBeta, mask);
                // 两段式计算 tanh(gate/beta)，结果存入 sigmoidReg。
                ACTIVATION_IMPL_COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, mask, oneReg, absReg, sqrReg, polyReg,
                                                     tanhTmpReg, expReg, c1Reg, c2Reg, cmpMaskReg);
                // beta * sigmoid(gate) = beta / (exp(-gate)+1)
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask); // gate 取负
                AscendC::MicroAPI::Exp(expReg, negReg, mask);                 // 计算 exp(-gate)
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);    // 计算 exp(-gate) + 1
                AscendC::MicroAPI::Duplicate(betaReg, situParams.beta, mask); // 将 beta 写入 betaReg
                AscendC::MicroAPI::Div(expReg, betaReg, addsReg, mask);       // 计算 beta / (exp(-gate) + 1)
                // situ_a = tanh(gate/beta) * beta * sigmoid(gate)
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, expReg, mask); // outFReg 保存 situ_a

                if constexpr (IsLinear) {
                    // 上投影路径：计算 alpha * tanh(up/alpha)，使用 vregX1F 暂存门控结果。
                    AscendC::MicroAPI::Muls(vregX1F, outFReg, scalarOne, mask);        // vregX1F 保存门控结果
                    AscendC::MicroAPI::Muls(zReg, vregX2F, situParams.invAlpha, mask); // zReg 保存 up / alpha
                    ACTIVATION_IMPL_COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, mask, oneReg, absReg, sqrReg, polyReg,
                                                         tanhTmpReg, expReg, c1Reg, c2Reg, cmpMaskReg);
                    // alpha * tanh(up/alpha)
                    AscendC::MicroAPI::Muls(sigmoidReg, sigmoidReg, situParams.alpha, mask);
                    // 将门控结果与变换后的上投影结果相乘。
                    AscendC::MicroAPI::Mul(outFReg, vregX1F, sigmoidReg, mask);
                } else {
                    // SITU 默认子模式：out = situ_a * up
                    AscendC::MicroAPI::Mul(outFReg, outFReg, vregX2F, mask);
                }
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(
                    dim0vfLoopIdx, ctx.nDstUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.gluResAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 =
                AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, ctx.nSrcUbAligned);
            AscendC::MicroAPI::AddrReg outOffset1 =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, ctx.nDstUbAligned);
            for (uint16_t tailIdx = 0; tailIdx < dim1TailTimes; tailIdx++) {
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask1);

                AscendC::MicroAPI::Mins(vregX1F, vregX1F, situParams.clampLimit, mask1);
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, situParams.clampLimit, mask1);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -situParams.clampLimit, mask1);

                // SiTU 激活尾循环。
                AscendC::MicroAPI::Muls(zReg, vregX1F, situParams.invBeta, mask1);
                ACTIVATION_IMPL_COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, mask1, oneReg, absReg, sqrReg, polyReg,
                                                     tanhTmpReg, expReg, c1Reg, c2Reg, cmpMaskReg);
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Duplicate(betaReg, situParams.beta, mask1);
                AscendC::MicroAPI::Div(expReg, betaReg, addsReg, mask1);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, expReg, mask1);

                if constexpr (IsLinear) {
                    AscendC::MicroAPI::Muls(vregX1F, outFReg, scalarOne, mask1);
                    AscendC::MicroAPI::Muls(zReg, vregX2F, situParams.invAlpha, mask1);
                    ACTIVATION_IMPL_COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, mask1, oneReg, absReg, sqrReg, polyReg,
                                                         tanhTmpReg, expReg, c1Reg, c2Reg, cmpMaskReg);
                    AscendC::MicroAPI::Muls(sigmoidReg, sigmoidReg, situParams.alpha, mask1);
                    AscendC::MicroAPI::Mul(outFReg, vregX1F, sigmoidReg, mask1);
                } else {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, vregX2F, mask1);
                }
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask1);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.activationTailAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t padIdx = 0; padIdx < dim1Tail2; padIdx++) {
                AscendC::MicroAPI::Duplicate(zeroReg, numZero);
                AscendC::MicroAPI::DataCopy<bfloat16_t>(ctx.activationTailAddr2, zeroReg, outOffset1, mask3);
            }
        }
    }
}

/*
 * RunSwiGLUOai：SwiGLU-OAI 激活计算流程
 *   公式：(up + beta) × min(gate, clamp) × sigmoid(alpha × min(gate, clamp))
 *        = (clip(up,±clamp)+beta) × min(gate,clamp) / (1+exp(-alpha×min(gate,clamp)))
 */
template <typename DataTypeIn, bool TopkWeightsPrefetch, bool IsInterleaved>
__aicore__ inline void RunSwiGLUOai(const ActivationContext<DataTypeIn> &ctx, const SwiGLUOaiParams &swiGluOaiParams)
{
    const float scalarOne = 1.0f;
    bfloat16_t numZero = 0;
    uint32_t mask1Num = ctx.mask1Num;
    uint32_t mask2Num = ctx.mask2Num;
    uint32_t mask3Num = ctx.mask3Num;
    uint16_t dim0VfTimes = ctx.dim0VfTimes;
    uint16_t dim1VfTimes = ctx.dim1VfTimes;
    uint16_t dim1TailTimes = ctx.dim1TailTimes;
    uint16_t dim1Tail2 = ctx.dim1Tail2;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX1;
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<float> weightReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> zeroReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(mask3Num);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            if constexpr (TopkWeightsPrefetch) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    weightReg, ctx.weightUbAddr + dim0vfLoopIdx * INT32_PER_256B + WEIGHT_INDEX);
            }
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(
                    dim0vfLoopIdx, ctx.nSrcUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstSrc, srcIdxOffset);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondSrc, srcIdxOffset);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask);

                // 对门控输入作上限裁剪。
                AscendC::MicroAPI::Mins(vregX1F, vregX1F, swiGluOaiParams.clampLimit, mask);
                // 对上投影输入作对称裁剪。
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, swiGluOaiParams.clampLimit, mask);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -swiGluOaiParams.clampLimit, mask);

                // 对上投影输入加 beta。
                AscendC::MicroAPI::Adds(vregX2F, vregX2F, swiGluOaiParams.beta, mask);

                // 计算 gate * sigmoid(alpha * gate)。
                AscendC::MicroAPI::Muls(negReg, vregX1F, -swiGluOaiParams.alpha, mask); // 计算 -alpha * gate
                AscendC::MicroAPI::Exp(expReg, negReg, mask);                           // 计算 exp(-alpha * gate)
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);              // 计算 exp(-alpha * gate) + 1
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask); // 计算 gate * sigmoid(alpha * gate)
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask); // 计算完整激活结果
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(
                    dim0vfLoopIdx, ctx.nDstUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.gluResAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 =
                AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, ctx.nSrcUbAligned);
            AscendC::MicroAPI::AddrReg outOffset1 =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, ctx.nDstUbAligned);
            for (uint16_t tailIdx = 0; tailIdx < dim1TailTimes; tailIdx++) {
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask1);

                AscendC::MicroAPI::Mins(vregX1F, vregX1F, swiGluOaiParams.clampLimit, mask1);
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, swiGluOaiParams.clampLimit, mask1);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -swiGluOaiParams.clampLimit, mask1);

                // 对上投影输入加 beta。
                AscendC::MicroAPI::Adds(vregX2F, vregX2F, swiGluOaiParams.beta, mask1);

                // 计算 SwiGLU-OAI 尾块。
                AscendC::MicroAPI::Muls(negReg, vregX1F, -swiGluOaiParams.alpha, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask1);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask1);
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask1);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.activationTailAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t padIdx = 0; padIdx < dim1Tail2; padIdx++) {
                AscendC::MicroAPI::Duplicate(zeroReg, numZero);
                AscendC::MicroAPI::DataCopy<bfloat16_t>(ctx.activationTailAddr2, zeroReg, outOffset1, mask3);
            }
        }
    }
}

#undef ACTIVATION_IMPL_COMPUTE_TANH_TWOPATH

} // namespace ActivationImpl
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_GATED_ACTIVATION_H
