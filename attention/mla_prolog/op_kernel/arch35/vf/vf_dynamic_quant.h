/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_dynamic_quant.h
 * \brief
 */

#ifndef VF_DYNAMIC_QUANT_H
#define VF_DYNAMIC_QUANT_H

#include "kernel_tensor.h"

namespace MlaProlog {
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float FP8_E4M3FN_MIN_VALUE = -448.0f;
constexpr float HIFLOAT8_MAX_VALUE = 32768.0f;
constexpr uint32_t FP8_E4M3FN_BLOCK_SIZE = 32;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr int64_t DIGIT_TWO = 2;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint8_t PER_TILE_QUANT_MODE = 1;
#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
constexpr float NEG_INFINITY = -INFINITY;
constexpr uint16_t REDUCE_SIZE = 8;

template <typename T, typename C, typename O, uint8_t M = 0> // M=0为fp8全量化pertoken量化；M=1为fp8全量化pertile量化
__simd_vf__ void ComputeVFImpl(__ubuf__ T *xAddr, __ubuf__ O *yAddr, __ubuf__ float *scaleAddr, uint32_t rowIndex,
                               uint32_t rowCount, uint32_t dtypeSize, uint16_t VL, uint16_t vfLoop,
                               const float alphaValue)
{
    constexpr static AscendC::Reg::CastTrait castTraitB16ToF32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::Reg::CastTrait castTraitPack2 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT};
    constexpr static AscendC::Reg::CastTrait castTraitF32ToHalf = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::CAST_ODD};
    constexpr static AscendC::Reg::CastTrait castTraitF32ToHif8 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::CAST_ROUND};
    static constexpr AscendC::Reg::DivSpecificMode mode = {AscendC::Reg::MaskMergeMode::ZEROING, true};
    AscendC::Reg::RegTensor<T> xInput;       // 搬入的x
    AscendC::Reg::RegTensor<float> xFp32;    // cast成float之后的x
    AscendC::Reg::RegTensor<float> xFp32Abs; // x的绝对值absvalue
    AscendC::Reg::RegTensor<float> xMaxAbs; // x的abs与-inf比较的结果，可以认为是x的absvalue的一些最大值
    AscendC::Reg::RegTensor<float> xReduceMax; // reduceMax
    AscendC::Reg::RegTensor<float> xScale;     // scale
    AscendC::Reg::RegTensor<float> xScaleDup;  // Duplicate之后的scale，为了和input一起得到y
    AscendC::Reg::RegTensor<float> xNorm;      // input/scale
    AscendC::Reg::RegTensor<half> yHalf;       // float-->half-->int8
    AscendC::Reg::RegTensor<O> yOutput;        // 最终y

    AscendC::Reg::MaskReg validMask0; // 有效掩码
    AscendC::Reg::MaskReg fullMask1 =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>(); // 启用所有通道，全掩码
    AscendC::Reg::MaskReg validMask2;

    AscendC::Reg::UnalignRegForStore ureg0;
    AscendC::Reg::Duplicate(xMaxAbs, NEG_INFINITY, fullMask1);
    uint32_t sreg0 = rowCount;

    // 计算量化参数
    for (uint16_t j = 0; j < vfLoop; j++) {
        validMask0 = AscendC::Reg::UpdateMask<float>(sreg0); // 有效元素
        if constexpr (!std::is_same<T, float>::value) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xInput,
                                                                                xAddr + rowIndex * rowCount + j * VL);
            AscendC::Reg::Cast<float, T, castTraitB16ToF32>(xFp32, xInput, validMask0);
        } else {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(xFp32, xAddr + rowIndex * rowCount + j * VL);
        }
        AscendC::Reg::Abs(xFp32Abs, xFp32, validMask0);
        AscendC::Reg::Max(xMaxAbs, xFp32Abs, xMaxAbs, fullMask1);
    }
    AscendC::Reg::Reduce<Reg::ReduceType::MAX, float, float, Reg::MaskMergeMode::ZEROING>(xReduceMax, xMaxAbs,
                                                                                          fullMask1);
    if constexpr (M == PER_TILE_QUANT_MODE) {
        constexpr float epsilonValue = 1e-4f;
        AscendC::Reg::RegTensor<float> epsilonReg;
        AscendC::Reg::Duplicate(epsilonReg, epsilonValue, fullMask1);
        AscendC::Reg::Max(xReduceMax, xReduceMax, epsilonReg, fullMask1); // regtensor类型
    }
    AscendC::Reg::Muls(xScale, xReduceMax, alphaValue, fullMask1);
    AscendC::Reg::Duplicate(xScaleDup, xScale, fullMask1);
    AscendC::Reg::StoreUnAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(scaleAddr, xScale, ureg0, 1);

    uint32_t sreg1 = rowCount;
    for (uint16_t j = 0; j < vfLoop; j++) {
        auto addr = yAddr + rowIndex * rowCount + j * VL;
        validMask2 = AscendC::Reg::UpdateMask<float>(sreg1);
        if constexpr (!std::is_same<T, float>::value) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xInput,
                                                                                xAddr + rowIndex * rowCount + j * VL);
            AscendC::Reg::Cast<float, T, castTraitB16ToF32>(xFp32, xInput, validMask2);
        } else {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(xFp32, xAddr + rowIndex * rowCount + j * VL);
        }
        AscendC::Reg::Div(xNorm, xFp32, xScaleDup, validMask2);
        if constexpr (std::is_same<O, fp8_e4m3fn_t>::value) {
            AscendC::Reg::Cast<O, float, castTraitPack2>(yOutput, xNorm, validMask2);
        } else if constexpr (std::is_same<O, hifloat8_t>::value) {
            AscendC::Reg::Cast<O, float, castTraitF32ToHif8>(yOutput, xNorm, validMask2);
        } else {
            AscendC::Reg::Cast<half, float, castTraitF32ToHalf>(yHalf, xNorm, validMask2);
            AscendC::Reg::Cast<O, half, castTraitPack2>(yOutput, yHalf, validMask2);
        }
        AscendC::Reg::StoreAlign<O, AscendC::Reg::StoreDist::DIST_PACK4_B32>(addr, yOutput, validMask2);
    }
    AscendC::Reg::StoreUnAlignPost(scaleAddr, ureg0, 0);
}

/**
 * @brief DynamicQuantPerTokenVf 对row行进行dynamicquant, BF16 ---> int8/FP8E4M3, 每一行出一个系数。
 * @param outputLocal 输出tensor [row , col]
 * @param scale 输出每行的反量化系数 [row]
 * @param inputLocal 输入tensor [row , col]
 * @param row 待处理的行数
 * @param col 待处理的列数
 */
template <typename T, typename C, typename O>
__aicore__ inline void DynamicQuantPerTokenVf(const LocalTensor<O> &outputLocal, const LocalTensor<C> &scale,
                                              const LocalTensor<T> &inputLocal, uint64_t row, uint64_t col)
{
    auto xAddr = (__local_mem__ T *)inputLocal.GetPhyAddr();
    auto yAddr = (__local_mem__ O *)outputLocal.GetPhyAddr();
    auto scaleAddr = (__local_mem__ C *)scale.GetPhyAddr();
    uint32_t dtypeSize = sizeof(float);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint32_t rowCount = col;
    uint16_t vfLoop = (rowCount + VL - 1) / VL;

    constexpr float maxValue = std::is_same<O, fp8_e4m3fn_t>::value ? FP8_E4M3FN_MAX_VALUE :
                               std::is_same<O, hifloat8_t>::value   ? HIFLOAT8_MAX_VALUE :
                                                                      INT8_MAX_VALUE;
    const float alphaValue = static_cast<float>(1.0) / maxValue;
    for (int32_t i = 0; i < row; i++) {
        ComputeVFImpl<T, C, O>(xAddr, yAddr, scaleAddr + i, i, rowCount, dtypeSize, VL, vfLoop, alphaValue);
    }
}

template <typename T, typename U>
__simd_vf__ void ComputeMaxExpVF(__ubuf__ T *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCountInUB,
                                 uint16_t loopNum, uint16_t vecLen)
{
    AscendC::Reg::RegTensor<T> vdExp0;
    AscendC::Reg::RegTensor<T> vdExp1;
    AscendC::Reg::RegTensor<uint16_t> vdExpExtract0;
    AscendC::Reg::RegTensor<uint16_t> vdExpExtract1;

    AscendC::Reg::RegTensor<uint16_t> expMaskBF16;
    AscendC::Reg::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

    AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
    AscendC::Reg::MaskReg scaleMask1;
    AscendC::Reg::MaskReg scaleMask2;
    AscendC::Reg::UnalignRegForStore u1;

    for (uint16_t i = 0; i < loopNum; i++) {
        scaleMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
        scaleMask2 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
        AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vecLen * DIGIT_TWO);
        // 通过位与运算得到bf16的指数位保留，尾数位置0所对应的值, 0x7f80是bf16 8个指数位为1，7个尾数位为0对应的值
        AscendC::Reg::And(vdExpExtract0, (AscendC::Reg::RegTensor<uint16_t> &)vdExp0, expMaskBF16, scaleMask1);

        AscendC::Reg::And(vdExpExtract1, (AscendC::Reg::RegTensor<uint16_t> &)vdExp1, expMaskBF16, scaleMask1);
        // 得到指数位最大的值
        AscendC::Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
        // 得到每个block(32个元素)中的指数位最大的值
        AscendC::Reg::ReduceDataBlock<Reg::ReduceType::MAX, uint16_t, Reg::MaskMergeMode::ZEROING>(vdMaxExp, vdMaxExp,
                                                                                                   scaleMask1);
        AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, vdMaxExp, u1,
                                                                                          REDUCE_SIZE);
    }
    AscendC::Reg::StoreUnAlignPost(maxExpAddr, u1, 0);
}

template <typename T, typename U>
__simd_vf__ void ComputeScaleVF(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB, uint16_t loopNumScale,
                                uint16_t vecLen)
{
    AscendC::Reg::RegTensor<uint16_t> expMask;
    AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
    AscendC::Reg::RegTensor<uint16_t> vdMaxExp;

    AscendC::Reg::MaskReg cmpResult;
    AscendC::Reg::MaskReg zeroMask;
    AscendC::Reg::MaskReg preMaskScale;
    AscendC::Reg::MaskReg invalidDataMask;
    AscendC::Reg::MaskReg specialDataMask;

    AscendC::Reg::RegTensor<uint16_t> maxExpValue;
    AscendC::Reg::Duplicate(maxExpValue, FP8_E4M3_MAX_EXP);
    AscendC::Reg::RegTensor<uint16_t> sharedExp;
    AscendC::Reg::RegTensor<uint16_t> scaleValue;
    AscendC::Reg::RegTensor<uint16_t> scaleBias;
    AscendC::Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
    AscendC::Reg::RegTensor<uint16_t> halfScale;
    AscendC::Reg::RegTensor<uint16_t> fp8NanRegTensor;
    AscendC::Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
    AscendC::Reg::RegTensor<uint16_t> zeroRegTensor;
    AscendC::Reg::Duplicate(zeroRegTensor, 0);
    AscendC::Reg::RegTensor<uint16_t> nanRegTensor;
    AscendC::Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
    AscendC::Reg::RegTensor<uint16_t> specialExpRegTensor;
    AscendC::Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);

    for (uint16_t i = 0; i < loopNumScale; i++) {
        preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(totalScaleInUB);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr, vecLen);
        AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale); // INF/NAN
        AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
        // 如果vdMaxExp小于等于maxExpValue, 则置为maxExpValue, maxExpValue为FP8E4M3最大正整数的指数位8左移7位是0x400
        AscendC::Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
        AscendC::Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
        // vdMaxExp - maxExpValue后右移7位得到FP8E8M0的值，右移7位是因为bf16的指数位从第7位开始
        AscendC::Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
        AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);

        AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
        AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                         vecLen / DIGIT_TWO, preMaskScale);

        AscendC::Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
        // 0x7f00 - sharedExp得到1/sharedExp
        AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
        AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
        AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
        AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(halfScaleLocalAddr, halfScale,
                                                                                        vecLen, preMaskScale);
    }
}

template <typename T, typename U>
__simd_vf__ void ComputeDataVF(__ubuf__ T *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
                               __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB, uint32_t totalCountInUB2,
                               uint16_t loopNum, uint16_t vecLen)
{
    AscendC::Reg::MaskReg dataMask1;
    AscendC::Reg::MaskReg dataMask2;
    AscendC::Reg::MaskReg dataMask3;
    AscendC::Reg::MaskReg dataMask4;
    AscendC::Reg::MaskReg nanResult;

    AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
    AscendC::Reg::RegTensor<float> floatScaleForMul;
    AscendC::Reg::RegTensor<T> vdExp0;
    AscendC::Reg::RegTensor<T> vdExp1;
    AscendC::Reg::RegTensor<T> vdExp0Convert;
    AscendC::Reg::RegTensor<T> vdExp1Convert;
    AscendC::Reg::RegTensor<float> vdExp0FP32Zero;
    AscendC::Reg::RegTensor<float> vdExp0FP32One;
    AscendC::Reg::RegTensor<float> vdExp1FP32Zero;
    AscendC::Reg::RegTensor<float> vdExp1FP32One;
    AscendC::Reg::RegTensor<float> maxFp8Value;
    AscendC::Reg::Duplicate(maxFp8Value, FP8_E4M3FN_MAX_VALUE);
    AscendC::Reg::RegTensor<float> minFp8Value;
    AscendC::Reg::Duplicate(minFp8Value, FP8_E4M3FN_MIN_VALUE);
    AscendC::Reg::RegTensor<U> vdExp0FP8Zero;
    AscendC::Reg::RegTensor<U> vdExp0FP8One;
    AscendC::Reg::RegTensor<U> vdExp1FP8Zero;
    AscendC::Reg::RegTensor<U> vdExp1FP8One;

    static constexpr AscendC::Reg::CastTrait castTraitZero = {AscendC::Reg::RegLayout::ZERO,
                                                              AscendC::Reg::SatMode::UNKNOWN,
                                                              AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait castTraitOne = {AscendC::Reg::RegLayout::ONE,
                                                             AscendC::Reg::SatMode::UNKNOWN,
                                                             AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait castTrait32to8 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT};
    for (uint16_t i = 0; i < loopNum; i++) {
        dataMask1 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
        dataMask2 = AscendC::Reg::UpdateMask<T>(totalCountInUB);
        dataMask3 = AscendC::Reg::UpdateMask<T>(totalCountInUB2);
        dataMask4 = AscendC::Reg::UpdateMask<T>(totalCountInUB2);
        AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vecLen * DIGIT_TWO);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr, REDUCE_SIZE);

        // X / mxscale
        AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<T> &)halfScaleForMul, dataMask1);
        AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<T> &)halfScaleForMul, dataMask1);
        AscendC::Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
        AscendC::Reg::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
        AscendC::Reg::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
        AscendC::Reg::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
        // 大于448.0的值设为448.0
        AscendC::Reg::Compare<float, CMPMODE::GT>(nanResult, vdExp0FP32Zero, maxFp8Value, dataMask1);
        AscendC::Reg::Select<float>(vdExp0FP32Zero, maxFp8Value, vdExp0FP32Zero, nanResult);
        AscendC::Reg::Compare<float, CMPMODE::GT>(nanResult, vdExp0FP32One, maxFp8Value, dataMask1);
        AscendC::Reg::Select<float>(vdExp0FP32One, maxFp8Value, vdExp0FP32One, nanResult);
        // 小于-448.0的值设为-448。0
        AscendC::Reg::Compare<float, CMPMODE::LT>(nanResult, vdExp0FP32Zero, minFp8Value, dataMask1);
        AscendC::Reg::Select<float>(vdExp0FP32Zero, minFp8Value, vdExp0FP32Zero, nanResult);
        AscendC::Reg::Compare<float, CMPMODE::LT>(nanResult, vdExp0FP32One, minFp8Value, dataMask1);
        AscendC::Reg::Select<float>(vdExp0FP32One, minFp8Value, vdExp0FP32One, nanResult);
        // 将结果转为FP8E4M3
        AscendC::Reg::Cast<U, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
        AscendC::Reg::Cast<U, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);

        AscendC::Reg::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
        AscendC::Reg::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
        AscendC::Reg::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
        // 大于448.0的值设为448.0
        AscendC::Reg::Compare<float, CMPMODE::GT>(nanResult, vdExp1FP32Zero, maxFp8Value, dataMask2);
        AscendC::Reg::Select<float>(vdExp1FP32Zero, maxFp8Value, vdExp1FP32Zero, nanResult);
        AscendC::Reg::Compare<float, CMPMODE::GT>(nanResult, vdExp1FP32One, maxFp8Value, dataMask2);
        AscendC::Reg::Select<float>(vdExp1FP32One, maxFp8Value, vdExp1FP32One, nanResult);
        // 小于-448.0的值设为-448。0
        AscendC::Reg::Compare<float, CMPMODE::LT>(nanResult, vdExp1FP32Zero, minFp8Value, dataMask2);
        AscendC::Reg::Select<float>(vdExp1FP32Zero, minFp8Value, vdExp1FP32Zero, nanResult);
        AscendC::Reg::Compare<float, CMPMODE::LT>(nanResult, vdExp1FP32One, minFp8Value, dataMask2);
        AscendC::Reg::Select<float>(vdExp1FP32One, minFp8Value, vdExp1FP32One, nanResult);
        // 将结果转为FP8E4M3
        AscendC::Reg::Cast<U, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
        AscendC::Reg::Cast<U, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
        AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::Reg::RegTensor<int8_t> &)vdExp0FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask3);
        AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::Reg::RegTensor<int8_t> &)vdExp0FP8One, OUT_ELE_NUM_ONE_BLK, dataMask3);
        AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::Reg::RegTensor<int8_t> &)vdExp1FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask4);
        AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (AscendC::Reg::RegTensor<int8_t> &)vdExp1FP8One, OUT_ELE_NUM_ONE_BLK, dataMask4);
    }
}

/**
 * @brief DynamicQuantPerBlockMxfp8Vf 对row进行dynamicquant, BF16 ---> FP8E4M3, 每个BLOCK出一个系数。
 * @param outputLocal 输出tensor [row, col]
 * @param outputScale 输出每行的反量化系数 [row, col/32]
 * @param inputLocal 输入tensor [row, col]
 * @param tmpLocal 临时buffer,所需空间大小row * col * 2
 * @param row 待处理的行数
 * @param col 待处理的列数
 */
/**
 shared_exp = floor(log2(max(|Vi|))) - emax
 mxscale = 2^shared_exp
 Pi = cast_to_dst_type(Vi/mxscale, round_mode)
**/
template <typename T, typename U>
__aicore__ inline void DynamicQuantPerBlockMxfp8Vf(const LocalTensor<int8_t> &outputLocal,
                                                   const LocalTensor<uint16_t> &outputScale,
                                                   const LocalTensor<T> &inputLocal,
                                                   const LocalTensor<uint8_t> &tmpLocal, uint32_t row, uint32_t col)
{
    LocalTensor<uint16_t> maxExpLocal = tmpLocal.ReinterpretCast<uint16_t>();
    uint32_t totalScaleInUB = row * col / FP8_E4M3FN_BLOCK_SIZE;
    uint32_t totalCountInUB = row * col;
    uint16_t vecLen = AscendC::VECTOR_REG_WIDTH / sizeof(T);
    uint16_t loopNum = (totalCountInUB + vecLen * DIGIT_TWO - 1) / (vecLen * DIGIT_TWO);
    uint16_t loopNumScale = (totalScaleInUB + vecLen - 1) / vecLen;

    auto srcAddr = reinterpret_cast<__ubuf__ T *>(inputLocal.GetPhyAddr());
    auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(maxExpLocal.GetPhyAddr());
    auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(outputScale.GetPhyAddr());
    LocalTensor<uint16_t> halfScaleLocal = maxExpLocal[totalCountInUB].template ReinterpretCast<uint16_t>();
    auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(halfScaleLocal.GetPhyAddr());
    auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t *>(outputLocal.GetPhyAddr());
    ComputeMaxExpVF<T, U>(srcAddr, maxExpAddr, totalCountInUB, loopNum, vecLen);
    ComputeScaleVF<T, U>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale, vecLen);

    srcAddr = reinterpret_cast<__ubuf__ T *>(inputLocal.GetPhyAddr());
    halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(halfScaleLocal.GetPhyAddr());

    uint32_t totalCountInUB2 = totalCountInUB * DIGIT_TWO;
    ComputeDataVF<T, U>(srcAddr, halfScaleLocalAddr, outLocalAddr, totalCountInUB, totalCountInUB2, loopNum, vecLen);
}

/**
 * @brief QuantPerTileVF 计算一行中每个tile的最大值，并计算量化参数和量化后的激活
 * @param outputLocal 输出tensor [row * col]，row为rmsnorm输出的结果，均为1
 * @param inputLocal 输入tensor [row * col]
 * @param quantScaleLocal 量化参数tensor [row, col / tileSize]
 * @param row 处理数据的行数，默认为1 后续可拓展
 * @param col 处理数据的列数
 * @param tileSize tile的大小 当前只支持128，且col可被tileSize整除
 */
template <typename T, typename C, typename O>
__aicore__ inline void QuantPerTileVF(const LocalTensor<O> &outputLocal, const LocalTensor<T> &inputLocal,
                                      const LocalTensor<T> &quantScaleLocal, const uint32_t row, const uint32_t col,
                                      const uint32_t tileSize)
{
    uint32_t cnt = row * col;
    __ubuf__ T *inputBuf = (__ubuf__ T *)inputLocal.GetPhyAddr();
    __ubuf__ T *quantScaleBuf = (__ubuf__ T *)quantScaleLocal.GetPhyAddr();
    __ubuf__ O *outputBuf = (__ubuf__ O *)outputLocal.GetPhyAddr();
    uint32_t dtypeSize = sizeof(float);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoop = (tileSize + VL - 1) / VL;
    constexpr float maxValue = std::is_same<O, fp8_e4m3fn_t>::value ? FP8_E4M3FN_MAX_VALUE :
                               std::is_same<O, hifloat8_t>::value   ? HIFLOAT8_MAX_VALUE :
                                                                      INT8_MAX_VALUE;
    const float alphaValue = static_cast<float>(1.0) / maxValue;
    uint32_t loopCount = cnt / tileSize;
    for (uint32_t rowIndex = 0; rowIndex < loopCount; rowIndex++) {
        ComputeVFImpl<T, C, O, PER_TILE_QUANT_MODE>(inputBuf, outputBuf, quantScaleBuf + rowIndex, rowIndex, tileSize,
                                                    dtypeSize, VL, vfLoop, alphaValue);
    }
}

} // namespace MlaProlog
#endif
