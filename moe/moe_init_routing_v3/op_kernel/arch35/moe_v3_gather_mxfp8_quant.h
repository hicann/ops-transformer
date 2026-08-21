/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_gather_mxfp8_quant.h
 * \brief
 */
#ifndef MOE_V3_GATHER_MXFP8_QUANT_H_REGBASE
#define MOE_V3_GATHER_MXFP8_QUANT_H_REGBASE

#include "moe_v3_common.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr uint16_t MXFP8_BF16_ABS_MASK = 0x7fff;
constexpr uint16_t MXFP8_BF16_AMAX_CLAMP = 0x38d2;
constexpr uint16_t MXFP8_BF16_MANT_MASK = 0x007f;
constexpr uint16_t MXFP8_FP8_MAX_MANT = 0x0060;
constexpr uint16_t MXFP8_E8M0_INV_EXP_SUB = 0x00fe;
constexpr uint16_t MXFP8_E5M2_EMAX = 15;
constexpr uint16_t MXFP8_E4M3_EMAX = 8;

template <typename T, typename U, bool CLAMP_AMAX>
__simd_vf__ inline void vfComputeAmax(__ubuf__ T *xAddr, __ubuf__ uint16_t *amaxOutAddr, uint32_t xElemNum,
                                      uint16_t vfLoopNum, uint32_t vlForT, uint32_t numVRegBlocks)
{
    using namespace AscendC::MicroAPI;

    // 用于把float16转为bfloat16，不涉及宽度变化
    static constexpr CastTrait traitFP16ToBF16 = {RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_TRUNC};

    // 0存奇数位元素，1存偶数位元素
    RegTensor<T> x0, x1;
    RegTensor<bfloat16_t> x0BF16, x1BF16;
    RegTensor<uint16_t> abs0, abs1, exp0FP16, exp1FP16, amax;
    // 存储FP16/BF16的指数位为1的mask
    RegTensor<uint16_t> emaskFP16, emaskBF16, absMask;
    Duplicate(emaskFP16, FP16_EMASK_AND_INF_VAL);
    Duplicate(emaskBF16, BF16_EMASK_AND_INF_VAL);
    Duplicate(absMask, MXFP8_BF16_ABS_MASK);
    // 2字节Reg的MaskALL
    MaskReg maskAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg mask0, mask1, mask0FP16NanInf, mask1FP16NanInf;
    // 非对齐搬出至UB用
    UnalignRegForStore uReg;

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        mask0 = UpdateMask<T>(xElemNum);
        mask1 = UpdateMask<T>(xElemNum);
        MaskDeInterleave<T>(mask0, mask1, mask0, mask1);

        // 双搬入，奇数位放x0、偶数位放x1
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr, vlForT * 2);

        if constexpr (IsSameType<T, half>::value) {
            // 单独提取fp16中inf/nan的元素
            And(exp0FP16, (RegTensor<uint16_t> &)x0, emaskFP16, mask0);
            And(exp1FP16, (RegTensor<uint16_t> &)x1, emaskFP16, mask1);
            Compare<uint16_t, CMPMODE::EQ>(mask0FP16NanInf, exp0FP16, emaskFP16, mask0);
            Compare<uint16_t, CMPMODE::EQ>(mask1FP16NanInf, exp1FP16, emaskFP16, mask1);

            // fp16要先转成bf16
            Cast<bfloat16_t, T, traitFP16ToBF16>(x0BF16, x0, mask0);
            Cast<bfloat16_t, T, traitFP16ToBF16>(x1BF16, x1, mask1);
            // 截断为BF16后取绝对值，保留完整指数和尾数，供RoundScale判断尾数进位。
            And(abs0, (RegTensor<uint16_t> &)x0BF16, absMask, mask0);
            And(abs1, (RegTensor<uint16_t> &)x1BF16, absMask, mask1);

            // FP16的Inf/NaN在截断前检测，统一按BF16 Inf参与后续特殊值处理。
            Select(abs0, emaskBF16, abs0, mask0FP16NanInf);
            Select(abs1, emaskBF16, abs1, mask1FP16NanInf);
        } else {
            // BF16直接取绝对值，NaN/Inf会在scale计算阶段统一识别。
            And(abs0, (RegTensor<uint16_t> &)x0, absMask, mask0);
            And(abs1, (RegTensor<uint16_t> &)x1, absMask, mask1);
        }
        // 先计算奇数位、偶数位之间的最大值，即每2个元素的最大值，放在maxExp
        // 由于mask0和mask1相同或是比mask1多一个元素，使用mask0保证不漏元素
        Max(amax, abs0, abs1, mask0);
        // 再每32B（对于T来说就是每16个元素）取一个最大值，此时amax里放的是每2个元素的最大值，且末尾数据为0，因此用MaskALL
        Reg::ReduceDataBlock<Reg::ReduceType::MAX>(amax, amax, maskAllB16);
        // 非对齐搬出，每次写出numVRegBlocks个元素（ReduceMaxWithDataBlock后放在amax头部）
        StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(amaxOutAddr, amax, uReg, numVRegBlocks);
    }
    // 非对齐搬出收尾，尾部多出的位置写0
    StoreUnAlignPost(amaxOutAddr, uReg, 0);
}

template <typename T, typename U, bool CLAMP_AMAX>
__simd_vf__ inline void vfComputeRoundScale(__ubuf__ uint16_t *amaxInAddr, __ubuf__ uint16_t *mxScaleOutAddr,
                                            __ubuf__ uint16_t *invScaleOutAddr, uint32_t scaleElemNum,
                                            uint32_t validScaleElemNum, uint16_t vfLoopNum, uint32_t vlForT,
                                            uint16_t fp8EmaxValue)
{
    using namespace AscendC::MicroAPI;

    RegTensor<uint16_t> amax, expBits, mantBits, mantAdd, roundedScale, mxScale, invExp, invScale;
    // infBF16存放bf16指数的inf值
    RegTensor<uint16_t> infBF16;
    Duplicate(infBF16, BF16_EMASK_AND_INF_VAL);
    // zeroB16存放B16长度的0值
    RegTensor<uint16_t> zeroB16;
    Duplicate(zeroB16, 0);
    RegTensor<uint16_t> oneB16;
    Duplicate(oneB16, 1);
    RegTensor<uint16_t> fp8Emax;
    Duplicate(fp8Emax, fp8EmaxValue);
    RegTensor<uint16_t> mantMask;
    Duplicate(mantMask, MXFP8_BF16_MANT_MASK);
    RegTensor<uint16_t> fp8MaxMant;
    Duplicate(fp8MaxMant, MXFP8_FP8_MAX_MANT);
    RegTensor<uint16_t> amaxClamp;
    Duplicate(amaxClamp, MXFP8_BF16_AMAX_CLAMP);
    // nanForE8M0以16位的低8位存放float8_e8m0的nan值
    RegTensor<uint16_t> nanForE8M0;
    Duplicate(nanForE8M0, FP8_E8M0_NAN_VAL);
    RegTensor<uint16_t> invExpSub;
    Duplicate(invExpSub, MXFP8_E8M0_INV_EXP_SUB);
    // nanBF16存放bf16的nan值
    RegTensor<uint16_t> nanBF16;
    Duplicate(nanBF16, BF16_NAN_VAL);
    // specialMinE8M0存放float8_e8m0的特殊极小值
    RegTensor<uint16_t> specialMinE8M0;
    Duplicate(specialMinE8M0, FP8_E8M0_SPECIAL_MIN);

    // 循环用mask
    MaskReg maskLoop, maskValid;
    // 存储compare后的结果用的mask
    MaskReg maskInfBF16, maskZero, maskLowExp, maskSpecialMin, maskMantHigh, maskClamp;

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        maskLoop = UpdateMask<uint16_t>(scaleElemNum);
        maskValid = UpdateMask<uint16_t>(validScaleElemNum);
        // 拷入vfComputeAmax算好的amax
        LoadAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(amax, amaxInAddr, vlForT);

        Duplicate(mxScale, 0, maskLoop);
        Duplicate(invScale, 0, maskLoop);

        if constexpr (CLAMP_AMAX) {
            Compare<uint16_t, CMPMODE::LT>(maskClamp, amax, amaxClamp, maskValid);
            Select<uint16_t>(amax, amaxClamp, amax, maskClamp);
        }

        // 1.计算并拷出mxScale（float8_e8m0）

        // roundedScale = ceil(log2(amax / fp8Max)) + E8M0_BIAS。
        // 对BF16正数，若尾数大于FP8最大值尾数(1.75)，scale指数需要向上进1。
        ShiftRights(expBits, amax, BF16_EXP_SHR_BITS, maskValid);
        And(mantBits, amax, mantMask, maskValid);
        Compare<uint16_t, CMPMODE::GT>(maskMantHigh, mantBits, fp8MaxMant, maskValid);
        Select<uint16_t>(mantAdd, oneB16, zeroB16, maskMantHigh);
        Compare<uint16_t, CMPMODE::LE>(maskLowExp, expBits, fp8Emax, maskValid);
        Sub<uint16_t>(roundedScale, expBits, fp8Emax, maskValid);
        Add<uint16_t>(roundedScale, roundedScale, mantAdd, maskValid);
        Select<uint16_t>(roundedScale, oneB16, roundedScale, maskLowExp);
        Select<uint16_t>(mxScale, roundedScale, mxScale, maskValid);

        // mxScale[amax>=infBF16]=nanE8M0
        Compare<uint16_t, CMPMODE::GE>(maskInfBF16, amax, infBF16, maskValid);
        Select<uint16_t>(mxScale, nanForE8M0, mxScale, maskInfBF16);
        // mxScale[amax==0]=0（CLAMP_AMAX=true时，0已被钳位到0.0001）
        Compare<uint16_t, CMPMODE::EQ>(maskZero, amax, zeroB16, maskValid);
        Select<uint16_t>(mxScale, zeroB16, mxScale, maskZero);

        // 拷出算好的mxScale，即为算子输出expandedScale的值。
        // 对于每个uint16的mxScale，都只拷出其低8位，也就是以uint8的长度拷出，实际上存储的就是float8_e8m0类型的二进制值，因此POST_MODE_UPDATE的数目为vlForT/2。
        StoreAlign<uint16_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B16>(mxScaleOutAddr, mxScale,
                                                                                      vlForT / 2, maskLoop);

        // 2.计算并拷出invScale（bfloat16）

        // invScale = 2^(-roundScaleExp)，以BF16指数位表示。
        Sub<uint16_t>(invExp, invExpSub, roundedScale, maskValid);
        ShiftLefts(invScale, invExp, BF16_EXP_SHR_BITS, maskValid);
        // invScale[amax==Inf/NaN]=nanBF16
        Select<uint16_t>(invScale, nanBF16, invScale, maskInfBF16);
        // invScale[amax==0]=0
        Select<uint16_t>(invScale, zeroB16, invScale, maskZero);
        // invScale指数下溢边界写为特殊极小值
        Compare<uint16_t, CMPMODE::EQ>(maskSpecialMin, roundedScale, invExpSub, maskValid);
        Select<uint16_t>(invScale, specialMinE8M0, invScale, maskSpecialMin);

        // 拷出算好的invScale，用于在后续量化xQuant=invScale*x，以uint16的长度拷出，实际存储的为bfloat16类型的二进制值
        StoreAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(invScaleOutAddr, invScale, vlForT, maskLoop);
    }
}

template <typename T, typename U>
__simd_vf__ inline void vfComputeData(__ubuf__ T *xAddr, __ubuf__ uint16_t *invScaleInAddr,
                                      __ubuf__ int8_t *xQuantOutAddr, uint32_t xElemNum, uint16_t vfLoopNum,
                                      uint32_t vlForT, uint32_t numVRegBlocks)
{
    using namespace AscendC::MicroAPI;

    // fp16转bf16，不涉及宽度变化，精度截断
    static constexpr CastTrait traitFP16ToBF16 = {RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_TRUNC};
    // b16转b32，奇数位的2字节->4字节，元素个数减半
    static constexpr CastTrait traitB16ToB32Layout0 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                       RoundMode::UNKNOWN};
    // b16转b32，偶数位的2字节->4字节，元素个数减半
    static constexpr CastTrait traitB16ToB32Layout1 = {RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                       RoundMode::UNKNOWN};
    // b32转b8，每个元素4字节->1字节，放在原来4字节的低位，超出b8的值写b8max，精度截断
    static constexpr CastTrait traitB32ToB8Layout0 = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                      RoundMode::CAST_RINT};

    RegTensor<uint16_t> invScale;
    RegTensor<float> invScaleFP32;
    RegTensor<T> x0, x1;
    RegTensor<float> x0FP32Layout0, x0FP32Layout1, x1FP32Layout0, x1FP32Layout1;
    RegTensor<U> xQuant0, xQuant1, xQuant2, xQuant3;

    MaskReg maskXQuant0B32, maskXQuant1B32, maskXQuant2B32, maskXQuant3B32;
    MaskReg maskAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg maskAllB32 = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        maskXQuant0B32 = UpdateMask<float>(xElemNum);
        maskXQuant1B32 = UpdateMask<float>(xElemNum);
        maskXQuant2B32 = UpdateMask<float>(xElemNum);
        maskXQuant3B32 = UpdateMask<float>(xElemNum);

        // 拷入待量化的x：x0存放偶数位的值，x1存放奇数位的值，双搬共拷入vlForT*2个元素
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr, vlForT * 2);
        // 拷入invScale，类型为uint16（计算时翻译成bfloat16），其中每个元素广播至32B，这里就是每个元素重复16次，共256B/32B=8个元素，也就是numVRegBlocks的值
        // 应该是加载(256/32=8)B的元素
        LoadAlign<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(invScale, invScaleInAddr,
                                                                                   numVRegBlocks);
        if constexpr (IsSameType<T, half>::value) {
            // 对于T为float16，先把x和invScale转成float32，再计算x*invScale后转为float8对应类型

            // 0.invScaleFP32存放invScale转为FP32后的值，invScale每个元素重复16次->invScaleFP32每个元素重复8次
            Cast<float, bfloat16_t, traitB16ToB32Layout0>(invScaleFP32, (RegTensor<bfloat16_t> &)invScale, maskAllB16);

            // 1.对奇数位元素x0进行量化：
            // x0FP32Layout0放x0的奇数位转为fp32的值，x0FP32Layout1放x0的偶数位转fp32的值
            Cast<float, T, traitB16ToB32Layout0>(x0FP32Layout0, x0, maskAllB16);
            Cast<float, T, traitB16ToB32Layout1>(x0FP32Layout1, x0, maskAllB16);
            // x0FP32*=invScaleFP32，这里每8个x0FP32Layout0、x0FP32Layout1对应一个invScaleFP32，加起来共16个x0FP32对应1个invScale
            Mul(x0FP32Layout0, x0FP32Layout0, invScaleFP32, maskAllB32);
            Mul(x0FP32Layout1, x0FP32Layout1, invScaleFP32, maskAllB32);
            // 把x0FP32Layout0、x0FP32Layout1由奇偶交织恢复为连续存储
            Interleave(x0FP32Layout0, x0FP32Layout1, x0FP32Layout0, x0FP32Layout1);

            // 2.对偶数位元素x1进行量化，与x0的做法一致：
            Cast<float, T, traitB16ToB32Layout0>(x1FP32Layout0, x1, maskAllB16);
            Cast<float, T, traitB16ToB32Layout1>(x1FP32Layout1, x1, maskAllB16);
            Mul(x1FP32Layout0, x1FP32Layout0, invScaleFP32, maskAllB32);
            Mul(x1FP32Layout1, x1FP32Layout1, invScaleFP32, maskAllB32);
            Interleave(x1FP32Layout0, x1FP32Layout1, x1FP32Layout0, x1FP32Layout1);

            // 3.把量化后的x0FP32Layout[i]、x1FP32Layout0[i]恢复顺序：
            // 按x的原始顺序依次为：x0FP32Layout0、x1FP32Layout0、x0FP32Layout1、x1FP32Layout1
            Interleave(x0FP32Layout0, x1FP32Layout0, x0FP32Layout0, x1FP32Layout0);
            Interleave(x0FP32Layout1, x1FP32Layout1, x0FP32Layout1, x1FP32Layout1);

            // 4.挨个cast回目标fp8类型
            Cast<U, float, traitB32ToB8Layout0>(xQuant0, x0FP32Layout0, maskXQuant0B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant1, x1FP32Layout0, maskXQuant1B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant2, x0FP32Layout1, maskXQuant2B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant3, x1FP32Layout1, maskXQuant3B32);
        } else {
            // 对于T为bfloat16，直接计算x*invScale，然后将结果先转为float32，再转为float8对应类型

            // 1.乘以invScale
            // x[i]*= invScale，这里invScale每元素重复16次，刚好对于x0、x1就是每32元素一个invScale
            Mul(x0, x0, (RegTensor<T> &)invScale, maskAllB16);
            Mul(x1, x1, (RegTensor<T> &)invScale, maskAllB16);
            // 恢复x0、x1为原始顺序
            Interleave(x0, x1, x0, x1);

            // 2.x0部分转为FP32，再转为FP8目标类型：
            // x0FP32Layout0存x0转为fp32的奇数位元素，x0FP32Layout1存x0转为fp32的偶数位元素，各64个
            Cast<float, T, traitB16ToB32Layout0>(x0FP32Layout0, x0, maskAllB16);
            Cast<float, T, traitB16ToB32Layout1>(x0FP32Layout1, x0, maskAllB16);
            // 恢复x0FP32Layout0、x0FP32Layout1的顺序
            Interleave(x0FP32Layout0, x0FP32Layout1, x0FP32Layout0, x0FP32Layout1);
            // 转为FP8对应类型，每个4字节x->1字节xQuant，但个数不增加，还是各64个
            Cast<U, float, traitB32ToB8Layout0>(xQuant0, x0FP32Layout0, maskXQuant0B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant1, x0FP32Layout1, maskXQuant1B32);

            // 3.x1部分转为FP32，再转为FP8目标类型：
            Cast<float, T, traitB16ToB32Layout0>(x1FP32Layout0, x1, maskAllB16);
            Cast<float, T, traitB16ToB32Layout1>(x1FP32Layout1, x1, maskAllB16);
            Interleave(x1FP32Layout0, x1FP32Layout1, x1FP32Layout0, x1FP32Layout1);
            Cast<U, float, traitB32ToB8Layout0>(xQuant2, x1FP32Layout0, maskXQuant2B32);
            Cast<U, float, traitB32ToB8Layout0>(xQuant3, x1FP32Layout1, maskXQuant3B32);
        }

        // 拷出xQuant到xOutAddr。OUT_ELE_NUM_ONE_BLK=64，即一个VL下有64个float32->fp8的元素，一次拷出64个元素
        StoreAlign<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant0, OUT_ELE_NUM_ONE_BLK, maskXQuant0B32);
        StoreAlign<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant1, OUT_ELE_NUM_ONE_BLK, maskXQuant1B32);
        StoreAlign<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant2, OUT_ELE_NUM_ONE_BLK, maskXQuant2B32);
        StoreAlign<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant3, OUT_ELE_NUM_ONE_BLK, maskXQuant3B32);
    }
}

template <typename T, typename U, bool CLAMP_AMAX = false>
class MoeGatherOutMxfp8Quant {
public:
    __aicore__ inline MoeGatherOutMxfp8Quant(){};
    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr, GM_ADDR workspace, GM_ADDR expandedRowIdxAddr,
                                GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitKernelTiling(GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData);
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void ScatterCopyExpandedXandMXQuant(int64_t progress);
    __aicore__ inline void GatherCopyExpandedXandMXQuant(int64_t progress);
    __aicore__ inline void CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols);
    __aicore__ inline void Compute(uint32_t xElemNum, uint32_t scaleElemNum, uint32_t validScaleElemNum);
    __aicore__ inline void CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols, int64_t loopScaleCols);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, GATHER_OUT_BUFFER_NUM> xInQueue_;
    TQue<QuePosition::VECIN, GATHER_OUT_BUFFER_NUM> sortedRowIdxInQueue_;
    TQue<QuePosition::VECOUT, GATHER_OUT_BUFFER_NUM> xQuantOutQueue_;
    TQue<QuePosition::VECOUT, GATHER_OUT_BUFFER_NUM> mxScaleOutQueue_;
    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> invScaleBuffer_;

    GlobalTensor<T> xInGm_;
    GlobalTensor<uint8_t> expandedXOutGm_;
    GlobalTensor<uint8_t> expandedScaleOutGm_;
    GlobalTensor<int32_t> sortedRowIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t validScaleCols_; // 一个token实际有意义的scale有多少个元素（列）
    int64_t
        scaleCols_; // 一个token的scale有多少个元素（列），即CeilAlign(CeliDiv(h,32),2)，在actualScaleCols_为奇数时，scaleCols_=validScaleCols_+1
    int64_t n_;
    int64_t k_;
    int64_t perCoreRow_;
    int64_t currentLoopRows_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t perLoopScaleCols_;
    int64_t lastLoopValidScaleCols_;
    int64_t lastLoopScaleCols_;
    int64_t indicesOffset_;
    int64_t rowIdxType_ = 0;
    int64_t useGatherCopy_ = 0;

    uint16_t fp8Emax_ = 0;

    // 一个RegTensor的长度Bytes
    const uint32_t vRegSize_ = Ops::Base::GetVRegSize();
    // ub的一个块大小，通常是32B
    const uint32_t ubBlockSize_ = Ops::Base::GetUbBlockSize();
    // 一个RegTensor能塞下多少T(B16)类型的元素
    const uint32_t vlForB16_ = vRegSize_ / sizeof(T);
    // 一个RegTensor有几个UbBlock，即ReduceWithBlock后的元素个数（256B/32B=8）
    // 这里如果ubBlockSize_获取的值为0，应该让kernel挂掉
    const uint32_t numUbBlocksPerVReg_ = vRegSize_ / ubBlockSize_; // 8个元素
    // 每个ubBlock对应多少T类型的元素
    const uint32_t numElemPerUbBlock_ = ubBlockSize_ / sizeof(T);
};

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, CLAMP_AMAX>::Init(
    GM_ADDR xAddr, GM_ADDR unused_ScaleAddr, GM_ADDR workspace, GM_ADDR expandedRowIdxAddr, GM_ADDR expandedXAddr,
    GM_ADDR expandedScaleAddr, const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
#endif

    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    InitKernelTiling(workspace, tilingData);

    xInGm_.SetGlobalBuffer((__gm__ T *)xAddr);
    expandedXOutGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedXAddr);
    if (useGatherCopy_) {
        if (rowIdxType_ == SCATTER) {
            sortedRowIdxGm_.SetGlobalBuffer(
                (__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) + blockIdx_ * perCoreRow_,
                Align(perCoreRow_, sizeof(int32_t)));
        } else {
            sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxAddr + blockIdx_ * perCoreRow_,
                                            Align(perCoreRow_, sizeof(int32_t)));
        }
    } else {
        if (rowIdxType_ == SCATTER) {
            sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxAddr + blockIdx_ * perCoreRow_,
                                            Align(perCoreRow_, sizeof(int32_t)));
        } else {
            sortedRowIdxGm_.SetGlobalBuffer(
                (__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) + blockIdx_ * perCoreRow_,
                Align(perCoreRow_, sizeof(int32_t)));
        }
    }
    expandedScaleOutGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedScaleAddr);

    // perrows * 2 * 2 * 4 expandRowIdx + sortedExpertId
    pipe_->InitBuffer(sortedRowIdxInQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(xInQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(xQuantOutQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopCols_ / 4, sizeof(int8_t)) * 4);
    pipe_->InitBuffer(mxScaleOutQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopScaleCols_, sizeof(int8_t)));
    pipe_->InitBuffer(maxExpBuffer_, AlignBytes(perLoopScaleCols_, sizeof(T)));
    pipe_->InitBuffer(invScaleBuffer_, AlignBytes(perLoopScaleCols_, sizeof(T)));
    if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        fp8Emax_ = MXFP8_E4M3_EMAX;
    } else {
        fp8Emax_ = MXFP8_E5M2_EMAX;
    }
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, CLAMP_AMAX>::InitKernelTiling(
    GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData)
{
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    validScaleCols_ = Ops::Base::CeilDiv<int64_t>(cols_, MX_BLOCK_SIZE);
    scaleCols_ = Ops::Base::CeilAlign<int64_t>(validScaleCols_, 2); // CeilDiv(h, 32)后再向上到2的倍数（偶数）
    n_ = tilingData->n;
    k_ = tilingData->k;
    rowIdxType_ = tilingData->rowIdxType;
    useGatherCopy_ = tilingData->useGatherCopy;

    // core split
    int64_t actualExpertNum_ = tilingData->actualExpertNum;
    int64_t scanRowCount = n_ * k_;
    if (!useGatherCopy_) {
        expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                                Align(actualExpertNum_, sizeof(int32_t)),
                                            1);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expertTotalCountGm_);
        scanRowCount = expertTotalCountGm_.GetValue(0);
    }
    perCoreRow_ = Ceil(scanRowCount, tilingData->coreNum);
    needCoreNum_ = Ceil(scanRowCount, perCoreRow_);
    int64_t lastCoreIndicesElements = scanRowCount - (needCoreNum_ - 1) * perCoreRow_;

    // inner core split
    coreRows_ = perCoreRow_;
    int64_t originPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements;
        originPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    }
    perLoopRows_ = Min(coreRows_, originPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;

    // cols split
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;
    perLoopScaleCols_ = perLoopCols_ / MX_BLOCK_SIZE; // perLoopCols_在tiling侧计算，已经对齐到32的整数倍了
    lastLoopValidScaleCols_ = validScaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
    lastLoopScaleCols_ = scaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, CLAMP_AMAX>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
            CopyInExpandedExpertIdx(loop);
            if (!useGatherCopy_) {
                ScatterCopyExpandedXandMXQuant(loop);
            } else {
                GatherCopyExpandedXandMXQuant(loop);
            }
        }

        currentLoopRows_ = lastLoopRows_;
        CopyInExpandedExpertIdx(rowLoops_ - 1);
        if (!useGatherCopy_) {
            ScatterCopyExpandedXandMXQuant(rowLoops_ - 1);
        } else {
            GatherCopyExpandedXandMXQuant(rowLoops_ - 1);
        }
    }
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, CLAMP_AMAX>::CopyInExpandedExpertIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, sortedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    sortedRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, CLAMP_AMAX>::ScatterCopyExpandedXandMXQuant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    for (int64_t index = 0; index < currentLoopRows_; index++) {
        // 对加载进ub的indicesLocal按行循环
        int32_t srcIdx = indicesLocal.GetValue(index);
        int64_t dstIdx = perCoreRow_ * blockIdx_ + perLoopRows_ * progress + index;
        for (int64_t j = 0; j < colLoops_; j++) {
            // 每行切分成cols，按cols读入-计算量化-拷出
            int64_t loopCols = (j == colLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
            uint32_t loopScaleCols = (j == colLoops_ - 1) ? lastLoopScaleCols_ : perLoopScaleCols_;
            uint32_t loopValidScaleCols = (j == colLoops_ - 1) ? lastLoopValidScaleCols_ : perLoopScaleCols_;
            CopyIn(srcIdx / k_, j, loopCols);
            Compute(loopCols, loopScaleCols, loopValidScaleCols);
            CopyOut(dstIdx, j, loopCols, loopScaleCols);
        }
    }
    sortedRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, CLAMP_AMAX>::GatherCopyExpandedXandMXQuant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    for (int64_t j = 0; j < colLoops_; j++) {
        int64_t loopCols = (j == colLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
        uint32_t loopScaleCols = (j == colLoops_ - 1) ? lastLoopScaleCols_ : perLoopScaleCols_;
        uint32_t loopValidScaleCols = (j == colLoops_ - 1) ? lastLoopValidScaleCols_ : perLoopScaleCols_;

        int64_t globalSortIdx = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int64_t curLoopRow = 0;
        int64_t currentLoopStartRow = globalSortIdx / k_;
        int64_t currentLoopLastRow = (globalSortIdx + currentLoopRows_ - 1) / k_;

        for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
            bool hasValidOut = false;
            while (curLoopRow < currentLoopRows_ && globalSortIdx / k_ == row) {
                if (indicesLocal.GetValue(curLoopRow) >= 0) {
                    hasValidOut = true;
                    break;
                }
                curLoopRow++;
                globalSortIdx++;
            }
            if (!hasValidOut) {
                continue;
            }

            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            CopyIn(row, j, loopCols);
            Compute(loopCols, loopScaleCols, loopValidScaleCols);

            LocalTensor<uint8_t> outLocal = xQuantOutQueue_.DeQue<uint8_t>();
            LocalTensor<uint8_t> mxScaleLocal = mxScaleOutQueue_.DeQue<uint8_t>();

            DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(loopCols * sizeof(uint8_t)), 0, 0, 0};
            DataCopyExtParams copyScaleParams = {1, static_cast<uint32_t>(loopScaleCols * sizeof(uint8_t)), 0, 0, 0};
            while (curLoopRow < currentLoopRows_ && globalSortIdx / k_ == row) {
                int32_t outIndex = indicesLocal.GetValue(curLoopRow);
                curLoopRow++;
                globalSortIdx++;
                if (outIndex < 0) {
                    continue;
                }
                int64_t outOffset = static_cast<int64_t>(outIndex) * cols_ + j * perLoopCols_;
                DataCopyPad<uint8_t>(expandedXOutGm_[outOffset], outLocal, copyOutParams);

                int64_t outScaleOffset = static_cast<int64_t>(outIndex) * scaleCols_ + j * perLoopScaleCols_;
                DataCopyPad<uint8_t>(expandedScaleOutGm_[outScaleOffset], mxScaleLocal, copyScaleParams);
            }

            xQuantOutQueue_.FreeTensor(outLocal);
            mxScaleOutQueue_.FreeTensor(mxScaleLocal);
        }
    }
    sortedRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, CLAMP_AMAX>::CopyIn(int64_t srcIdx, int64_t colIdx,
                                                                        int64_t loopCols)
{
    LocalTensor<T> inLocal = xInQueue_.AllocTensor<T>();
    DataCopyExtParams copyInParam = {1, static_cast<uint32_t>(loopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    int64_t loopColsTail = loopCols % MX_BLOCK_SIZE;
    if (loopColsTail != 0) {
        padParams.isPad = true;
        if (loopColsTail > numElemPerUbBlock_) {
            padParams.rightPadding = numElemPerUbBlock_ - (loopColsTail - numElemPerUbBlock_);
        } else {
            padParams.rightPadding = numElemPerUbBlock_;
        }
    }
    DataCopyPad(inLocal, xInGm_[srcIdx * cols_ + colIdx * perLoopCols_], copyInParam, padParams);
    xInQueue_.EnQue(inLocal);
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, CLAMP_AMAX>::Compute(uint32_t xElemNum, uint32_t scaleElemNum,
                                                                         uint32_t validScaleElemNum)
{
    // deque input
    LocalTensor<T> xLocal = xInQueue_.DeQue<T>();
    auto xLocalAddr = reinterpret_cast<__ubuf__ T *>(xLocal.GetPhyAddr());
    // alloc outputs
    LocalTensor<int8_t> xQuantLocal = xQuantOutQueue_.AllocTensor<int8_t>();
    auto xQuantLocalAddr = reinterpret_cast<__ubuf__ int8_t *>(xQuantLocal.GetPhyAddr());
    LocalTensor<uint16_t> mxScaleLocal = mxScaleOutQueue_.AllocTensor<uint16_t>();
    auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(mxScaleLocal.GetPhyAddr());
    // get tmp buffers
    LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();
    auto maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(maxExpLocal.GetPhyAddr());
    LocalTensor<uint16_t> invScaleLocal = invScaleBuffer_.Get<uint16_t>();
    auto invScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(invScaleLocal.GetPhyAddr());

    // 按每次双搬vlForHalfNumber_*2来计算loopNum
    uint16_t vfLoopNumForX = (xElemNum + vlForB16_ * 2 - 1) / (vlForB16_ * 2);
    uint16_t vfLoopNumForScale = (scaleElemNum + vlForB16_ - 1) / vlForB16_;

    VF_CALL<vfComputeAmax<T, U, CLAMP_AMAX>>(xLocalAddr, maxExpLocalAddr, xElemNum, vfLoopNumForX, vlForB16_,
                                             numUbBlocksPerVReg_);
    VF_CALL<vfComputeRoundScale<T, U, CLAMP_AMAX>>(maxExpLocalAddr, mxScaleLocalAddr, invScaleLocalAddr, scaleElemNum,
                                                   validScaleElemNum, vfLoopNumForScale, vlForB16_, fp8Emax_);
    VF_CALL<vfComputeData<T, U>>(xLocalAddr, invScaleLocalAddr, xQuantLocalAddr, xElemNum, vfLoopNumForX, vlForB16_,
                                 numUbBlocksPerVReg_);

    // free input
    xInQueue_.FreeTensor(xLocal);
    // enque outputs
    xQuantOutQueue_.EnQue(xQuantLocal);
    mxScaleOutQueue_.EnQue(mxScaleLocal);
}

template <typename T, typename U, bool CLAMP_AMAX>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, CLAMP_AMAX>::CopyOut(int64_t dstIdx, int64_t colIdx,
                                                                         int64_t loopCols, int64_t loopScaleCols)
{
    LocalTensor<uint8_t> mxScaleLocal = mxScaleOutQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> outLocal = xQuantOutQueue_.DeQue<uint8_t>();

    DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(loopCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad<uint8_t>(expandedXOutGm_[dstIdx * cols_ + colIdx * perLoopCols_], outLocal, copyOutParams);

    DataCopyExtParams copyScaleParams = {1, static_cast<uint32_t>(loopScaleCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad<uint8_t>(expandedScaleOutGm_[dstIdx * scaleCols_ + colIdx * perLoopScaleCols_], mxScaleLocal,
                         copyScaleParams);

    xQuantOutQueue_.FreeTensor(outLocal);
    mxScaleOutQueue_.FreeTensor(mxScaleLocal);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_MXFP8_QUANT_H_REGBASE
