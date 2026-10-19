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
 * \file vf_flashupdate_trans_fp16.h
 * \brief FP16 softmax V2: VfFlashUpdateAcc（跨S2累积）+ VfDivAndCast（除cast转置带pad输出）
 */

#ifndef VF_FLASHUPDATE_TRANS_FP16_H_
#define VF_FLASHUPDATE_TRANS_FP16_H_

#include "kernel_tensor.h"
#include "vf_basic_block_utils.h"

namespace FaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

template <bool isUpdate>
__simd_vf__ inline void VfFlashUpdateAccVF(__ubuf__ float* dstOut, __ubuf__ float* dstSum, __ubuf__ float* cur,
                                           __ubuf__ float* rowsum, __ubuf__ float* factor)
{
    RegTensor<float> vreg_factor_a;
    RegTensor<float> vreg_factor_b;
    RegTensor<float> vreg_sum_acc;
    RegTensor<float> vreg_sum_new;
    RegTensor<float> vreg_out_0, vreg_out_1, vreg_out_2, vreg_out_3;
    RegTensor<float> vreg_cur_0, vreg_cur_1, vreg_cur_2, vreg_cur_3;
    MaskReg preg_f32 = CreateMask<float, MaskPattern::ALL>();

    // 首块：纯 Load→Store 拷贝起步（消费者统一 V 管道）
    if constexpr (!isUpdate) {
        LoadAlign(vreg_sum_new, rowsum);
        StoreAlign(dstSum, vreg_sum_new, preg_f32);
        LoadAlign(vreg_sum_new, rowsum + 64);
        StoreAlign(dstSum + 64, vreg_sum_new, preg_f32);
        for (uint32_t row = 0; row < 128; row += 4) {
            uint32_t base0 = row * 128;
            uint32_t base1 = (row + 1) * 128;
            uint32_t base2 = (row + 2) * 128;
            uint32_t base3 = (row + 3) * 128;
            LoadAlign(vreg_out_0, cur + base0);
            LoadAlign(vreg_out_1, cur + base1);
            LoadAlign(vreg_out_2, cur + base2);
            LoadAlign(vreg_out_3, cur + base3);
            StoreAlign(dstOut + base0, vreg_out_0, preg_f32);
            StoreAlign(dstOut + base1, vreg_out_1, preg_f32);
            StoreAlign(dstOut + base2, vreg_out_2, preg_f32);
            StoreAlign(dstOut + base3, vreg_out_3, preg_f32);
            LoadAlign(vreg_out_0, cur + base0 + 64);
            LoadAlign(vreg_out_1, cur + base1 + 64);
            LoadAlign(vreg_out_2, cur + base2 + 64);
            LoadAlign(vreg_out_3, cur + base3 + 64);
            StoreAlign(dstOut + base0 + 64, vreg_out_0, preg_f32);
            StoreAlign(dstOut + base1 + 64, vreg_out_1, preg_f32);
            StoreAlign(dstOut + base2 + 64, vreg_out_2, preg_f32);
            StoreAlign(dstOut + base3 + 64, vreg_out_3, preg_f32);
        }
        return;
    }

    // 非首块：乘加累积
    LoadAlign(vreg_factor_a, factor);
    LoadAlign(vreg_factor_b, factor + 64);

    // dstSum = dstSum × factor + rowsum
    LoadAlign(vreg_sum_acc, dstSum);
    Mul(vreg_sum_acc, vreg_sum_acc, vreg_factor_a, preg_f32);
    LoadAlign(vreg_sum_new, rowsum);
    Add(vreg_sum_acc, vreg_sum_acc, vreg_sum_new, preg_f32);
    StoreAlign(dstSum, vreg_sum_acc, preg_f32);

    LoadAlign(vreg_sum_acc, dstSum + 64);
    Mul(vreg_sum_acc, vreg_sum_acc, vreg_factor_b, preg_f32);
    LoadAlign(vreg_sum_new, rowsum + 64);
    Add(vreg_sum_acc, vreg_sum_acc, vreg_sum_new, preg_f32);
    StoreAlign(dstSum + 64, vreg_sum_acc, preg_f32);

    // dstOut = dstOut × factor + cur，128行×2列半，4路行展开
    for (uint32_t row = 0; row < 128; row += 4) {
        uint32_t base0 = row * 128;
        uint32_t base1 = (row + 1) * 128;
        uint32_t base2 = (row + 2) * 128;
        uint32_t base3 = (row + 3) * 128;

        LoadAlign(vreg_out_0, dstOut + base0);
        LoadAlign(vreg_out_1, dstOut + base1);
        LoadAlign(vreg_out_2, dstOut + base2);
        LoadAlign(vreg_out_3, dstOut + base3);
        Mul(vreg_out_0, vreg_out_0, vreg_factor_a, preg_f32);
        Mul(vreg_out_1, vreg_out_1, vreg_factor_a, preg_f32);
        Mul(vreg_out_2, vreg_out_2, vreg_factor_a, preg_f32);
        Mul(vreg_out_3, vreg_out_3, vreg_factor_a, preg_f32);
        LoadAlign(vreg_cur_0, cur + base0);
        LoadAlign(vreg_cur_1, cur + base1);
        LoadAlign(vreg_cur_2, cur + base2);
        LoadAlign(vreg_cur_3, cur + base3);
        Add(vreg_out_0, vreg_out_0, vreg_cur_0, preg_f32);
        Add(vreg_out_1, vreg_out_1, vreg_cur_1, preg_f32);
        Add(vreg_out_2, vreg_out_2, vreg_cur_2, preg_f32);
        Add(vreg_out_3, vreg_out_3, vreg_cur_3, preg_f32);
        StoreAlign(dstOut + base0, vreg_out_0, preg_f32);
        StoreAlign(dstOut + base1, vreg_out_1, preg_f32);
        StoreAlign(dstOut + base2, vreg_out_2, preg_f32);
        StoreAlign(dstOut + base3, vreg_out_3, preg_f32);

        LoadAlign(vreg_out_0, dstOut + base0 + 64);
        LoadAlign(vreg_out_1, dstOut + base1 + 64);
        LoadAlign(vreg_out_2, dstOut + base2 + 64);
        LoadAlign(vreg_out_3, dstOut + base3 + 64);
        Mul(vreg_out_0, vreg_out_0, vreg_factor_b, preg_f32);
        Mul(vreg_out_1, vreg_out_1, vreg_factor_b, preg_f32);
        Mul(vreg_out_2, vreg_out_2, vreg_factor_b, preg_f32);
        Mul(vreg_out_3, vreg_out_3, vreg_factor_b, preg_f32);
        LoadAlign(vreg_cur_0, cur + base0 + 64);
        LoadAlign(vreg_cur_1, cur + base1 + 64);
        LoadAlign(vreg_cur_2, cur + base2 + 64);
        LoadAlign(vreg_cur_3, cur + base3 + 64);
        Add(vreg_out_0, vreg_out_0, vreg_cur_0, preg_f32);
        Add(vreg_out_1, vreg_out_1, vreg_cur_1, preg_f32);
        Add(vreg_out_2, vreg_out_2, vreg_cur_2, preg_f32);
        Add(vreg_out_3, vreg_out_3, vreg_cur_3, preg_f32);
        StoreAlign(dstOut + base0 + 64, vreg_out_0, preg_f32);
        StoreAlign(dstOut + base1 + 64, vreg_out_1, preg_f32);
        StoreAlign(dstOut + base2 + 64, vreg_out_2, preg_f32);
        StoreAlign(dstOut + base3 + 64, vreg_out_3, preg_f32);
    }
}

template <bool isUpdate>
__aicore__ inline void VfFlashUpdateAcc(const LocalTensor<float>& stage2Out, const LocalTensor<float>& accRowsum,
                                        const LocalTensor<float>& mm2Res, const LocalTensor<float>& expF32)
{
    constexpr uint32_t rowsumOff = 128 * 128;
    __ubuf__ float* dstOutPtr = reinterpret_cast<__ubuf__ float*>(stage2Out.GetPhyAddr());
    __ubuf__ float* dstSumPtr = reinterpret_cast<__ubuf__ float*>(accRowsum.GetPhyAddr());
    __ubuf__ float* curPtr = reinterpret_cast<__ubuf__ float*>(mm2Res.GetPhyAddr());
    __ubuf__ float* rowsumPtr = curPtr + rowsumOff;
    __ubuf__ float* factorPtr = reinterpret_cast<__ubuf__ float*>(expF32.GetPhyAddr());
    VfFlashUpdateAccVF<isUpdate>(dstOutPtr, dstSumPtr, curPtr, rowsumPtr, factorPtr);
}

/* **************************************************************************************************
 * VfDivAndCast — V2 末块：逐元素除 rowsum + cast bf16 + 转置带 pad 写
 * dst = bf16(src / rowsum)，融合转置输出到 mm2Res 别名区
 * rowsum 沿 S1 列方向逐元素（[DV,S1] 行主序红利：rowsum 分 2 regs 全程复用）
 * ************************************************************************************************* */
__simd_vf__ inline void VfDivAndCastVF(__ubuf__ bfloat16_t* dstT, __ubuf__ float* src, __ubuf__ float* rowsum)
{
    RegTensor<float> vreg_src_1, vreg_src_2, vreg_src_3, vreg_src_4;
    RegTensor<float> vreg_src_5, vreg_src_6, vreg_src_7, vreg_src_8;
    RegTensor<float> vreg_src_9, vreg_src_10, vreg_src_11, vreg_src_12;
    RegTensor<float> vreg_src_13, vreg_src_14, vreg_src_15, vreg_src_16;
    RegTensor<float> vreg_recip;
    RegTensor<float> vreg_ones;
    RegTensor<bfloat16_t> vreg_bf16_1;
    RegTensor<bfloat16_t> vreg_bf16_2;
    MaskReg preg_f32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_bf16 = CreateMask<bfloat16_t, MaskPattern::ALL>();

    Duplicate(vreg_ones, 1.0f);

    // 按列分成前后两半（每半 64 列）
    for (uint32_t colHalf = 0; colHalf < 2; colHalf++) {
        LoadAlign(vreg_recip, rowsum + colHalf * 64);
        Div(vreg_recip, vreg_ones, vreg_recip, preg_f32);

        // 按行分成 8 块（每块 16 行，覆盖 128 个头维度行）
        for (uint32_t block = 0; block < 8; block++) {
            uint32_t blockBase = block * 16 * 128 + colHalf * 64;
            LoadAlign(vreg_src_1, src + blockBase + 0 * 128);
            LoadAlign(vreg_src_2, src + blockBase + 1 * 128);
            LoadAlign(vreg_src_3, src + blockBase + 2 * 128);
            LoadAlign(vreg_src_4, src + blockBase + 3 * 128);
            LoadAlign(vreg_src_5, src + blockBase + 4 * 128);
            LoadAlign(vreg_src_6, src + blockBase + 5 * 128);
            LoadAlign(vreg_src_7, src + blockBase + 6 * 128);
            LoadAlign(vreg_src_8, src + blockBase + 7 * 128);
            LoadAlign(vreg_src_9, src + blockBase + 8 * 128);
            LoadAlign(vreg_src_10, src + blockBase + 9 * 128);
            LoadAlign(vreg_src_11, src + blockBase + 10 * 128);
            LoadAlign(vreg_src_12, src + blockBase + 11 * 128);
            LoadAlign(vreg_src_13, src + blockBase + 12 * 128);
            LoadAlign(vreg_src_14, src + blockBase + 13 * 128);
            LoadAlign(vreg_src_15, src + blockBase + 14 * 128);
            LoadAlign(vreg_src_16, src + blockBase + 15 * 128);
            Mul(vreg_src_1, vreg_src_1, vreg_recip, preg_f32);
            Mul(vreg_src_2, vreg_src_2, vreg_recip, preg_f32);
            Mul(vreg_src_3, vreg_src_3, vreg_recip, preg_f32);
            Mul(vreg_src_4, vreg_src_4, vreg_recip, preg_f32);
            Mul(vreg_src_5, vreg_src_5, vreg_recip, preg_f32);
            Mul(vreg_src_6, vreg_src_6, vreg_recip, preg_f32);
            Mul(vreg_src_7, vreg_src_7, vreg_recip, preg_f32);
            Mul(vreg_src_8, vreg_src_8, vreg_recip, preg_f32);
            Mul(vreg_src_9, vreg_src_9, vreg_recip, preg_f32);
            Mul(vreg_src_10, vreg_src_10, vreg_recip, preg_f32);
            Mul(vreg_src_11, vreg_src_11, vreg_recip, preg_f32);
            Mul(vreg_src_12, vreg_src_12, vreg_recip, preg_f32);
            Mul(vreg_src_13, vreg_src_13, vreg_recip, preg_f32);
            Mul(vreg_src_14, vreg_src_14, vreg_recip, preg_f32);
            Mul(vreg_src_15, vreg_src_15, vreg_recip, preg_f32);
            Mul(vreg_src_16, vreg_src_16, vreg_recip, preg_f32);
            // 8 对跨行 Cast+Or
            Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src_1, preg_f32);
            Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src_2, preg_f32);
            Or((RegTensor<uint16_t>&)vreg_src_1, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
               preg_bf16);
            Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src_3, preg_f32);
            Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src_4, preg_f32);
            Or((RegTensor<uint16_t>&)vreg_src_2, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
               preg_bf16);
            Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src_5, preg_f32);
            Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src_6, preg_f32);
            Or((RegTensor<uint16_t>&)vreg_src_3, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
               preg_bf16);
            Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src_7, preg_f32);
            Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src_8, preg_f32);
            Or((RegTensor<uint16_t>&)vreg_src_4, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
               preg_bf16);
            Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src_9, preg_f32);
            Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src_10, preg_f32);
            Or((RegTensor<uint16_t>&)vreg_src_5, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
               preg_bf16);
            Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src_11, preg_f32);
            Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src_12, preg_f32);
            Or((RegTensor<uint16_t>&)vreg_src_6, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
               preg_bf16);
            Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src_13, preg_f32);
            Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src_14, preg_f32);
            Or((RegTensor<uint16_t>&)vreg_src_7, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
               preg_bf16);
            Cast<bfloat16_t, float, castTraitZero>(vreg_bf16_1, vreg_src_15, preg_f32);
            Cast<bfloat16_t, float, castTraitOne>(vreg_bf16_2, vreg_src_16, preg_f32);
            Or((RegTensor<uint16_t>&)vreg_src_8, (RegTensor<uint16_t>&)vreg_bf16_1, (RegTensor<uint16_t>&)vreg_bf16_2,
               preg_bf16);
            // Interleave 三轮蝶形（8×8 单元网格转置）
            Interleave(vreg_src_1, vreg_src_5, vreg_src_1, vreg_src_5);
            Interleave(vreg_src_2, vreg_src_6, vreg_src_2, vreg_src_6);
            Interleave(vreg_src_3, vreg_src_7, vreg_src_3, vreg_src_7);
            Interleave(vreg_src_4, vreg_src_8, vreg_src_4, vreg_src_8);
            Interleave(vreg_src_1, vreg_src_3, vreg_src_1, vreg_src_3);
            Interleave(vreg_src_5, vreg_src_7, vreg_src_5, vreg_src_7);
            Interleave(vreg_src_2, vreg_src_4, vreg_src_2, vreg_src_4);
            Interleave(vreg_src_6, vreg_src_8, vreg_src_6, vreg_src_8);
            Interleave(vreg_src_1, vreg_src_2, vreg_src_1, vreg_src_2);
            Interleave(vreg_src_3, vreg_src_4, vreg_src_3, vreg_src_4);
            Interleave(vreg_src_5, vreg_src_6, vreg_src_5, vreg_src_6);
            Interleave(vreg_src_7, vreg_src_8, vreg_src_7, vreg_src_8);
            // DATA_BLOCK_COPY散布存储
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float*)dstT + colHalf * 4608 + 8 * block + 576 * 0, (RegTensor<float>&)vreg_src_1, 8 + 1,
                preg_f32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float*)dstT + colHalf * 4608 + 8 * block + 576 * 1, (RegTensor<float>&)vreg_src_2, 8 + 1,
                preg_f32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float*)dstT + colHalf * 4608 + 8 * block + 576 * 2, (RegTensor<float>&)vreg_src_3, 8 + 1,
                preg_f32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float*)dstT + colHalf * 4608 + 8 * block + 576 * 3, (RegTensor<float>&)vreg_src_4, 8 + 1,
                preg_f32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float*)dstT + colHalf * 4608 + 8 * block + 576 * 4, (RegTensor<float>&)vreg_src_5, 8 + 1,
                preg_f32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float*)dstT + colHalf * 4608 + 8 * block + 576 * 5, (RegTensor<float>&)vreg_src_6, 8 + 1,
                preg_f32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float*)dstT + colHalf * 4608 + 8 * block + 576 * 6, (RegTensor<float>&)vreg_src_7, 8 + 1,
                preg_f32);
            StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                (__ubuf__ float*)dstT + colHalf * 4608 + 8 * block + 576 * 7, (RegTensor<float>&)vreg_src_8, 8 + 1,
                preg_f32);
        }
    }
}

__aicore__ inline void VfDivAndCast(const LocalTensor<bfloat16_t>& outputT, const LocalTensor<float>& stage2Out,
                                    const LocalTensor<float>& accRowsum)
{
    __ubuf__ bfloat16_t* dstTPtr = reinterpret_cast<__ubuf__ bfloat16_t*>(outputT.GetPhyAddr());
    __ubuf__ float* srcPtr = reinterpret_cast<__ubuf__ float*>(stage2Out.GetPhyAddr());
    __ubuf__ float* rowsumPtr = reinterpret_cast<__ubuf__ float*>(accRowsum.GetPhyAddr());
    VfDivAndCastVF(dstTPtr, srcPtr, rowsumPtr);
}

/* **************************************************************************************************
 * ComputeLseTrans — LSE = alignedMax + ln(rowSum/pScale)，无效行输出3e+99
 * ************************************************************************************************* */
template <typename T>
__simd_vf__ inline void ComputeLseTransVF(__ubuf__ T* dstUb, __ubuf__ T* sumUb, __ubuf__ T* maxUb,
                                          const uint32_t dealCount, const float pScaleRec, const float minValue)
{
    static_assert(IsSameType<T, float>::value, "VF ComputeLseTrans, T must be float");
    RegTensor<float> vregSum;
    RegTensor<float> vregMax;
    RegTensor<float> vregRes;
    RegTensor<float> vregResFinal;
    RegTensor<float> vregMinValue;
    RegTensor<float> vregInfValue;
    RegTensor<float> vregScaleRec;
    MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregCompare;
    constexpr uint16_t floatRepSize = 64;
    constexpr float infValue = 3e+99;

    Duplicate(vregMinValue, minValue);
    Duplicate(vregInfValue, infValue);
    Duplicate(vregScaleRec, pScaleRec);

    uint16_t loops = (dealCount + floatRepSize - 1) / floatRepSize;
    for (uint16_t l = 0; l < loops; ++l) {
        uint32_t cnt = dealCount - l * floatRepSize;
        cnt = cnt < floatRepSize ? cnt : floatRepSize;
        MaskReg preg = UpdateMask<float>(cnt);
        LoadAlign(vregSum, sumUb + l * floatRepSize);
        LoadAlign(vregMax, maxUb + l * floatRepSize);
        Mul(vregSum, vregSum, vregScaleRec, preg);
        Log<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregSum, preg);
        Add<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, preg);
        Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, preg);
        Select<T>(vregResFinal, vregInfValue, vregRes, pregCompare);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(dstUb + l * floatRepSize, vregResFinal, preg);
    }
}

template <typename T>
__aicore__ inline void ComputeLseTrans(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor,
                                       const LocalTensor<T>& maxTensor, const uint32_t dealCount, const float pScaleRec,
                                       const float minValue)
{
    __ubuf__ T* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ T* sumUb = (__ubuf__ T*)sumTensor.GetPhyAddr();
    __ubuf__ T* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    ComputeLseTransVF<T>(dstUb, sumUb, maxUb, dealCount, pScaleRec, minValue);
}

} // namespace FaVectorApi

#endif // VF_FLASHUPDATE_TRANS_FP16_H_
