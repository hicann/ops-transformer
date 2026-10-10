/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_flashupdate_acc.h
 * \brief V2 跨 S2 分块累积 rescale：dstOut = dstOut×factor + cur，[DV,S1] 段加载
 */

#ifndef VF_FLASHUPDATE_ACC_H_
#define VF_FLASHUPDATE_ACC_H_

#include "kernel_tensor.h"
#include "vf_common_def_mxfp8_softmax_fp16.h"

namespace QFA_KERNEL {
namespace QfaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

/*
 * @ingroup VfaVectorApi
 * @brief V2 跨 S2 分块累积：分子/分母 rescale 后累加新分块结果
 *
 * dstSum = dstSum × factor + rowsum（分母 [128] per S1）
 * dstOut[i][:] = dstOut[i][:] × factor[:] + cur[i][:]（分子 [128,128] [DV,S1]）
 *
 * factor 沿 S1 列方向逐元素（[DV,S1] 行主序红利：factor 向量与数据行形状对齐，无需广播）。
 * factor 分 2 个寄存器（列前半/后半），载入一次全程复用。
 *
 * @param [in/out] dstOut   stage2OutUB [128,128] fp32（[DV,S1]）
 * @param [in/out] dstSum   accRowsumUB [128] fp32（per S1）
 * @param [in]     cur      mm2ResUB[0:128] fp32（当前分块 PV）
 * @param [in]     rowsum   mm2ResUB[128] fp32（当前分块 rowsum）
 * @param [in]     factor   softmaxExpUB[loop%3] fp32（VfCalcRescaleFactor 输出）
 */
template <bool isUpdate>
__simd_vf__ inline void VfFlashUpdateAccVF(__ubuf__ float* dstOut, __ubuf__ float* dstSum, __ubuf__ float* cur,
                                           __ubuf__ float* rowsum, __ubuf__ float* factor)
{
    RegTensor<float> vreg_factor_a; // factor[0..63]，列前半，全程复用（仅 isUpdate）
    RegTensor<float> vreg_factor_b; // factor[64..127]，列后半，全程复用（仅 isUpdate）
    RegTensor<float> vreg_sum_acc;  // dstSum 累积值（Mul+Add 中间结果）
    RegTensor<float> vreg_sum_new;  // rowsum 新值
    RegTensor<float> vreg_out_0, vreg_out_1, vreg_out_2, vreg_out_3; // 4 路行展开
    RegTensor<float> vreg_cur_0, vreg_cur_1, vreg_cur_2, vreg_cur_3;
    MaskReg preg_f32 = CreateMask<float, MaskPattern::ALL>(); // 64 lanes fp32

    // ===== 首块：纯 Load→Store 拷贝起步（消费者统一 V 管道）=====
    if constexpr (!isUpdate) {
        LoadAlign(vreg_sum_new, rowsum);
        StoreAlign(dstSum, vreg_sum_new, preg_f32);
        LoadAlign(vreg_sum_new, rowsum + 64);
        StoreAlign(dstSum + 64, vreg_sum_new, preg_f32);
        for (uint16_t group = 0; group < 32U; ++group) {
            uint32_t row = static_cast<uint32_t>(group) * 4U;
            uint32_t base0 = row * 128;
            uint32_t base1 = (row + 1) * 128;
            uint32_t base2 = (row + 2) * 128;
            uint32_t base3 = (row + 3) * 128;
            // 列前半 [0..63]
            LoadAlign(vreg_out_0, cur + base0);
            LoadAlign(vreg_out_1, cur + base1);
            LoadAlign(vreg_out_2, cur + base2);
            LoadAlign(vreg_out_3, cur + base3);
            StoreAlign(dstOut + base0, vreg_out_0, preg_f32);
            StoreAlign(dstOut + base1, vreg_out_1, preg_f32);
            StoreAlign(dstOut + base2, vreg_out_2, preg_f32);
            StoreAlign(dstOut + base3, vreg_out_3, preg_f32);
            // 列后半 [64..127]
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

    // ===== 非首块：乘加累积 =====
    // ① factor 载入（2 regs，全程复用）
    LoadAlign(vreg_factor_a, factor);
    LoadAlign(vreg_factor_b, factor + 64);

    //    列前半 [0..63]
    LoadAlign(vreg_sum_acc, dstSum);

    LoadAlign(vreg_sum_new, rowsum);
    MulDstAdd(vreg_sum_acc, vreg_factor_a, vreg_sum_new, preg_f32);
    StoreAlign(dstSum, vreg_sum_acc, preg_f32);

    //    列后半 [64..127]
    LoadAlign(vreg_sum_acc, dstSum + 64);

    LoadAlign(vreg_sum_new, rowsum + 64);
    MulDstAdd(vreg_sum_acc, vreg_factor_b, vreg_sum_new, preg_f32);
    StoreAlign(dstSum + 64, vreg_sum_acc, preg_f32);

    // ③ dstOut 更新：128 行 × 2 列半，4 路行展开（32 迭代 × 2 列半）
    //    [DV,S1] 行主序，行 stride = 128 fp32；列前半 offset+0 用 factorA，后半 +64 用 factorB
    for (uint16_t group = 0; group < 32U; ++group) {
        uint32_t row = static_cast<uint32_t>(group) * 4U;
        uint32_t base0 = row * 128; // 行 row 的起始偏移
        uint32_t base1 = (row + 1) * 128;
        uint32_t base2 = (row + 2) * 128;
        uint32_t base3 = (row + 3) * 128;

        // ===== 列前半 [0..63]（factorA）=====
        LoadAlign(vreg_out_0, dstOut + base0);
        LoadAlign(vreg_out_1, dstOut + base1);
        LoadAlign(vreg_out_2, dstOut + base2);
        LoadAlign(vreg_out_3, dstOut + base3);

        LoadAlign(vreg_cur_0, cur + base0);
        LoadAlign(vreg_cur_1, cur + base1);
        LoadAlign(vreg_cur_2, cur + base2);
        LoadAlign(vreg_cur_3, cur + base3);
        MulDstAdd(vreg_out_0, vreg_factor_a, vreg_cur_0, preg_f32);
        MulDstAdd(vreg_out_1, vreg_factor_a, vreg_cur_1, preg_f32);
        MulDstAdd(vreg_out_2, vreg_factor_a, vreg_cur_2, preg_f32);
        MulDstAdd(vreg_out_3, vreg_factor_a, vreg_cur_3, preg_f32);
        StoreAlign(dstOut + base0, vreg_out_0, preg_f32);
        StoreAlign(dstOut + base1, vreg_out_1, preg_f32);
        StoreAlign(dstOut + base2, vreg_out_2, preg_f32);
        StoreAlign(dstOut + base3, vreg_out_3, preg_f32);

        // ===== 列后半 [64..127]（factorB）=====
        LoadAlign(vreg_out_0, dstOut + base0 + 64);
        LoadAlign(vreg_out_1, dstOut + base1 + 64);
        LoadAlign(vreg_out_2, dstOut + base2 + 64);
        LoadAlign(vreg_out_3, dstOut + base3 + 64);

        LoadAlign(vreg_cur_0, cur + base0 + 64);
        LoadAlign(vreg_cur_1, cur + base1 + 64);
        LoadAlign(vreg_cur_2, cur + base2 + 64);
        LoadAlign(vreg_cur_3, cur + base3 + 64);
        MulDstAdd(vreg_out_0, vreg_factor_b, vreg_cur_0, preg_f32);
        MulDstAdd(vreg_out_1, vreg_factor_b, vreg_cur_1, preg_f32);
        MulDstAdd(vreg_out_2, vreg_factor_b, vreg_cur_2, preg_f32);
        MulDstAdd(vreg_out_3, vreg_factor_b, vreg_cur_3, preg_f32);
        StoreAlign(dstOut + base0 + 64, vreg_out_0, preg_f32);
        StoreAlign(dstOut + base1 + 64, vreg_out_1, preg_f32);
        StoreAlign(dstOut + base2 + 64, vreg_out_2, preg_f32);
        StoreAlign(dstOut + base3 + 64, vreg_out_3, preg_f32);
    }
}

/*
 * @ingroup VfaVectorApi
 * @brief VfFlashUpdateAcc 的 __aicore__ wrapper：接收 LocalTensor，内部转指针调 vf
 *
 * @param [in,out] stage2OutUB  跨 S2 分块累积分子（[128,128] fp32）
 * @param [in,out] accRowsumUB  累积分母（[128] per S1 列）
 * @param [in]     mm2ResUB     本块 PV 结果（[129,128]——前 128 行 cur + 第 128 行 rowsum）
 * @param [in]     softmaxExpUB 跨块 rescale 因子 ring（[taskSlot][128] 布局）
 * @param [in]     taskSlot     任务槽（runInfo.loop % 3）
 */
template <bool isUpdate>
__aicore__ inline void VfFlashUpdateAcc(const LocalTensor<float>& stage2OutUB, const LocalTensor<float>& accRowsumUB,
                                        const LocalTensor<float>& mm2ResUB, const LocalTensor<float>& softmaxExpUB,
                                        uint32_t taskSlot)
{
    constexpr uint32_t rowsumOff = 128 * 128; // rowsum = mm2Res 第 128 行
    constexpr uint32_t expSlot = 128;         // softmaxExpUB 每 task 槽 128 float
    __ubuf__ float* dstOutPtr = reinterpret_cast<__ubuf__ float*>(stage2OutUB.GetPhyAddr());
    __ubuf__ float* dstSumPtr = reinterpret_cast<__ubuf__ float*>(accRowsumUB.GetPhyAddr());
    __ubuf__ float* curPtr = reinterpret_cast<__ubuf__ float*>(mm2ResUB.GetPhyAddr());
    __ubuf__ float* rowsumPtr = curPtr + rowsumOff;
    __ubuf__ float* factorPtr = reinterpret_cast<__ubuf__ float*>(softmaxExpUB.GetPhyAddr()) + taskSlot * expSlot;
    VfFlashUpdateAccVF<isUpdate>(dstOutPtr, dstSumPtr, curPtr, rowsumPtr, factorPtr);
}

} // namespace QfaVectorApi
} // namespace QFA_KERNEL

#endif // VF_FLASHUPDATE_ACC_H_
