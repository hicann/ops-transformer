/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef VF_SOFTMAX_DN_CAST_NZ_MXFP8_ALIGN_QS128_KVS32_H_
#define VF_SOFTMAX_DN_CAST_NZ_MXFP8_ALIGN_QS128_KVS32_H_
#include "vf_common_def_mxfp8.h"
#include "../../bsa_epilogue_dispatch_policy.hpp"

namespace NpuArch::Epilogue::Block::Mxfp8VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

template <MXQuantMode MX_QUANT_MODE = MXQuantMode::OCP, bool clear_gmax, typename T, typename T2,
          uint16_t KvsBaseAlign = 32, uint16_t QsBase = 128>
__simd_vf__ inline void softmax_with_group_max_align_qs128_kvs32_vf(__ubuf__ T2* pDest, __ubuf__ T* s,
                                                                    __ubuf__ T* local_group_max, __ubuf__ T* global_max,
                                                                    __ubuf__ uint8_t* indexesUb)
{
    // ====================== 寄存器定义 ======================
    RegTensor<half> src_c0, src_c1, src_c2, src_c3;
    RegTensor<half> curr_group_max;
    RegTensor<half> group_gmax;
    RegTensor<half> min_val_reg;
    RegTensor<uint8_t> idx_nd2nz;

    // 量化专用寄存器（四路 float→e4m3）
    RegTensor<float> src_f0, src_f1, src_f2, src_f3, src_f4, src_f5, src_f6, src_f7;

    // ====================== 分块常量 ======================
    const uint16_t ROWS_PER_GROUP = 32;
    const uint16_t GROUP_COUNT = KvsBaseAlign / ROWS_PER_GROUP;
    const uint16_t ROW_SUB_LOOP = 4;
    const uint16_t ITER_PER_GROUP = ROWS_PER_GROUP / ROW_SUB_LOOP;

    // ====================== 掩码定义 ======================
    MaskReg preg_all_16bit = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_all_8bit = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg preg_all_fp32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_vl128 = CreateMask<uint8_t, MaskPattern::VL128>();
    MaskReg preg_vl128_not;
    MaskNot(preg_vl128_not, preg_vl128, preg_all_8bit);
    MaskReg preg_invalid_max;

    // ====================== 全局最大值初始化 ======================
    LoadAlign(group_gmax, global_max);

    LoadAlign(idx_nd2nz, indexesUb);

    // ====================== 预计算：第一个分组的最大值 ======================
    Duplicate(curr_group_max, MIN_VALUE);
    for (uint16_t iter = 0; iter < ITER_PER_GROUP; ++iter) {
        LoadAlign(src_c0, s + (iter * QsBase * ROW_SUB_LOOP + 0 * QsBase) * 2);
        LoadAlign(src_c1, s + (iter * QsBase * ROW_SUB_LOOP + 1 * QsBase) * 2);
        LoadAlign(src_c2, s + (iter * QsBase * ROW_SUB_LOOP + 2 * QsBase) * 2);
        LoadAlign(src_c3, s + (iter * QsBase * ROW_SUB_LOOP + 3 * QsBase) * 2);

        Max(src_c0, src_c0, src_c1, preg_all_16bit);
        Max(src_c2, src_c2, src_c3, preg_all_16bit);
        Max(curr_group_max, curr_group_max, src_c0, preg_all_16bit);
        Max(curr_group_max, curr_group_max, src_c2, preg_all_16bit);
    }

    Muls(curr_group_max, curr_group_max, INV_LN2, preg_all_16bit);
    Truncate<T, RoundMode::CAST_FLOOR>(curr_group_max, curr_group_max, preg_all_16bit);
    Max(group_gmax, group_gmax, curr_group_max, preg_all_16bit);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max, curr_group_max, preg_all_16bit);
    Adds(curr_group_max, curr_group_max, NEG_EIGHT_VALE, preg_all_16bit);
    Muls(curr_group_max, curr_group_max, LN2, preg_all_16bit);

    // ====================== 核心：双块流水分组循环 ======================
    for (uint16_t i = 0; i < GROUP_COUNT; i++) {
        // ========== 第一个内循环：处理偶数子块 j = 0,2,4,6 ==========
        for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
            // 当前块计算：FusedExpSub
            uint16_t rowOffset_cur = i * ROWS_PER_GROUP + j * ROW_SUB_LOOP;
            LoadAlign(src_c0, s + (rowOffset_cur * QsBase + 0 * QsBase) * 2);
            LoadAlign(src_c1, s + (rowOffset_cur * QsBase + 1 * QsBase) * 2);
            LoadAlign(src_c2, s + (rowOffset_cur * QsBase + 2 * QsBase) * 2);
            LoadAlign(src_c3, s + (rowOffset_cur * QsBase + 3 * QsBase) * 2);

            BSA_MXFP8_EXPSUB_SPLIT_4WAY_MINS(src_f0, src_f1, src_f2, src_f3, src_f4, src_f5, src_f6, src_f7, src_c0,
                                             src_c1, src_c2, src_c3, curr_group_max, preg_all_16bit, preg_all_fp32);
            uint32_t pOff = i * 2048 + j * 256;
            BSA_MXFP8_PACK_STORE_E4M3_VL128(pDest, pOff, pOff + 8064, src_f0, src_f1, src_f2, src_f3, idx_nd2nz,
                                            preg_all_fp32, preg_all_8bit, preg_vl128, preg_vl128_not);
            BSA_MXFP8_PACK_STORE_E4M3_VL128(pDest, pOff + 16384, pOff + 24448, src_f4, src_f5, src_f6, src_f7,
                                            idx_nd2nz, preg_all_fp32, preg_all_8bit, preg_vl128, preg_vl128_not);
        }

        // ========== 第二个内循环：处理奇数子块 j+1 = 1,3,5,7 ==========
        for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
            // 当前块计算：FusedExpSub
            uint16_t rowOffset_cur = i * ROWS_PER_GROUP + (j + 1) * ROW_SUB_LOOP;
            LoadAlign(src_c0, s + (rowOffset_cur * QsBase + 0 * QsBase) * 2);
            LoadAlign(src_c1, s + (rowOffset_cur * QsBase + 1 * QsBase) * 2);
            LoadAlign(src_c2, s + (rowOffset_cur * QsBase + 2 * QsBase) * 2);
            LoadAlign(src_c3, s + (rowOffset_cur * QsBase + 3 * QsBase) * 2);

            BSA_MXFP8_EXPSUB_SPLIT_4WAY_MINS(src_f0, src_f1, src_f2, src_f3, src_f4, src_f5, src_f6, src_f7, src_c0,
                                             src_c1, src_c2, src_c3, curr_group_max, preg_all_16bit, preg_all_fp32);
            uint32_t pOff = i * 2048 + j * 256;
            BSA_MXFP8_PACK_STORE_E4M3_VL128(pDest, pOff + 128, pOff + 8192, src_f0, src_f1, src_f2, src_f3, idx_nd2nz,
                                            preg_all_fp32, preg_all_8bit, preg_vl128, preg_vl128_not);
            BSA_MXFP8_PACK_STORE_E4M3_VL128(pDest, pOff + 16384 + 128, pOff + 24576, src_f4, src_f5, src_f6, src_f7,
                                            idx_nd2nz, preg_all_fp32, preg_all_8bit, preg_vl128, preg_vl128_not);
        }

        // ====================== 全局/局部最大值更新 ======================
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(global_max, group_gmax, preg_all_16bit);
    }

    // pscale 固定读 4 组 groupmax（ulmaxLoopRow=m*4）。kvs32 只写 1 组，必须把 2/3/4 组 pad 成 MIN，
    // 否则残留 Inf/NaN 会进 e8m0（255）把对应 qs 打成 NaN。
    Duplicate(min_val_reg, MIN_VALUE);
    constexpr uint16_t PSCALE_GROUP_CNT = 4;
    for (uint16_t g = GROUP_COUNT; g < PSCALE_GROUP_CNT; ++g) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max + g * QsBase, min_val_reg, preg_all_16bit);
    }
}

template <MXQuantMode MX_QUANT_MODE = MXQuantMode::OCP, bool clear_gmax, typename T, typename T2,
          uint16_t KvsBaseAlign = 32, uint16_t QsBase = 128>
__aicore__ inline void SoftmaxWithGroupMaxAlignQs128Kvs32CallVF(const LocalTensor<T2>& dstTensor,
                                                                const LocalTensor<T>& srcTensor,
                                                                const LocalTensor<T>& local_group_max,
                                                                const LocalTensor<T>& global_max,
                                                                const LocalTensor<uint8_t>& indexesBuf)
{
    __ubuf__ T2* pDest = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T* input_x_local_UB = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T* localGroupMax = (__ubuf__ T*)local_group_max.GetPhyAddr();
    __ubuf__ T* globalMax = (__ubuf__ T*)global_max.GetPhyAddr();
    __ubuf__ uint8_t* indexesUb = (__ubuf__ uint8_t*)indexesBuf.GetPhyAddr();

    softmax_with_group_max_align_qs128_kvs32_vf<MX_QUANT_MODE, clear_gmax, T, T2, KvsBaseAlign, QsBase>(
        pDest, input_x_local_UB, localGroupMax, globalMax, indexesUb);
}

} // namespace NpuArch::Epilogue::Block::Mxfp8VF
#endif // VF_SOFTMAX_DN_CAST_NZ_MXFP8_ALIGN_QS128_KVS32_H_
