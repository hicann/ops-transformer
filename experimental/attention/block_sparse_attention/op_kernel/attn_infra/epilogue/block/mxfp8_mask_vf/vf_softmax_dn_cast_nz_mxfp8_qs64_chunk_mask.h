/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef VF_SOFTMAX_DN_CAST_NZ_MXFP8_QS64_CHUNK_MASK_H_
#define VF_SOFTMAX_DN_CAST_NZ_MXFP8_QS64_CHUNK_MASK_H_
#include "kernel_tensor.h"
#include "../mxfp8_vf/vf_common_def_mxfp8.h"
#include "../../bsa_epilogue_dispatch_policy.hpp"

namespace NpuArch::Epilogue::Block::Mxfp8VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

template <MXQuantMode MX_QUANT_MODE = MXQuantMode::OCP, bool clear_gmax, typename T, typename T2, uint16_t QsBase = 128>
__simd_vf__ inline void softmax_with_group_max_qs64_chunk_mask_vf(__ubuf__ T2* pDest, __ubuf__ T* s,
                                                                  __ubuf__ T* local_group_max, __ubuf__ T* global_max,
                                                                  __ubuf__ uint8_t* indexesUb, uint16_t groupBase,
                                                                  uint16_t effY, uint16_t groupsInChunk)
{
    RegTensor<half> src_c0, src_c1, src_c2, src_c3;
    RegTensor<half> src_n0, src_n1, src_n2, src_n3;
    RegTensor<half> curr_group_max, next_group_max, group_gmax, min_val_reg;
    RegTensor<uint8_t> idx_nd2nz, zero8;
    RegTensor<float> src_f0, src_f1, src_f2, src_f3;

    constexpr uint16_t ROWS_PER_GROUP = 32;
    constexpr uint16_t ROW_SUB_LOOP = 4;
    constexpr uint16_t ITER_PER_GROUP = ROWS_PER_GROUP / ROW_SUB_LOOP;
    const uint16_t chunkRows = groupsInChunk * ROWS_PER_GROUP;
    const uint16_t effClamped = effY < chunkRows ? effY : chunkRows;
    const uint16_t validGroups = (effClamped + ROWS_PER_GROUP - 1) / ROWS_PER_GROUP;
    uint32_t MID_VALID_CNT = QsBase * (validGroups > 0 ? (validGroups - 1) : 0);

    MaskReg preg_all_16bit = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_VL64_16bit = CreateMask<uint16_t, MaskPattern::VL64>();
    MaskReg preg_all_8bit = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg preg_all_fp32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_vl128 = CreateMask<uint8_t, MaskPattern::VL128>();
    MaskReg preg_vl128_not;
    MaskNot(preg_vl128_not, preg_vl128, preg_all_8bit);
    MaskReg preg_invalid_max;

    LoadAlign(group_gmax, global_max);
    LoadAlign(idx_nd2nz, indexesUb);
    Duplicate(min_val_reg, MIN_VALUE);
    Duplicate(zero8, static_cast<uint8_t>(0));

    const uint16_t rowBase = groupBase * ROWS_PER_GROUP;
    Duplicate(curr_group_max, MIN_VALUE);
    for (uint16_t iter = 0; iter < ITER_PER_GROUP; ++iter) {
        LoadAlign(src_c0, s + ((rowBase + iter * ROW_SUB_LOOP) * QsBase + 0 * QsBase) * 2);
        LoadAlign(src_c1, s + ((rowBase + iter * ROW_SUB_LOOP) * QsBase + 1 * QsBase) * 2);
        LoadAlign(src_c2, s + ((rowBase + iter * ROW_SUB_LOOP) * QsBase + 2 * QsBase) * 2);
        LoadAlign(src_c3, s + ((rowBase + iter * ROW_SUB_LOOP) * QsBase + 3 * QsBase) * 2);
        Max(src_c0, src_c0, src_c1, preg_VL64_16bit);
        Max(src_c2, src_c2, src_c3, preg_VL64_16bit);
        Max(curr_group_max, curr_group_max, src_c0, preg_VL64_16bit);
        Max(curr_group_max, curr_group_max, src_c2, preg_VL64_16bit);
    }

    Muls(curr_group_max, curr_group_max, INV_LN2, preg_VL64_16bit);
    Truncate<T, RoundMode::CAST_FLOOR>(curr_group_max, curr_group_max, preg_VL64_16bit);
    Max(group_gmax, group_gmax, curr_group_max, preg_VL64_16bit);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max + groupBase * QsBase, curr_group_max,
                                                      preg_VL64_16bit);
    Adds(curr_group_max, curr_group_max, NEG_EIGHT_VALE, preg_VL64_16bit);
    Muls(curr_group_max, curr_group_max, LN2, preg_VL64_16bit);

    for (uint16_t g = 0; g < validGroups; ++g) {
        MaskReg preg_valid_max = UpdateMask<half>(MID_VALID_CNT);
        MaskNot(preg_invalid_max, preg_valid_max, preg_all_16bit);
        Duplicate(next_group_max, MIN_VALUE);
        const uint16_t gi = groupBase + g;
        const uint16_t hasNext = static_cast<uint16_t>(g + 1 < validGroups);

        for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
            if (hasNext) {
                uint16_t rowOffset_next = (gi + 1) * ROWS_PER_GROUP + j * ROW_SUB_LOOP;
                LoadAlign(src_n0, s + (rowOffset_next * QsBase + 0 * QsBase) * 2);
                LoadAlign(src_n1, s + (rowOffset_next * QsBase + 1 * QsBase) * 2);
                LoadAlign(src_n2, s + (rowOffset_next * QsBase + 2 * QsBase) * 2);
                LoadAlign(src_n3, s + (rowOffset_next * QsBase + 3 * QsBase) * 2);
                Max(src_n0, src_n0, src_n1, preg_valid_max);
                Max(src_n2, src_n2, src_n3, preg_valid_max);
                Max(next_group_max, next_group_max, src_n0, preg_valid_max);
                Max(next_group_max, next_group_max, src_n2, preg_valid_max);
            }
            uint16_t rowOffset_cur = gi * ROWS_PER_GROUP + j * ROW_SUB_LOOP;
            LoadAlign(src_c0, s + (rowOffset_cur * QsBase + 0 * QsBase) * 2);
            LoadAlign(src_c1, s + (rowOffset_cur * QsBase + 1 * QsBase) * 2);
            LoadAlign(src_c2, s + (rowOffset_cur * QsBase + 2 * QsBase) * 2);
            LoadAlign(src_c3, s + (rowOffset_cur * QsBase + 3 * QsBase) * 2);
            BSA_MXFP8_EXPSUB_EVEN64_4WAY_MINS(src_f0, src_f1, src_f2, src_f3, src_c0, src_c1, src_c2, src_c3,
                                              curr_group_max, preg_VL64_16bit, preg_all_fp32);
            uint32_t pOff = gi * 2048 + j * 256;
            BSA_MXFP8_PACK_STORE_E4M3_VL128(pDest, pOff, pOff + 8064, src_f0, src_f1, src_f2, src_f3, idx_nd2nz,
                                            preg_all_fp32, preg_all_8bit, preg_vl128, preg_vl128_not);
        }

        for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
            if (hasNext) {
                uint16_t rowOffset_next = (gi + 1) * ROWS_PER_GROUP + (j + 1) * ROW_SUB_LOOP;
                LoadAlign(src_n0, s + (rowOffset_next * QsBase + 0 * QsBase) * 2);
                LoadAlign(src_n1, s + (rowOffset_next * QsBase + 1 * QsBase) * 2);
                LoadAlign(src_n2, s + (rowOffset_next * QsBase + 2 * QsBase) * 2);
                LoadAlign(src_n3, s + (rowOffset_next * QsBase + 3 * QsBase) * 2);
                Max(src_n0, src_n0, src_n1, preg_valid_max);
                Max(src_n2, src_n2, src_n3, preg_valid_max);
                Max(next_group_max, next_group_max, src_n0, preg_valid_max);
                Max(next_group_max, next_group_max, src_n2, preg_valid_max);
            }
            uint16_t rowOffset_cur = gi * ROWS_PER_GROUP + (j + 1) * ROW_SUB_LOOP;
            LoadAlign(src_c0, s + (rowOffset_cur * QsBase + 0 * QsBase) * 2);
            LoadAlign(src_c1, s + (rowOffset_cur * QsBase + 1 * QsBase) * 2);
            LoadAlign(src_c2, s + (rowOffset_cur * QsBase + 2 * QsBase) * 2);
            LoadAlign(src_c3, s + (rowOffset_cur * QsBase + 3 * QsBase) * 2);
            BSA_MXFP8_EXPSUB_EVEN64_4WAY_MINS(src_f0, src_f1, src_f2, src_f3, src_c0, src_c1, src_c2, src_c3,
                                              curr_group_max, preg_VL64_16bit, preg_all_fp32);
            uint32_t pOff = gi * 2048 + j * 256;
            BSA_MXFP8_PACK_STORE_E4M3_VL128(pDest, pOff + 128, pOff + 8192, src_f0, src_f1, src_f2, src_f3, idx_nd2nz,
                                            preg_all_fp32, preg_all_8bit, preg_vl128, preg_vl128_not);
        }

        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(global_max, group_gmax, preg_invalid_max);
        Muls(next_group_max, next_group_max, INV_LN2, preg_valid_max);
        Truncate<T, RoundMode::CAST_FLOOR>(next_group_max, next_group_max, preg_valid_max);
        Max(group_gmax, group_gmax, next_group_max, preg_valid_max);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max + ((gi + 1) * QsBase), next_group_max,
                                                          preg_valid_max);
        Adds(next_group_max, next_group_max, NEG_EIGHT_VALE, preg_VL64_16bit);
        Muls(curr_group_max, next_group_max, LN2, preg_valid_max);
    }

    for (uint16_t g = validGroups; g < groupsInChunk; ++g) {
        const uint16_t gi = groupBase + g;
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max + gi * QsBase, min_val_reg, preg_VL64_16bit);
        for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
            uint32_t pOff = gi * 2048 + j * 256;
            StoreAlign(((__ubuf__ uint8_t*&)pDest) + pOff, zero8, preg_vl128);
            StoreAlign(((__ubuf__ uint8_t*&)pDest) + pOff + 128, zero8, preg_vl128);
            StoreAlign(((__ubuf__ uint8_t*&)pDest) + pOff + 8192, zero8, preg_vl128);
            StoreAlign(((__ubuf__ uint8_t*&)pDest) + pOff + 8192 + 128, zero8, preg_vl128);
        }
    }

    // pscale 固定读 4 组。align32 < align64 时 PV 仍按 align64 取 K，
    // 多出来的组不写 MIN，e8m0 scale 会变成 NaN。
    constexpr uint16_t PSCALE_GROUP_CNT = 4;
    const uint16_t padBegin = static_cast<uint16_t>(groupBase + groupsInChunk);
    for (uint16_t gi = padBegin; gi < PSCALE_GROUP_CNT; ++gi) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max + gi * QsBase, min_val_reg, preg_all_16bit);
    }
}

template <MXQuantMode MX_QUANT_MODE = MXQuantMode::OCP, bool clear_gmax, typename T, typename T2, uint16_t QsBase = 128>
__aicore__ inline void SoftmaxWithGroupMaxQs64ChunkMaskCallVF(const LocalTensor<T2>& dstTensor,
                                                              const LocalTensor<T>& srcTensor,
                                                              const LocalTensor<T>& localGroupMax,
                                                              const LocalTensor<T>& globalMax,
                                                              const LocalTensor<uint8_t>& indexesBuf,
                                                              uint16_t groupBase, uint16_t effY, uint16_t groupsInChunk)
{
    softmax_with_group_max_qs64_chunk_mask_vf<MX_QUANT_MODE, clear_gmax, T, T2, QsBase>(
        (__ubuf__ T2*)dstTensor.GetPhyAddr(), (__ubuf__ T*)srcTensor.GetPhyAddr(),
        (__ubuf__ T*)localGroupMax.GetPhyAddr(), (__ubuf__ T*)globalMax.GetPhyAddr(),
        (__ubuf__ uint8_t*)indexesBuf.GetPhyAddr(), groupBase, effY, groupsInChunk);
}

} // namespace NpuArch::Epilogue::Block::Mxfp8VF
#endif // VF_SOFTMAX_DN_CAST_NZ_MXFP8_QS64_CHUNK_MASK_H_
