/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VF_MASK_INVALID_ROWS_MXFP4_H
#define VF_MASK_INVALID_ROWS_MXFP4_H
#include "vf_common_def.h"

namespace NpuArch::Epilogue::Block::Mxfp4VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

// 将validRows所在的32行以及下一个32行无效部分置为-inf，softmax无效行得到0。
template <typename T, bool QS64 = false>
__simd_vf__ inline void mask_invalid_rows_to_min_value_vf(__ubuf__ T *s, uint16_t chunkRowBase, uint16_t validRows,
                                                          uint16_t rows)
{
    constexpr uint16_t QsBase = 128;
    constexpr uint16_t ROWS_PER_GROUP = 32;

    const uint16_t effClamped = validRows < rows ? validRows : rows;
    const uint16_t validGroups = (effClamped + ROWS_PER_GROUP - 1) / ROWS_PER_GROUP;
    const uint16_t writeEnd = ((validGroups + 1) * ROWS_PER_GROUP) < rows ? ((validGroups + 1) * ROWS_PER_GROUP) : rows;

    MaskReg mask = CreateMask<uint16_t, (QS64 ? MaskPattern::VL64 : MaskPattern::ALL)>();
    RegTensor<T> min_val_reg;
    Duplicate(min_val_reg, MIN_VALUE, mask);

    for (uint16_t r = effClamped; validGroups > 0 && r < writeEnd; ++r) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(s + ((chunkRowBase + r) * QsBase) * 2, min_val_reg, mask);
    }
}

template <typename T, bool QS64 = false>
__aicore__ inline void MaskInvalidRowsToMinValueCallVF(const LocalTensor<T> &srcTensor, uint16_t chunkRowBase,
                                                       uint16_t validRows, uint16_t rows)
{
    __ubuf__ T *s_ub = (__ubuf__ T *)srcTensor.GetPhyAddr();

    mask_invalid_rows_to_min_value_vf<T, QS64>(s_ub, chunkRowBase, validRows, rows);
}

} // namespace NpuArch::Epilogue::Block::Mxfp4VF
#endif // VF_MASK_INVALID_ROWS_MXFP4_H
