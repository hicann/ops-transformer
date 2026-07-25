/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
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
 * \file vf_nd2nz_indexes_dn_mxfp4.h
 * \brief
 */

#ifndef ND2NZ_INDEXES_DN_MXFP4_H_
#define ND2NZ_INDEXES_DN_MXFP4_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
namespace Mxfp4Api {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

template <typename T>
__simd_vf__ void inline vf_init_indexs_and_duplicate(__ubuf__ uint8_t *index_nd2xz, __ubuf__ T *dupDest)
{
    // Index
    RegTensor<uint8_t> index_reg, index_reg_1, index_reg_2;
    MaskReg mask_32 = CreateMask<int8_t, MaskPattern::VL32>();

    Arange((RegTensor<int8_t> &)index_reg, 0);

    ShiftLefts(index_reg, index_reg, NUM_2, mask_32);
    for (uint16_t i = 0; i < 4; ++i) {
        Adds(index_reg_1, index_reg, i, mask_32);
        Adds(index_reg_2, index_reg_1, NUM_128, mask_32);
        StoreAlign(index_nd2xz + indexSubLength * i, (RegTensor<uint8_t> &)index_reg_1, mask_32);
        StoreAlign(index_nd2xz + indexSubLength * i + NUM_128, (RegTensor<uint8_t> &)index_reg_2, mask_32);
    }

    // Duplicate
    RegTensor<T> src;
    MaskReg preg_all_16bit = CreateMask<uint16_t, MaskPattern::ALL>();
    Duplicate(src, MIN_VALUE);
    for (uint16_t i = 0; i < 4; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(dupDest + i * 128, src, preg_all_16bit);
    }
}

template <typename T, uint16_t S1Base = 128>
__aicore__ inline void InitIndexesAndDuplicateCallVF(LocalTensor<uint8_t> &nd2nzIndexes,
                                                     const LocalTensor<T> &localGlobalMaxUB)
{
    __ubuf__ uint8_t *index_nd2xz = (__ubuf__ uint8_t *)nd2nzIndexes.GetPhyAddr();

    __ubuf__ T *localGlobalMax1Buf = (__ubuf__ T *)localGlobalMaxUB.GetPhyAddr();

    vf_init_indexs_and_duplicate<T>(index_nd2xz, localGlobalMax1Buf);
}

} // namespace Mxfp4Api
#endif // ND2NZ_INDEXES_DN_MXFP4_H_
