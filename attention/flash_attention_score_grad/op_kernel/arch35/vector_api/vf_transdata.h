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
 * \file vf_cast_transdata.h
 */
#ifndef MULS_CAST_TRANSDATA_H
#define MULS_CAST_TRANSDATA_H

#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;
/* **************************************************************************************************
 * nz2nd                                             *
 * ************************************************************************************************* */
/*
 * @ingroup TransData
 * @brief compute :res = nz2nd(x)
 * @param [out] dstTensor output LocalTensor
 * @param [in] srcTensor input src LocalTensor
 */

template <typename T>
__aicore__ inline void Transdata(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, uint32_t srcM,
                                 uint32_t srcN)
{
    const uint32_t blockSize = 32;
    const uint32_t blockN = blockSize / sizeof(T);
    const uint32_t fullExeSize = srcN;
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint32_t blockStride = (srcM * blockN) * sizeof(T) / blockSize;
    uint32_t repeatStride = 1;

    if (srcN <= 128) {
        __VEC_SCOPE__
        {
            RegTensor<T> vregSrc1;
            MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    vregSrc1, (__ubuf__ T *&)srcLocalInt, blockStride, repeatStride, pregFullExe);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>((__ubuf__ T *&)dstLocalInt, vregSrc1, srcN, pregFullExe);
            }
        }
    } else if (srcN <= 192) {
        uint64_t srcLocalInt2 = srcLocalInt + srcM * VREG_SIZE;
        uint32_t tailSize = srcN - (VREG_SIZE >> 1);
        __VEC_SCOPE__
        {
            RegTensor<T> vregSrc1;
            RegTensor<T> vregSrc2;
            UnalignRegForStore ureg1;
            UnalignRegForStore ureg2;
            MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    vregSrc1, (__ubuf__ T *&)srcLocalInt, blockStride, repeatStride, pregFullExe);
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    vregSrc2, (__ubuf__ T *&)srcLocalInt2, blockStride, repeatStride, pregFullExe);
                StoreUnAlign<T, PostLiteral::POST_MODE_UPDATE>((__ubuf__ T *&)dstLocalInt, vregSrc1, ureg1,
                                                               VREG_SIZE >> 1);
                StoreUnAlign<T, PostLiteral::POST_MODE_UPDATE>((__ubuf__ T *&)dstLocalInt, vregSrc2, ureg2, tailSize);
            }
            StoreUnAlignPost<T>((__ubuf__ T *&)dstLocalInt, ureg1, 0);
            StoreUnAlignPost<T>((__ubuf__ T *&)dstLocalInt, ureg2, 0);
        }
    }
}
#else
template <typename T>
__aicore__ inline void Transdata(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, uint32_t srcM,
                                 uint32_t srcN)
{}
#endif
} // namespace AscendC

#endif // MULS_CAST_TRANSDATA_H
