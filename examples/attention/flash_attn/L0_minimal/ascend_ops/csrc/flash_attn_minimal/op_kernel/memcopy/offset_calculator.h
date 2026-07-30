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
 * \file offset_calculator.h
 * \brief
 */
#ifndef OFFSET_CALCULATOR_H
#define OFFSET_CALCULATOR_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

using AscendC::GlobalTensor;
using AscendC::LocalTensor;

enum class GmFormat {
    BSNGD = 0,
    BNGSD = 1,
    NGBSD = 2,
    TNGD = 3,
    NGTD = 4,
    BSND = 5,
    BNSD = 6,
    TND = 7,
    NTD = 8,
    PA_BnBsND = 9,
    PA_BnNBsD = 10,
    PA_NZ = 11,
    NGD = 12, // post_quant
    ND = 13, //antiquant no PA
    BS2 = 14,
    BNS2 = 15,
    PA_BnBs = 16, //antiquant PA
    PA_BnNBs = 17,
    BN2GS1S2 = 18, //PSE_GmFormat
    SBNGD = 19,
    SBND = 20,
    NTGD = 21,
    TND2 = 22, // VSCALE, 尾轴2
    PA_NZ_K_SCALE = 23,
};

struct BsndOffsetInfo {
    uint64_t strideB;
    uint64_t strideN;
    uint64_t strideS;
    uint64_t strideD;
    uint64_t dimB;
    uint64_t dimN;
    uint64_t dimS;
    uint64_t dimD;
};

template <GmFormat FORMAT>
struct OffsetInfoType {};

template <>
struct OffsetInfoType<GmFormat::BSND> {
    using Type = BsndOffsetInfo;
};

template <typename INPUT_T, GmFormat FORMAT, typename ACTLEN_T = uint64_t, bool WITH_ZERO_HEAD = false>
struct FaGmTensor {
    GlobalTensor<INPUT_T> gmTensor;
    typename OffsetInfoType<FORMAT>::Type offsetInfo;

    __aicore__ inline FaGmTensor() {}

    __aicore__ inline FaGmTensor(__gm__ INPUT_T *gm, uint32_t b, uint32_t n, uint32_t s, uint32_t d) {
        gmTensor.SetGlobalBuffer(gm);
        InitOffset(offsetInfo, b, n, s, d);
    }
};

template <typename OUT_T>
struct FaUbTensor {
    LocalTensor<OUT_T> tensor;
    uint32_t rowCount;
    uint32_t colCount;
};

template <typename INPUT_T>
struct FaL1Tensor {
    LocalTensor<INPUT_T> tensor;
    uint32_t rowCount;
};

template <typename OffsetInfoT>
__aicore__ inline void InitOffset(OffsetInfoT &info, uint32_t b, uint32_t n, uint32_t s, uint32_t d)
{
    info.dimB = b;
    info.dimN = n;
    info.dimS = s;
    info.dimD = d;
    info.strideD = 1;
    info.strideN = d;
    info.strideS = (uint64_t)n * d;
    info.strideB = (uint64_t)s * n * d;
}

template <typename OffsetInfoT>
__aicore__ inline uint64_t GetOffset(const OffsetInfoT &info,
    uint32_t bIdx, uint32_t nIdx, uint32_t sIdx, uint32_t dIdx)
{
    return bIdx * info.strideB + nIdx * info.strideN + sIdx * info.strideS + dIdx * info.strideD;
}

#endif
