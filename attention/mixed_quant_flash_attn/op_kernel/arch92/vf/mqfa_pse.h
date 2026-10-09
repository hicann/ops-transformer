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
 * \file mqfa_pse.h
 * \brief
 */

#ifndef MQFA_PSE_H
#define MQFA_PSE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace regbaseutil {
enum class PseLayoutTypeEnum : uint32_t {
    PSE_S1S2 = 0,
    PSE_1S2 = 1,
    PSE_SLOPE_BN = 2,
    PSE_SLOPE_N = 3
};

constexpr static uint8_t pseEncodeALibiS2Full = 0x11;

enum class PseTypeEnum {
    PSE_OUTER_MUL_ADD_TYPE = 0, // default
    PSE_OUTER_ADD_MUL_TYPE,
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE,
    PSE_NONE_TYPE = 9
};

struct PseInfo {
    int64_t pseBSize;         // pse输入batch大小
    int64_t pseS1Size;        // for alibi
    int64_t pseS2ComputeSize; // for alibi, do not need assignment
    int64_t pseS2Size;        // for alibi
    int64_t readS2Size;       // for alibi, do not need assignment
    uint32_t pseLayoutType;   // pse输入shape的layout
    uint32_t pseEncodeType;   // for distinguish alibi
    uint32_t pseType;         // 0: outer, mul-add   1:outer, add-mul   2:inner, mul-add   3:inner, mul-add-sqrt
    uint32_t pseStride;
    int64_t qStartIdx;
    int64_t kvStartIdx;
};
} // namespace regbaseutil
#endif
