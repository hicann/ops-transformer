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
 * \file rotary_position_embedding_grad_tiling_data.h
 * \brief reduce tiling data struct
 */

#ifndef _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_
#define _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_

#include "atvoss/reduce/reduce_tiling_data.h"

struct RopeGradRegbaseParams {
    int64_t b;
    int64_t s;
    int64_t d;
    int64_t n;
    int64_t blockNumB;
    int64_t blockFactorB;
    int64_t blockNumS;
    int64_t blockFactorS;
    int64_t ubFactorS;
    int64_t ubFactorB;
    int64_t ubLoopNumN;
    int64_t ubFactorN;
    int64_t ubTailFactorN;
    int64_t usedCoreNum;
    int64_t rotaryMode;
};

struct RopeGradRegbaseABParams {
    int64_t b;
    int64_t s;
    int64_t d;
    int64_t n;
    int64_t dAlign;
    int64_t dSplitCoef;
    int64_t blockNumBS;
    int64_t blockFactorBS;
    int64_t blockTailBS;
    int64_t blockNumN;
    int64_t blockFactorN;
    int64_t blockTailN;
    int64_t ubFactorBS;
    int64_t ubFactorN;
    int64_t usedCoreNum;
    int64_t rotaryMode;
};

struct RotaryXParams {
    int64_t b;
    int64_t s;
    int64_t d;
    int64_t n;
    int64_t usedCoreNum;
    int64_t blockFactorN;     // 每个核处理多少个D
    int64_t blockTailFactorN; // 尾核处理多少个D
    int64_t ubFactorN;        // UB 一次处理多少个D
    int64_t rotaryMode;
};
struct RopeGradTilingData {
    Ops::Base::ReduceOpTilingData reduceTiling;
    RopeGradRegbaseParams ropeGradParams;
    RopeGradRegbaseABParams ropeGradABParams;
    RotaryXParams rotaryXParams;
    uint32_t dCosFlag;
    uint32_t isEmptyDy;   // dy/x 为空 tensor：kernel 跳过全部有效计算，仅清零 dcos/dsin
    int64_t cosShapeSize; // dcos/dsin 的元素个数（清零范围）
};

#endif
