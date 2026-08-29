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
 * \file apply_rotary_pos_emb_grad_tiling_data.h
 * \brief
 */

#ifndef _APPLY_ROTARY_POS_EMB_GRAD_TILING_DATA_H_
#define _APPLY_ROTARY_POS_EMB_GRAD_TILING_DATA_H_

#include "atvoss/reduce/reduce_tiling_data.h"

struct ApplyRopeGradRegbaseParams {
    int64_t b;             // batch size
    int64_t s;             // seq_len
    int64_t d;             // head_dim
    int64_t nQ;            // Q num_heads
    int64_t nK;            // K num_heads
    int64_t blockNumB;     // B axis split count
    int64_t blockFactorB;  // B rows per core
    int64_t blockNumS;     // S axis split count
    int64_t blockFactorS;  // S rows per core
    int64_t ubFactorS;     // UB S rows per iteration
    int64_t ubLoopNumN;    // N loop count
    int64_t ubFactorN;     // UB N rows per iteration
    int64_t ubTailFactorN; // UB N tail size
    int64_t usedCoreNum;   // actual cores used
    int64_t rotaryMode;    // HALF = 0
    int64_t layout;        // internal Layout: 0=BSND, 1=SBND, 2=BNSD(reserved)
    uint32_t dCosFlag;     // 0=no grad_cos/sin, 1=compute inline
};

struct ApplyRopeGradRegbaseABParams {
    int64_t b;             // batch size
    int64_t s;             // seq_len
    int64_t d;             // head_dim
    int64_t nQ;            // Q num_heads
    int64_t nK;            // K num_heads
    int64_t dAlign;        // D aligned size
    int64_t dSplitCoef;    // half mode: 2
    int64_t blockNumBS;    // BS merged axis split count
    int64_t blockFactorBS; // BS rows per core
    int64_t blockTailBS;   // BS tail size
    int64_t blockNumN;     // N axis split count
    int64_t blockFactorN;  // N rows per core
    int64_t blockTailN;    // N tail size
    int64_t ubFactorBS;    // UB BS rows per iteration
    int64_t ubFactorN;     // UB N rows per iteration
    int64_t usedCoreNum;   // actual cores used
    int64_t rotaryMode;    // HALF = 0
};

struct ApplyRopeGradTilingData {
    Ops::Base::ReduceOpTilingData reduceTiling;
    ApplyRopeGradRegbaseParams ropeGradParams;     // BAB / A shared
    ApplyRopeGradRegbaseABParams ropeGradABParams; // AB专用
    uint32_t dCosFlag; // 0=不计算grad_cos/grad_sin, 1=需要计算(内联在kernel中同步完成)
    uint32_t layout;   // 0=BSND, 1=SBND, 2=BNSD(reserved)
};

#endif // _APPLY_ROTARY_POS_EMB_GRAD_TILING_DATA_H_
