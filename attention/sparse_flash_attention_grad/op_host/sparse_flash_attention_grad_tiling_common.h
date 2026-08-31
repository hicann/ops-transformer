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
 * \file sparse_flash_attention_grad_tiling_common.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <vector>
#include <register/tilingdata_base.h>
#include <register/op_impl_registry.h>
#include <tiling/tiling_api.h>
#include "err/ops_err.h"
#include "log/log.h"

namespace optiling {
namespace sfag {
constexpr const char *BSH_STR = "BSH";
constexpr const char *SBH_STR = "SBH";
constexpr const char *BNSD_STR = "BNSD";
constexpr const char *BSND_STR = "BSND";
constexpr const char *TND_STR = "TND";
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t DIM_4 = 4;
// N<=64 有做L1常驻性能优化
constexpr int64_t SFAG_HEAD_N_L1_RESIDENT = 64;

enum class InputLayout : uint32_t {
    BSH = 0,
    SBH = 1,
    BNSD = 2,
    BSND = 3,
    TND
};

enum class InputIndex : uint32_t {
    QUERY = 0,
    KEY,
    TOPK_INDICES,
    ATTENTION_OUT_GRAD,
    ATTENTION_OUT,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    SINKS, // oss-sink 输入 [N1] FP32，OPTIONAL
    VALUE,
    ACTUAL_SEQ_Q_LEN,
    ACTUAL_SEQ_KV_LEN,
    Q_ROPE,
    K_ROPE
};

enum class OutputIndex : uint32_t {
    DQ = 0,
    DK,
    DV
};

enum class AttrIndex : uint32_t {
    SCALE_VALUE = 0,
    SELECTED_BLOCK_SIZE,
    INPUT_LAYOUT,
    SPARSE_MODE,
    PRE_TOKENS,
    NEXT_TOKENS,
    DETERMINISTIC
};

struct SparseFlashAttentionGradCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    int64_t coreNum;
};

inline int64_t CeilCommon(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

inline uint32_t AlignData(const uint32_t a, const uint32_t b)
{
    if (b == 0U) {
        return a;
    }
    return (a + b - 1U) / b * b;
}

inline int64_t AlignData(const int64_t a, const int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

inline uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
{
    uint32_t ration;
    if (aicCoreNum == 0 || aivCoreNum == 0 || aicCoreNum > aivCoreNum) {
        return sliceNum;
    }
    ration = aivCoreNum / aicCoreNum;
    return (sliceNum + (ration - 1)) / ration;
}

ge::graphStatus CheckTndSoftmaxMaxShape(gert::TilingContext *context, int64_t t1, int64_t n2, int64_t g);
ge::graphStatus CheckTndSoftmaxSumShape(gert::TilingContext *context, int64_t t1, int64_t n2, int64_t g);
ge::graphStatus CheckAttentionInShape(gert::TilingContext *context);
ge::graphStatus CheckSoftmaxDtype(gert::TilingContext *context);
ge::graphStatus CheckAttentionInDtype(gert::TilingContext *context);
ge::graphStatus CheckTndShapeValid(gert::TilingContext *context, int64_t t1, int64_t n1, int64_t d, int64_t d2,
                                   int64_t n2);
ge::graphStatus CheckDtypeValid(gert::TilingContext *context);
bool IsSameShape(const gert::StorageShape *aShape, const gert::StorageShape *bShape);

} // namespace sfag
} // namespace optiling
