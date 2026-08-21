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
 * \file sparse_lightning_indexer_grad_kl_loss_tiling_common.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_COMMON_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_COMMON_H

#include <cstdint>
#include "err/ops_err.h"
#include "tiling/platform/platform_ascendc.h"
#include <sstream>
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>

using namespace ge;

namespace optiling {

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t QUERY_INDEX_INPUT_INDEX = 2;
constexpr uint32_t KEY_INDEX_INPUT_INDEX = 3;
constexpr uint32_t WEIGHT_INPUT_INDEX = 4;
constexpr uint32_t SPARSE_INDICES_INPUT_INDEX = 5;
constexpr uint32_t SOFTMAX_MAX_INPUT_INDEX = 6;
constexpr uint32_t SOFTMAX_SUM_INPUT_INDEX = 7;
constexpr uint32_t QUERY_ROPE_INPUT_INDEX = 8;
constexpr uint32_t KEY_ROPE_INPUT_INDEX = 9;
constexpr uint32_t ACTUAL_SEQ_LENGTHS_QUERY_INPUT_INDEX = 10;
constexpr uint32_t ACTUAL_SEQ_LENGTHS_KEY_INPUT_INDEX = 11;

// Outputs Index
constexpr uint32_t D_QUERY_INDEX_OUTPUT_INDEX = 0;
constexpr uint32_t D_KEY_INDEX_OUTPUT_INDEX = 1;
constexpr uint32_t D_WEIGHTS_OUTPUT_INDEX = 2;
constexpr uint32_t LOSS_OUTPUT_INDEX = 3;

// Attributes Index
constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 0;
constexpr uint32_t LAYOUT_ATTR_INDEX = 1;
constexpr uint32_t SPARSE_MODE_ATTR_INDEX = 2;
constexpr uint32_t DETERMINISTIC_ATTR_INDEX = 3;

// Dim Num
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t DIM_NUM_THREE = 3;
constexpr size_t DIM_NUM_FOUR = 4;
// 常量
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t BYTE_BLOCK = 32;

struct InnerSplitParams {
    uint32_t s1GBaseSize = 1;
    uint32_t s2BaseSize = 1;
};

enum class SparseMode : uint32_t {
    RIGHT_DOWN_CAUSAL = 3  // 右下角点划分的下三角部分
};

struct AiCoreParams {
    uint64_t ubSize = 0;
    uint64_t blockDim = 0;
    uint64_t aicNum = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
};

// SLIGKLLOSS tiling类定义 继承TilingBaseClass
struct SparseLightningIndexerGradKLLossCompileInfo {
    int64_t core_num;
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch;
};

template <typename T> inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

} // namespace optiling

#endif
