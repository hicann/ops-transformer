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
 * \file nsa_compress_attention_infer_tiling_utils.h
 * \brief
 */

#ifndef NSA_COMPRESS_ATTENTION_INFER_TILING_UTILS_H
#define NSA_COMPRESS_ATTENTION_INFER_TILING_UTILS_H

#include "nsa_compress_attention_infer_tiling.h"

namespace optiling {

constexpr uint32_t ZERO = 0;
constexpr uint32_t ONE = 1;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;

constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t ATTEN_MASK_INPUT_INDEX = 3;
constexpr uint32_t BLOCK_TABLE_INPUT_INDEX = 4;
constexpr uint32_t ACT_Q_SEQ_LEN_INPUT_INDEX = 5;
constexpr uint32_t ACT_CMP_KV_SEQ_LEN_INPUT_INDEX = 6;
constexpr uint32_t ACT_SEL_KV_SEQ_LEN_INPUT_INDEX = 7;
constexpr uint32_t TOPK_MASK_INPUT_INDEX = 8;

constexpr uint32_t ATTEN_OUTPUT_INDEX = 0;
constexpr uint32_t SELECT_OUTPUT_INDEX = 1;

constexpr uint32_t NUM_HEADS_ATTR_INDEX = 0;
constexpr uint32_t KV_NUM_HEADS_ATTR_INDEX = 1;
constexpr uint32_t SELECT_SIZE_ATTR_INDEX = 2;
constexpr uint32_t SELECT_NUM_ATTR_INDEX = 3;
constexpr uint32_t COMP_SIZE_ATTR_INDEX = 4;
constexpr uint32_t COMP_STRIDE_ATTR_INDEX = 5;
constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 6;

constexpr uint32_t LAYOUT_ATTR_INDEX = 7;
constexpr uint32_t BLOCK_SIZE_ATTR_INDEX = 8;
constexpr uint32_t SPARSE_MODE_ATTR_INDEX = 9;

constexpr uint32_t ROW_LIMIT_QP = 128;
constexpr uint32_t SEQLEN_LIMIT = 8192;

constexpr uint32_t SINGLE_UB_SIZE_BFLOAT16 = 4; // 分成4份
constexpr uint32_t SOFTMAX_INPUT_DATA_BYTE = 4;
constexpr uint32_t COMPARE_INT = 64;

constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t COLLEN_ALIGNED_REQUIRED = 16;
constexpr uint32_t NUM_VECTOR_PER_CUBE = 2;
constexpr uint32_t ALIGNED_32 = 32;

constexpr int64_t MAX_ACTUALQSEQLEN = 4;
constexpr int64_t MIN_ACTUALQSEQLEN = 1;

} // namespace optiling

#endif // NSA_COMPRESS_ATTENTION_INFER_TILING_UTILS_H
