/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file norm_rope_concat_grad_base_host.h
 * \brief
 */

#include <cstddef>
namespace NormRopeConcatGrad {
constexpr size_t INPUT_DIM_NUM = 4;  // B, S, H, D
constexpr size_t ROPE_DIM_NUM = 2;   // S, D
constexpr size_t OUTPUT_DIM_NUM = 4; // B, S, H, D
constexpr size_t BATCH_DIM = 0;
constexpr size_t SEQ_DIM = 1;
constexpr size_t HEAD_DIM = 2;
constexpr size_t DIM_DIM = 3;
constexpr size_t GRAD_SEQ_DIM = 2; // B, H, S, D
constexpr size_t GRAD_HEAD_DIM = 1;
constexpr size_t ROPE_SEQ_DIM = 0;
constexpr size_t ROPE_DIM_DIM = 1;
constexpr size_t T_SEQ_DIM = 2;
constexpr size_t T_HEAD_DIM = 1;
constexpr size_t TOTAL_NORM_NUM = 8;
constexpr size_t NUM_TWO = 2;
constexpr size_t NUM_THREE = 3;
constexpr size_t AVGHEAD_MAX_NUM = 4096;
constexpr size_t MIN_SHARE_BUFFER = 256;
constexpr size_t RES_UB_BUFFER = 1024;
constexpr size_t MAX_AFFINE_BLOCK_NUM = 40;
constexpr int64_t BATCH_MODE_MIN_KEY = 100000;

enum class NormType : int64_t {
    NONE = 0,
    LAYER_NORM,
    LAYER_NORM_AFFINE,
    LAYER_NORM_ACROSS_HEADS,
    LAYER_NORM_AFFINE_ACROSS_HEADS,
    RMS_NORM,
    RMS_NORM_AFFINE,
    RMS_NORM_ACROSS_HEADS,
    RMS_NORM_AFFINE_ACROSS_HEADS,
    L2_NORM,
};

inline bool IsNormTypeValid(int64_t normType)
{
    return normType >= static_cast<int64_t>(NormType::NONE) &&
           normType <= static_cast<int64_t>(NormType::LAYER_NORM_AFFINE);
}

// read_store
enum class RopeType : int64_t {
    NONE = 0,
    INTERLEAVE,
    HALF,
};

inline bool IsRopeTypeValid(int64_t ropeType)
{
    return ropeType >= static_cast<int64_t>(RopeType::NONE) && ropeType <= static_cast<int64_t>(RopeType::HALF);
}

enum class ConcatOrder : int64_t {
    BEFORE_ENCODER = 0,
    AFTER_ENCODER,
};

inline bool IsConcatOrderValid(int64_t concatOrder)
{
    return concatOrder >= static_cast<int64_t>(ConcatOrder::BEFORE_ENCODER) &&
           concatOrder <= static_cast<int64_t>(ConcatOrder::AFTER_ENCODER);
}

enum class InputIndexBackward : size_t {
    GRAD_QUERY_OUTPUT_INDEX = 0,
    GRAD_KEY_OUTPUT_INDEX,
    GRAD_VALUE_OUTPUT_INDEX,
    QUERY_INDEX,
    KEY_INDEX,
    ENCODER_QUERY_INDEX,
    ENCODER_KEY_INDEX,
    NORM_QUERY_WEIGHT_INDEX,
    NORM_QUERY_MEAN_INDEX,
    NORM_QUERY_RSTD_INDEX,
    NORM_KEY_WEIGHT_INDEX,
    NORM_KEY_MEAN_INDEX,
    NORM_KEY_RSTD_INDEX,
    NORM_ADDED_QUERY_WEIGHT_INDEX,
    NORM_ADDED_QUERY_MEAN_INDEX,
    NORM_ADDED_QUERY_RSTD_INDEX,
    NORM_ADDED_KEY_WEIGHT_INDEX,
    NORM_ADDED_KEY_MEAN_INDEX,
    NORM_ADDED_KEY_RSTD_INDEX,
    ROPE_SIN_INDEX,
    ROPE_COS_INDEX
};

enum class OutputIndexBackward : size_t {
    GRAD_QUERY_INDEX = 0,
    GRAD_KEY_INDEX,
    GRAD_VALUE_INDEX,
    GRAD_ENCODER_QUERY_INDEX,
    GRAD_ENCODER_KEY_INDEX,
    GRAD_ENCODER_VALUE_INDEX,
    GRAD_NORM_QUERY_WEIGHT_INDEX,
    GRAD_NORM_QUERY_BIAS_INDEX,
    GRAD_NORM_KEY_WEIGHT_INDEX,
    GRAD_NORM_KEY_BIAS_INDEX,
    GRAD_NORM_ADDED_QUERY_WEIGHT_INDEX,
    GRAD_NORM_ADDED_QUERY_BIAS_INDEX,
    GRAD_NORM_ADDED_KEY_WEIGHT_INDEX,
    GRAD_NORM_ADDED_KEY_BIAS_INDEX
};

enum class AttrIndexBackward : size_t {
    NORM_TYPE_INDEX = 0,
    NORM_ADDED_TYPE_INDEX,
    ROPE_TYPE_INDEX,
    CONCAT_ORDER_INDEX
};
} // namespace NormRopeConcatGrad
