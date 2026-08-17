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
 * \file norm_rope_concat_grad_tiling.h
 * \brief
 */
#ifndef NORM_ROPE_CONCAT_GRAD_TILING_H_
#define NORM_ROPE_CONCAT_GRAD_TILING_H_

#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "norm_rope_concat_grad_base_host.h"

using namespace NormRopeConcatGrad;

namespace optiling {
struct NormRopeConcatGradCompileInfo {
    uint64_t aivNum;
    uint64_t ubSize;
    uint64_t sysWorkspace;
};

BEGIN_TILING_DATA_DEF(NormRopeConcatGradTilingData)
TILING_DATA_FIELD_DEF(uint64_t, tilingKey)
TILING_DATA_FIELD_DEF(uint64_t, batch)
TILING_DATA_FIELD_DEF(uint64_t, querySeq)
TILING_DATA_FIELD_DEF(uint64_t, keySeq)
TILING_DATA_FIELD_DEF(uint64_t, valueSeq)
TILING_DATA_FIELD_DEF(uint64_t, encoderQuerySeq)
TILING_DATA_FIELD_DEF(uint64_t, encoderKeySeq)
TILING_DATA_FIELD_DEF(uint64_t, encoderValueSeq)
TILING_DATA_FIELD_DEF(uint64_t, totalQuerySeq)
TILING_DATA_FIELD_DEF(uint64_t, totalKeySeq)
TILING_DATA_FIELD_DEF(uint64_t, totalValueSeq)
TILING_DATA_FIELD_DEF(uint64_t, ropeActualSeq)
TILING_DATA_FIELD_DEF(uint64_t, splitHeadNum)
TILING_DATA_FIELD_DEF(uint64_t, avgHeads)
TILING_DATA_FIELD_DEF(uint64_t, tailHeads)
TILING_DATA_FIELD_DEF(uint64_t, normDim)
TILING_DATA_FIELD_DEF(uint64_t, ropeDim)
TILING_DATA_FIELD_DEF(uint64_t, headNum)
TILING_DATA_FIELD_DEF(uint64_t, headDim)
TILING_DATA_FIELD_DEF(uint64_t, usedCore)
TILING_DATA_FIELD_DEF(float, eps)
TILING_DATA_FIELD_DEF(float, scale)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NormRopeConcatGrad, NormRopeConcatGradTilingData)

enum class TensorType {
    INPUT_TENSOR,
    ENCODER_TENSOR,
    GRADIENT_TENSOR,
    NORM_TENSOR,
    GRADIENT_NORM_TENSOR,
    ROPE_TENSOR,
};

struct InputTensorInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
    TensorType type;
    size_t index;
    bool isRequired;
};

constexpr size_t MAX_FORWARD_INPUT_NUM = 16;
constexpr size_t MAX_BACKWARD_INPUT_NUM = 21;
constexpr size_t REQUIRED_FORWARD_INPUT_NUM = 3;
constexpr size_t REQUIRED_BACKWARD_INPUT_NUM = 5;
static constexpr TensorType FORWARD_INPUT_TYPE[MAX_FORWARD_INPUT_NUM] = {
    TensorType::INPUT_TENSOR,   // query
    TensorType::INPUT_TENSOR,   // key
    TensorType::INPUT_TENSOR,   // value
    TensorType::ENCODER_TENSOR, // encoder_query
    TensorType::ENCODER_TENSOR, // encoder_key
    TensorType::ENCODER_TENSOR, // encoder_value
    TensorType::NORM_TENSOR,    // norm_query_weight
    TensorType::NORM_TENSOR,    // norm_query_bias
    TensorType::NORM_TENSOR,    // norm_key_weight
    TensorType::NORM_TENSOR,    // norm_key_bias
    TensorType::NORM_TENSOR,    // norm_added_query_weight
    TensorType::NORM_TENSOR,    // norm_added_query_bias
    TensorType::NORM_TENSOR,    // norm_added_key_weight
    TensorType::NORM_TENSOR,    // norm_added_key_bias
    TensorType::ROPE_TENSOR,    // rope_sin
    TensorType::ROPE_TENSOR,    // rope_cos
};
static constexpr TensorType BACKWARD_INPUT_TYPE[MAX_BACKWARD_INPUT_NUM] = {
    TensorType::GRADIENT_TENSOR,      // grad_query_output
    TensorType::GRADIENT_TENSOR,      // grad_key_output
    TensorType::GRADIENT_TENSOR,      // grad_value_output
    TensorType::INPUT_TENSOR,         // query
    TensorType::INPUT_TENSOR,         // key
    TensorType::ENCODER_TENSOR,       // encoder_query
    TensorType::ENCODER_TENSOR,       // encoder_key
    TensorType::NORM_TENSOR,          // norm_query_weight
    TensorType::GRADIENT_NORM_TENSOR, // norm_query_mean
    TensorType::GRADIENT_NORM_TENSOR, // norm_query_rstd
    TensorType::NORM_TENSOR,          // norm_key_weight
    TensorType::GRADIENT_NORM_TENSOR, // norm_key_mean
    TensorType::GRADIENT_NORM_TENSOR, // norm_key_rstd
    TensorType::NORM_TENSOR,          // norm_added_query_weight
    TensorType::GRADIENT_NORM_TENSOR, // norm_added_query_mean
    TensorType::GRADIENT_NORM_TENSOR, // norm_added_query_rstd
    TensorType::NORM_TENSOR,          // norm_added_key_weight
    TensorType::GRADIENT_NORM_TENSOR, // norm_added_key_mean
    TensorType::GRADIENT_NORM_TENSOR, // norm_added_key_rstd
    TensorType::ROPE_TENSOR,          // rope_sin
    TensorType::ROPE_TENSOR,          // rope_cos
};

// tiling key
constexpr int64_t IS_TRAINING_TILING_KEY = 1;
constexpr int64_t CONCAT_ORDER_TILING_KEY = 10;
constexpr int64_t ROPE_TYPE_TILING_KEY = 100;
constexpr int64_t NORM_ADDED_TYPE_TILING_KEY = 100000;
constexpr int64_t NORM_TYPE_TILING_KEY = 100000000;

constexpr int64_t SINGLE_BUFFER = 1;
constexpr int64_t DOUBLE_BUFFER = 1;
constexpr int64_t BLOCK_SIZE = 32;

template <typename T>
inline T CeilAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

static int64_t CeilDiv(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

struct NormRopeConcatGradContext {
    const char *opName;

    InputTensorInfo inputs[MAX_BACKWARD_INPUT_NUM];

    const size_t requiredInputNum{REQUIRED_BACKWARD_INPUT_NUM};
    const size_t maxInputNum{MAX_BACKWARD_INPUT_NUM};

    const int64_t *normType;
    const int64_t *normAddedType;
    const int64_t *ropeType;
    const int64_t *concatOrder;
    const float *eps;
};

class NormRopeConcatGradTiling {
public:
    NormRopeConcatGradTiling() = default;
    ~NormRopeConcatGradTiling() = default;

    ge::graphStatus DoTiling(gert::TilingContext *ctx);
    ge::graphStatus SetTiling(gert::TilingContext *ctx);

private:
    ge::graphStatus GetShapeInfo();
    ge::graphStatus CheckInput();
    ge::graphStatus CheckInput(const InputTensorInfo &x);
    ge::graphStatus ComputeTilingKey();
    ge::graphStatus ComputeCoreTilingStrategy();
    ge::graphStatus ComputeUBTilingStrategy();
    ge::graphStatus ComputeWorkSpace();
    ge::graphStatus FillTilingData();
    ge::graphStatus PrintTilingData(gert::TilingContext *ctx);
    ge::graphStatus ConvertContextOne(gert::TilingContext *ctx, InputTensorInfo &x);

private:
    uint64_t batchSize_{0};
    uint64_t headNum_{0};
    uint64_t headDim_{0};
    uint64_t querySeq_{0};
    uint64_t keySeq_{0};
    uint64_t valueSeq_{0};
    uint64_t encoderQuerySeq_{0};
    uint64_t encoderKeySeq_{0};
    uint64_t encoderValueSeq_{0};
    uint64_t totalQuerySeq_{0};
    uint64_t totalKeySeq_{0};
    uint64_t totalValueSeq_{0};
    uint64_t ropeSeq_{0};
    uint64_t ropeDim_{0};
    uint64_t normDim_{0};
    uint64_t splitHeadNum_{1};
    uint64_t usedCore_{0};
    uint64_t maxSeq_{0};
    uint64_t avgHeads_{0};
    uint64_t tailHeads_{0};
    float scale_{1.0f};
    float eps_{1e-5f};

    uint64_t numBlocks_{0};
    int64_t tilingKey_{0};
    size_t *workspace_{nullptr};
    ge::DataType inputDtype_{ge::DT_FLOAT};

    NormRopeConcatGradContext context_;
    NormRopeConcatGradTilingData tilingData_;

    NormRopeConcatGradCompileInfo compileInfo_;
};

} // namespace optiling

#endif // NORM_ROPE_CONCAT_GRAD_TILING_H_
