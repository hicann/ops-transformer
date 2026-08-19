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
 * \file kv_rms_norm_rope_cache_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_KV = 0;
constexpr size_t INPUT_IDX_GAMMA = 1;
constexpr size_t INPUT_IDX_COS = 2;
constexpr size_t INPUT_IDX_SIN = 3;
constexpr size_t INPUT_IDX_INDEX = 4;
constexpr size_t INPUT_IDX_K_CACHE = 5;
constexpr size_t INPUT_IDX_V_CACHE = 6;
constexpr size_t OUTPUT_IDX_K_CACHE = 0;
constexpr size_t OUTPUT_IDX_CKV_CACHE = 1;
constexpr size_t OUTPUT_IDX_K_ROPE = 2;
constexpr size_t OUTPUT_IDX_C_KV = 3;
constexpr size_t HEAD_SIZE_SHAPE_IDX = 3;
constexpr size_t EXPECT_DIM_NUM = 4;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
constexpr int64_t RMS_NORM_LENGTH_V2 = 192;

inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape *output_shape)
{
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    return ge::GRAPH_SUCCESS;
}

graphStatus InferShape4KvRmsNormRopeCache(gert::InferShapeContext *context)
{
    OP_LOGI(context, "Begin to do InferShape4KvRmsNormRopeCache.");

    const gert::Shape *kCacheInputShape = context->GetInputShape(INPUT_IDX_K_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheInputShape);
    const gert::Shape *vCacheInputShape = context->GetInputShape(INPUT_IDX_V_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheInputShape);
    const gert::Shape *kvInputShape = context->GetInputShape(INPUT_IDX_KV);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvInputShape);
    const gert::Shape *cosInputShape = context->GetInputShape(INPUT_IDX_COS);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosInputShape);
    const gert::Shape *gammaInputShape = context->GetInputShape(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaInputShape);

    gert::Shape *kCacheShape = context->GetOutputShape(OUTPUT_IDX_K_CACHE);
    gert::Shape *vCacheShape = context->GetOutputShape(OUTPUT_IDX_CKV_CACHE);
    gert::Shape *kRopeShape = context->GetOutputShape(OUTPUT_IDX_K_ROPE);
    gert::Shape *cKvShape = context->GetOutputShape(OUTPUT_IDX_C_KV);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, kRopeShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, cKvShape);

    int64_t kvDimSize = kvInputShape->GetDimNum();
    int64_t cosDimSize = cosInputShape->GetDimNum();
    int64_t gammaDimSize = gammaInputShape->GetDimNum();
    int64_t methodMode = gammaInputShape->GetDim(gammaDimSize - 1) == RMS_NORM_LENGTH_V2 ? 1 : 0;

    *kCacheShape = *kCacheInputShape;
    *vCacheShape = *vCacheInputShape;
    *kRopeShape = *kvInputShape;
    *cKvShape = *kvInputShape;

    // -2 场景将输出kRopeShape和cKvShape设置为[-1, -1, -1, -1]
    if (Ops::Base::IsUnknownRank(*kvInputShape)) {
        SetAllUnknownDim(EXPECT_DIM_NUM, kRopeShape);
        SetAllUnknownDim(EXPECT_DIM_NUM, cKvShape);
        // 非-2场景校验输入shape为4维
    } else {
        if (kvDimSize != EXPECT_DIM_NUM) {
            OP_LOGE(context, "don't support kv dimSize != 4, infershape failed");
            return ge::GRAPH_FAILED;
        }
    }

    if (methodMode == 0) {
        // v1版本根据gamma和cos的最后1维设置kRopeShape和cKvShape的最后1维
        if (!Ops::Base::IsUnknownRank(*cosInputShape)) {
            kRopeShape->SetDim(HEAD_SIZE_SHAPE_IDX, cosInputShape->GetDim(cosDimSize - 1));
        }
        if (!Ops::Base::IsUnknownRank(*gammaInputShape)) {
            cKvShape->SetDim(HEAD_SIZE_SHAPE_IDX, gammaInputShape->GetDim(gammaDimSize - 1));
        }
    } else {
        // v2版本根据gamma和vCache的最后1维设置kRopeShape和cKvShape的最后1维
        if (!Ops::Base::IsUnknownRank(*gammaInputShape)) {
            kRopeShape->SetDim(HEAD_SIZE_SHAPE_IDX, gammaInputShape->GetDim(gammaDimSize - 1));
        }
        if (!Ops::Base::IsUnknownRank(*vCacheInputShape)) {
            cKvShape->SetDim(HEAD_SIZE_SHAPE_IDX, vCacheInputShape->GetDim(kvDimSize - 1));
        }
    }

    OP_LOGD(context, "End to do InferShape4KvRmsNormRopeCache.");
    return GRAPH_SUCCESS;
}

graphStatus InferDtype4KvRmsNormRopeCache(gert::InferDataTypeContext *context)
{
    OP_LOGD(context, "InferDtype4KvRmsNormRopeCache enter");

    auto k_input_dtype = context->GetInputDataType(INPUT_IDX_K_CACHE);
    auto v_input_dtype = context->GetInputDataType(INPUT_IDX_V_CACHE);
    auto kv_input_dtype = context->GetInputDataType(INPUT_IDX_KV);

    context->SetOutputDataType(OUTPUT_IDX_K_CACHE, k_input_dtype);
    context->SetOutputDataType(OUTPUT_IDX_CKV_CACHE, v_input_dtype);
    context->SetOutputDataType(OUTPUT_IDX_K_ROPE, kv_input_dtype);
    context->SetOutputDataType(OUTPUT_IDX_C_KV, kv_input_dtype);
    OP_LOGD(context, "InferDtype4KvRmsNormRopeCache end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvRmsNormRopeCache)
    .InferShape(InferShape4KvRmsNormRopeCache)
    .InferDataType(InferDtype4KvRmsNormRopeCache);
} // namespace ops
