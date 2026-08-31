/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>

#include "fallback/fallback.h"
#include "fallback/fallback_comm.h"
#include "log/log.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;

constexpr size_t INDEX_INPUT_X1 = 0;
constexpr size_t INDEX_INPUT_X2 = 1;
constexpr size_t INDEX_INPUT_SCALE2 = 2;
constexpr size_t INDEX_INPUT_GROUP_LIST = 3;
constexpr size_t INDEX_INPUT_Y = 4;
constexpr size_t INDEX_INPUT_SCALE1 = 5;
constexpr size_t INDEX_OUTPUT_Y = 0;
constexpr size_t INDEX_ATTR_GROUP_LIST_TYPE = 0;
constexpr size_t DIM_TWO = 2;

static inline aclTensor *GeTensor2AclTensor(const gert::Tensor *geTensor, bool enableTranspose, size_t index)
{
    if (geTensor == nullptr) {
        return nullptr;
    }
    auto storageShape = geTensor->GetStorageShape();
    if (storageShape.GetDimNum() <= 1) {
        return ConvertType(geTensor);
    }
    std::vector<int64_t> storageShapeVec;
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        storageShapeVec.push_back(storageShape.GetDim(i));
    }
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    OP_CHECK_IF(aclCreateTensor == nullptr, OP_LOGE("aclnnfallback", "aclCreateTensor nullptr"), return nullptr);
    void *deviceAddr = const_cast<void *>(geTensor->GetAddr());
    auto dataType_ge = geTensor->GetDataType();
    aclDataType dataType = ACL_DT_UNDEFINED;
    if (dataType_ge == ge::DataType::DT_FLOAT8_E4M3FN || dataType_ge == ge::DataType::DT_FLOAT8_E5M2 ||
        dataType_ge == ge::DataType::DT_HIFLOAT8 || dataType_ge == ge::DataType::DT_FLOAT8_E8M0) {
        dataType = static_cast<aclDataType>(dataType_ge);
    } else {
        dataType = ToAclDataType(dataType_ge);
    }
    auto origin_shape = geTensor->GetOriginShape();
    std::vector<int64_t> viewShape;
    for (size_t i = 0; i < origin_shape.GetDimNum(); ++i) {
        viewShape.push_back(origin_shape.GetDim(i));
    }
    std::vector<int64_t> strides(viewShape.size(), 1);
    // Compute the strides of contiguous tensor
    OP_CHECK_IF(viewShape.size() < DIM_TWO,
                OP_LOGE("QuantGroupedMatmulInplaceAdd aclnnfallback",
                        "The dim num of viewshape should be greater than or equal to 2, but the actual is %zu.",
                        viewShape.size()),
                return nullptr);
    for (int64_t i = viewShape.size() - 2; i >= 0; i--) {
        strides[i] = viewShape[i + 1] * strides[i + 1];
    }

    if (index == INDEX_INPUT_SCALE1 && dataType_ge == ge::DataType::DT_FLOAT8_E8M0 && enableTranspose) {
        OP_CHECK_IF(viewShape.size() < 3,
                    OP_LOGE("aclnnfallback",
                            "Mx type: wrong perTokenScale size, dim num should be greater than or equal to 3."),
                    return nullptr);
        auto swap = viewShape[0];
        viewShape[0] = viewShape[1];
        viewShape[1] = swap;
        strides[0] = 2;                // 2 in shape(k//64 + g, M, 2)
        strides[1] = viewShape[0] * 2; // since last dim is contiguous 2
        strides[2] = 1;                // last axis of stride with index 2 has velue 1
    } else if (enableTranspose) {      // when tensor is transposed, last two dims in strides and viewShape should swap
        // dimM the second-to-last dim， dimN the last dim
        auto dimM = viewShape.size() - 2;
        auto dimN = viewShape.size() - 1;
        if (viewShape[dimM] != 1 && viewShape[dimN] != 0) {
            auto swap = strides[dimN];
            strides[dimN] = strides[dimM];
            strides[dimM] = swap;
            swap = viewShape[dimN];
            viewShape[dimN] = viewShape[dimM];
            viewShape[dimM] = swap;
        }
    }
    auto aclFormat = aclFormat::ACL_FORMAT_ND;
    aclTensor *out = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, strides.data(), 0, aclFormat,
                                     storageShapeVec.data(), storageShapeVec.size(), deviceAddr);
    OP_CHECK_IF(out == nullptr, OP_LOGE("aclnnfallback", "out nullptr"), return nullptr);
    return out;
}

static graphStatus PrepareAclTensor(const OpExecuteContext *host_api_ctx, const aclTensor *&tensor, size_t index,
                                    bool enableTranspose)
{
    if (index != INDEX_INPUT_SCALE1) {
        tensor = GeTensor2AclTensor(host_api_ctx->GetInputTensor(index), enableTranspose, index);
    } else {
        tensor = GeTensor2AclTensor(host_api_ctx->GetOptionalInputTensor(INDEX_INPUT_SCALE1), enableTranspose, index);
    }
    OP_CHECK_IF(tensor == nullptr, OP_LOGE("QuantGroupedMatmulInplaceAdd aclnnfallback", "PrepareAclTensor failed."),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus QuantGroupedMatmulInplaceAddExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);
    const int64_t *groupListTypeGe = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_GROUP_LIST_TYPE);
    OP_CHECK_IF(groupListTypeGe == nullptr, OP_LOGE("aclnnfallback", "groupListTypeGe is null"), return GRAPH_FAILED);

    const aclTensor *aclTensorX1 = nullptr;
    OP_CHECK_IF(PrepareAclTensor(host_api_ctx, aclTensorX1, INDEX_INPUT_X1, true) != GRAPH_SUCCESS,
                OP_LOGE("aclnnfallback", "PrepareAclTensor x1 failed"), return GRAPH_FAILED);

    const aclTensor *aclTensorX2 = nullptr;
    OP_CHECK_IF(PrepareAclTensor(host_api_ctx, aclTensorX2, INDEX_INPUT_X2, false) != GRAPH_SUCCESS,
                OP_LOGE("aclnnfallback", "PrepareAclTensor x2 failed"), return GRAPH_FAILED);

    const aclTensor *aclTensorScale2 = nullptr;
    OP_CHECK_IF(PrepareAclTensor(host_api_ctx, aclTensorScale2, INDEX_INPUT_SCALE2, false) != GRAPH_SUCCESS,
                OP_LOGE("aclnnfallback", "PrepareAclTensor scale2 failed"), return GRAPH_FAILED);

    auto scale1Tensor = host_api_ctx->GetOptionalInputTensor(INDEX_INPUT_SCALE1);
    const aclTensor *aclTensorScale1 = nullptr;

    if (scale1Tensor != nullptr && scale1Tensor->GetDataType() == ge::DataType::DT_FLOAT8_E8M0) {
        OP_CHECK_IF(PrepareAclTensor(host_api_ctx, aclTensorScale1, INDEX_INPUT_SCALE1, true) != GRAPH_SUCCESS,
                    OP_LOGE("aclnnfallback", "PrepareAclTensor scale1 failed"), return GRAPH_FAILED);
    } else if (scale1Tensor != nullptr && scale1Tensor->GetDataType() == ge::DataType::DT_FLOAT) {
        aclTensorScale1 = ConvertType(scale1Tensor);
        OP_CHECK_IF(aclTensorScale1 == nullptr, OP_LOGE("aclnnfallback", "scale1 convert failed"), return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(scale1Tensor == nullptr, OP_LOGE("aclnnfallback", "scale1 nullptr"), return GRAPH_FAILED);
        OP_LOGE("aclnnfallback", "scale1 dtype is invalid");
        return GRAPH_FAILED;
    }

    auto groupListTensor = host_api_ctx->GetInputTensor(INDEX_INPUT_GROUP_LIST);
    auto inputYTensor = host_api_ctx->GetInputTensor(INDEX_INPUT_Y);
    OP_CHECK_IF(groupListTensor == nullptr, OP_LOGE("aclnnfallback", "groupListTensor nullptr"), return GRAPH_FAILED);
    OP_CHECK_IF(inputYTensor == nullptr, OP_LOGE("aclnnfallback", "inputYTensor nullptr"), return GRAPH_FAILED);

    int64_t groupsize = 0;

    // execute opapi
    auto api_ret_gmm = EXEC_OPAPI_CMD(aclnnQuantGroupedMatmulInplaceAdd, aclTensorX1, aclTensorX2, aclTensorScale1,
                                      aclTensorScale2, groupListTensor, inputYTensor, *groupListTypeGe, groupsize);
    OP_CHECK_IF(api_ret_gmm != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "api_ret failed:%u", api_ret_gmm),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

IMPL_OP(QuantGroupedMatmulInplaceAdd).OpExecuteFunc(QuantGroupedMatmulInplaceAddExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
