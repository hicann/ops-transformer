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
 * \file kv_quant_sparse_flash_attention.cpp
 * \brief C++ wrapper for aclnnKvQuantSparseFlashAttentionV2
 */

#include <torch/extension.h>
#include "aclnn_common.h"
#include <limits>

namespace op_api {

const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;

inline TensorWrapper MakeWrapper(const at::Tensor &tensor, aclDataType tensorAclType)
{
    return {tensor, tensorAclType};
}

int64_t GetKvHeadNum(const at::Tensor &key, const std::string &layoutKvStr)
{
    if (layoutKvStr == "TND") {
        return key.size(DIM_1);
    }
    return key.size(DIM_2);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> KvQuantSparseFlashAttention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &sparseIndices,
    const c10::optional<at::Tensor> &keyDequantScale, const c10::optional<at::Tensor> &valueDequantScale,
    const c10::optional<at::Tensor> &blockTable, const c10::optional<at::Tensor> &actualSeqLengthsQuery,
    const c10::optional<at::Tensor> &actualSeqLengthsKv, double scaleValue, int64_t keyQuantMode,
    int64_t valueQuantMode, int64_t sparseBlockSize, c10::string_view layoutQuery, c10::string_view layoutKv,
    int64_t sparseMode, int64_t preTokens, int64_t nextTokens, int64_t attentionMode, int64_t quantScaleRepoMode,
    int64_t tileSize, int64_t ropeHeadDim, c10::optional<int64_t> keyDtype, c10::optional<int64_t> valueDtype,
    const c10::optional<at::Tensor> &sinks, bool returnSoftmaxLse)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")
    TORCH_CHECK(sparseIndices.numel() > 0, "Tensor sparse_indices is empty.")

    std::string layoutQStr(layoutQuery);
    std::string layoutKvStr(layoutKv);
    const char *layoutQPtr = layoutQStr.c_str();
    const char *layoutKvPtr = layoutKvStr.c_str();

    auto outSizes = query.sizes().vec();
    outSizes.back() -= ropeHeadDim;
    at::Tensor attenOut = at::empty(outSizes, query.options().dtype(query.dtype()));

    int64_t numHeads = (layoutQStr == "BSND") ? query.size(DIM_2) : query.size(DIM_1);
    at::Tensor sinksVal = (sinks.has_value() && sinks->defined()) ?
                              sinks.value() :
                              at::zeros({numHeads}, query.options().dtype(torch::kFloat32));

    at::Tensor softmaxMax;
    at::Tensor softmaxSum;
    auto opts_f32 = query.options().dtype(torch::kFloat32);
    if (returnSoftmaxLse) {
        int64_t kvHeadNum = GetKvHeadNum(key, layoutKvStr);
        int64_t g = (layoutQStr == "BSND") ? query.size(DIM_2) / kvHeadNum : query.size(DIM_1) / kvHeadNum;
        at::SmallVector<int64_t, SIZE> lseSize;
        if (layoutQStr == "BSND") {
            lseSize = {query.size(DIM_0), kvHeadNum, query.size(DIM_1), g};
        } else {
            lseSize = {kvHeadNum, query.size(DIM_0), g};
        }
        softmaxMax = at::full(lseSize, -std::numeric_limits<float>::infinity(), opts_f32);
        softmaxSum = at::zeros(lseSize, opts_f32);
    } else {
        softmaxMax = at::empty({0}, opts_f32);
        softmaxSum = at::empty({0}, opts_f32);
    }

    bool isHifloat8Kv = keyDtype.has_value() && GetAclDataType(keyDtype.value()) == ACL_HIFLOAT8;
    if (isHifloat8Kv) {
        TensorWrapper keyWrapper = MakeWrapper(key, GetAclDataType(keyDtype.value()));
        TensorWrapper valueWrapper = MakeWrapper(value, GetAclDataType(valueDtype.value()));
        ACLNN_CMD(aclnnKvQuantSparseFlashAttentionV2, query, keyWrapper, valueWrapper, sparseIndices, keyDequantScale,
                  valueDequantScale, blockTable, actualSeqLengthsQuery, actualSeqLengthsKv, sinksVal, scaleValue,
                  keyQuantMode, valueQuantMode, sparseBlockSize, layoutQPtr, layoutKvPtr, sparseMode, preTokens,
                  nextTokens, attentionMode, quantScaleRepoMode, tileSize, ropeHeadDim, returnSoftmaxLse, attenOut,
                  softmaxMax, softmaxSum);
    } else {
        ACLNN_CMD(aclnnKvQuantSparseFlashAttentionV2, query, key, value, sparseIndices, keyDequantScale,
                  valueDequantScale, blockTable, actualSeqLengthsQuery, actualSeqLengthsKv, sinksVal, scaleValue,
                  keyQuantMode, valueQuantMode, sparseBlockSize, layoutQPtr, layoutKvPtr, sparseMode, preTokens,
                  nextTokens, attentionMode, quantScaleRepoMode, tileSize, ropeHeadDim, returnSoftmaxLse, attenOut,
                  softmaxMax, softmaxSum);
    }

    return std::make_tuple(attenOut, softmaxMax, softmaxSum);
}

} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_kv_quant_sparse_flash_attention", &op_api::KvQuantSparseFlashAttention,
          "kv_quant_sparse_flash_attention");
}
