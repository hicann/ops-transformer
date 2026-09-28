/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Adapted from op-plugin/op_plugin/ops/opapi/LightningIndexerNpuOpApi.cpp.
#include <acl/acl.h>
#include <torch/extension.h>
#include <string>
#include <tuple>
#include <vector>

#include "aclnn_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor> LightningIndexer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &curSeqLengthsQuery, const c10::optional<at::Tensor> &curSeqLengthsKey,
    const c10::optional<at::Tensor> &blockTable, const std::string &layoutQuery, const std::string &layoutKey,
    int64_t sparseCount, int64_t kvBlockLen, int64_t qBlockLen, int64_t initNum, int64_t localNum, int64_t sparseMode,
    int64_t preTokens, int64_t nextTokens, bool returnValue)
{
    TORCH_CHECK(query.device().type() == at::kPrivateUse1, "query must be on NPU");
    TORCH_CHECK(key.device() == query.device() && weights.device() == query.device(),
                "query, key and weights must be on the same device");
    TORCH_CHECK(query.is_contiguous(), "query should be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key should be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights should be contiguous");
    TORCH_CHECK(layoutQuery == "BSND" || layoutQuery == "TND", "unsupported layout_query");
    TORCH_CHECK(layoutKey == "BSND" || layoutKey == "TND" || layoutKey == "PA_BSND", "unsupported layout_key");
    TORCH_CHECK(query.dim() == (layoutQuery == "BSND" ? 4 : 3), "invalid query rank");
    TORCH_CHECK(key.dim() == (layoutKey == "TND" ? 3 : 4), "invalid key rank");
    TORCH_CHECK(sparseCount > 0, "sparse_count must be positive");

    const int64_t n2 = key.size(layoutKey == "TND" ? 1 : 2);
    std::vector<int64_t> shape;
    if (layoutQuery == "BSND") {
        shape = {query.size(0), query.size(1), n2, sparseCount};
    } else {
        shape = {query.size(0), n2, sparseCount};
    }
    auto indices = at::empty(shape, query.options().dtype(at::kInt));
    auto values = at::empty(shape, query.options());

    // Preserve the original adapter's int64 conversion without converting an
    // undefined tensor when a valid BSND invocation omits the optional lengths.
    c10::optional<at::Tensor> seqQ = c10::nullopt;
    c10::optional<at::Tensor> seqK = c10::nullopt;
    if (curSeqLengthsQuery.has_value() && curSeqLengthsQuery->defined()) {
        seqQ = curSeqLengthsQuery->to(at::kLong);
    }
    if (curSeqLengthsKey.has_value() && curSeqLengthsKey->defined()) {
        seqK = curSeqLengthsKey->to(at::kLong);
    }
    char *queryLayout = const_cast<char *>(layoutQuery.c_str());
    char *keyLayout = const_cast<char *>(layoutKey.c_str());
    ACLNN_CMD(aclnnLightningIndexer, query, key, weights, seqQ, seqK, blockTable, queryLayout, keyLayout, sparseCount,
              kvBlockLen, qBlockLen, initNum, localNum, sparseMode, preTokens, nextTokens, returnValue, indices,
              values);
    return {indices, values};
}
} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lightning_indexer", &op_api::LightningIndexer,
          "Experimental LightningIndexer with block, sink and local-window options");
}
