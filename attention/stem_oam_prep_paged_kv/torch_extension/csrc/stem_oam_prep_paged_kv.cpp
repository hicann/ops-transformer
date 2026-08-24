/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <string>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

const int64_t DIM_QK = 128;

std::tuple<at::Tensor, at::Tensor> StemOamPrepPagedKv(const at::Tensor &kCache, const at::Tensor &vCache,
                                                      const at::Tensor &kvIndices, at::IntArrayRef kvSeqLens,
                                                      const c10::optional<at::Tensor> &kScaleCacheOpt,
                                                      const c10::optional<at::Tensor> &vScaleOpt, double lambdaMag,
                                                      const std::string &kvLayout, int64_t stemBlockSize,
                                                      int64_t stemStride)
{
    TORCH_CHECK(kCache.numel() > 0, "kCache is empty");
    TORCH_CHECK(vCache.numel() > 0, "vCache is empty");
    TORCH_CHECK(kvIndices.numel() > 0, "kvIndices is empty");
    TORCH_CHECK(kvSeqLens.size() > 0, "kvSeqLens is empty");
    TORCH_CHECK(stemBlockSize % 32 == 0 && stemBlockSize <= 256, "stemBlockSize must be %32==0 and <=256, got ",
                stemBlockSize);
    TORCH_CHECK(stemStride % 16 == 0 && stemStride <= 64 && stemStride <= stemBlockSize,
                "stemStride must be %16==0, <=64, and <=stemBlockSize, got ", stemStride);
    TORCH_CHECK(kvLayout == "BBND" || kvLayout == "BNBD", "kvLayout must be BBND or BNBD, got ", kvLayout);

    at::Tensor kScaleCache = kScaleCacheOpt.value_or(at::Tensor());
    at::Tensor vScale = vScaleOpt.value_or(at::Tensor());
    if (kCache.scalar_type() == at::kFloat8_e4m3fn) {
        TORCH_CHECK(kScaleCache.defined(), "kScaleCache is required when kCache is FP8");
        TORCH_CHECK(kScaleCache.numel() > 0, "kScaleCache is empty");
        TORCH_CHECK(vScale.defined(), "vScale is required when kCache is FP8");
        TORCH_CHECK(vScale.numel() > 0, "vScale is empty");
    }

    int64_t batch = kvIndices.size(0);
    int64_t numHeads = (kvLayout == "BBND") ? kCache.size(2) : kCache.size(1);
    int64_t kflatDim = stemStride * DIM_QK;

    // read kvSeqLens from NPU to host to compute max_Kb
    int64_t maxKvSeqLens = *std::max_element(kvSeqLens.begin(), kvSeqLens.end());
    int64_t maxKb = (maxKvSeqLens + stemBlockSize - 1) / stemBlockSize;
    if (maxKb < 1)
        maxKb = 1;

    at::Tensor kFlat = at::empty({batch, numHeads, maxKb, kflatDim}, kCache.options().dtype(at::kBFloat16));
    at::Tensor vBias = at::empty({batch, numHeads, maxKb}, kCache.options().dtype(at::kFloat));

    StorageShapeTensor kCacheWrapped{kCache};
    StorageShapeTensor vCacheWrapped{vCache};
    StorageShapeTensor kvIndicesWrapped{kvIndices};
    StorageShapeTensor kScaleCacheWrapped{kScaleCache};
    StorageShapeTensor kFlatWrapped{kFlat};
    StorageShapeTensor vBiasWrapped{vBias};
    const char *kvLayoutPtr = kvLayout.c_str();
    ACLNN_CMD(aclnnStemOamPrepPagedKv, kCacheWrapped, vCacheWrapped, kvIndicesWrapped, kvSeqLens, kScaleCacheWrapped,
              vScale, lambdaMag, kvLayoutPtr, stemBlockSize, stemStride, kFlatWrapped, vBiasWrapped);

    return std::tuple<at::Tensor, at::Tensor>(kFlat, vBias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("stem_oam_prep_paged_kv", &StemOamPrepPagedKv, "stem_oam_prep_paged_kv");
}
} // namespace op_api
