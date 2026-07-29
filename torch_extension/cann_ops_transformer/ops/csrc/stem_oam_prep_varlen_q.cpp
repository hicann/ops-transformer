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
 * \file stem_oam_prep_varlen_q.cpp
 * \brief
 */

#include <algorithm>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

static constexpr int64_t DIM_QK = 128;

at::Tensor stem_oam_prep_varlen_q(const at::Tensor &q, at::IntArrayRef qSeqLens, at::IntArrayRef cuSeqLensQ,
                                  const c10::optional<at::Tensor> &qScale, int64_t stemBlockSize, int64_t stemStride)
{
    TORCH_CHECK(q.size(2) == DIM_QK, "q last dim must be 128, but got ", q.size(2));
    TORCH_CHECK(stemBlockSize % 32 == 0, "stemBlockSize must be multiple of 32");
    TORCH_CHECK(stemBlockSize <= 256, "stemBlockSize must be <= 256");
    TORCH_CHECK(stemStride <= stemBlockSize, "stemStride must be <= stemBlockSize");
    TORCH_CHECK(stemStride % 16 == 0, "stemStride must be multiple of 16");
    TORCH_CHECK(stemStride <= 64, "stemStride must be <= 64");

    int64_t H_q = q.size(1);
    int64_t batch = static_cast<int64_t>(qSeqLens.size());
    int64_t qflat_dim = stemStride * DIM_QK;

    int64_t maxQLen = batch > 0 ? *std::max_element(qSeqLens.begin(), qSeqLens.end()) : 0;
    int64_t maxQPadded = ((maxQLen + stemBlockSize - 1) / stemBlockSize) * stemBlockSize;
    int64_t maxQb = maxQPadded / stemBlockSize;
    at::Tensor qFlat = at::empty({batch, H_q, maxQb, qflat_dim}, q.options().dtype(at::kBFloat16));

    ACLNN_CMD(aclnnStemOamPrepVarlenQ, q, qSeqLens, cuSeqLensQ, qScale, stemBlockSize, stemStride, qFlat);
    return qFlat;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("stem_oam_prep_varlen_q", &stem_oam_prep_varlen_q, "stem_oam_prep_varlen_q");
}

} // namespace op_api
