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
 * \file stem_oam_prep_paged_kv.cpp
 * \brief
 */

#include "stem_oam_prep_paged_kv.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(StemOamPrepPagedKv);

std::tuple<const aclTensor *, const aclTensor *> StemOamPrepPagedKv(
    const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kvIndices, const aclTensor *kvSeqLens,
    const aclTensor *kScaleCache, const aclTensor *vScale, float lambdaMag, const std::string &kvLayout,
    int64_t stemBlockSize, int64_t stemStride, const aclTensor *kFlat, const aclTensor *vBias,
    aclOpExecutor *executor)
{
    L0_DFX(StemOamPrepPagedKv, kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale, lambdaMag, kvLayout,
           stemBlockSize, stemStride);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        StemOamPrepPagedKv, OP_INPUT(kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale),
        OP_ATTR(lambdaMag, kvLayout, stemBlockSize, stemStride), OP_OUTPUT(kFlat, vBias));

    auto kNullResult = std::tuple<const aclTensor *, const aclTensor *>{nullptr, nullptr};
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "StemOamPrepPagedKv ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return kNullResult);
    return std::tuple<const aclTensor *, const aclTensor *>(kFlat, vBias);
}
} // namespace l0op
