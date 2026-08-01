/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_flash_attn_inner.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

void QuantFlashAttnProcessSoftmaxLse(bool returnSoftmaxLse, const aclTensor *softmaxLse,
                                     const aclTensor *&tempTensor, const aclTensor *&placeHolder)
{
    if (!returnSoftmaxLse) {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), static_cast<void *>(&addr));
        placeHolder = tempTensor;
    } else {
        placeHolder = softmaxLse;
    }
}

void QuantFlashAttnProcessSinks(const aclTensor *&sinksOptional)
{
    if (sinksOptional != nullptr) {
        const auto &shape = sinksOptional->GetViewShape();
        if (shape.GetDimNum() == 1U && shape[0] == 0) {
            OP_LOGD("sinks shape is {0}, treat as nullptr.");
            sinksOptional = nullptr;
        }
    }
}

} // namespace

#ifdef __cplusplus
}
#endif
