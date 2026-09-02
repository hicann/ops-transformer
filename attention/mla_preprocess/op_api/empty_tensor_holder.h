/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_ATTENTION_MLA_PREPROCESS_OP_API_EMPTY_TENSOR_HOLDER_H
#define OPS_TRANSFORMER_ATTENTION_MLA_PREPROCESS_OP_API_EMPTY_TENSOR_HOLDER_H

#include <cstdint>
#include <vector>

#include "aclnn/acl_meta.h"

namespace MlaPreprocessApi {

class EmptyTensorHolder {
public:
    EmptyTensorHolder(const aclTensor *&tensor, aclDataType dataType)
    {
        if (tensor == nullptr) {
            const std::vector<int64_t> shape = {0};
            tensor_ = aclCreateTensor(shape.data(), shape.size(), dataType, shape.data(), 0, ACL_FORMAT_ND,
                                      shape.data(), shape.size(), static_cast<void *>(&placeholderAddr_));
            tensor = tensor_;
        }
    }

    ~EmptyTensorHolder()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
        }
    }

    bool IsValid() const
    {
        return tensor_ != nullptr;
    }

private:
    const aclTensor *tensor_ = nullptr;
    int64_t placeholderAddr_ = 0xff;
};

} // namespace MlaPreprocessApi

#endif // OPS_TRANSFORMER_ATTENTION_MLA_PREPROCESS_OP_API_EMPTY_TENSOR_HOLDER_H
