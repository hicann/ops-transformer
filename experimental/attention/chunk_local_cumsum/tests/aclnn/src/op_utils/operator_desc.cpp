/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file operator_desc.cpp
 */
#include "operator_desc.h"

#include "common.h"

using namespace std;

OperatorDesc::OperatorDesc() {}

OperatorDesc::~OperatorDesc()
{
    for (auto* desc : inputDesc) {
        aclDestroyTensorDesc(desc);
    }

    for (auto* desc : outputDesc) {
        aclDestroyTensorDesc(desc);
    }
}

OperatorDesc& OperatorDesc::AddInputTensorDesc(aclDataType dataType, int numDims, const int64_t* dims, aclFormat format)
{
    aclTensorDesc* desc = aclCreateTensorDesc(dataType, numDims, dims, format);
    if (desc == nullptr) {
        ERROR_LOG("create tensor failed");
        return *this;
    }
    inputDesc.emplace_back(desc);
    return *this;
}

OperatorDesc& OperatorDesc::AddOutputTensorDesc(aclDataType dataType, int numDims, const int64_t* dims,
                                                aclFormat format)
{
    aclTensorDesc* desc = aclCreateTensorDesc(dataType, numDims, dims, format);
    if (desc == nullptr) {
        ERROR_LOG("create tensor failed");
        return *this;
    }

    outputDesc.emplace_back(desc);
    return *this;
}
