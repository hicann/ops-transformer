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
 * \file quant_grouped_matmul_inplace_add_param.cpp
 * \brief QuantGroupedMatmulInplaceAdd 参数信息.
 */

#include "quant_grouped_matmul_inplace_add_param.h"
#include <vector>

using namespace ops::adv::tests::quant_grouped_matmul_inplace_add;

Param::Param(std::vector<TensorList> inputs, Tensor perTokenScale, Tensor groupList, std::vector<int64_t> groupListData,
             int32_t groupListType, int32_t groupSize)
    : mPerTokenScale(perTokenScale),
      mGroupListData(std::move(groupListData)),
      mGroupListType(groupListType),
      mGroupSize(groupSize)
{
    for (auto &tensorList : inputs) {
        mTensorLists[tensorList.Name()] = tensorList;
    }
    mGroupList = groupList;
}

Tensor ops::adv::tests::quant_grouped_matmul_inplace_add::GenTensor(const char *name, const std::vector<int64_t> &shape,
                                                                    ge::DataType dType, ge::Format format)
{
    return Tensor(name, shape, "", dType, format);
}

TensorList ops::adv::tests::quant_grouped_matmul_inplace_add::GenTensorList(
    const char *name, const std::vector<std::vector<int64_t>> &shapes, ge::DataType dType, ge::Format format)
{
    return TensorList(name, shapes, "", dType, format);
}
