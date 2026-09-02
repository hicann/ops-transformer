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
 * \file quant_grouped_matmul_inplace_add_param.h
 * \brief QuantGroupedMatmulInplaceAdd 参数信息.
 */

#ifndef UTEST_QUANT_GROUPED_MATMUL_INPLACE_ADD_PARAM_H
#define UTEST_QUANT_GROUPED_MATMUL_INPLACE_ADD_PARAM_H

#include <register/op_impl_registry.h>
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"
#include "tests/utils/aclnn_tensor.h"

namespace ops::adv::tests::quant_grouped_matmul_inplace_add {

using ops::adv::tests::utils::Tensor;
using ops::adv::tests::utils::TensorList;

class Param {
public:
    std::map<std::string, TensorList> mTensorLists;
    Tensor mGroupList;
    Tensor mPerTokenScale;
    std::vector<int64_t> mGroupListData = {};
    int32_t mGroupListType = 0;
    int32_t mGroupSize = 0;

public:
    Param() = default;
    Param(std::vector<TensorList> inputs, Tensor perTokenScale, Tensor groupList, std::vector<int64_t> groupListData,
          int32_t groupListType, int32_t groupSize);
};

Tensor GenTensor(const char *name, const std::initializer_list<int64_t> &shape, ge::DataType dType,
                 ge::Format format = ge::FORMAT_ND);

TensorList GenTensorList(const char *name, const std::vector<std::vector<int64_t>> &shapes, ge::DataType dType,
                         ge::Format format = ge::FORMAT_ND);

} // namespace ops::adv::tests::quant_grouped_matmul_inplace_add
#endif // UTEST_GROUPED_MATMUL_PARAM_H
