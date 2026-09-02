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
 * \file aclnn_quant_grouped_matmul_inplace_add_param.h
 * \brief QGMMInplaceAdd Aclnn 参数信息.
 */

#ifndef UTEST_ACLNN_QGMM_INPLACE_ADD_PARAM_H
#define UTEST_ACLNN_QGMM_INPLACE_ADD_PARAM_H

#include "quant_grouped_matmul_inplace_add_case.h"
#include "tests/utils/aclnn_tensor.h"
#include "tests/utils/aclnn_tensor_list.h"

namespace ops::adv::tests::quant_grouped_matmul_inplace_add {

class AclnnQGMMInplaceAddParam : public ops::adv::tests::quant_grouped_matmul_inplace_add::Param {
public:
    using AclnnTensor = ops::adv::tests::utils::AclnnTensor;

public:
    /* 输入输出 */
    AclnnTensor aclnnX1, aclnnX2, aclnnScale2, aclnnGroupList, aclnnScale1, aclnnY;
    aclTensor *x1 = nullptr;
    aclTensor *scale1 = nullptr;
    std::map<std::string, Tensor> mTensors;

public:
    AclnnQGMMInplaceAddParam() = default;
    AclnnQGMMInplaceAddParam(std::vector<Tensor> inputs, std::vector<int64_t> groupListData, int32_t groupListType,
                             int32_t groupSize);

    ~AclnnQGMMInplaceAddParam();

    bool Init();
    void InitAclTensor();

private:
    bool InitGroupList();
    bool CreateX1AclTensor();
    bool CreateScale1AclTensor();
};

} // namespace ops::adv::tests::quant_grouped_matmul_inplace_add
#endif // UTEST_ACLNN_GROUPEDMATMUL_PARAM_H
