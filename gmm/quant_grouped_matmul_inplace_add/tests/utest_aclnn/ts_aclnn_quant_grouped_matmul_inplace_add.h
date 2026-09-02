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
 * \file ts_quant_grouped_matmul_inplace_add.h
 * \brief QGMMInplaceAdd UTest 相关基类定义.
 */

#ifndef UTEST_TS_QGMM_INPLACE_ADD_H
#define UTEST_TS_QGMM_INPLACE_ADD_H

#include "tests/utest/ts.h"
#include "quant_grouped_matmul_inplace_add_case.h"
#include "aclnn_quant_grouped_matmul_inplace_add_case.h"

using ops::adv::tests::quant_grouped_matmul_inplace_add::AclnnQGMMInplaceAddCase;
using ops::adv::tests::quant_grouped_matmul_inplace_add::GenTensor;
using AclnnQGMMInplaceAddParam = ops::adv::tests::quant_grouped_matmul_inplace_add::AclnnQGMMInplaceAddParam;

namespace qgmmInplaceAddTestParam {
class Ts_Aclnn_QGMMInplaceAdd_WithParam_Ascend910_9591 : public Ts_WithParam_Ascend910_9591<AclnnQGMMInplaceAddCase> {};
} // namespace qgmmInplaceAddTestParam

#endif // UTEST_TS_QGMM_INPLACE_ADD_H
