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
 * \file quant_grouped_matmul_dequant.cpp
 * \brief
 */

#include "quant_grouped_matmul_dequant.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(QuantGroupedMatmulDequant);

const aclTensor* QuantGroupedMatmulDequant(const aclTensor* x, const aclTensor* weight, const aclTensor* weightScale,
                                           const aclTensor* groupList, const aclTensor* bias, const aclTensor* xScale,
                                           const aclTensor* xOffset, const aclTensor* smoothScale, char* xQuantMode,
                                           bool transposeWeight, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(QuantGroupedMatmulDequant, x, weight, weightScale, groupList, bias, xScale, xOffset, smoothScale, xQuantMode,
           transposeWeight, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        QuantGroupedMatmulDequant, OP_INPUT(x, weight, weightScale, groupList, bias, xScale, xOffset, smoothScale),
        OP_OUTPUT(out), OP_ATTR(xQuantMode, transposeWeight));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "QuantGroupedMatmulDequantAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return out;
}
} // namespace l0op
