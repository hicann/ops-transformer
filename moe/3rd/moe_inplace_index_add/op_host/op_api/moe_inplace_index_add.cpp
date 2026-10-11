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
 * \file inplace_index_add.cpp
 * \brief
 */
#include "moe_inplace_index_add.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "tiling/platform/platform_ascendc.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeInplaceIndexAdd);
OP_TYPE_REGISTER(InplaceIndexAdd);
// AICORE算子kernel
const aclTensor* MoeInplaceIndexAddAiCore(const aclTensor* self, const int64_t dim, const aclTensor* index,
                                          const aclTensor* source, const aclTensor* alphaTensor,
                                          aclOpExecutor* executor)
{
    L0_DFX(MoeInplaceIndexAddAiCore, self, dim, index, source, alphaTensor);
    auto indexAddOut = const_cast<aclTensor*>(self);
    auto ret = ACLNN_SUCCESS;
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        ret = ADD_TO_LAUNCHER_LIST_AICORE(MoeInplaceIndexAdd, OP_INPUT(self, index, source, alphaTensor),
                                          OP_OUTPUT(indexAddOut), OP_ATTR(dim));
    } else {
        ret = ADD_TO_LAUNCHER_LIST_AICORE(InplaceIndexAdd, OP_INPUT(self, index, source, alphaTensor),
                                          OP_OUTPUT(indexAddOut), OP_ATTR(dim));
    }
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MoeInplaceIndexAddAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return indexAddOut;
}

} // namespace l0op
