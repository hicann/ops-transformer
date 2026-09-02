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
 * \file norm_rope_concat_grad.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_OP_Norm_Rope_Concat_GRAD_OP_H
#define OP_API_INC_LEVEL0_OP_Norm_Rope_Concat_GRAD_OP_H

#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 14> NormRopeConcatGrad(
    const aclTensor *gradQueryOutput, const aclTensor *gradKeyOutput, const aclTensor *gradValueOutput,
    const aclTensor *query, const aclTensor *key, const aclTensor *encoderQuery, const aclTensor *encoderKey,
    const aclTensor *normQueryWeight, const aclTensor *normQueryMean, const aclTensor *normQueryRstd,
    const aclTensor *normKeyWeight, const aclTensor *normKeyMean, const aclTensor *normKeyRstd,
    const aclTensor *normAddedQueryWeight, const aclTensor *normAddedQueryMean, const aclTensor *normAddedQueryRstd,
    const aclTensor *normAddedKeyWeight, const aclTensor *normAddedKeyMean, const aclTensor *normAddedKeyRstd,
    const aclTensor *ropeSin, const aclTensor *ropeCos, int64_t normType, int64_t normAddedType, int64_t ropeType,
    int64_t concatOrder, aclOpExecutor *executor);
}

// namespace l0op

#endif // OP_API_INC_LEVEL0_OP_Norm_Rope_Concat_GRAD_OP_H
