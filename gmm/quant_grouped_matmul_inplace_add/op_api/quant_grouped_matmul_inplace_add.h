/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_OP_QUANT_GROUPED_MATMUL_INPLACE_ADD_OP_H
#define OP_API_INC_LEVEL0_OP_QUANT_GROUPED_MATMUL_INPLACE_ADD_OP_H

#include "opdev/op_executor.h"

namespace l0op {
aclTensor *QuantGroupedMatmulInplaceAdd(const aclTensor *x1, const aclTensor *x2, const aclTensor *scale1Optional,
                                        const aclTensor *scale2, const aclTensor *groupList, aclTensor *yRef,
                                        int64_t groupListType, int64_t groupSize, aclOpExecutor *executor);
}

#endif
