/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_QUANT_GROUPED_MATMUL_INPLACE_ADD_UTIL_H
#define OP_API_INC_QUANT_GROUPED_MATMUL_INPLACE_ADD_UTIL_H
#include "opdev/common_types.h"

namespace QGmmInPlaceAdd {
using namespace op;
struct QuantGroupedMatmulInplaceAddParams {
    const aclTensor *x1 = nullptr;
    const aclTensor *x2 = nullptr;
    const aclTensor *scale1Optional = nullptr;
    const aclTensor *scale2 = nullptr;
    const aclTensor *groupList = nullptr;
    aclTensor *yRef = nullptr;
    int64_t groupListType = 0;
    int64_t groupSize = 0;
};
} // namespace QGmmInPlaceAdd
#endif
