/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_ADD_UTIL_H
#define OP_API_INC_GROUPED_MATMUL_ADD_UTIL_H
#include "opdev/common_types.h"

namespace gmm_add_advanced {
struct GroupedMatmulAddParams {
    const aclTensor *x = nullptr;
    const aclTensor *weight = nullptr;
    const aclTensor *groupList = nullptr;
    aclTensor *yRef = nullptr;
    bool transposeX = true;
    bool transposeWeight = false;
    int64_t groupType = 2;
    int64_t groupListType = 0;
};
} // namespace gmm_add_advanced
#endif
