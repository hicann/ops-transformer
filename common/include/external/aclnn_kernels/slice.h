/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_INC_EXTERNAL_ACLNN_KERNELS_SLICE_H
#define COMMON_INC_EXTERNAL_ACLNN_KERNELS_SLICE_H

#include "opdev/op_def.h"

namespace l0op {

const aclTensor* Slice(
    const aclTensor* x, const aclTensor* y, const aclTensor* offset, const aclTensor* size, aclOpExecutor* executor);

const aclTensor* Slice(
    const aclTensor* x, const aclIntArray* offsets, const aclIntArray* size, aclOpExecutor* executor);
} // namespace l0op

#endif // COMMON_INC_EXTERNAL_ACLNN_KERNELS_SLICE_H
