/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_compress_epilog.cpp
 * \brief L0 operator implementation for KvCompressEpilog
 */

#include "kv_compress_epilog.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(KvCompressEpilog);

const aclTensor* KvCompressEpilog(aclTensor* cache, const aclTensor* x, const aclTensor* slotMapping,
                                  int64_t quantGroupSize, int64_t quantMode, bool roundScale, float xScale,
                                  aclOpExecutor* executor)
{
    L0_DFX(KvCompressEpilog, cache, x, slotMapping, quantGroupSize, quantMode, roundScale, xScale);

    ADD_TO_LAUNCHER_LIST_AICORE(KvCompressEpilog, OP_INPUT(cache, x, slotMapping), OP_OUTPUT(cache),
                                OP_ATTR(quantGroupSize, quantMode, roundScale, xScale));

    return cache;
}

} // namespace l0op
