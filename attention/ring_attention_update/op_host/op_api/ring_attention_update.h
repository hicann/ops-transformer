/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_OPS_ADV_RING_ATTENTION_UPDATE_H
#define CANN_OPS_ADV_RING_ATTENTION_UPDATE_H

#include "opdev/op_executor.h"

namespace l0op {

const std::array<const aclTensor *, 3> RingAttentionUpdate(
    const aclTensor *prevAttnOut, const aclTensor *prevSoftmaxMax, const aclTensor *prevSoftmaxSum,
    const aclTensor *curAttnOut, const aclTensor *curSoftmaxMax, const aclTensor *curSoftmaxSum,
    const aclTensor *actualSeqQlen, const char *inputLayout, const char *inputSoftmaxLayout, aclOpExecutor *executor);

}

#endif // CANN_OPS_ADV_RING_ATTENTION_UPDATE_H
