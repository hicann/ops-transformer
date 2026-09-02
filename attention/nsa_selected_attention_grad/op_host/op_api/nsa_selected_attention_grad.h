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
 * \file nsa_selected_attention_grad.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_OP_NSA_SELECTED_ATTENTION_GRAD_OP_H
#define OP_API_INC_LEVEL0_OP_NSA_SELECTED_ATTENTION_GRAD_OP_H

#include "opdev/op_executor.h"

namespace l0op {
const std::size_t NSA_GRAD_OUTPUT_CNT = 3;

const std::array<const aclTensor *, NSA_GRAD_OUTPUT_CNT> NsaSelectedAttentionGrad(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *attentionOut,
    const aclTensor *attentionOutGrad, const aclTensor *softmaxMax, const aclTensor *softmaxSum,
    const aclTensor *topkIndices, const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKVLlenOptional,
    const aclTensor *attenMskOptional, double scaleValue, int64_t selectedBlockCount, int64_t selectedBlockSize,
    int64_t headNum, const char *inputLayout, int64_t sparseMode, aclOpExecutor *executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_NSA_SELECTED_ATTENTION_GRAD_OP_H
