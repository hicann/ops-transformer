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
 * \file moe_fused_topk.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_MOE_FUSED_TOPK_H_
#define OP_API_INC_LEVEL0_MOE_FUSED_TOPK_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 2> MoeFusedTopk(const aclTensor *x, const aclTensor *addNum,
                                                    const aclTensor *mappingNum, const aclTensor *mappingTable,
                                                    uint32_t groupNum, uint32_t groupTopk, uint32_t topN, uint32_t topK,
                                                    uint32_t activateType, bool isNorm, float scale,
                                                    bool enableExpertMapping, aclOpExecutor *executor);
}

#endif // OP_API_INC_LEVEL0_MOE_FUSED_TOPK_H_
