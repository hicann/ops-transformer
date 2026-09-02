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
 * \file moe_ep_combine_base.h
 * \brief
 */

#ifndef MOE_EP_COMBINE_BASE_H
#define MOE_EP_COMBINE_BASE_H
#include <cstdint>

#if __has_include("../common/mc2_moe_context.h")
#include "../common/mc2_moe_context.h"
#else
#include "../../common/op_kernel/mc2_moe_context.h"
#endif

__aicore__ inline GM_ADDR GetWinCombineDataAddrByRankId(__gm__ Mc2Aclnn::MoeCommContext *ctx)
{
    return (GM_ADDR)ctx->epHcclBuffer[0];
}

__aicore__ inline GM_ADDR GetWinCombineStateAddrByRankId(__gm__ Mc2Aclnn::MoeCommContext *ctx)
{
    return (GM_ADDR)ctx->epHcclBuffer[0];
}

#endif // MOE_EP_COMBINE_BASE_H
