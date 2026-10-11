/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_inplace_index_add_tiling_arch35.cpp
 * \brief
 */

#include "op_host/tiling_templates_registry.h"
#include "moe_inplace_index_add_tiling_arch35.h"

namespace optiling {
ge::graphStatus MoeInplaceIndexAddTilingForAscendC(gert::TilingContext* context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

bool MoeGetTilingCoreNum(const gert::TilingParseContext* context, uint32_t& core_num)
{
    auto platform_info = context->GetPlatformInfo();
    MOE_OPS_CHECK_NULL_WITH_CONTEXT_RET(context, platform_info, false);

    core_num = platform_info->GetCoreNum();
    OP_LOGD(context->GetNodeName(), "get tiling core num is %u", core_num);
    return true;
}

} // namespace optiling
