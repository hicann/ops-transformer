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
 * \file moe_distribute_dispatch_teardown_tiling_arch35.cpp
 * \brief
 */

#include "mc2_log.h"
#include "moe_distribute_dispatch_teardown_tiling_arch35.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(MoeDistributeDispatchTeardown, MoeDistributeDispatchTeardownTilingA5, 0);
bool MoeDistributeDispatchTeardownTilingA5::IsCapable()
{
    if (npuArch_ == NpuArch::DAV_3510) {
        OP_LOGD(nodeName_, "Do MoeDistributeDispatchTeardownTilingA5 tiling.");
        return true;
    }
    return false;
}

ge::graphStatus MoeDistributeDispatchTeardownTilingA5::DoOpTiling()
{
    return MoeDistributeDispatchTeardownTilingFuncImpl();
}
} // namespace optiling
