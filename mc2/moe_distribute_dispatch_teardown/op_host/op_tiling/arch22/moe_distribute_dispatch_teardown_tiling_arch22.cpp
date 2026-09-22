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
 * \file moe_distribute_dispatch_teardown_tiling_arch22.cpp
 * \brief
 */

#include "moe_distribute_dispatch_teardown_tiling_arch22.h"

namespace optiling {

bool MoeDistributeDispatchTeardownTilingA3::IsCapable()
{
    OP_LOGD(nodeName_, "Do MoeDistributeDispatchTeardownTilingA3 tiling.");
    return false;
}

ge::graphStatus MoeDistributeDispatchTeardownTilingA3::DoOpTiling()
{
    return MoeDistributeDispatchTeardownTilingFuncImpl();
}
} // namespace optiling
