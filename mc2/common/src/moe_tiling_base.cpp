/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tiling_base.cc
 * \brief
 */

#include "tiling/moe_tiling_base.h"

// Default implementation
// Every thing is done by DoOptiling.
namespace optiling {
ge::graphStatus MoeTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    const char *nodeName = context_->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE(nodeName, "Fail to get nodeName."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "fail to get platform info"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTilingBase::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTilingBase::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
