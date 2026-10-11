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
 * \file engram_fetch_wait_tiling.cpp
 * \brief host侧tiling实现
 */

#include <string>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../../../op_kernel/engram_fetch_wait_tiling_data.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace optiling {
constexpr uint32_t COMM_CONTEXT_INDEX = 0U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;

static ge::graphStatus CheckInputDataType(const gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();

    auto commContextDesc = context->GetInputDesc(COMM_CONTEXT_INDEX);
    OP_TILING_CHECK(commContextDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commContext"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commContextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "commContext",
                                              Ops::Base::ToString(commContextDesc->GetDataType()).c_str(), "INT32"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static void SetTilingKey(gert::TilingContext* context)
{
    context->SetTilingKey(0);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EngramFetchWaitTilingFunc(gert::TilingContext* context)
{
    OP_LOGD("EngramFetchWait", "tiling start.");
    OP_TILING_CHECK(context == nullptr, OP_LOGE("engram_fetch_wait_tiling", "failed to get tiling context."),
                    return ge::GRAPH_FAILED);
    const char* nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE("engram_fetch_wait_tiling", "failed to get nodeName."),
                    return ge::GRAPH_FAILED);

    EngramFetchWaitTilingData* tilingData = context->GetTilingData<EngramFetchWaitTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputDataType(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "Check input dtype failed."),
                    return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    OP_LOGD(nodeName, "aicNum=%u, aivNum=%u, numBlocks=%u", aicNum, aivNum, numBlocks);

    OP_TILING_CHECK(SetWorkSpace(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "SetWorkSpace failed."),
                    return ge::GRAPH_FAILED);
    SetTilingKey(context);

    OP_LOGD(nodeName, "EngramFetchWait tiling end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramFetchWait).Tiling(EngramFetchWaitTilingFunc);

} // namespace optiling
