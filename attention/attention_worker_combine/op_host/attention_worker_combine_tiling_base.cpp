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
 * \file attention_worker_combine_tiling_base.cpp
 * \brief Implements the common AttentionWorkerCombine tiling lifecycle, workspace and platform preparation.
 */
#include "attention_worker_combine_tiling_base.h"
#include "err/ops_err.h"

namespace optiling {
// 合法输入下 Tiling 失败时，按正式错误码 EZ1008（Arglist: opName, reason）首报，并输出 PLOG 上下文。
// static 内部链接：避免与其他算子文件的同名 inline 函数 ODR 去重导致覆盖率归因不确定。
static void ReportTilingFailure(const char* opName, const char* reason)
{
    OP_LOGE_WITHOUT_REPORT(opName, "%s\n", reason);
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName, reason};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;
constexpr uint32_t BATCH_MODE = 1;

ge::graphStatus AttentionWorkerCombineTilingBase::GetPlatformInfo()
{
    return DoGetPlatformInfo();
}

ge::graphStatus AttentionWorkerCombineTilingBase::GetShapeAttrsInfo()
{
    return DoGetShapeAttrsInfo();
}

ge::graphStatus AttentionWorkerCombineTilingBase::DoOpTiling()
{
    OP_CHECK_IF((CalcOpTiling() != ge::GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "CalcOpTiling()", "GRAPH_FAILED",
                                                      "CalcOpTiling failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((CalcTilingKey() != ge::GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "CalcTilingKey()", "GRAPH_FAILED",
                                                      "CalcTilingKey failed."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionWorkerCombineTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AttentionWorkerCombineTilingBase::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus AttentionWorkerCombineTilingBase::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionWorkerCombineTilingBase::PostTiling()
{
    DoPostTiling();
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(BATCH_MODE);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = static_cast<size_t>(DEFAULT_WORKSPACE_SIZE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4AttentionWorkerCombine(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingForAttentionWorkerCombine running.");
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4AttentionWorkerCombine(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4AttentionWorkerCombine running.");
    auto compileInfo = context->GetCompiledInfo<AttentionWorkerCombineCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        compileInfo->coreNum <= 0,
        ReportTilingFailure(context->GetNodeName(), ("PlatformAscendC::GetCoreNumAiv returned " +
                                                     std::to_string(compileInfo->coreNum) + ", platform query failure.")
                                                        .c_str()),
        return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF(
        compileInfo->ubSize <= 0,
        ReportTilingFailure(context->GetNodeName(), ("PlatformAscendC::GetCoreMemSize(UB) returned " +
                                                     std::to_string(compileInfo->ubSize) + ", platform query failure.")
                                                        .c_str()),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "coreNum: %ld, ubSize: %ld", compileInfo->coreNum, compileInfo->ubSize);
    OP_LOGD(context->GetNodeName(), "TilingPrepare4AttentionWorkerCombine success.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttentionWorkerCombine)
    .Tiling(Tiling4AttentionWorkerCombine)
    .TilingParse<AttentionWorkerCombineCompileInfo>(TilingPrepare4AttentionWorkerCombine);

} // namespace optiling
