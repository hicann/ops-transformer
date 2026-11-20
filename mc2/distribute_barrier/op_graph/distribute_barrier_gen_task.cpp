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
 * \file distribute_barrier_gen_task.cpp
 * \brief
 */
#include "common/utils/mc2_gen_task_common_inc.h"
#include "fallback/fallback_2stages.h"

namespace fallback {
constexpr size_t X_REF_INDEX = 0;
constexpr size_t TIME_OUT_INDEX = 1;
constexpr size_t ELASTIC_INFO_INDEX = 2;
constexpr size_t GROUP_INDEX = 0;
constexpr size_t WORLD_SIZE_INDEX = 1;

static void ReleaseBarrierParams(gert::OpExecutePrepareContext* context)
{
    auto* chain = reinterpret_cast<gert::KernelContext*>(context)->GetOutput(
        static_cast<size_t>(gert::OpExecutePrepareOutputIndex::kParams));
    auto* params = chain == nullptr ? nullptr : chain->GetPointer<OpApiParams>();
    if (params == nullptr) {
        return;
    }
    for (auto& value : params->converted_params) {
        if (value.deleter != nullptr) {
            value.deleter(value.pointer);
        }
    }
    params->converted_params.clear();
}

static ge::graphStatus DistributeBarrierPrepareFunc(gert::OpExecutePrepareContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("DistributeBarrier", "Execute context is null"),
                return ge::GRAPH_FAILED);
    const auto* xRef = host_api_ctx->GetInputTensor(X_REF_INDEX);
    const auto* attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(xRef == nullptr || attrs == nullptr,
                OP_LOGE("DistributeBarrier", "Required input or attributes are absent"), return ge::GRAPH_FAILED);
    const auto* group = attrs->GetStr(GROUP_INDEX);
    const auto* worldSize = attrs->GetInt(WORLD_SIZE_INDEX);
    OP_CHECK_IF(group == nullptr || worldSize == nullptr,
                OP_LOGE("DistributeBarrier", "Required group or world_size attribute is absent"),
                return ge::GRAPH_FAILED);
    const auto* timeOut = host_api_ctx->GetOptionalInputTensor(TIME_OUT_INDEX);
    const auto* elasticInfo = host_api_ctx->GetOptionalInputTensor(ELASTIC_INFO_INDEX);
    OP_LOGD("DistributeBarrier", "ACLNN graph prepare, world_size=%ld, timeout=%d, elastic=%d", *worldSize,
            timeOut != nullptr, elasticInfo != nullptr);
    auto apiRet = EXEC_OPAPI_PREPARE_CMD(aclnnDistributeBarrierV2, xRef, timeOut, elasticInfo, group, *worldSize);
    if (apiRet != ge::GRAPH_SUCCESS) {
        ReleaseBarrierParams(host_api_ctx);
    }
    return apiRet;
}

IMPL_OP(DistributeBarrier).Op2StageExecuteFuncs(DistributeBarrierPrepareFunc, ExecuteOpLaunch);
} // namespace fallback

namespace ops {

ge::Status DistributeBarrierCalcParamFunc(gert::ExeResGenerationContext* context)
{
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status DistributeBarrierGenTaskFunc(const gert::ExeResGenerationContext* context,
                                        std::vector<std::vector<uint8_t>>& tasks)
{
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
}

// new ver
IMPL_OP(DistributeBarrier).CalcOpParam(DistributeBarrierCalcParamFunc).GenerateTask(DistributeBarrierGenTaskFunc);
} // namespace ops
