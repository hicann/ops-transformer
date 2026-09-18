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
 * \file allto_allv_quant_grouped_mat_mul_gen_task_training.cpp
 * \brief
 */
#include <vector>

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/mc2_moe_gen_task_ops_utils.h"
#include "op_graph/mc2_gen_task_ops_utils_arch35.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "mc2_platform_info.h"
#include "mc2_comm_utils.h"

namespace ops {
static const size_t ATTR_COMM_MODE_INDEX = 14;
static const size_t RANK_DIM_BOUNDARY = 8;
static ge::Status GetRankSizeAndSetCommmode(const gert::ExeResGenerationContext *context, std::string &commMode)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Attrs pointer is null.");
        return ge::GRAPH_FAILED;
    }
    const char *commModePtr = attrs->GetStr(ATTR_COMM_MODE_INDEX);
    if (commModePtr == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "commModePtr pointer is null.");
        return ge::GRAPH_FAILED;
    }
    commMode = commModePtr;
    return ge::GRAPH_SUCCESS;
}

ge::Status AlltoAllvQuantGroupedMatMulCalcParamFunc(gert::ExeResGenerationContext *context)
{
    std::string commMode;
    ge::Status status = GetRankSizeAndSetCommmode(context, commMode);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    const char *serverType = nullptr;
    const char *streamType = nullptr;
    if ((isArch35 && commMode == "ai_cpu")) {
        serverType = "aicpu kfc server";
        streamType = "kfc_stream";
        OPS_LOG_D(context->GetNodeName(), "AlltoAllvQuantGroupedMatMulCalcParamFunc use AICPU GenTask");
    } else if ((isArch35 && commMode == "ccu")) {
        serverType = "ccu server";
        streamType = "ccu_stream";
        OPS_LOG_D(context->GetNodeName(), "AlltoAllvQuantGroupedMatMulCalcParamFunc use CCU GenTask");
    }
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, serverType, streamType);
}

ge::Status AlltoAllvQuantGroupedMatMulGenTaskFunc(const gert::ExeResGenerationContext *context,
                                                  std::vector<std::vector<uint8_t>> &tasks)
{
    std::string commMode;
    ge::Status status = GetRankSizeAndSetCommmode(context, commMode);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    if ((isArch35 && commMode == "ai_cpu")) {
        OPS_LOG_D(context->GetNodeName(), "AlltoAllvQuantGroupedMatMulGenTaskFunc use AICPU GenTask");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    } else if ((isArch35 && commMode == "ccu")) {
        OPS_LOG_D(context->GetNodeName(), "AlltoAllvQuantGroupedMatMulGenTaskFunc use CCU GenTask");
        return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
    }
    OPS_LOG_E(context->GetNodeName(), "Unsupported arch or commMode %s for AlltoAllvQuantGroupedMatMulGenTaskFunc.",
              commMode.c_str());
    return ge::GRAPH_FAILED;
}

// new ver
IMPL_OP(AlltoAllvQuantGroupedMatMul)
    .CalcOpParam(AlltoAllvQuantGroupedMatMulCalcParamFunc)
    .GenerateTask(AlltoAllvQuantGroupedMatMulGenTaskFunc);
} // namespace ops
