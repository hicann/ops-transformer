/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_reduce_scatter_v2_gen_task.cpp
 * \brief
 */

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/mc2_gen_task_ops_utils_arch35.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "mc2_platform_info.h"
#include "mc2_comm_utils.h"

namespace {
constexpr int64_t MAX_CCU_RANKSIZE = 8;
}

namespace ops {
static ge::Status MatmulReduceScatterV2CalcOpParam(gert::ExeResGenerationContext* context)
{
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        ge::AscendString comm_mode;
        if (!context->GetStrAttrVal("comm_mode", comm_mode)) {
            OPS_LOG_E(context->GetNodeName(), "GetStrAttrVal failed to get comm_mode");
            return ge::GRAPH_FAILED;
        }
        if (comm_mode == ge::AscendString("ai_cpu")) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 AICPU GenTask CalcOpParam");
            return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
        } else if (comm_mode == ge::AscendString("ccu")) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTask CalcOpParam");
            return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
        }
        OPS_LOG_E(context->GetNodeName(), "Unsupported comm_mode: %s, comm_mode must be 'ai_cpu' or 'ccu'.",
                  comm_mode.GetString());
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_E(context->GetNodeName(), "Only support A5");
    return ge::GRAPH_FAILED;
}

static ge::Status MatmulReduceScatterV2GenTask(const gert::ExeResGenerationContext* context,
                                               std::vector<std::vector<uint8_t>>& tasks)
{
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 GenTaskFunc");
        ge::AscendString comm_mode;
        if (!context->GetStrAttrVal("comm_mode", comm_mode)) {
            OPS_LOG_E(context->GetNodeName(), "GetStrAttrVal failed to get comm_mode");
            return ge::GRAPH_FAILED;
        }
        if (comm_mode == ge::AscendString("ai_cpu")) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 AiCPU GenTaskFunc");
            return Mc2GenTaskOpsUtils::CommonKFCMc2GenTask(context, tasks);
        } else if (comm_mode == ge::AscendString("ccu")) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTaskFunc");
            return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
        }
        OPS_LOG_E(context->GetNodeName(), "Unsupported comm_mode: %s, comm_mode must be 'ai_cpu' or 'ccu'.",
                  comm_mode.GetString());
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_E(context->GetNodeName(), "Only support A5 AICPU or CCU");
    return ge::GRAPH_FAILED;
}

IMPL_OP(MatmulReduceScatterV2).CalcOpParam(MatmulReduceScatterV2CalcOpParam).GenerateTask(MatmulReduceScatterV2GenTask);
} // namespace ops
