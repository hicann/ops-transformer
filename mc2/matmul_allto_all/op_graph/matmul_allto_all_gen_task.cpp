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
 * \file matmul_allto_all_gen_task.cpp
 * \brief
 */
#include <vector>
#include <platform/platform_info.h>
#include "common/utils/op_mc2.h"

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "register/op_impl_registry.h"
#include "op_graph/mc2_gen_task_ops_utils_arch35.h"
#include "mc2_log.h"
#include "mc2_platform_info.h"
#include "mc2_comm_utils.h"

namespace ops {

static ge::Status GetCommModeFromAttrs(const gert::ExeResGenerationContext* context, uint8_t& commMode)
{
    ge::AscendString commModeStr;
    if (!context->GetStrAttrVal("comm_mode", commModeStr)) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get comm_mode attr.");
        return ge::GRAPH_FAILED;
    }

    constexpr int64_t MAX_CCU_RANKSIZE = 8;
    if (commModeStr == ge::AscendString("ccu")) {
        commMode = Mc2Comm::COMM_MODE_CCU;
    } else if (commModeStr == ge::AscendString("ai_cpu")) {
        commMode = Mc2Comm::COMM_MODE_AICPU;
    } else {
        OPS_LOG_E(context->GetNodeName(), "Unsupported comm_mode: [%s].", commModeStr.GetString());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::Status MatmulAlltoAllGenTaskCallback(const gert::ExeResGenerationContext* context,
                                         std::vector<std::vector<uint8_t>>& tasks)
{
    return Mc2GenTaskOpsUtils::CommonKFCMc2GenTask(context, tasks);
}

static ge::Status MatmulAlltoAllCalcOpParamFunc(gert::ExeResGenerationContext* context)
{
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        uint8_t commMode = Mc2Comm::COMM_MODE_CCU;
        auto ret = GetCommModeFromAttrs(context, commMode);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        if (commMode == Mc2Comm::COMM_MODE_CCU) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTask CalcOpParam");
            return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
        } else {
            OPS_LOG_D(context->GetNodeName(), "Do A5 AICPU GenTask CalcOpParam");
            return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
        }
    }
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

static ge::Status MatmulAlltoAllGenTaskFunc(const gert::ExeResGenerationContext* context,
                                            std::vector<std::vector<uint8_t>>& tasks)
{
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        uint8_t commMode = Mc2Comm::COMM_MODE_CCU;
        auto ret = GetCommModeFromAttrs(context, commMode);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        if (commMode == Mc2Comm::COMM_MODE_CCU) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTaskFunc");
            return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
        } else {
            OPS_LOG_D(context->GetNodeName(), "Do A5 AICPU GenTaskFunc");
            return MatmulAlltoAllGenTaskCallback(context, tasks);
        }
    }

    return MatmulAlltoAllGenTaskCallback(context, tasks);
}

IMPL_OP(MatmulAlltoAll).CalcOpParam(MatmulAlltoAllCalcOpParamFunc).GenerateTask(MatmulAlltoAllGenTaskFunc);
} // namespace ops
