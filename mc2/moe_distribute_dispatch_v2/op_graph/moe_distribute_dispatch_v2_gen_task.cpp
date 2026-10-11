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
 * \file moe_distribute_dispatch_v2_gen_task.cpp
 * \brief
 */
#include <vector>
#include <set>
#include <string>

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/mc2_moe_gen_task_ops_utils.h"
#include "op_graph/mc2_gen_task_ops_utils_arch35.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "mc2_platform_info.h"

namespace ops {
static const size_t ATTR_INDEX_COMM_ALG_DISTRIBUTE_DISPATCH_V2 = 13;
ge::Status MoeDistributeDispatchV2CalcParamFunc(gert::ExeResGenerationContext* context)
{
    if ((IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) &&
        (Mc2Arch35GenTaskOpsUtils::GetCommAlg(context, ATTR_INDEX_COMM_ALG_DISTRIBUTE_DISPATCH_V2) == COMM_ALG_CCU)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 ccu calc param.");
        return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
    }
    OPS_LOG_D(context->GetNodeName(), "Do general calc param.");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

ge::Status MoeDistributeDispatchV2GenTaskFunc(const gert::ExeResGenerationContext* context,
                                              std::vector<std::vector<uint8_t>>& tasks)
{
    const char* nodeName = context->GetNodeName();
    if (IsTargetPlatformSocVersion(nodeName, PLATFORM_A2)) {
        OPS_LOG_D(nodeName, "Do A2 mte gen task.");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    if (IsTargetPlatformNpuArch(nodeName, NPUARCH_A5)) {
        const std::string commAlg =
            Mc2Arch35GenTaskOpsUtils::GetCommAlg(context, ATTR_INDEX_COMM_ALG_DISTRIBUTE_DISPATCH_V2);
        if ((commAlg == COMM_ALG_MTE) || (commAlg == COMM_ALG_FULLMESH_V1) || (commAlg == COMM_ALG_FULLMESH_V2)) {
            OPS_LOG_D(nodeName, "Do A5 mte gen task.");
            return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
        }
        if (commAlg == COMM_ALG_CCU) {
            OPS_LOG_D(nodeName, "Do A5 CCU gen task.");
            return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
        }
        OPS_LOG_E(nodeName, "Got unsupported commAlg %s.", commAlg.c_str());
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_D(nodeName, "Do A3 gen task.");
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
}

// new ver
IMPL_OP(MoeDistributeDispatchV2)
    .CalcOpParam(MoeDistributeDispatchV2CalcParamFunc)
    .GenerateTask(MoeDistributeDispatchV2GenTaskFunc);
} // namespace ops
