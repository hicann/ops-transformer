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
 * \file inplace_matmul_all_reduce_add_rms_norm_gen_task.cpp
 * \brief
 */
#include <vector>

#include "common/utils/op_mc2.h"
#include "platform/platform_info.h"

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/matmul_all_reduce_gen_task_ops_utils.h" // in transformer dev
#include "graph/arg_desc_info.h"
#include "graph/kernel_launch_info.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"

namespace ops {

ge::Status InplaceMatmulAllReduceAddRmsNormCalcParamFunc(gert::ExeResGenerationContext* context)
{
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status InplaceMatmulAllReduceAddRmsNormGenTaskFunc(const gert::ExeResGenerationContext* context,
                                                       std::vector<std::vector<uint8_t>>& tasks)
{
    return MatmulAllReduceGenTaskOpsUtils::MatmulAllReduceGenTaskCallback(context, tasks);
}

// new ver
IMPL_OP(InplaceMatmulAllReduceAddRmsNorm)
    .CalcOpParam(InplaceMatmulAllReduceAddRmsNormCalcParamFunc)
    .GenerateTask(InplaceMatmulAllReduceAddRmsNormGenTaskFunc);
} // namespace ops
