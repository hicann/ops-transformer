/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mhc_pre_sinkhorn_backward.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MhcPreSinkhornBackward);

const std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *> MhcPreSinkhornBackward(
    const aclTensor *gradHin, const aclTensor *gradHPost, const aclTensor *gradHRes, const aclTensor *x,
    const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias, const aclTensor *hPre,
    const aclTensor *hcBeforeNorm, const aclTensor *invRms, const aclTensor *sumOut, const aclTensor *normOut,
    float hcEps, aclOpExecutor *executor)
{
    L0_DFX(MhcPreSinkhornBackward, gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms,
           sumOut, normOut, hcEps);

    DataType outType = DataType::DT_FLOAT; // 输出类型
    Format format = Format::FORMAT_ND;     // 输出分形
    // 直接根据输入shape创建输出tensor，不依赖INFER_SHAPE
    // gradX: shape与x一致，dtype与gradHin一致
    auto outGradX = executor->AllocTensor(x->GetViewShape(), gradHin->GetDataType(), format);
    // gradPhi: shape与phi一致，dtype为FP32
    auto outGradPhi = executor->AllocTensor(phi->GetViewShape(), outType, format);
    // gradAlpha: shape与alpha一致，dtype为FP32
    auto outGradAlpha = executor->AllocTensor(alpha->GetViewShape(), outType, format);
    // gradBias: shape与bias一致，dtype为FP32
    auto outGradBias = executor->AllocTensor(bias->GetViewShape(), outType, format);

    auto ret1 = ADD_TO_LAUNCHER_LIST_AICORE(
        MhcPreSinkhornBackward,
        OP_INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut, normOut),
        OP_OUTPUT(outGradX, outGradPhi, outGradAlpha, outGradBias), OP_ATTR(hcEps));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret1 != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr, nullptr, nullptr),
                                         "MhcPreSinkhornBackward ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return std::tuple(outGradX, outGradPhi, outGradAlpha, outGradBias);
}
} // namespace l0op
