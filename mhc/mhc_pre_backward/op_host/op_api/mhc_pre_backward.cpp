/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mhc_pre_backward.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MhcPreBackward);

const std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *> MhcPreBackward(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *gradHIn,
    const aclTensor *gradHPost, const aclTensor *gradHRes, const aclTensor *invRms, const aclTensor *hMix,
    const aclTensor *hPre, const aclTensor *hPost, const aclTensor *gamma, const aclTensor *gradXPostOptional,
    float hcEps, int64_t opImplMode, aclOpExecutor *executor)
{
    L0_DFX(MhcPreBackward, x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma,
           gradXPostOptional, hcEps, opImplMode);

    DataType outType = DataType::DT_FLOAT; // 输出类型
    Format format = Format::FORMAT_ND;     // 输出分形
    auto outGradX = executor->AllocTensor(gradHIn->GetDataType(), format, format);
    auto outGradPhi = executor->AllocTensor(outType, format, format);
    auto outGradAlpha = executor->AllocTensor(outType, format, format);
    auto outGradBias = executor->AllocTensor(outType, format, format);
    auto outGradGamma = executor->AllocTensor(outType, format, format);

    auto ret = INFER_SHAPE(
        MhcPreBackward,
        OP_INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, gradXPostOptional),
        OP_OUTPUT(outGradX, outGradPhi, outGradAlpha, outGradBias, outGradGamma), OP_ATTR(hcEps, opImplMode));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr),
                        "MhcPreBackward InferShape failed.");
    auto ret1 = ADD_TO_LAUNCHER_LIST_AICORE(
        MhcPreBackward,
        OP_INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, gradXPostOptional),
        OP_OUTPUT(outGradX, outGradPhi, outGradAlpha, outGradBias, outGradGamma), OP_ATTR(hcEps, opImplMode));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret1 != ACLNN_SUCCESS,
                                         return std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr),
                                         "MhcPreBackward ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return std::tuple(outGradX, outGradPhi, outGradAlpha, outGradBias, outGradGamma);
}
} // namespace l0op
