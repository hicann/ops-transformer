/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mhc_pre.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MhcPre);

const std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *>
MhcPre(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
       const aclTensor *gammaOptional, int64_t outFlag, double normEps, double hcEps, int64_t opImplMode,
       aclOpExecutor *executor)
{
    L0_DFX(MhcPre, x, phi, alpha, bias, gammaOptional, outFlag, normEps, hcEps, opImplMode);

    DataType outType = DataType::DT_FLOAT;
    Format format = Format::FORMAT_ND;
    auto outHin = executor->AllocTensor(x->GetDataType(), format, format);
    auto outHpost = executor->AllocTensor(outType, format, format);
    auto outHres = executor->AllocTensor(outType, format, format);

    auto outInvRms = executor->AllocTensor(outType, format, format);
    auto outMmRes = executor->AllocTensor(outType, format, format);
    auto outHpre = executor->AllocTensor(outType, format, format);

    auto ret = INFER_SHAPE(MhcPre, OP_INPUT(x, phi, alpha, bias, gammaOptional),
                           OP_OUTPUT(outHin, outHpost, outHres, outInvRms, outMmRes, outHpre),
                           OP_ATTR(outFlag, static_cast<float>(normEps), static_cast<float>(hcEps), opImplMode));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
                        "MhcPre InferShape failed.");
    auto ret1 = ADD_TO_LAUNCHER_LIST_AICORE(MhcPre, OP_INPUT(x, phi, alpha, bias, gammaOptional),
                                            OP_OUTPUT(outHin, outHpost, outHres, outInvRms, outMmRes, outHpre),
                                            OP_ATTR(outFlag, static_cast<float>(normEps), static_cast<float>(hcEps),
                                                    opImplMode));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret1 != ACLNN_SUCCESS,
                                         return std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
                                         "MhcPre ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return std::tuple(outHin, outHpost, outHres, outInvRms, outMmRes, outHpre);
}
} // namespace l0op
