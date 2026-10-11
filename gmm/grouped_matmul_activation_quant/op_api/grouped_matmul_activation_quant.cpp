/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_activation_quant.h"

#include <string>

#include "grouped_matmul_activation_quant_common.h"
#include "gmm/common/op_host/log_format_util.h"
#include "opdev/make_op_executor.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "util/math_util.h"
#include "log/log.h"

using namespace op;
using Ops::Transformer::Gmm::FormatString;

namespace l0op {
OP_TYPE_REGISTER(GroupedMatmulActivationQuant);

const std::tuple<aclTensor*, aclTensor*> GroupedMatmulActivationQuant(
    const aclTensor* x, const aclTensor* groupList, const aclTensorList* weight, const aclTensorList* weightScale,
    const aclTensorList* bias, const aclTensor* xScale, const char* activationType, bool transposeWeight,
    int64_t groupListType, const aclIntArray* tuningConfig, const char* quantMode, int64_t yDtype,
    const char* roundMode, int64_t scaleAlg, float dstTypeMax, aclOpExecutor* executor)
{
    L0_DFX(GroupedMatmulActivationQuant, x, groupList, weight, weightScale, bias, xScale, activationType,
           transposeWeight, groupListType, tuningConfig, quantMode, yDtype, roundMode, scaleAlg, dstTypeMax);
    if (executor == nullptr || x == nullptr || weightScale == nullptr || weightScale->Size() == 0 ||
        (*weightScale)[gmaq::FIRST_TENSOR_INDEX] == nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "GroupedMatmulActivationQuant", "executor, x and weightScale",
            FormatString("executor=%s, x=%s, weightScale=%s", executor == nullptr ? "nullptr" : "not nullptr",
                         x == nullptr ? "nullptr" : "not nullptr",
                         weightScale == nullptr ? "nullptr" : "invalid tensorList"),
            "executor, x and weightScale can not be nullptr, and weightScale can not be empty");
        return std::tuple(nullptr, nullptr);
    }

    int64_t m = x->GetViewShape().GetDim(gmaq::X_M_DIM_INDEX);
    const auto& weightScaleShape = (*weightScale)[gmaq::FIRST_TENSOR_INDEX]->GetViewShape();
    if (weightScaleShape.GetDimNum() != gmaq::WEIGHT_SCALE_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("GroupedMatmulActivationQuant", "weightScale",
                                                 std::to_string(weightScaleShape.GetDimNum()),
                                                 "the shape dim of weightScale must be 4");
        return std::tuple(nullptr, nullptr);
    }
    int64_t n = transposeWeight ? weightScaleShape.GetDim(gmaq::WEIGHT_SCALE_TRANS_N_DIM_INDEX) :
                                  weightScaleShape.GetDim(gmaq::WEIGHT_SCALE_N_DIM_INDEX);
    gert::Shape outShape({m, n});
    int64_t scaleN = static_cast<int64_t>(Ops::Base::CeilDiv(n, gmaq::MX_GROUP_SIZE));
    gert::Shape scaleOutShape({m, scaleN, gmaq::MX_SCALE_PAIR});
    auto out = executor->AllocTensor(outShape, static_cast<ge::DataType>(yDtype), ge::FORMAT_ND);
    auto scaleOut = executor->AllocTensor(scaleOutShape, DataType::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    if (out == nullptr || scaleOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to allocate output tensors.");
        return std::tuple(nullptr, nullptr);
    }

    auto ret = INFER_SHAPE(GroupedMatmulActivationQuant, OP_INPUT(x, groupList, weight, weightScale, bias, xScale),
                           OP_OUTPUT(out, scaleOut),
                           OP_ATTR(activationType, transposeWeight, groupListType, tuningConfig, quantMode, yDtype,
                                   roundMode, scaleAlg, dstTypeMax));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return std::tuple(nullptr, nullptr);
    }

    ret =
        ADD_TO_LAUNCHER_LIST_AICORE(GroupedMatmulActivationQuant,
                                    OP_INPUT(x, groupList, weight, weightScale, bias, xScale), OP_OUTPUT(out, scaleOut),
                                    OP_ATTR(activationType, transposeWeight, groupListType, tuningConfig, quantMode,
                                            yDtype, roundMode, scaleAlg, dstTypeMax));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple(nullptr, nullptr);
    }

    return std::tie(out, scaleOut);
}
} // namespace l0op
