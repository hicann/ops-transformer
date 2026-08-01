/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "util/math_util.h"
#include "grouped_matmul_swiglu_quant_utils.h"
#include "grouped_matmul_swiglu_quant_v2.h"

using namespace op;
using namespace gmm_dsq;

namespace l0op {
OP_TYPE_REGISTER(GroupedMatmulSwigluQuantV2);

constexpr int64_t SWIGLU_SPLIT_SIZE = 64L;
constexpr int64_t QUANT_MODE_PERTOKEN = 0L;
constexpr size_t MX_MULTI_WEIGHT_DIM = 2UL;
constexpr size_t MX_MULTI_WEIGHT_SCALE_N_DIM = 1UL;
constexpr size_t MX_WEIGHT_SCALE_N_DIM = 2UL;

static bool IsMxWeightNzMultiTensor(const aclTensorList *weight)
{
    return weight != nullptr && weight->Size() > 0 && (*weight)[0] != nullptr &&
           op::IsPrivateFormat((*weight)[0]->GetStorageFormat()) &&
           (*weight)[0]->GetViewShape().GetDimNum() == MX_MULTI_WEIGHT_DIM;
}

const std::tuple<aclTensor *, aclTensor *> GroupedMatmulSwigluQuantV2(const aclTensor *x, const aclTensorList *weight,
                         const aclTensorList *weightScale,
                         const aclTensor *xScale, const aclTensorList *weightAssistanceMatrix,
                         const aclTensor *bias, const aclTensor *smoothScale,
                         const aclTensor *groupList, int64_t dequantMode, int64_t dequantDtype,
                         int64_t quantMode, int64_t quantDtype, bool transposeWeight, int64_t groupListType,
                         const aclIntArray *tuningConfigOptional, aclOpExecutor *executor)
{
    L0_DFX(GroupedMatmulSwigluQuantV2, x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale,
           groupList, dequantMode, dequantDtype, quantMode, quantDtype, transposeWeight, tuningConfigOptional);
    if (x == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x is nullptr.");
        return std::tuple(nullptr, nullptr);
    }
    int64_t m = xScale->GetViewShape().GetDim(0);
    auto weightScaleShape = (*weightScale)[0]->GetViewShape();
    int64_t weightScaleDimNum = static_cast<int64_t>(weightScaleShape.GetDimNum());
    int64_t n = weightScaleShape.GetDim(weightScaleDimNum - 1);
    int64_t nAfterHalve = static_cast<int64_t>(n / 2);
    gert::Shape outShape({m, nAfterHalve});
    gert::Shape scaleOutShape({m});
    auto out = executor->AllocTensor(outShape, DataType::DT_INT8, ge::FORMAT_ND);
    auto scaleOut = executor->AllocTensor(scaleOutShape, DataType::DT_FLOAT, ge::FORMAT_ND);
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        bool isMxWeightNzMultiTensor = IsMxWeightNzMultiTensor(weight);
        if (quantMode == QUANT_MODE_PERTOKEN) {
            // A8W4/A4W4 weightScale is [E, N] for per-channel and [E, KGroup, N] for per-group.
            // WeightNZ may set transposeWeight, but weightScale remains ND and its last dimension is always N.
            n = weightScaleShape.GetDim(weightScaleDimNum - 1);
        } else {
            n = isMxWeightNzMultiTensor ?
                    (transposeWeight ? weightScaleShape.GetDim(0) :
                                       weightScaleShape.GetDim(MX_MULTI_WEIGHT_SCALE_N_DIM)) :
                    (transposeWeight ? weightScaleShape.GetDim(MX_MULTI_WEIGHT_SCALE_N_DIM) :
                                       weightScaleShape.GetDim(MX_WEIGHT_SCALE_N_DIM));
        }
        nAfterHalve = static_cast<int64_t>(n / 2); // outShape需要为[M, N / 2]
        gert::Shape outShapeV2({m, nAfterHalve});
        gert::Shape scaleOutShapeV2;
        // 当quantMode等于2时，out_scale 的形状为三维
        if (quantMode == 2) {
            int64_t nAfterSplit = static_cast<int64_t>(Ops::Base::CeilDiv(nAfterHalve, SWIGLU_SPLIT_SIZE));
            scaleOutShapeV2 = gert::Shape({m, nAfterSplit, 2});
        } else {
            scaleOutShapeV2 = gert::Shape({m});
        }
        out = executor->AllocTensor(outShapeV2, static_cast<ge::DataType>(quantDtype), ge::FORMAT_ND);
        // 当quantMode等于2时，outScale的DataType为FLOAT8_E8M0
        scaleOut = quantMode == 2 ? executor->AllocTensor(scaleOutShapeV2, DataType::DT_FLOAT8_E8M0, ge::FORMAT_ND) :
                                    executor->AllocTensor(scaleOutShapeV2, DataType::DT_FLOAT, ge::FORMAT_ND);
    }
    auto ret = INFER_SHAPE(GroupedMatmulSwigluQuantV2,
                    OP_INPUT(x, xScale, groupList, weight, weightScale, weightAssistanceMatrix, bias, smoothScale),
                    OP_OUTPUT(out, scaleOut), OP_ATTR(dequantMode, dequantDtype, quantMode, quantDtype, transposeWeight,
                    groupListType, tuningConfigOptional));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return std::tuple(nullptr, nullptr);
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        GroupedMatmulSwigluQuantV2,
        OP_INPUT(x, xScale, groupList, weight, weightScale, weightAssistanceMatrix, bias, smoothScale),
        OP_OUTPUT(out, scaleOut), OP_ATTR(dequantMode, dequantDtype, quantMode, quantDtype, transposeWeight,
                        groupListType, tuningConfigOptional));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple(nullptr, nullptr);
    }

    return std::tie(out, scaleOut);
}

} // namespace l0op
