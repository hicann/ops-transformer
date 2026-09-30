/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <unordered_map>
#include "../../common/op_api/gmm_tensor_storage_check.h"
#include "gmm_dsq_base.h"
#include "grouped_matmul_swiglu_quant_v2_utils.h"
#include "grouped_matmul_swiglu_quant_api_check_utils.h"
#include "grouped_matmul_swiglu_quant_v2.h"
#include "aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "aclnn_grouped_matmul_swiglu_quant_v2.h"

using namespace op;
using namespace gmm_dsq;
using namespace gmm_dsq_base;

namespace {
constexpr char GMM_SWIGLU_QUANT_V2_OP_NAME[] = "grouped_matmul_swiglu_quant_v2";
constexpr char ACLNN_GMM_SWIGLU_QUANT_V2_API_NAME[] = "aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize";
constexpr char ACLNN_GMM_SWIGLU_QUANT_WEIGHT_NZ_V2_API_NAME[] =
    "aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize";
} // namespace

class GmmDsqHandlerFactory {
private:
    std::unordered_map<NpuArch, std::unique_ptr<GroupedMatmulSwigluQuantHandler>> handlers_;

public:
    void registerHandler(NpuArch npuArch, std::unique_ptr<GroupedMatmulSwigluQuantHandler> handler)
    {
        handlers_[npuArch] = std::move(handler);
    }

    GroupedMatmulSwigluQuantHandler* getHandler(NpuArch npuArch)
    {
        auto it = handlers_.find(npuArch);
        return it != handlers_.end() ? it->second.get() : nullptr;
    }
};

static aclnnStatus aclnnGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(const char* apiName,
                                                                       GroupedMatmulSwigluQuantParamsBase& params,
                                                                       uint64_t* workspaceSize,
                                                                       aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    GmmDsqHandlerFactory factory;
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    factory.registerHandler(NpuArch::DAV_2201, std::make_unique<gmm_dsq_base::GroupedMatmulSwigluQuantBaseHandler>());
    factory.registerHandler(NpuArch::DAV_3510,
                            std::make_unique<gmmSwigluQuantV2::GroupedMatmulSwigluQuantBaseHandler>());

    if (auto* handler = factory.getHandler(npuArch)) {
        handler->Initialize(apiName, params, workspaceSize, executor);
        return handler->Process();
    } else {
        std::ostringstream reason;
        reason << "SoC version " << static_cast<int32_t>(npuArch) << " is not supported";
        std::string reasonStr = reason.str();
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In op [%s], %s.", GMM_SWIGLU_QUANT_V2_OP_NAME, reasonStr.c_str());
    }

    return ACLNN_ERR_PARAM_INVALID;
}

static aclnnStatus CheckMxfp4WeightNzViewShape(const char* apiName, const aclTensor* weight, const op::Shape& viewShape,
                                               size_t expectedViewDimNum)
{
    if (weight->GetDataType() != DataType::DT_FLOAT4_E2M1 && weight->GetDataType() != DataType::DT_FLOAT4_E1M2) {
        return ACLNN_SUCCESS;
    }
    if (unlikely(!(viewShape.GetDimNum() == expectedViewDimNum))) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            apiName, "weight viewShape", std::to_string(viewShape.GetDimNum()),
            "when the dtype of weight is DT_FLOAT4, the dim num of weight viewShape is invalid");
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t lastSecondDim = viewShape.GetDim(viewShape.GetDimNum() - 2);
    int64_t lastDim = viewShape.GetDim(viewShape.GetDimNum() - 1);
    if (unlikely(!(lastSecondDim != 1 && lastDim != 1))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "weight viewShape", op::ToString(viewShape).GetString(),
            "when the dtype of weight is DT_FLOAT4, the last two dimensions of weight viewShape can not be 1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale,
    const aclTensorList* weightAssistMatrix, const aclTensor* bias, const aclTensor* xScale,
    const aclTensor* smoothScale, const aclTensor* groupList, int64_t dequantMode, int64_t dequantDtype,
    int64_t quantMode, int64_t groupListType, const aclIntArray* tuningConfigOptional, aclTensor* output,
    aclTensor* outputScale, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    auto status = gmm_swiglu_quant_api_check::CheckRequiredInputs(ACLNN_GMM_SWIGLU_QUANT_V2_API_NAME, x, weight,
                                                                  weightScale, xScale, groupList, output, outputScale);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    CHECK_RET(gmm::CheckTensorStorageBounds(x, "GroupedMatmulSwigluQuant", "x"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorListStorageBounds(weight, "GroupedMatmulSwigluQuant", "weight"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorListStorageBounds(weightScale, "GroupedMatmulSwigluQuant", "weightScale"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorListStorageBounds(weightAssistMatrix, "GroupedMatmulSwigluQuant", "weightAssistMatrix"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(bias, "GroupedMatmulSwigluQuant", "bias"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(xScale, "GroupedMatmulSwigluQuant", "xScale"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(smoothScale, "GroupedMatmulSwigluQuant", "smoothScale"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(groupList, "GroupedMatmulSwigluQuant", "groupList"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(output, "GroupedMatmulSwigluQuant", "output"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(outputScale, "GroupedMatmulSwigluQuant", "outputScale"),
              ACLNN_ERR_PARAM_INVALID);
    L2_DFX_PHASE_1(aclnnGroupedMatmulSwigluQuantV2,
                   DFX_IN(x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList, dequantMode,
                          dequantDtype, quantMode, groupListType, tuningConfigOptional),
                   DFX_OUT(output, outputScale));
    GroupedMatmulSwigluQuantParamsBase params =
        GroupedMatmulSwigluQuantParamsBuilder::Create(x, weight, weightScale, output, outputScale)
            .SetXScale(xScale)
            .SetSmoothScale(smoothScale)
            .SetGroupList(groupList)
            .SetGroupListType(groupListType)
            .SetWeightAssistMatrix(weightAssistMatrix)
            .SetDequantAttr(dequantMode, dequantDtype)
            .SetQuantAttr(quantMode, static_cast<int64_t>(output->GetDataType()))
            .SetTransposeAttr(false)
            .SetBias(bias)
            .SetScenario()
            .SetTuningConfig(tuningConfigOptional)
            .Build();

    // 调用公共接口
    return aclnnGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(ACLNN_GMM_SWIGLU_QUANT_V2_API_NAME, params,
                                                               workspaceSize, executor);
}

static aclnnStatus ProcessSingleWeightNz(const aclTensorList* weight)
{
    auto w = (*weight)[0];
    auto storgeShape = w->GetStorageShape();
    auto viewShape = w->GetViewShape();
    aclTensor* weightNZ = const_cast<aclTensor*>(w);
    std::ostringstream gotShape;
    gotShape << op::ToString(storgeShape).GetString() << " with dim num " << storgeShape.GetDimNum();
    std::string gotShapeStr = gotShape.str();
    CHECK_COND((storgeShape.GetDimNum() == WEIGHT_NZ_DIM_LIMIT), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], the shape of [%s] is not supported, got [%s]. Constraint:[%s]", GMM_SWIGLU_QUANT_V2_OP_NAME,
               "weight", gotShapeStr.c_str(), "storage shape dim num must be 5 when weight NZ v2");
    CHECK_RET(CheckMxfp4WeightNzViewShape(ACLNN_GMM_SWIGLU_QUANT_WEIGHT_NZ_V2_API_NAME, w, viewShape,
                                          WEIGHT_ND_DIM_LIMIT) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    weightNZ->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    if (viewShape.GetDimNum() == WEIGHT_NZ_DIM_LIMIT) {
        weightNZ->SetViewFormat(op::Format::FORMAT_FRACTAL_NZ);
    } else if (viewShape.GetDimNum() == WEIGHT_ND_DIM_LIMIT) {
        weightNZ->SetViewFormat(op::Format::FORMAT_ND);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessMultiWeightNz(const aclTensorList* weight)
{
    size_t wLength = weight->Size();
    for (size_t i = 0; i < wLength; i++) {
        auto w = (*weight)[i];
        CHECK_RET(w != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        auto storgeShape = w->GetStorageShape();
        auto viewShape = w->GetViewShape();
        aclTensor* weightNZ = const_cast<aclTensor*>(w);
        std::ostringstream gotShape;
        gotShape << op::ToString(storgeShape).GetString() << " with dim num " << storgeShape.GetDimNum();
        std::string gotShapeStr = gotShape.str();
        CHECK_COND((storgeShape.GetDimNum() == MULTI_WEIGHT_NZ_DIM_LIMIT), ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], the shape of [%s] is not supported, got [%s]. Constraint:[%s]",
                   GMM_SWIGLU_QUANT_V2_OP_NAME, "weight", gotShapeStr.c_str(),
                   "storage shape dim num must be 4 when multi-weight NZ v2");
        CHECK_RET(CheckMxfp4WeightNzViewShape(ACLNN_GMM_SWIGLU_QUANT_WEIGHT_NZ_V2_API_NAME, w, viewShape,
                                              MULTI_WEIGHT_ND_DIM_LIMIT) == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
        // weight的StorageFormat无条件视为NZ
        weightNZ->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
        if (viewShape.GetDimNum() == MULTI_WEIGHT_NZ_DIM_LIMIT) {
            weightNZ->SetViewFormat(op::Format::FORMAT_FRACTAL_NZ);
        } else if (viewShape.GetDimNum() == MULTI_WEIGHT_ND_DIM_LIMIT) {
            weightNZ->SetViewFormat(op::Format::FORMAT_ND);
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale,
    const aclTensorList* weightAssistMatrix, const aclTensor* bias, const aclTensor* xScale,
    const aclTensor* smoothScale, const aclTensor* groupList, int64_t dequantMode, int64_t dequantDtype,
    int64_t quantMode, int64_t groupListType, const aclIntArray* tuningConfigOptional, aclTensor* output,
    aclTensor* outputScale, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    auto status = gmm_swiglu_quant_api_check::CheckRequiredInputs(
        ACLNN_GMM_SWIGLU_QUANT_WEIGHT_NZ_V2_API_NAME, x, weight, weightScale, xScale, groupList, output, outputScale);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    CHECK_RET(gmm::CheckTensorStorageBounds(x, "GroupedMatmulSwigluQuant", "x"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorListStorageBounds(weight, "GroupedMatmulSwigluQuant", "weight"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorListStorageBounds(weightScale, "GroupedMatmulSwigluQuant", "weightScale"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorListStorageBounds(weightAssistMatrix, "GroupedMatmulSwigluQuant", "weightAssistMatrix"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(bias, "GroupedMatmulSwigluQuant", "bias"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(xScale, "GroupedMatmulSwigluQuant", "xScale"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(smoothScale, "GroupedMatmulSwigluQuant", "smoothScale"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(groupList, "GroupedMatmulSwigluQuant", "groupList"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(output, "GroupedMatmulSwigluQuant", "output"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(outputScale, "GroupedMatmulSwigluQuant", "outputScale"),
              ACLNN_ERR_PARAM_INVALID);
    L2_DFX_PHASE_1(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                   DFX_IN(x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList, dequantMode,
                          dequantDtype, quantMode, groupListType, tuningConfigOptional),
                   DFX_OUT(output, outputScale));
    size_t wLength = weight->Size();
    auto firstWeightViewDimNum = (*weight)[0]->GetViewShape().GetDimNum();
    bool isSingleWeightTensor =
        wLength == 1 && (firstWeightViewDimNum == WEIGHT_ND_DIM_LIMIT || firstWeightViewDimNum == WEIGHT_NZ_DIM_LIMIT);
    if (isSingleWeightTensor) {
        CHECK_RET(ProcessSingleWeightNz(weight) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(ProcessMultiWeightNz(weight) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }

    GroupedMatmulSwigluQuantParamsBase params =
        GroupedMatmulSwigluQuantParamsBuilder::Create(x, weight, weightScale, output, outputScale)
            .SetXScale(xScale)
            .SetSmoothScale(smoothScale)
            .SetGroupList(groupList)
            .SetGroupListType(groupListType)
            .SetWeightAssistMatrix(weightAssistMatrix)
            .SetDequantAttr(dequantMode, dequantDtype)
            .SetQuantAttr(quantMode, static_cast<int64_t>(output->GetDataType()))
            .SetTransposeAttr(false)
            .SetBias(bias)
            .SetScenario()
            .SetTuningConfig(tuningConfigOptional)
            .Build();

    // 调用公共接口
    return aclnnGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(ACLNN_GMM_SWIGLU_QUANT_WEIGHT_NZ_V2_API_NAME, params,
                                                               workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulSwigluQuantV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulSwigluQuantV2);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in GroupedMatmulSwigluQuantV2 launch aicore");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulSwigluQuantWeightNzV2);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in GroupedMatmulSwigluQuantWeightNzV2 launch aicore");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
