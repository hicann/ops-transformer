// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "aclnn_dual_level_grouped_matmul_weight_nz.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "log/log.h"

#include "dual_level_grouped_matmul.h"
#include "dual_level_grouped_matmul_checker.h"
using namespace op;

namespace {
constexpr char API_NAME[] = "aclnnDualLevelGroupedMatmulWeightNz";
}

extern "C" aclnnStatus aclnnDualLevelGroupedMatmulWeightNzGetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight, const aclTensorList* biasOptional,
    const aclTensorList* scaleOptional, const aclTensorList* perTokenScaleOptional, const aclTensor* groupListOptional,
    const aclTensorList* xLevel0Scale, const aclTensorList* weightLevel0Scale, int64_t level0GroupSize,
    int64_t level1GroupSize, int64_t splitItem, int64_t groupType, int64_t groupListType,
    aclIntArray* tuningConfigOptional, aclTensorList* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDualLevelGroupedMatmulWeightNz,
                   DFX_IN(x, weight, biasOptional, scaleOptional, perTokenScaleOptional, groupListOptional,
                          xLevel0Scale, weightLevel0Scale, level0GroupSize, level1GroupSize, splitItem, groupType,
                          groupListType, tuningConfigOptional),
                   DFX_OUT(out));
    if (workspaceSize == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "workspaceSize", "must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "executor", "must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    bool transposeWeight = false;
    const dual_level_gmm::DualLevelGroupedMatmulParams params{x,
                                                              weight,
                                                              biasOptional,
                                                              scaleOptional,
                                                              perTokenScaleOptional,
                                                              groupListOptional,
                                                              xLevel0Scale,
                                                              weightLevel0Scale,
                                                              level0GroupSize,
                                                              level1GroupSize,
                                                              splitItem,
                                                              groupType,
                                                              groupListType,
                                                              tuningConfigOptional,
                                                              out};
    const auto checkRet = dual_level_gmm::CheckParams(params, transposeWeight);
    if (checkRet != ACLNN_SUCCESS) {
        return checkRet;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if ((*out)[0]->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    const auto ret =
        l0op::DualLevelGroupedMatmul(x, weight, biasOptional, scaleOptional, perTokenScaleOptional, groupListOptional,
                                     xLevel0Scale, weightLevel0Scale, transposeWeight, groupListType,
                                     tuningConfigOptional, level0GroupSize, level1GroupSize, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor.get()->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnDualLevelGroupedMatmulWeightNz(void* workspace, uint64_t workspaceSize,
                                                           aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDualLevelGroupedMatmulWeightNz);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
