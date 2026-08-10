/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <tuple>
#include <cstddef>
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/cast.h"
#include "opdev/common_types.h"
#include "moe_init_routing_v4.h"
#include "aclnn_moe_init_routing_v4.h"
#include "aclnn_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_X_REGBASE = {
    DataType::DT_FLOAT16,  DataType::DT_BF16,        DataType::DT_FLOAT,         DataType::DT_INT8,
    DataType::DT_HIFLOAT8, DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_EXPERT_IDX = {DataType::DT_INT32};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_SCALE_REGBASE = {DataType::DT_FLOAT,
    DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_OFFSET = {DataType::DT_FLOAT};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_ACTIVE_NUM = {DataType::DT_INT64};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_TOPK_WEIGHT = {DataType::DT_FLOAT};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_EXPANDED_X_OUT_REGBASE = {
    DataType::DT_FLOAT16,       DataType::DT_BF16,        DataType::DT_FLOAT,
    DataType::DT_INT8,          DataType::DT_HIFLOAT8,    DataType::DT_FLOAT8_E5M2,
    DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT4_E2M1, DataType::DT_INT4};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_EXPANDED_ROW_IDX_OUT = {DataType::DT_INT32};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_EXPERT_TOKENS_COUNT_OR_CUMSUMOUT = {DataType::DT_INT64};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_EXPANDED_SCALE_OUT_REGBASE = {
    DataType::DT_FLOAT, DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_EXPANDED_TOPK_WEIGHT_OUT = {DataType::DT_FLOAT};

static inline bool CheckNotNull(const aclTensor *x, const aclTensor *expertIdx, const aclTensor *expandedXOut,
                                const aclTensor *expandedRowIdxOut, const aclTensor *expertTokensCountOrCumsumOut,
                                const aclTensor *expandedScaleOut)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expertIdx, return false);
    OP_CHECK_NULL(expandedXOut, return false);
    OP_CHECK_NULL(expandedRowIdxOut, return false);
    OP_CHECK_NULL(expertTokensCountOrCumsumOut, return false);
    OP_CHECK_NULL(expandedScaleOut, return false);

    return true;
}

static inline bool CheckDtypeValidRegbase(const aclTensor *x, const aclTensor *expertIdx,
                                          const aclTensor *scaleOptional, const aclTensor *offsetOptional,
                                          const aclTensor *activeNumOptional, const aclTensor *topkWeightOptional,
                                          const aclTensor *expandedXOut, const aclTensor *expandedRowIdxOut,
                                          const aclTensor *expertTokensCountOrCumsumOut,
                                          const aclTensor *expandedScaleOut,
                                          const aclTensor *expandedTopkWeightOut)
{
    if (x != nullptr && x->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST_X_REGBASE, return false);
    }
    if (expertIdx != nullptr && expertIdx->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertIdx, DTYPE_SUPPORT_LIST_EXPERT_IDX, return false);
    }
    if (scaleOptional != nullptr && scaleOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scaleOptional, DTYPE_SUPPORT_LIST_SCALE_REGBASE, return false);
    }
    if (offsetOptional != nullptr && offsetOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(offsetOptional, DTYPE_SUPPORT_LIST_OFFSET, return false);
    }
    if (activeNumOptional != nullptr && activeNumOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(activeNumOptional, DTYPE_SUPPORT_LIST_ACTIVE_NUM, return false);
    }
    if (topkWeightOptional != nullptr && topkWeightOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(topkWeightOptional, DTYPE_SUPPORT_LIST_TOPK_WEIGHT, return false);
    }
    if (expandedXOut != nullptr && expandedXOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedXOut, DTYPE_SUPPORT_LIST_EXPANDED_X_OUT_REGBASE, return false);
    }
    if (expandedRowIdxOut != nullptr && expandedRowIdxOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedRowIdxOut, DTYPE_SUPPORT_LIST_EXPANDED_ROW_IDX_OUT, return false);
    }
    if (expertTokensCountOrCumsumOut != nullptr && expertTokensCountOrCumsumOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertTokensCountOrCumsumOut, DTYPE_SUPPORT_LIST_EXPERT_TOKENS_COUNT_OR_CUMSUMOUT,
                                   return false);
    }
    if (expandedScaleOut != nullptr && expandedScaleOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedScaleOut, DTYPE_SUPPORT_LIST_EXPANDED_SCALE_OUT_REGBASE, return false);
    }
    if (expandedTopkWeightOut != nullptr && expandedTopkWeightOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedTopkWeightOut, DTYPE_SUPPORT_LIST_EXPANDED_TOPK_WEIGHT_OUT, return false);
    }
    return true;
}

ACLNN_API aclnnStatus aclnnMoeInitRoutingV4GetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIdx,
    const aclTensor *scaleOptional, const aclTensor *offsetOptional,
    const aclTensor *activeNumOptional, const aclTensor *topkWeightOptional,
    int64_t expertCapacity, int64_t expertNum, int64_t dropPadMode,
    int64_t expertTokensNumType, bool expertTokensNumFlag,
    int64_t quantMode, const aclIntArray *activeExpertRangeOptional,
    int64_t rowIdxType, const aclTensor *expandedXOut,
    const aclTensor *expandedRowIdxOut,
    const aclTensor *expertTokensCountOrCumsumOut,
    const aclTensor *expandedScaleOut,
    const aclTensor *expandedTopkWeightOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMoeInitRoutingV4,
                   DFX_IN(x, expertIdx, scaleOptional, offsetOptional, activeNumOptional,
                          topkWeightOptional, expertCapacity, expertNum, dropPadMode,
                          expertTokensNumType, expertTokensNumFlag, quantMode,
                          activeExpertRangeOptional, rowIdxType),
                   DFX_OUT(expandedXOut, expandedRowIdxOut,
                           expertTokensCountOrCumsumOut, expandedScaleOut,
                           expandedTopkWeightOut));

    if ((topkWeightOptional != nullptr && expandedTopkWeightOut == nullptr) ||
        (topkWeightOptional == nullptr && expandedTopkWeightOut != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "topk_weight and expanded_topk_weight must be both present or both absent.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto ret =
        CheckNotNull(x, expertIdx, expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOut, expandedScaleOut);

    CHECK_RET(ret, ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(CheckDtypeValidRegbase(x, expertIdx, scaleOptional, offsetOptional,
                                     activeNumOptional, topkWeightOptional,
                                     expandedXOut, expandedRowIdxOut,
                                     expertTokensCountOrCumsumOut, expandedScaleOut, expandedTopkWeightOut),
              ACLNN_ERR_PARAM_INVALID);

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto expertIdxContiguous = l0op::Contiguous(expertIdx, uniqueExecutor.get());
    CHECK_RET(expertIdxContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const aclTensor *scaleContiguous = nullptr;
    const aclTensor *offsetContiguous = nullptr;
    if (scaleOptional != nullptr) {
        scaleContiguous = l0op::Contiguous(scaleOptional, uniqueExecutor.get());
        CHECK_RET(scaleContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }

    if (offsetOptional != nullptr) {
        offsetContiguous = l0op::Contiguous(offsetOptional, uniqueExecutor.get());
        CHECK_RET(offsetContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }

    const aclTensor *activeNumContiguous = nullptr;
    if (activeNumOptional != nullptr) {
        activeNumContiguous = l0op::Contiguous(activeNumOptional, uniqueExecutor.get());
        CHECK_RET(activeNumContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }

    const aclTensor *topkWeightContiguous = nullptr;
    if (topkWeightOptional != nullptr) {
        topkWeightContiguous = l0op::Contiguous(topkWeightOptional, uniqueExecutor.get());
        CHECK_RET(topkWeightContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }

    auto routingResult =
        std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(
            nullptr, nullptr, nullptr, nullptr, nullptr);
    routingResult = l0op::MoeInitRoutingV4(
        xContiguous, expertIdxContiguous, scaleContiguous, offsetContiguous,
        activeNumContiguous, topkWeightContiguous,
        expertCapacity, expertNum, dropPadMode,
        expertTokensNumType, expertTokensNumFlag,
        quantMode, activeExpertRangeOptional, rowIdxType, expandedXOut,
        expandedRowIdxOut, expertTokensCountOrCumsumOut, expandedScaleOut,
        expandedTopkWeightOut, uniqueExecutor.get());
    auto [expandedXOut_, expandedRowIdxOut_, expertTokensCountOrCumsumOut_,
          expandedScaleOut_, expandedTopkWeightOut_] = routingResult;
    bool hasNullptr = (expandedXOut_ == nullptr) || (expandedRowIdxOut_ == nullptr) ||
        (expertTokensCountOrCumsumOut_ == nullptr) || (expandedScaleOut_ == nullptr);

    CHECK_RET(hasNullptr != true, ACLNN_ERR_INNER_NULLPTR);

    if (topkWeightOptional != nullptr) {
        CHECK_RET(expandedTopkWeightOut_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewCopyExpandedXOutResult = l0op::ViewCopy(expandedXOut_, expandedXOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpandedXOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyExpandedRowIdxOutResult = l0op::ViewCopy(expandedRowIdxOut_, expandedRowIdxOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpandedRowIdxOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyExpertTokensCountOrCumsumOutResult =
        l0op::ViewCopy(expertTokensCountOrCumsumOut_, expertTokensCountOrCumsumOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpertTokensCountOrCumsumOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyExpandedScaleOutResult = l0op::ViewCopy(expandedScaleOut_, expandedScaleOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpandedScaleOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (expandedTopkWeightOut != nullptr && expandedTopkWeightOut_ != nullptr) {
        auto viewCopyExpandedTopkWeightOutResult =
            l0op::ViewCopy(expandedTopkWeightOut_, expandedTopkWeightOut, uniqueExecutor.get());
        CHECK_RET(viewCopyExpandedTopkWeightOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
ACLNN_API aclnnStatus aclnnMoeInitRoutingV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeInitRoutingV4);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
