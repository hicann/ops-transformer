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
 * \file aclnn_moe_init_routing_v2.cpp
 * \brief
 */
#include <string>
#include <tuple>
#include "aclnn_moe_init_routing_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "aclnn_util.h"
#include "moe_init_routing_v2.h"

using namespace op;

namespace MoeInitRoutingV2Check {

static const std::initializer_list<op::DataType> MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_X = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_X_310P = {DataType::DT_FLOAT16,
                                                                                                  DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_EXPERT_IDX = {
    DataType::DT_INT32};
static const std::initializer_list<op::DataType> MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_EXPERT_IDX_REGBASE = {
    DataType::DT_INT32, DataType::DT_INT64};
static const std::initializer_list<op::DataType> MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX = {DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor* x, const aclTensor* expertIdx, const aclTensor* expandedXOut,
                                const aclTensor* expandedRowIdxOut,
                                const aclTensor* expertTokensCountOrCumsumOutOptional,
                                const aclTensor* expertTokensBeforeCapacityOutOptional)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expertIdx, return false);
    OP_CHECK_NULL(expandedXOut, return false);
    OP_CHECK_NULL(expandedRowIdxOut, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* x, const aclTensor* expertIdx, const aclTensor* expandedXOut,
                                   const aclTensor* expandedRowIdxOut,
                                   const aclTensor* expertTokensCountOrCumsumOutOptional,
                                   const aclTensor* expertTokensBeforeCapacityOutOptional)
{
    if (x != nullptr && x->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (expertIdx != nullptr && expertIdx->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertIdx, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_EXPERT_IDX, return false);
    }
    if (expandedXOut != nullptr && expandedXOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedXOut, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(x, expandedXOut, return false);
    }
    if (expandedRowIdxOut != nullptr && expandedRowIdxOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedRowIdxOut, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (expertTokensCountOrCumsumOutOptional != nullptr &&
        expertTokensCountOrCumsumOutOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertTokensCountOrCumsumOutOptional, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX,
                                   return false);
    }
    if (expertTokensBeforeCapacityOutOptional != nullptr &&
        expertTokensBeforeCapacityOutOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertTokensBeforeCapacityOutOptional,
                                   MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    return true;
}

static inline bool CheckDtypeValid310P(const aclTensor* x, const aclTensor* expertIdx, const aclTensor* expandedXOut,
                                       const aclTensor* expandedRowIdxOut,
                                       const aclTensor* expertTokensCountOrCumsumOutOptional,
                                       const aclTensor* expertTokensBeforeCapacityOutOptional)
{
    if (x != nullptr && x->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_X_310P, return false);
    }
    if (expertIdx != nullptr && expertIdx->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertIdx, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_EXPERT_IDX, return false);
    }
    if (expandedXOut != nullptr && expandedXOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedXOut, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_X_310P, return false);
        OP_CHECK_DTYPE_NOT_SAME(x, expandedXOut, return false);
    }
    if (expandedRowIdxOut != nullptr && expandedRowIdxOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedRowIdxOut, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (expertTokensCountOrCumsumOutOptional != nullptr &&
        expertTokensCountOrCumsumOutOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertTokensCountOrCumsumOutOptional, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX,
                                   return false);
    }
    if (expertTokensBeforeCapacityOutOptional != nullptr &&
        expertTokensBeforeCapacityOutOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertTokensBeforeCapacityOutOptional,
                                   MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    return true;
}

static inline bool CheckDtypeValidRegbase(const aclTensor* x, const aclTensor* expertIdx, const aclTensor* expandedXOut,
                                          const aclTensor* expandedRowIdxOut,
                                          const aclTensor* expertTokensCountOrCumsumOutOptional,
                                          const aclTensor* expertTokensBeforeCapacityOutOptional)
{
    if (x != nullptr && x->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (expertIdx != nullptr && expertIdx->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertIdx, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_EXPERT_IDX_REGBASE, return false);
    }
    if (expandedXOut != nullptr && expandedXOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedXOut, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(x, expandedXOut, return false);
    }
    if (expandedRowIdxOut != nullptr && expandedRowIdxOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedRowIdxOut, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (expertTokensCountOrCumsumOutOptional != nullptr &&
        expertTokensCountOrCumsumOutOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertTokensCountOrCumsumOutOptional, MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX,
                                   return false);
    }
    if (expertTokensBeforeCapacityOutOptional != nullptr &&
        expertTokensBeforeCapacityOutOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertTokensBeforeCapacityOutOptional,
                                   MOE_INIT_ROUTING_V2_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* expertIdx, const aclTensor* expandedXOut,
                               const aclTensor* expandedRowIdxOut,
                               const aclTensor* expertTokensCountOrCumsumOutOptional,
                               const aclTensor* expertTokensBeforeCapacityOutOptional)
{
    CHECK_RET(CheckNotNull(x, expertIdx, expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOutOptional,
                           expertTokensBeforeCapacityOutOptional),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, expertIdx, expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOutOptional,
                              expertTokensBeforeCapacityOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams310P(const aclTensor* x, const aclTensor* expertIdx, const aclTensor* expandedXOut,
                                   const aclTensor* expandedRowIdxOut,
                                   const aclTensor* expertTokensCountOrCumsumOutOptional,
                                   const aclTensor* expertTokensBeforeCapacityOutOptional)
{
    CHECK_RET(CheckNotNull(x, expertIdx, expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOutOptional,
                           expertTokensBeforeCapacityOutOptional),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid310P(x, expertIdx, expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOutOptional,
                                  expertTokensBeforeCapacityOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsRegbase(const aclTensor* x, const aclTensor* expertIdx, const aclTensor* expandedXOut,
                                      const aclTensor* expandedRowIdxOut,
                                      const aclTensor* expertTokensCountOrCumsumOutOptional,
                                      const aclTensor* expertTokensBeforeCapacityOutOptional)
{
    CHECK_RET(CheckNotNull(x, expertIdx, expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOutOptional,
                           expertTokensBeforeCapacityOutOptional),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValidRegbase(x, expertIdx, expandedXOut, expandedRowIdxOut,
                                     expertTokensCountOrCumsumOutOptional, expertTokensBeforeCapacityOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeInitRoutingV2Check

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnMoeInitRoutingV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* expertIdx, int64_t activeNumOptional, int64_t expertCapacityOptional,
    int64_t expertNumOptional, int64_t dropPadModeOptional, int64_t expertTokensCountOrCumsumFlagOptional,
    bool expertTokensBeforeCapacityFlagOptional, const aclTensor* expandedXOut, const aclTensor* expandedRowIdxOut,
    const aclTensor* expertTokensCountOrCumsumOutOptional, const aclTensor* expertTokensBeforeCapacityOutOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(
        aclnnMoeInitRoutingV2,
        DFX_IN(x, expertIdx, activeNumOptional, expertCapacityOptional, expertNumOptional, dropPadModeOptional,
               expertTokensCountOrCumsumFlagOptional, expertTokensBeforeCapacityFlagOptional),
        DFX_OUT(expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOutOptional,
                expertTokensBeforeCapacityOutOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    bool is310P = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
    aclnnStatus ret;
    if (Ops::Transformer::AclnnUtil::IsRegbase()) {
        ret = MoeInitRoutingV2Check::CheckParamsRegbase(x, expertIdx, expandedXOut, expandedRowIdxOut,
                                                        expertTokensCountOrCumsumOutOptional,
                                                        expertTokensBeforeCapacityOutOptional);
    } else if (is310P) {
        ret = MoeInitRoutingV2Check::CheckParams310P(x, expertIdx, expandedXOut, expandedRowIdxOut,
                                                     expertTokensCountOrCumsumOutOptional,
                                                     expertTokensBeforeCapacityOutOptional);
    } else {
        ret = MoeInitRoutingV2Check::CheckParams(x, expertIdx, expandedXOut, expandedRowIdxOut,
                                                 expertTokensCountOrCumsumOutOptional,
                                                 expertTokensBeforeCapacityOutOptional);
    }
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，将输入转换成连续的tensor
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto expertIdxContiguous = l0op::Contiguous(expertIdx, uniqueExecutor.get());
    CHECK_RET(expertIdxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0接口进行计算，传入输出参数
    auto result = l0op::MoeInitRoutingV2(xContiguous, expertIdxContiguous, activeNumOptional, expertCapacityOptional,
                                         expertNumOptional, dropPadModeOptional, expertTokensCountOrCumsumFlagOptional,
                                         expertTokensBeforeCapacityFlagOptional, expandedXOut, expandedRowIdxOut,
                                         expertTokensCountOrCumsumOutOptional, expertTokensBeforeCapacityOutOptional,
                                         uniqueExecutor.get());
    auto [expandedXOut_, expandedRowIdxOut_, expertTokensCountOrCumsumOut_, expertTokensBeforeCapacityOut_] = result;
    bool hasNullptr = (expandedXOut_ == nullptr) || (expandedRowIdxOut_ == nullptr);
    CHECK_RET(hasNullptr != true, ACLNN_ERR_INNER_NULLPTR);

    // copyout结果，如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyExpandedXOutResult = l0op::ViewCopy(expandedXOut_, expandedXOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpandedXOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyExpandedRowIdxOutResult = l0op::ViewCopy(expandedRowIdxOut_, expandedRowIdxOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpandedRowIdxOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 处理可选输出
    if (expertTokensCountOrCumsumOutOptional != nullptr && expertTokensCountOrCumsumOut_ != nullptr) {
        auto viewCopyExpertTokensCountOrCumsumOutResult =
            l0op::ViewCopy(expertTokensCountOrCumsumOut_, expertTokensCountOrCumsumOutOptional, uniqueExecutor.get());
        CHECK_RET(viewCopyExpertTokensCountOrCumsumOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (expertTokensBeforeCapacityOutOptional != nullptr && expertTokensBeforeCapacityOut_ != nullptr) {
        auto viewCopyExpertTokensBeforeCapacityOutResult =
            l0op::ViewCopy(expertTokensBeforeCapacityOut_, expertTokensBeforeCapacityOutOptional, uniqueExecutor.get());
        CHECK_RET(viewCopyExpertTokensBeforeCapacityOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus aclnnMoeInitRoutingV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeInitRoutingV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
