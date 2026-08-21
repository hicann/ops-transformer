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
 * \file aclnn_moe_fused_topk.cpp
 * \brief
 */

#include "aclnn_moe_fused_topk.h"
#include "moe_fused_topk.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const int64_t X_NUM_TOKEN_DIM = 0;
static const int64_t X_EXPERT_NUM_DIM = 1;

static const int64_t MAPPING_ATBLE_EXPERT_NUM_DIM = 0;
static const int64_t MAPPING_ATBLE_MAX_MAPPING_NUM_DIM = 1;

static const int64_t Y_NUM_TOKEN_DIM = 0;
static const int64_t Y_TOPK_DIM = 1;

static const int64_t INDICES_NUM_TOKEN_DIM = 0;
static const int64_t INDICES_TOPK_DIM = 1;

static const int64_t X_DIM_NUM = 2;
static const int64_t ADD_NUM_DIM_NUM = 1;
static const int64_t MAPPING_NUM_DIM_NUM = 1;
static const int64_t MAPPING_ATBLE_DIM_NUM = 2;
static const int64_t Y_DIM_NUM = 2;

static const uint32_t GROUP_NUM_MAX = 256U;
static const uint32_t EXPERT_NUM_MAX = 1024U;
static const uint32_t MAX_MAPPING_NUM_MAX = 128U;

static const std::initializer_list<op::DataType> X_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                   DataType::DT_BF16};

static const std::initializer_list<op::DataType> INT_DTYPE_SUPPORT_LIST = {DataType::DT_INT32};

static const std::initializer_list<op::DataType> Y_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};

static inline bool CheckNotNull(const aclTensor *x, const aclTensor *addNum, const aclTensor *mappingNum,
                                const aclTensor *mappingTable, const aclTensor *y, const aclTensor *indices,
                                bool enableExpertMapping)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(addNum, return false);

    if (enableExpertMapping) {
        OP_CHECK_NULL(mappingNum, return false);
        OP_CHECK_NULL(mappingTable, return false);
    }

    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(indices, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *x, const aclTensor *addNum, const aclTensor *mappingNum,
                                   const aclTensor *mappingTable, const aclTensor *y, const aclTensor *indices,
                                   bool enableExpertMapping)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, X_SUPPORT_LIST, return false);
    OP_CHECK(x->GetDataType() == addNum->GetDataType(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x dtype %s and addNum dtype %s should be same.",
                     op::ToString(x->GetDataType()).GetString(), op::ToString(addNum->GetDataType()).GetString()),
             return false);

    if (enableExpertMapping) {
        OP_CHECK_DTYPE_NOT_SUPPORT(mappingNum, INT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mappingTable, INT_DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, INT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, Y_DTYPE_SUPPORT_LIST, return false);

    return true;
}

static inline bool CheckShape(const aclTensor *x, const aclTensor *addNum, const aclTensor *mappingNum,
                              const aclTensor *mappingTable, uint32_t groupNum, uint32_t groupTopk, uint32_t topN,
                              uint32_t topK, uint32_t activateType, bool isNorm, float scale, bool enableExpertMapping,
                              const aclTensor *y, const aclTensor *indices)
{
    if (x->IsEmpty()) {
        return true;
    }
    OP_CHECK(groupNum > 0 && groupTopk > 0 && topN > 0 && topK > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "all of groupNum: %u, groupTopk: %u, topN: %u and topK: %u must be greater than 0.", groupNum,
                     groupTopk, topN, topK),
             return false);
    OP_CHECK(groupNum <= GROUP_NUM_MAX,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupNum must be less than %u, which is %u.", GROUP_NUM_MAX, groupNum),
             return false);
    auto xShape = x->GetViewShape();
    auto addNumShape = addNum->GetViewShape();

    auto yShape = y->GetViewShape();
    auto indicesShape = indices->GetViewShape();
    OP_CHECK(
        xShape.GetDimNum() == X_DIM_NUM,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The Dims of x must be %ld, but which is %ld.", X_DIM_NUM, xShape.GetDimNum()),
        return false);
    OP_CHECK(addNumShape.GetDimNum() == ADD_NUM_DIM_NUM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The Dims of addNum must be %ld, but which is %ld.", ADD_NUM_DIM_NUM,
                     addNumShape.GetDimNum()),
             return false);
    OP_CHECK(
        yShape.GetDimNum() == Y_DIM_NUM,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The Dims of y must be %ld, but which is %ld.", Y_DIM_NUM, yShape.GetDimNum()),
        return false);
    OP_CHECK(yShape == indicesShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of y %s must be same as the shape of indices %s.",
                     op::ToString(yShape).GetString(), op::ToString(indicesShape).GetString()),
             return false);

    auto numToken = xShape.GetDim(X_NUM_TOKEN_DIM);
    auto expertNum = xShape.GetDim(X_EXPERT_NUM_DIM);

    OP_CHECK(yShape.GetDim(Y_NUM_TOKEN_DIM) == numToken,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size y[%ld]: %ld must equal to size of x[%ld]: %ld.",
                     Y_NUM_TOKEN_DIM, yShape.GetDim(Y_NUM_TOKEN_DIM), X_NUM_TOKEN_DIM, numToken),
             return false);
    OP_CHECK(yShape.GetDim(Y_TOPK_DIM) == topK,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size y[%ld]: %ld must equal to topK: %u.", Y_TOPK_DIM,
                     yShape.GetDim(Y_TOPK_DIM), topK),
             return false);
    OP_CHECK(addNumShape.GetDim(0) == expertNum,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size addNum[0]: %ld must equal to size of x[1]: %ld.",
                     addNumShape.GetDim(0), expertNum),
             return false);

    OP_CHECK(expertNum % groupNum == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The expertNum: %ld must be an integer multiple of groupNum: %u.",
                     expertNum, groupNum),
             return false);
    OP_CHECK(groupTopk <= groupNum,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupTopk: %u must be less than groupNum: %u.", groupTopk, groupNum),
             return false);
    OP_CHECK(
        topK <= expertNum,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TopK must be less than expertNum: %ld, but which is %u.", expertNum, topK),
        return false);
    OP_CHECK(topN <= expertNum / groupNum,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TopN must be less than expertNum / groupNum: %ld, but which is %u.",
                     expertNum / groupNum, topN),
             return false);
    OP_CHECK(expertNum <= EXPERT_NUM_MAX,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ExpertNum must be less than %u, but which is %ld.", EXPERT_NUM_MAX,
                     expertNum),
             return false);

    if (enableExpertMapping) {
        auto mappingNumShape = mappingNum->GetViewShape();
        auto mappingTableShape = mappingTable->GetViewShape();
        OP_CHECK(mappingNumShape.GetDimNum() == MAPPING_NUM_DIM_NUM,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The Dims of mappingNum must be %ld, but which is %ld.",
                         MAPPING_NUM_DIM_NUM, mappingNumShape.GetDimNum()),
                 return false);
        OP_CHECK(mappingTableShape.GetDimNum() == MAPPING_ATBLE_DIM_NUM,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The Dims of mappingTable must be %ld, but which is %ld.",
                         MAPPING_ATBLE_DIM_NUM, mappingTableShape.GetDimNum()),
                 return false);
        auto maxMappingNum = mappingTableShape.GetDim(MAPPING_ATBLE_MAX_MAPPING_NUM_DIM);
        OP_CHECK(
            maxMappingNum <= MAX_MAPPING_NUM_MAX,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of mappingTable[%ld] must be less than %u, but which is %ld.",
                    MAPPING_ATBLE_MAX_MAPPING_NUM_DIM, MAX_MAPPING_NUM_MAX, maxMappingNum),
            return false);
        OP_CHECK(mappingNumShape.GetDim(0) == expertNum,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size mappingNum[0]: %ld must equal to size of x[1]: %ld.",
                         mappingNumShape.GetDim(0), expertNum),
                 return false);
        OP_CHECK(mappingTableShape.GetDim(0) == expertNum,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size mappingTable[0]: %ld must equal to size of x[1]: %ld.",
                         mappingTableShape.GetDim(0), expertNum),
                 return false);
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *addNum, const aclTensor *mappingNum,
                               const aclTensor *mappingTable, uint32_t groupNum, uint32_t groupTopk, uint32_t topN,
                               uint32_t topK, uint32_t activateType, bool isNorm, float scale, bool enableExpertMapping,
                               const aclTensor *y, const aclTensor *indices)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, addNum, mappingNum, mappingTable, y, indices, enableExpertMapping),
              ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x, addNum, mappingNum, mappingTable, y, indices, enableExpertMapping),
              ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和out的shape是否一致
    CHECK_RET(CheckShape(x, addNum, mappingNum, mappingTable, groupNum, groupTopk, topN, topK, activateType, isNorm,
                         scale, enableExpertMapping, y, indices),
              ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeFusedTopkGetWorkspaceSize(const aclTensor *x, const aclTensor *addNum, const aclTensor *mappingNum,
                                              const aclTensor *mappingTable, uint32_t groupNum, uint32_t groupTopk,
                                              uint32_t topN, uint32_t topK, uint32_t activateType, bool isNorm,
                                              float scale, bool enableExpertMapping, aclTensor *y, aclTensor *indices,
                                              uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeFusedTopk,
                   DFX_IN(x, addNum, mappingNum, mappingTable, groupNum, groupTopk, topN, topK, activateType, isNorm,
                          scale, enableExpertMapping),
                   DFX_OUT(y, indices));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, addNum, mappingNum, mappingTable, groupNum, groupTopk, topN, topK, activateType, isNorm,
                           scale, enableExpertMapping, y, indices);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 空Tensor处理
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto addNumContiguous = l0op::Contiguous(addNum, uniqueExecutor.get());
    CHECK_RET(addNumContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *mappingNumContiguous = mappingNum;
    const aclTensor *mappingTableContiguous = mappingTable;
    if (enableExpertMapping) {
        mappingNumContiguous = l0op::Contiguous(mappingNum, uniqueExecutor.get());
        CHECK_RET(mappingNumContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        mappingTableContiguous = l0op::Contiguous(mappingTable, uniqueExecutor.get());
        CHECK_RET(mappingTableContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto MoeFusedTopkRes = l0op::MoeFusedTopk(xContiguous, addNumContiguous, mappingNumContiguous,
                                              mappingTableContiguous, groupNum, groupTopk, topN, topK, activateType,
                                              isNorm, scale, enableExpertMapping, uniqueExecutor.get());
    CHECK_RET(MoeFusedTopkRes[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MoeFusedTopkRes[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyY = l0op::ViewCopy(MoeFusedTopkRes[0], y, uniqueExecutor.get());
    CHECK_RET(viewCopyY != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyIndices = l0op::ViewCopy(MoeFusedTopkRes[1], indices, uniqueExecutor.get());
    CHECK_RET(viewCopyIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeFusedTopk(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeFusedTopk);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
