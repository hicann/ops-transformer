/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_nsa_compress_with_cache.h"

#include <dlfcn.h>
#include <new>

#include "nsa_compress_with_cache.h"
#include "aclnn_kernels/contiguous.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
struct CompressWithCacheShapeInfo {
    int64_t headDim;
    int64_t headNum;
    int64_t pagePerBatch;
    int64_t compressBlockSize;
    int64_t compressStride;
    int64_t pageBlockSize;
};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_BF16};

static aclnnStatus CheckNsaCompressWithCacheParam(const aclTensor *input, const aclTensor *weight,
                                                  const aclTensor *slotMapping, const aclTensor *outputCacheRef,
                                                  const uint64_t *workspaceSize, aclOpExecutor *const *executor)
{
    // 必须的参数指针判空
    CHECK_RET(input != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(slotMapping != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(outputCacheRef != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus Contiguous(const aclTensor *&input, const aclTensor *&weight, const aclTensor *&blockTableOptional,
                              aclTensor *&outputCache, aclOpExecutor *executor)
{
    input = l0op::Contiguous(input, executor);
    CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);

    weight = l0op::Contiguous(weight, executor);
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);

    outputCache = const_cast<aclTensor *>(l0op::Contiguous(outputCache, executor));
    CHECK_RET(outputCache != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (blockTableOptional != nullptr) {
        blockTableOptional = l0op::Contiguous(blockTableOptional, executor);
        CHECK_RET(blockTableOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus InputDtypeCheck(const aclTensor *input, const aclTensor *weight, const aclTensor *slotMapping,
                                   const aclTensor *blockTableOptional, const aclTensor *outputCache)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(weight, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(outputCache, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);

    CHECK_COND(slotMapping->GetDataType() == DataType::DT_INT32, false,
               "slotMapping only support dtype int32, but got %s.",
               op::ToString(slotMapping->GetDataType()).GetString());

    if (blockTableOptional != nullptr) {
        CHECK_COND(blockTableOptional->GetDataType() == DataType::DT_INT32, false,
                   "blockTableOptional only support dtype int32, but got %s.",
                   op::ToString(blockTableOptional->GetDataType()).GetString());
    }

    auto inputDtype = input->GetDataType();
    auto weightDtype = weight->GetDataType();
    auto outputCacheDtype = outputCache->GetDataType();
    if (inputDtype != weightDtype || inputDtype != outputCacheDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of input[%s], weight[%s], outputCache[%s] are not equal.",
                op::ToString(DataType(inputDtype)).GetString(), op::ToString(DataType(weightDtype)).GetString(),
                op::ToString(DataType(outputCacheDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static bool IsSupportedFormat(ge::Format format)
{
    return format == ge::FORMAT_ND || format == ge::FORMAT_NCL || format == ge::FORMAT_NCHW ||
           format == ge::FORMAT_NCDHW;
}

static aclnnStatus CheckNDFormat(const aclTensor *input, const aclTensor *weight, const aclTensor *slotMapping,
                                 const aclTensor *blockTableOptional, const aclTensor *outputCache)
{
    auto inputFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(input->GetStorageFormat()));
    auto weightFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(weight->GetStorageFormat()));
    auto slotMappingFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(slotMapping->GetStorageFormat()));
    ge::Format blockTableOptionalFormat = ge::FORMAT_ND;
    if (blockTableOptional != nullptr) {
        blockTableOptionalFormat =
            static_cast<ge::Format>(ge::GetPrimaryFormat(blockTableOptional->GetStorageFormat()));
    }
    auto outputCacheFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(outputCache->GetStorageFormat()));
    if (!IsSupportedFormat(inputFormat) || !IsSupportedFormat(weightFormat) || !IsSupportedFormat(slotMappingFormat) ||
        !IsSupportedFormat(blockTableOptionalFormat) || !IsSupportedFormat(outputCacheFormat)) {
        return false;
    }
    return true;
}

static bool CheckIsEmptyTensor(const aclTensor *input, const aclTensor *weight, const aclTensor *outputCache)
{
    int64_t inputShapeSize = input->GetViewShape().GetShapeSize();
    int64_t weightShapeSize = weight->GetViewShape().GetShapeSize();
    int64_t outputCacheShapeSize = outputCache->GetViewShape().GetShapeSize();
    if (inputShapeSize == 0 || weightShapeSize == 0 || outputCacheShapeSize == 0) {
        return true;
    }
    return false;
}

aclnnStatus aclnnNsaCompressWithCacheGetWorkspaceSize(
    const aclTensor *input, const aclTensor *weight, const aclTensor *slotMapping, const aclIntArray *actSeqLenOptional,
    const aclTensor *blockTableOptional, char *layoutOptional, int64_t compressBlockSize, int64_t compressStride,
    int64_t actSeqLenType, int64_t pageBlockSize, aclTensor *outputCache, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    // L2接口阶段1
    L2_DFX_PHASE_1(aclnnNsaCompressWithCache,
                   DFX_IN(input, weight, slotMapping, actSeqLenOptional, blockTableOptional, layoutOptional,
                          compressBlockSize, compressStride, actSeqLenType, pageBlockSize, outputCache),
                   DFX_OUT(outputCache));
    // 检查入参
    CHECK_RET(CheckNsaCompressWithCacheParam(input, weight, slotMapping, outputCache, workspaceSize, executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    // 检查空tensor
    if (CheckIsEmptyTensor(input, weight, outputCache)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[NSACompressWithCache] do not support empty input/weight/outputCache.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 检查是否支持格式
    if (!CheckNDFormat(input, weight, slotMapping, blockTableOptional, outputCache)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "[NSACompressWithCache] All input tensors must be in ND, NCL, NCHW or NCDHW format");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(InputDtypeCheck(input, weight, slotMapping, blockTableOptional, outputCache) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    // 获取executor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    CHECK_RET(Contiguous(input, weight, blockTableOptional, outputCache, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    // 调用L0接口
    auto l0NsaCompressWithCacheOuts = l0op::NsaCompressWithCache(
        input, weight, slotMapping, outputCache, actSeqLenOptional, blockTableOptional, layoutOptional,
        compressBlockSize, compressStride, actSeqLenType, pageBlockSize, l0Executor);

    // 检查输出
    CHECK_RET(l0NsaCompressWithCacheOuts != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Viewout
    auto viewCopyResult = l0op::ViewCopy(l0NsaCompressWithCacheOuts, outputCache, l0Executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaCompressWithCache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                      aclrtStream stream)
{
    // 固定写法
    L2_DFX_PHASE_2(aclnnNsaCompressWithCache);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in NsaCompressWithCache launch aicore");
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
}
#endif
