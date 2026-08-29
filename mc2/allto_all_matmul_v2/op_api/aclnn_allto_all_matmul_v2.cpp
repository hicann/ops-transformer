/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_allto_all_matmul_v2.cpp
 * \brief aclnn API implementation for AlltoAllMatmulV2 (MX-quant FP8, apace UDMA path)
 */

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include <cstring>
#include <string>
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/op_executor.h"
#include "mc2_log_compat.h"

#include "aclnnInner_allto_all_matmul_v2.h"

using namespace op;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

static bool IsTransposeLastTwoDims(const aclTensor *tensor)
{
    if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
        return false;
    }
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

static aclTensor *TransX2Tensor(const aclTensor *x2)
{
    uint64_t storageShapeDimNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = x2->GetStorageShape().GetDim(i);
    }
    uint64_t viewShapeDimNum = x2->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim(viewShapeDimNum);
    viewDim[0] = x2->GetViewShape().GetDim(1);
    viewDim[1] = x2->GetViewShape().GetDim(0);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (aclGetDataType(x2, &dataType) != ACL_SUCCESS) {
        OP_LOGE_WITH_INVALID_INPUT("AlltoAllMatmulV2", "x2 dtype");
        return nullptr;
    }
    std::vector<int64_t> stride(viewShapeDimNum);
    auto transStride = x2->GetViewStrides();
    stride = std::vector<int64_t>(transStride.begin(), transStride.end());
    stride[0] = transStride[1];
    stride[1] = transStride[0];
    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;
    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, x2->GetTensor()->GetAddr());
}

static bool CheckNullParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1ScaleOptional,
                            const aclTensor *x2ScaleOptional, const aclTensor *output, uint64_t *workspaceSize,
                            aclOpExecutor **executor)
{
    if (x1 == nullptr || x2 == nullptr || x1ScaleOptional == nullptr || x2ScaleOptional == nullptr ||
        output == nullptr || workspaceSize == nullptr || executor == nullptr) {
        if (x1 == nullptr)
            OP_LOGE_WITH_INVALID_INPUT("AlltoAllMatmulV2", "x1");
        if (x2 == nullptr)
            OP_LOGE_WITH_INVALID_INPUT("AlltoAllMatmulV2", "x2");
        if (x1ScaleOptional == nullptr)
            OP_LOGE_WITH_INVALID_INPUT("AlltoAllMatmulV2", "x1Scale");
        if (x2ScaleOptional == nullptr)
            OP_LOGE_WITH_INVALID_INPUT("AlltoAllMatmulV2", "x2Scale");
        if (output == nullptr)
            OP_LOGE_WITH_INVALID_INPUT("AlltoAllMatmulV2", "output");
        if (workspaceSize == nullptr)
            OP_LOGE_WITH_INVALID_INPUT("AlltoAllMatmulV2", "workspaceSize");
        if (executor == nullptr)
            OP_LOGE_WITH_INVALID_INPUT("AlltoAllMatmulV2", "executor");
        return false;
    }
    return true;
}

extern "C" aclnnStatus AlltoAllMatmulV2GetWorkspaceSize(
    const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional, const aclTensor *x2ScaleOptional, const char *group, int64_t worldSize,
    int64_t hcclBufferSize, int64_t x1QuantMode, int64_t x2QuantMode, int64_t groupSize, const char *commMode,
    int64_t precisionMode, const aclTensor *output, const aclTensor *alltoAllOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_LOGD("AlltoAllMatmulV2GetWorkspaceSize start");

    if (!CheckNullParams(x1, x2, x1ScaleOptional, x2ScaleOptional, output, workspaceSize, executor)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    bool transposeX2 = true; // required by tiling
    auto transX2 = x2;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        bool notContiguous = IsTransposeLastTwoDims(x2);
        if (notContiguous) {
            transX2 = TransX2Tensor(x2);
            if (transX2 == nullptr) {
                OP_LOGE_WITH_INVALID_INPUT("AlltoAllMatmulV2", "transX2");
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
    }

    const char *effectiveCommMode = commMode != nullptr ? commMode : "urma";
    if (strcmp(effectiveCommMode, "urma") != 0) {
        OP_LOGE_WITH_INVALID_ATTR("AlltoAllMatmulV2", "commMode", effectiveCommMode, "'urma'");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // y_dtype 从 output 的 tensor dtype 推导（acl 枚举与 ge 枚举一致），
    // 未显式指定 BF16/FP16（默认 fp32）时由 tiling 侧校验拒绝。
    aclDataType outputAclDtype = ACL_DT_UNDEFINED;
    if (aclGetDataType(output, &outputAclDtype) != ACL_SUCCESS) {
        OP_LOGE_WITH_INVALID_INPUT("AlltoAllMatmulV2", "output dtype");
        return ACLNN_ERR_INNER;
    }
    int64_t yDtype = static_cast<int64_t>(outputAclDtype);

    constexpr int64_t kDefaultX1QuantDtype = 28; // ge::DT_UNDEFINED
    constexpr bool kDefaultTransposeX1 = false;
    aclnnStatus ret = aclnnInnerAlltoAllMatmulV2GetWorkspaceSize(
        context, x1, transX2, biasOptional, x1ScaleOptional, x2ScaleOptional, const_cast<char *>(group), worldSize,
        hcclBufferSize, yDtype, x1QuantMode, x2QuantMode, kDefaultX1QuantDtype, kDefaultTransposeX1, transposeX2,
        groupSize, const_cast<char *>(effectiveCommMode), precisionMode, output, alltoAllOutOptional, workspaceSize,
        executor);

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnInnerAlltoAllMatmulV2GetWorkspaceSize failed, ret=%d", ret);
    }

    OP_LOGD("AlltoAllMatmulV2GetWorkspaceSize end: workspaceSize=%lu", *workspaceSize);
    return ret;
}

extern "C" aclnnStatus AlltoAllMatmulV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    OP_LOGD("AlltoAllMatmulV2 start");

    if (NnopbaseSetHcclServerType != nullptr) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerAlltoAllMatmulV2(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnInnerAlltoAllMatmulV2 failed, ret=%d", ret);
        return ACLNN_ERR_INNER;
    }

    OP_LOGD("AlltoAllMatmulV2 end");
    return ACLNN_SUCCESS;
}
