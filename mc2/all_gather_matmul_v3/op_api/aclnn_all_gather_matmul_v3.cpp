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
 * \file aclnn_all_gather_matmul_v3.cpp
 * \brief aclnn API implementation for AllGatherMatmulV3 (MX-quant FP8/FP4, apace UDMA path)
 */

#include "aclnnInner_all_gather_matmul_v3.h"

#include <cstring>
#include <initializer_list>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/op_api/mc2_aclnn_util.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "mc2_log_compat.h"
#include "graph/types.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/op_executor.h"

using namespace op;

namespace {

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

constexpr int64_t TWO_DIMS = 2;
constexpr int64_t SCALE_DIMS = 3;
constexpr int64_t MX_SCALE_BLOCK = 64;
constexpr int64_t SCALE_LAST_DIM = 2;
constexpr size_t HCCL_GROUP_NAME_LENGTH_MAX = 128U; // HCCL group 名长度上限

const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT4_E2M1};
const std::initializer_list<op::DataType> SCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E8M0};
const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

// 用户传 [N,K].t() (viewShape [K,N]) → 翻回 [N,K] 连续视图，保留原始物理存储
static const aclTensor *TransX2Tensor(const aclTensor *x2)
{
    uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDims(storageDimsNum);
    for (uint64_t i = 0; i < storageDimsNum; i++) {
        storageDims[i] = x2->GetStorageShape().GetDim(i);
    }

    uint64_t viewDimsNum = x2->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDims;
    viewDims.resize(viewDimsNum);
    for (uint64_t i = 0; i < viewDimsNum; i++) {
        viewDims[i] = x2->GetViewShape().GetDim(i);
    }
    // transpose the viewshape last two dimensions
    viewDims[0] = x2->GetViewShape().GetDim(1);
    viewDims[1] = x2->GetViewShape().GetDim(0);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    std::vector<int64_t> stride(viewDimsNum);
    auto transStride = x2->GetViewStrides();
    stride = std::vector<int64_t>(transStride.begin(), transStride.end());
    // transpose the two dimensions
    stride[0] = transStride[1];
    stride[1] = transStride[0];
    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDims.data(), viewDimsNum, dataType, stride.data(), offset, format, storageDims.data(),
                           storageDimsNum, x2->GetTensor()->GetAddr());
}

// CheckNullParams: 校验 REQUIRED 输入/输出非空（本次交付不使能 gather_out/amax_out，不做校验）
static bool CheckNullParams(const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *output,
                            uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (context == nullptr || x1 == nullptr || x2 == nullptr || output == nullptr || workspaceSize == nullptr ||
        executor == nullptr) {
        if (context == nullptr)
            OP_LOGE_WITH_INVALID_INPUT("aclnnAllGatherQuantMatmulV3", "context");
        if (x1 == nullptr)
            OP_LOGE_WITH_INVALID_INPUT("aclnnAllGatherQuantMatmulV3", "x1");
        if (x2 == nullptr)
            OP_LOGE_WITH_INVALID_INPUT("aclnnAllGatherQuantMatmulV3", "x2");
        if (output == nullptr)
            OP_LOGE_WITH_INVALID_INPUT("aclnnAllGatherQuantMatmulV3", "output");
        if (workspaceSize == nullptr)
            OP_LOGE_WITH_INVALID_INPUT("aclnnAllGatherQuantMatmulV3", "workspaceSize");
        if (executor == nullptr)
            OP_LOGE_WITH_INVALID_INPUT("aclnnAllGatherQuantMatmulV3", "executor");
        return false;
    }
    return true;
}

// CheckDtype: dtype 校验
aclnnStatus CheckDtype(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *x1Scale,
                       const aclTensor *x2Scale, const aclTensor *output)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, X_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, X_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    // FP4 要求 x1/x2 同时为 FLOAT4_E2M1（双向拦截，x1=e4m3/x2=fp4 或反之均报错）
    if (x1->GetDataType() == op::DataType::DT_FLOAT4_E2M1 || x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1) {
        OP_CHECK_DTYPE_NOT_SAME(x1, x2, return ACLNN_ERR_PARAM_INVALID);
    }
    // x1Scale/x2Scale 为 OPTIONAL，非空时校验 dtype
    if (x1Scale != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x1Scale, SCALE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    }
    if (x2Scale != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x2Scale, SCALE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    }
    // bias 不为空时校验 FLOAT
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, BIAS_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

// 通用维度数校验：tensor 维度数须等于 expectedDimNum（调用方保证 tensor 非空）
aclnnStatus CheckTensorDimNum(const aclTensor *tensor, const char *paramName, int64_t expectedDimNum)
{
    if (tensor->GetViewShape().GetDimNum() != expectedDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnAllGatherQuantMatmulV3", paramName,
            (std::to_string(tensor->GetViewShape().GetDimNum()) + "D").c_str(),
            ("The shape of " + std::string(paramName) + " must be " + std::to_string(expectedDimNum) + "D.").c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// 通用 format 校验：仅支持 ND；空指针放行（兼容 OPTIONAL 输入）
aclnnStatus CheckTensorFormatND(const aclTensor *tensor, const char *paramName)
{
    if (tensor != nullptr && tensor->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnAllGatherQuantMatmulV3", paramName,
                                                op::ToString(tensor->GetStorageFormat()).GetString(),
                                                "Only ND format is supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// CheckFormat: ND 校验（scale 兼容 NCL，同 v2 CheckScale）
aclnnStatus CheckFormat(const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                        const aclTensor *x1Scale, const aclTensor *x2Scale, const aclTensor *output)
{
    aclnnStatus ret = CheckTensorFormatND(context, "context");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorFormatND(x1, "x1");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorFormatND(x2, "x2");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorFormatND(output, "output");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorFormatND(bias, "bias");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (x1Scale != nullptr && x1Scale->GetStorageFormat() != op::Format::FORMAT_ND &&
        x1Scale->GetStorageFormat() != op::Format::FORMAT_NCL) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnAllGatherQuantMatmulV3", "x1Scale",
                                                op::ToString(x1Scale->GetStorageFormat()).GetString(),
                                                "Only ND/NCL format is supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2Scale != nullptr && x2Scale->GetStorageFormat() != op::Format::FORMAT_ND &&
        x2Scale->GetStorageFormat() != op::Format::FORMAT_NCL) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnAllGatherQuantMatmulV3", "x2Scale",
                                                op::ToString(x2Scale->GetStorageFormat()).GetString(),
                                                "Only ND/NCL format is supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// 校验 MX scale shape: [dim0, Ceil(K/64), 2]；scale 为 OPTIONAL，空指针直接放行
aclnnStatus CheckScaleShape(const aclTensor *scale, const char *paramName, int64_t dim0, int64_t scaleKDim)
{
    if (scale == nullptr) {
        return ACLNN_SUCCESS;
    }
    aclnnStatus ret = CheckTensorDimNum(scale, paramName, SCALE_DIMS);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (scale->GetViewShape().GetDim(0) != dim0 || scale->GetViewShape().GetDim(1) != scaleKDim ||
        scale->GetViewShape().GetDim(2) != SCALE_LAST_DIM) {
        OP_LOGE_FOR_INVALID_SHAPE("aclnnAllGatherQuantMatmulV3", paramName,
                                  op::ToString(scale->GetViewShape()).GetString(),
                                  ("[" + std::to_string(dim0) + ", " + std::to_string(scaleKDim) + ", " +
                                   std::to_string(SCALE_LAST_DIM) + "]")
                                      .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// CheckShape: scale dim(0) 校验（gather_out/amax_out 不使能，不校验）
aclnnStatus CheckShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale, const aclTensor *x2Scale,
                       const aclTensor *output)
{
    aclnnStatus ret = CheckTensorDimNum(x1, "x1", TWO_DIMS);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorDimNum(x2, "x2", TWO_DIMS);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorDimNum(output, "output", TWO_DIMS);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // x1: [M, K]; x2: [N, K]
    int64_t mVal = x1->GetViewShape().GetDim(0);
    int64_t kX1 = x1->GetViewShape().GetDim(1);
    int64_t nVal = x2->GetViewShape().GetDim(0);
    int64_t kX2 = x2->GetViewShape().GetDim(1);
    if (kX1 != kX2) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnAllGatherQuantMatmulV3", "x1, x2",
                                               (std::string(op::ToString(x1->GetViewShape()).GetString()) + ", " +
                                                op::ToString(x2->GetViewShape()).GetString())
                                                   .c_str(),
                                               "K mismatch, x2 must be [N, K] layout.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (output->GetViewShape().GetDim(1) != nVal) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnAllGatherQuantMatmulV3", "output",
                                              op::ToString(output->GetViewShape()).GetString(),
                                              "output.N must be equal to x2.N.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // scale: x1Scale [M, Ceil(K/64), 2]; x2Scale [N, Ceil(K/64), 2]
    int64_t scaleKDim = (kX1 + MX_SCALE_BLOCK - 1) / MX_SCALE_BLOCK;
    ret = CheckScaleShape(x1Scale, "x1Scale", mVal, scaleKDim);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckScaleShape(x2Scale, "x2Scale", nVal, scaleKDim);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

// CheckAttrs: 仅校验 REQUIRED 属性（group、hccl_buffer_size）；OPTIONAL 属性（comm_mode 等）不在此拦截
aclnnStatus CheckAttrs(const char *group, int64_t hcclBufferSize)
{
    if (group == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAllGatherQuantMatmulV3", "group");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    // group 名长度须在 [1, 128)：空串与超长（含未 \0 结尾）均拦截
    size_t groupLen = strnlen(group, HCCL_GROUP_NAME_LENGTH_MAX); // 长度 >= 128 时返回 128
    if (groupLen == 0 || groupLen >= HCCL_GROUP_NAME_LENGTH_MAX) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnAllGatherQuantMatmulV3", "group length", std::to_string(groupLen).c_str(),
                                  ("in range [1, " + std::to_string(HCCL_GROUP_NAME_LENGTH_MAX) + ")").c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hcclBufferSize <= 0) {
        OP_LOGE_WITH_INVALID_ATTR("aclnnAllGatherQuantMatmulV3", "hccl_buffer_size",
                                  std::to_string(hcclBufferSize).c_str(), "positive integer");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus aclnnAllGatherQuantMatmulV3GetWorkspaceSize(
    const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional, const aclTensor *x2ScaleOptional, const char *group, int64_t rankSize,
    int64_t hcclBufferSize, int64_t groupSize, const char *commMode, const aclTensor *output,
    const aclTensor *gatherOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnAllGatherQuantMatmulV3GetWorkspaceSize start");

    // 校验顺序：null -> dtype -> format -> attrs -> shape
    if (!CheckNullParams(context, x1, x2, output, workspaceSize, executor)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    aclnnStatus ret = CheckDtype(x1, x2, biasOptional, x1ScaleOptional, x2ScaleOptional, output);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckFormat(context, x1, x2, biasOptional, x1ScaleOptional, x2ScaleOptional, output);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckAttrs(group, hcclBufferSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    OP_LOGD("X1 is %s.", x1->ToString().GetString());
    OP_LOGD("X2 is %s.", x2->ToString().GetString());

    constexpr bool kIsTransA = false;
    constexpr bool kIsTransB = true;
    auto transX2 = x2;
    auto transX2Scale = x2ScaleOptional;
    if (Ops::Transformer::IsTransposeLastTwoDims(x2)) {
        OP_LOGI("x2 is a transposed [K,N] view, transpose back to [N,K].");
        transX2 = TransX2Tensor(x2);
        if (transX2 == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    if ((x2ScaleOptional != nullptr) && MC2Aclnn::IsNeedScaleTrans(x2ScaleOptional)) {
        transX2Scale = TransX2Tensor(x2ScaleOptional);
        if (transX2Scale == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
        OP_LOGI("x2Scale after trans: dim0=%ld dim1=%ld dim2=%ld", transX2Scale->GetViewShape().GetDim(0),
                transX2Scale->GetViewShape().GetDim(1), transX2Scale->GetViewShape().GetDim(2));
    }

    ret = CheckShape(x1, transX2, x1ScaleOptional, transX2Scale, output);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    uint64_t outDtype = static_cast<uint64_t>(output->GetDataType());
    OP_LOGD("outDtype value is: %lu.", outDtype);

    // gather_out 本次交付不使能（def 为 REQUIRED，torch 侧传 {0} 空占位 tensor），aclnn 原样透传 inner，
    // aclnn/tiling 不做处理；amax_out 为 OPTIONAL，inner 固定传 nullptr
    ret = aclnnInnerAllGatherMatmulV3GetWorkspaceSize(
        context, x1, transX2, biasOptional, x1ScaleOptional, transX2Scale, const_cast<char *>(group), hcclBufferSize,
        kIsTransA, kIsTransB, rankSize, groupSize, static_cast<int64_t>(outDtype), const_cast<char *>(commMode), output,
        gatherOut, nullptr, workspaceSize, executor);

    OP_LOGD("aclnnInnerAllGatherMatmulV3GetWorkspaceSize ret = %d.", ret);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnInnerAllGatherMatmulV3GetWorkspaceSize failed, ret=%d", ret);
    }

    OP_LOGD("aclnnAllGatherQuantMatmulV3GetWorkspaceSize end: workspaceSize=%lu", *workspaceSize);
    return ret;
}

extern "C" aclnnStatus aclnnAllGatherQuantMatmulV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                   aclrtStream stream)
{
    OP_LOGD("aclnnAllGatherQuantMatmulV3 start");

    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "aclnnAllGatherQuantMatmulV3: executor is null.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (NnopbaseSetHcclServerType != nullptr) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerAllGatherMatmulV3(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnInnerAllGatherMatmulV3 failed, ret=%d", ret);
        return ACLNN_ERR_INNER;
    }

    OP_LOGD("aclnnAllGatherQuantMatmulV3 end");
    return ACLNN_SUCCESS;
}
