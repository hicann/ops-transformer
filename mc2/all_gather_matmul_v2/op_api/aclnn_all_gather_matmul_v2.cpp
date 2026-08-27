/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

#include "aclnn_all_gather_matmul_v2.h"
#include "securec.h"
#include "acl/acl.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "mc2_log_compat.h"
#include "opdev/platform.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "common/utils/hccl_util.h"
#include "common/op_api/mc2_aclnn_util.h"
#include "aclnnInner_all_gather_matmul_v2.h"
#include "mc2_comm_utils.h"

using namespace op;
using namespace Ops::Transformer;

#ifdef __cplusplus
extern "C" {
#endif
static constexpr int64_t NUM_ACL_STOP_ON_FAILURE = 1;
static constexpr int64_t ONE_DIMS = 1;
static constexpr int64_t SCALAR = 1;
static constexpr int64_t TWO_DIMS = 2;
static constexpr int64_t KVALUE_MIN = 256;
static constexpr int64_t KVALUE_MAX = 65535;
static constexpr int64_t COMM_MODE_RANKSIZE = 8;
static constexpr int64_t CMP_MAX_LEN = 7;

typedef struct {
    uint32_t id;
    const char *funcName;
    bool hasReg;
} NnopbaseDfxId;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
extern "C" void *NnopbaseGetUserHandle(void *executor);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);

// check nullptr
static bool CheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *output)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(output, return false);
    return true;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16,     op::DataType::DT_BF16,     op::DataType::DT_FLOAT8_E4M3FN,
    op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT4_E2M1};

static const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> FP8_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_HIFLOAT8};

static const std::initializer_list<op::DataType> FP4_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT4_E2M1};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = BIAS_DTYPE_SUPPORT_LIST;

static const std::initializer_list<op::DataType> AIV_MODE_INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT8, op::DataType::DT_INT4};

static const std::initializer_list<op::DataType> AIV_MODE_OUTPUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                                       op::DataType::DT_BF16};

static bool CheckSupportDtype(const aclTensor *x1, const std::initializer_list<op::DataType> &supportTypes)
{
    return std::find(supportTypes.begin(), supportTypes.end(), x1->GetDataType()) != supportTypes.end();
}

static bool CheckOutDtypeValid(int64_t outDtype, const std::initializer_list<op::DataType> &validTypes)
{
    OP_LOGD("outDtype value is: %ld", outDtype);
    return std::find(validTypes.begin(), validTypes.end(), static_cast<op::DataType>(outDtype)) != validTypes.end();
}

static bool CheckDataTypeFp16Valid(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                   const aclTensor *output)
{
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);
    if (bias != nullptr) {
        // 当x1类型为fp16/bf16时, bias类型为与x1保持一致
        OP_CHECK_DTYPE_NOT_SAME(bias, x1, return false);
    }
    return true;
}

static bool CheckDataTypeFp8Fp4Valid(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias)
{
    if (x1->GetDataType() == op::DataType::DT_HIFLOAT8) {
        OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    }
    if (x1->GetDataType() == op::DataType::DT_FLOAT4_E2M1) {
        OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    }
    if (bias != nullptr) {
        // 当x1类型为fp16/bf16时, bias类型为与x1保持一致
        if (bias->GetDataType() != op::DataType::DT_FLOAT) {
            std::string dtypeStr = std::to_string(static_cast<uint32_t>(bias->GetDataType()));
            OP_LOGE_FOR_INVALID_DTYPE("aclnnAllGatherMatmulV2", "bias", dtypeStr.c_str(), "FLOAT");
            return false;
        }
    }
    return true;
}

static bool IsFp16orBf16Input(const aclTensor *x1)
{
    return (x1->GetDataType() == op::DataType::DT_FLOAT16) || (x1->GetDataType() == op::DataType::DT_BF16);
}

static bool CheckDtypeValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *output)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUT_DTYPE_SUPPORT_LIST, return false);
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, BIAS_DTYPE_SUPPORT_LIST, return false);
    }

    if (IsFp16orBf16Input(x1)) {
        return CheckDataTypeFp16Valid(x1, x2, bias, output);
    } else if (CheckSupportDtype(x1, FP8_DTYPE_SUPPORT_LIST) || CheckSupportDtype(x1, FP4_DTYPE_SUPPORT_LIST)) {
        return CheckDataTypeFp8Fp4Valid(x1, x2, bias);
    }
    return true;
}

static bool CheckAIVModeDtypeValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *output)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, AIV_MODE_INPUT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, AIV_MODE_INPUT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, AIV_MODE_OUTPUT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckAttr(int64_t streamMode)
{
    if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
        OP_LOGE_WITH_INVALID_ATTR("aclnnAllGatherMatmulV2", "streamMode", std::to_string(streamMode).c_str(), "1");
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor *x1, const aclTensor *x2, const aclTensor *output, const aclTensor *bias,
                        aclTensor *gatherOut, aclTensor *amaxOut)
{
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnAllGatherMatmulV2", "x1",
                                                op::ToString(x1->GetStorageFormat()).GetString(),
                                                "Only ND format is supported");
        return false;
    }
    if (x2->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnAllGatherMatmulV2", "x2",
                                                op::ToString(x2->GetStorageFormat()).GetString(),
                                                "Only ND format is supported");
        return false;
    }
    if (output->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnAllGatherMatmulV2", "output",
                                                op::ToString(output->GetStorageFormat()).GetString(),
                                                "Only ND format is supported");
        return false;
    }

    if (bias != nullptr && bias->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnAllGatherMatmulV2", "bias",
                                                op::ToString(bias->GetStorageFormat()).GetString(),
                                                "Only ND format is supported");
        return false;
    }
    if (gatherOut != nullptr && gatherOut->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnAllGatherMatmulV2", "gatherOut",
                                                op::ToString(gatherOut->GetStorageFormat()).GetString(),
                                                "Only ND format is supported");
        return false;
    }
    if (amaxOut != nullptr && amaxOut->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnAllGatherMatmulV2", "amaxOut",
                                                op::ToString(amaxOut->GetStorageFormat()).GetString(),
                                                "Only ND format is supported");
        return false;
    }
    return true;
}

static bool IsGatherOut(const aclTensor *gatherOut)
{
    OP_CHECK_NULL(gatherOut, return false);
    if (gatherOut->IsEmpty()) {
        OP_LOGD("AllGatherMatmulV2, get gather out is false.");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *output, const aclTensor *gatherOut,
                       bool isTransA)
{
    if (x1->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnAllGatherMatmulV2", "x1",
                                                 (std::to_string(x1->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x1 must be 2D.");
        return false;
    }
    if (x2->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnAllGatherMatmulV2", "x2",
                                                 (std::to_string(x2->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x2 must be 2D.");
        return false;
    }
    // A矩阵不能转置
    if (isTransA) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnAllGatherMatmulV2", "x1", "transposed", "not transposed");
        return false;
    }

    const auto kValX1 = x1->GetViewShape().GetDim(1);
    const auto kValX2 = x2->GetViewShape().GetDim(0);
    if (kValX1 != kValX2) {
        std::string shapeMsg = std::to_string(kValX1) + " and " + std::to_string(kValX2);
        std::string reasonMsg = "The k-axis of x1 and x2 must be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnAllGatherMatmulV2", "x1 and x2", shapeMsg.c_str(),
                                               reasonMsg.c_str());
        return false;
    }

    if ((kValX1 < KVALUE_MIN) || (kValX1 >= KVALUE_MAX)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnAllGatherMatmulV2", "x1", std::to_string(kValX1).c_str(),
                                              "The value of x1 k-axis must be in the range [256, 65535)");
        return false;
    }

    const auto nVal1 = x2->GetViewShape().GetDim(1);
    const auto nVal2 = output->GetViewShape().GetDim(1);
    if (nVal1 != nVal2) {
        std::string nAxisMsg = std::to_string(nVal1) + " and " + std::to_string(nVal2);
        std::string nReasonMsg = "The n-axis of x2 and output must be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnAllGatherMatmulV2", "x2 and output", nAxisMsg.c_str(),
                                               nReasonMsg.c_str());
        return false;
    }
    return true;
}

// 分别对输入类型为fp8/hif8和fp16/bf16数据类型进行分别校验
static aclnnStatus CheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, int64_t streamMode,
                               const aclTensor *output, aclTensor *gatherOut, aclTensor *amaxOut)
{
    CHECK_RET(CheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValid(x1, x2, bias, output), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckAttr(streamMode), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckFormat(x1, x2, output, bias, gatherOut, amaxOut), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsForAIVMode(const aclTensor *x1, const aclTensor *x2, const aclTensor *output,
                                         int64_t streamMode)
{
    CHECK_RET(CheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckAIVModeDtypeValid(x1, x2, output), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckAttr(streamMode), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShapeForAIVMode(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                        const aclTensor *output, const aclTensor *gatherOut, bool isTransA,
                                        bool isViewTransB)
{
    if (x1->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnAllGatherMatmulV2", "x1",
                                                 (std::to_string(x1->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x1 must be 2D.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnAllGatherMatmulV2", "x2",
                                                 (std::to_string(x2->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x2 must be 2D.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // A矩阵不能转置
    if (isTransA) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnAllGatherMatmulV2", "x1", "transposed", "not transposed");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const auto kValX1 = x1->GetViewShape().GetDim(1);
    const auto kValX2 = x2->GetViewShape().GetDim(isViewTransB ? 1 : 0);
    if (kValX1 != kValX2) {
        std::string shapeMsg = std::to_string(kValX1) + " and " + std::to_string(kValX2);
        std::string reasonMsg = "The k-axis of x1 and x2 must be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnAllGatherMatmulV2", "x1 and x2", shapeMsg.c_str(),
                                               reasonMsg.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (IsGatherOut(gatherOut)) {
        const auto kVal = gatherOut->GetViewShape().GetDim(1);
        if (kValX1 != kVal) {
            std::string kMsg = std::to_string(kValX1) + " and " + std::to_string(kVal);
            std::string kReason = "The k-axis of x1 and gatherOut must be the same";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnAllGatherMatmulV2", "x1 and gatherOut", kMsg.c_str(),
                                                   kReason.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    const auto nVal1 = x2->GetViewShape().GetDim(isViewTransB ? 0 : 1);
    const auto nVal2 = output->GetViewShape().GetDim(1);
    if (nVal1 != nVal2) {
        std::string nMsg = std::to_string(nVal1) + " and " + std::to_string(nVal2);
        std::string nReason = "The n-axis of x2 and output must be the same";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnAllGatherMatmulV2", "x2 and output", nMsg.c_str(),
                                               nReason.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static inline bool CheckParamDtypeFP8Vaild(const aclTensor *tensor)
{
    return tensor->GetDataType() == op::DataType::DT_FLOAT;
}

static aclnnStatus CheckScale(const aclTensor *x1Scale, const aclTensor *x2Scale, const aclTensor *quantScale)
{
    if (x1Scale == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAllGatherMatmulV2", "x1Scale");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2Scale == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAllGatherMatmulV2", "x2Scale");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x1Scale->GetStorageFormat() != op::Format::FORMAT_ND && x1Scale->GetStorageFormat() != op::Format::FORMAT_NCL) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnAllGatherMatmulV2", "x1Scale",
                                                op::ToString(x1Scale->GetStorageFormat()).GetString(),
                                                "Only ND/NCL format is supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2Scale->GetStorageFormat() != op::Format::FORMAT_ND && x2Scale->GetStorageFormat() != op::Format::FORMAT_NCL) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnAllGatherMatmulV2", "x2Scale",
                                                op::ToString(x2Scale->GetStorageFormat()).GetString(),
                                                "Only ND/NCL format is supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 如果scaleInV1 和 scaleInV2都不为空指针则为scalar类型数据
    auto x1ScaleLen = x1Scale->GetViewShape().GetDim(0);
    auto x2ScaleLen = x2Scale->GetViewShape().GetDim(0);
    OP_LOGD("AllGatherMatmulV2, x1ScaleLen is %ld.", x1ScaleLen);
    OP_LOGD("AllGatherMatmulV2, x2ScaleLen is %ld.", x2ScaleLen);

    // scale不为空指针为则scalar类型
    if (quantScale != nullptr) {
        if (quantScale->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnAllGatherMatmulV2", "quantScale",
                                                    op::ToString(quantScale->GetStorageFormat()).GetString(),
                                                    "Only ND format is supported");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (quantScale->GetViewShape().GetDimNum() != ONE_DIMS) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                "aclnnAllGatherMatmulV2", "quantScale",
                (std::to_string(quantScale->GetViewShape().GetDimNum()) + "D").c_str(),
                "The shape of quantScale must be 1D.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        auto scaleLen = quantScale->GetViewShape().GetDim(0);
        OP_LOGD("AllGatherMatmulV2, scaleLen is %ld.", scaleLen);
        if (scaleLen != SCALAR) {
            OP_LOGE_FOR_INVALID_SHAPESIZE("aclnnAllGatherMatmulV2", "quantScale", std::to_string(scaleLen).c_str(),
                                          "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        CHECK_RET(CheckParamDtypeFP8Vaild(quantScale), ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static bool IsAMaxOut(const aclTensor *amaxOut)
{
    if (amaxOut == nullptr) {
        return false;
    }
    if (amaxOut->IsEmpty()) {
        OP_LOGD("AllGatherMatmulV2, get amax out is false.");
        return false;
    }
    return true;
}

// fp16/bf16场景不支持amaxOut输入，fp8/hif8场景amaxOut为scalar且只支持float类型
static bool CheckAMaxOutVaild(const aclTensor *x1, const aclTensor *amaxOut)
{
    if (IsFp16orBf16Input(x1)) {
        if (amaxOut != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "aclnnAllGatherMatmulV2", "amaxOut", "non-null",
                "The value of amaxOut must be nullptr when the dtype of input is fp16/bf16");
            return false;
        }
    }
    if (IsAMaxOut(amaxOut)) {
        if (amaxOut->GetViewShape().GetDimNum() != ONE_DIMS) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                "aclnnAllGatherMatmulV2", "amaxOut",
                (std::to_string(amaxOut->GetViewShape().GetDimNum()) + "D").c_str(),
                "The shape of amaxOut must be 1D.");
            return false;
        }
        auto amaxOutLen = amaxOut->GetViewShape().GetDim(0);
        if (amaxOutLen != SCALAR) {
            return false;
        }
        CHECK_RET(CheckParamDtypeFP8Vaild(amaxOut), false);
    }
    return true;
}

static aclnnStatus DealWithX1Empty(uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("AllGatherMatmulV2 dealing with empty tensor.");
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    *workspaceSize = 0U;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

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

static bool checkX1InputEmptyTensor(const aclTensor *x1, const aclTensor *x2)
{
    const auto kValX1 = x1->GetViewShape().GetDim(1);
    if (kValX1 != 0) {
        if (x1->IsEmpty()) {
            return true;
        }
    } else {
        // k will be checked again in following CheckShape()
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnAllGatherMatmulV2", "x1", std::to_string(kValX1).c_str(),
                                              "The value of x1 k-axis must not be 0");
    }
    return false;
}

static aclnnStatus CheckAndHandleCommMode(const char *group, const char *commMode, uint8_t &commModeEnum)
{
    // 获取通信引擎参数
    if (std::strncmp(commMode, "ai_cpu", CMP_MAX_LEN) == 0) {
        commModeEnum = Mc2Comm::COMM_MODE_AICPU;
    } else if (std::strncmp(commMode, "ccu", CMP_MAX_LEN) == 0) {
        commModeEnum = Mc2Comm::COMM_MODE_CCU;
    } else {
        OP_LOGE_WITH_INVALID_ATTR("aclnnAllGatherMatmulV2", "commMode", commMode, "'ccu' or 'ai_cpu'");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus allGatherMatmulV2GetWorkspaceSizeCCUMode(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                                     const aclTensor *x1Scale, const aclTensor *x2Scale,
                                                     const aclTensor *quantScale, int64_t blockSize, const char *group,
                                                     int64_t gatherIndex, int64_t commTurn, int64_t streamMode,
                                                     int64_t groupSize, const char *commMode, aclTensor *output,
                                                     aclTensor *gatherOut, aclTensor *amaxOut, uint64_t *workspaceSize,
                                                     aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    auto retParam = CheckParams(x1, x2, bias, streamMode, output, gatherOut, amaxOut);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // 处理空tensor 如果x1不为空 x2为空 需要进行gatherOut
    if (checkX1InputEmptyTensor(x1, x2)) {
        return DealWithX1Empty(workspaceSize, executor);
    }

    OP_LOGD("X1 is %s.", x1->ToString().GetString());
    OP_LOGD("X2 is %s.", x2->ToString().GetString());

    if (CheckSupportDtype(x1, FP8_DTYPE_SUPPORT_LIST) || CheckSupportDtype(x1, FP4_DTYPE_SUPPORT_LIST)) {
        auto retScaleChk = CheckScale(x1Scale, x2Scale, quantScale);
        CHECK_RET(retScaleChk == ACLNN_SUCCESS, retScaleChk);
    }

    uint32_t rankSize = 0;
    bool transposeX1 = IsTransposeLastTwoDims(x1);
    bool transposeX2 = IsTransposeLastTwoDims(x2);

    CHECK_RET(CheckShape(x1, x2, output, gatherOut, transposeX1), ACLNN_ERR_PARAM_INVALID);
    bool isGatherOut = IsGatherOut(gatherOut);
    bool isAMaxOut = IsAMaxOut(amaxOut);
    // 如果为bf16/fp16的,不能输入amaxout, 如果为低精度，amaxout 数据类型只能为float类型且维度为1维
    CHECK_RET(CheckAMaxOutVaild(x1, amaxOut), ACLNN_ERR_PARAM_INVALID);
    // outDtype 通过output类型判断
    uint64_t outDtype = static_cast<uint64_t>(output->GetDataType());
    CHECK_RET(CheckOutDtypeValid(outDtype, OUT_DTYPE_SUPPORT_LIST), ACLNN_ERR_PARAM_INVALID);
    auto transX2 = x2;
    auto transX2Scale = x2Scale;
    if (transposeX2) {
        // x2转置时将两轴shape调换
        if (x2->GetTensor() == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT("aclnnAllGatherMatmulV2", "Tensor of x2 is null.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
        transX2 = TransX2Tensor(x2);
    }
    if ((x2Scale != nullptr) && MC2Aclnn::IsNeedScaleTrans(x2Scale)) {
        transX2Scale = TransX2Tensor(x2Scale);
    }

    uint8_t commModeEnum = 0;
    aclnnStatus checkCommModeRet = CheckAndHandleCommMode(group, commMode, commModeEnum);
    CHECK_RET(checkCommModeRet == ACLNN_SUCCESS, checkCommModeRet);

    aclnnStatus ret = aclnnInnerAllGatherMatmulV2GetWorkspaceSize(
        x1, transX2, bias, x1Scale, transX2Scale, quantScale, const_cast<char *>(group), transposeX1, transposeX2,
        gatherIndex, commTurn, rankSize, blockSize, groupSize, isGatherOut, isAMaxOut, outDtype,
        const_cast<char *>(commMode), output, gatherOut, amaxOut, workspaceSize, executor);
    OP_LOGD("AllGatherMatmulV2, aclnnInnerGetWorkspaceSize ret = %d.", ret);
    if (ret == ACLNN_SUCCESS && *executor != nullptr) {
        void *arg = reinterpret_cast<void *>(static_cast<uintptr_t>(commModeEnum));
        NnopbaseSetUserHandle(*executor, arg);
    }
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

bool IsViewTransposeX2(const aclTensor *x1, const aclTensor *x2, bool isTransX1)
{
    const auto kValX1 = x1->GetViewShape().GetDim(isTransX1 ? 0 : 1);
    const auto x2Dim0 = x2->GetViewShape().GetDim(0);
    const auto x2Dim1 = x2->GetViewShape().GetDim(1);
    return (kValX1 != x2Dim0) && (kValX1 == x2Dim1);
}

aclnnStatus allGatherMatmulV2GetWorkspaceSizeAIVMode(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                                     const aclTensor *x1Scale, const aclTensor *x2Scale,
                                                     const aclTensor *quantScale, int64_t blockSize, const char *group,
                                                     int64_t gatherIndex, int64_t commTurn, int64_t streamMode,
                                                     int64_t groupSize, const char *commMode, aclTensor *output,
                                                     aclTensor *gatherOut, aclTensor *amaxOut, uint64_t *workspaceSize,
                                                     aclOpExecutor **executor)
{
    OP_LOGD("allGatherMatmulV2GetWorkspaceSizeAIVMode start");
    // 先检查参数
    auto retParam = CheckParamsForAIVMode(x1, x2, output, streamMode);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // 校验comm_mode取值
    if (std::strncmp(commMode, "aiv", CMP_MAX_LEN) != 0) {
        OP_LOGE_WITH_INVALID_ATTR("aclnnAllGatherMatmulV2", "commMode", commMode, "'aiv'");
        return ACLNN_ERR_PARAM_INVALID;
    }
    bool transposeX1 = IsTransposeLastTwoDims(x1);
    bool viewTransposeX2 = IsViewTransposeX2(x1, x2, transposeX1);
    bool transposeX2 = viewTransposeX2 || IsTransposeLastTwoDims(x2);
    uint32_t rankSize = 0;
    bool isAmaxOut = false;
    bool isGatherOut = IsGatherOut(gatherOut);
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    retParam = CheckShapeForAIVMode(x1, x2, bias, output, gatherOut, transposeX1, viewTransposeX2);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // 【A2、A3】校验非连续入参合法性
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        if (!transposeX2 && !MC2Aclnn::IsTensorContiguous(x2)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnAllGatherMatmulV2", "x2", "non-contiguous",
                                                  "The value of x2 must be contiguous when not transposed");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    aclnnStatus ret = aclnnInnerAllGatherMatmulV2GetWorkspaceSize(
        x1, x2, bias, x1Scale, x2Scale, quantScale, const_cast<char *>(group), transposeX1, transposeX2, gatherIndex,
        commTurn, rankSize, blockSize, groupSize, isGatherOut, isAmaxOut, yDtype, const_cast<char *>(commMode), output,
        gatherOut, amaxOut, workspaceSize, executor);
    OP_LOGD("allGatherMatmulV2AIVMode, aclnnInnerGetWorkspaceSize ret = %d.", ret);
    return ret;
}

aclnnStatus aclnnAllGatherMatmulV2GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                                   const aclTensor *x1Scale, const aclTensor *x2Scale,
                                                   const aclTensor *quantScale, int64_t blockSize, const char *group,
                                                   int64_t gatherIndex, int64_t commTurn, int64_t streamMode,
                                                   int64_t groupSize, const char *commMode, aclTensor *output,
                                                   aclTensor *gatherOut, aclTensor *amaxOut, uint64_t *workspaceSize,
                                                   aclOpExecutor **executor)
{
    aclnnStatus ret = ACLNN_SUCCESS;
    if (commMode == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnAllGatherMatmulV2", "CommMode is null.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        ret = allGatherMatmulV2GetWorkspaceSizeCCUMode(x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, group,
                                                       gatherIndex, commTurn, streamMode, groupSize, commMode, output,
                                                       gatherOut, amaxOut, workspaceSize, executor);
    } else if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        ret = allGatherMatmulV2GetWorkspaceSizeAIVMode(x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, group,
                                                       gatherIndex, commTurn, streamMode, groupSize, commMode, output,
                                                       gatherOut, amaxOut, workspaceSize, executor);
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnAllGatherMatmulV2", "npuArch",
            std::to_string(static_cast<int>(GetCurrentPlatformInfo().GetCurNpuArch())).c_str(),
            "The value of npuArch must be DAV_3510 or DAV_2201");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ret;
}

aclnnStatus aclnnAllGatherMatmulV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if ((workspace == nullptr) || (workspaceSize == 0UL)) {
        OP_LOGD("Skip the api for empty tensor, workspace size %lu.", workspaceSize);
        return ACLNN_SUCCESS;
    }

    if (executor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnAllGatherMatmulV2", "executor is null.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            void *arg = NnopbaseGetUserHandle(executor);
            uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
            uint8_t commMode = static_cast<uint8_t>(handleVal);
            if (commMode == Mc2Comm::COMM_MODE_AICPU) {
                OP_LOGD("aclnnAllGatherMatmulV2: NnopbaseHcclServerType, use AICPU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            } else {
                OP_LOGD("aclnnAllGatherMatmulV2: NnopbaseHcclServerType, use CCU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            }
        } else if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }

    return aclnnInnerAllGatherMatmulV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
