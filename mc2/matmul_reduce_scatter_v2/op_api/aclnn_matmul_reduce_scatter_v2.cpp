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
 * \file aclnn_matmul_reduce_scatter_v2.cpp
 * \brief
 */
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "common/utils/hccl_util.h"
#include "mc2_comm_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "mc2_log_compat.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "common/op_api/mc2_aclnn_util.h"
#include "aclnn_matmul_reduce_scatter_v2.h"
#include "aclnnInner_matmul_reduce_scatter_v2.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
static constexpr int64_t NUM_ACL_STOP_ON_FAILURE = 1;
static constexpr int64_t SCALAR = 1;
static constexpr int64_t ONE_DIM = 1;
static constexpr int64_t TWO_DIMS = 2;
static constexpr int64_t KVALUE_MIN = 256;
static constexpr int64_t KVALUE_MAX = 65535;
static constexpr int64_t LAST_AXIS = -1;
static constexpr int64_t SECOND_TO_LAST_AXIS = -2;
static constexpr int64_t DIM_NUM_THREE = 3;
static constexpr int64_t DIM_NUM_TWO = 2;
static constexpr int64_t DIM_NUM_ONE = 1;
static constexpr int64_t AICPU_STRLEN = 6;
static constexpr int64_t CCU_STRLEN = 3;
static constexpr char ACLNN_CCU_PEER_ONLY_MODE[] = "ccu_peer_only";
static constexpr char GRAPH_CCU_MODE[] = "ccu_graph";
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

enum class CommType : uint64_t {
    AI_CPU = 0,
    CCU,
    AIV,
    INVALID
};

extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void *NnopbaseGetUserHandle(void *executor);
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

static inline bool IsAscend950(void)
{
    return op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
}

static void SetNnopbaseHcclServerTypeByArch(aclOpExecutor *executor, CommType commModeEnum)
{
    if ((executor == nullptr) || (NnopbaseSetHcclServerType == nullptr)) {
        return;
    }
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    } else if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        switch (commModeEnum) {
            case CommType::AI_CPU:
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
                OP_LOGD("SetHcclServerType to ai_cpu");
                break;
            case CommType::CCU:
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
                OP_LOGD("SetHcclServerType to ccu");
                break;
            default:
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("SetNnopbaseHcclServerTypeByArch", "commModeEnum", "invalid",
                                                      "Does not match any available case, \
                                                      please check the input commModeEnum.");
                return;
        }
    }
}

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *output)
{
    OP_CHECK_NULL(output, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(x1, return false);
    return true;
}

enum class CaseOption {
    HIGH_ACCURACY = 0,
    LOW_ACCURACY_PER_TENSOR_WITHOUT_QUANT_AMAX,
    LOW_ACCURACY_PER_TENSOR_WITH_QUANT_AMAX,
    LOW_ACCURACY_PER_BLOCK,
    INVALID,
};

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16,        op::DataType::DT_FLOAT16,  op::DataType::DT_FLOAT8_E4M3FN,
    op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> BIAS_OUTPUT_SUPPORT_TYPE = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> INPUT_SUPPORT_TYPE_HIGH_ACCURACY = {op::DataType::DT_FLOAT16,
                                                                                     op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> INPUT_SUPPORT_TYPE_LOW_ACCURACY = {
    op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_HIFLOAT8,
    op::DataType::DT_FLOAT4_E2M1};

static bool CheckDtypeValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *output)
{
    // 检查x1、x2、bias、output的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, BIAS_OUTPUT_SUPPORT_TYPE, return false);
    // 检查bias的数据类型是否在算子的支持列表内
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, BIAS_OUTPUT_SUPPORT_TYPE, return false);
    }
    return true;
}

static const std::initializer_list<op::DataType> AIV_MODE_INPUT_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> AIV_MODE_BIAS_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> AIV_MODE_OUTPUT_SUPPORT_TYPE = {op::DataType::DT_FLOAT16,
                                                                                 op::DataType::DT_BF16};
static bool CheckAivModeDtypeValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                   const aclTensor *output)
{
    // 检查x1、x2、output的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, AIV_MODE_INPUT_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, AIV_MODE_INPUT_SUPPORT_LIST, return false);
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, AIV_MODE_BIAS_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(output, AIV_MODE_OUTPUT_SUPPORT_TYPE, return false);
    return true;
}

// 检查传入的reduction数值是否在可选范围内
static bool CheckAttr(int64_t streamMode)
{
    if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulReduceScatterV2GetWorkspaceSize", "streamMode",
                                  std::to_string(streamMode).c_str(), std::to_string(NUM_ACL_STOP_ON_FAILURE).c_str());
        return false;
    }
    return true;
}
static aclnnStatus CheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, int64_t streamMode,
                               const aclTensor *output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(x1, x2, bias, output), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查attr是否符合规则
    CHECK_RET(CheckAttr(streamMode), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAivModeParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                      int64_t streamMode, const aclTensor *output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckAivModeDtypeValid(x1, x2, bias, output), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查attr是否符合规则
    CHECK_RET(CheckAttr(streamMode), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAndSetCommMode(const char *commMode, CommType &commModeEnum)
{
    if (commMode == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "[aclnnMatmulReduceScatterV2GetWorkspaceSize] commMode must not be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (strncmp(commMode, "ai_cpu", AICPU_STRLEN) == 0) {
        commModeEnum = CommType::AI_CPU;
    } else if (strncmp(commMode, "ccu", CCU_STRLEN) == 0) {
        commModeEnum = CommType::CCU;
    } else {
        OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulReduceScatterV2GetWorkspaceSize", "commMode", commMode,
                                  "\"ccu\" or \"ai_cpu\"");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static bool CheckShape(const aclTensor *x1, const aclTensor *x2, bool isTransA)
{
    if (x1->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnMatmulReduceScatterV2GetWorkspaceSize", "x1",
                                                 (std::to_string(x1->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x1 must be 2D.");
        return false;
    }
    if (x2->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnMatmulReduceScatterV2GetWorkspaceSize", "x2",
                                                 (std::to_string(x2->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x2 must be 2D.");
        return false;
    }
    int64_t kVal1 = 0, kVal2 = 0;
    if (isTransA) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMatmulReduceScatterV2GetWorkspaceSize", "x1", "transposed",
                                              "Does not support transpose x1 matrix.");
        return false;
    }
    kVal1 = x1->GetViewShape().GetDim(1);
    kVal2 = x2->GetViewShape().GetDim(0);
    OP_API_CHECK((kVal1 != kVal2), {
        OP_LOGE_FOR_INVALID_SHAPE("aclnnMatmulReduceScatterV2GetWorkspaceSize", "x1/x2",
                                  (std::string("k=") + std::to_string(kVal1) + "/" + std::to_string(kVal2)).c_str(),
                                  "equal k values");
        return false;
    });
    OP_API_CHECK((kVal1 < KVALUE_MIN || kVal1 >= KVALUE_MAX), {
        OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulReduceScatterV2GetWorkspaceSize", "k-axis", std::to_string(kVal1).c_str(),
                                  "[256, 65535)");
        return false;
    });
    return true;
}

static bool CheckEmptyTensor(const aclTensor *tensor, const char *tensorName)
{
    if (IsNullptr(tensor, tensorName)) {
        return false;
    }
    if (tensor->IsEmpty()) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnMatmulReduceScatterV2GetWorkspaceSize", tensorName);
        return false;
    }
    return true;
}

static bool CheckEmptyOptionalTensor(const aclTensor *tensor, const char *tensorName)
{
    if (tensor == nullptr) {
        OP_LOGD("Expecting valid tensor, name %s", tensorName);
        return false;
    }
    if (tensor->IsEmpty()) {
        OP_LOGD("Expecting non-empty tensor, name %s", tensorName);
        return false;
    }
    return true;
}

static enum CaseOption CheckHighAccuracyCase(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                             const aclTensor *y)
{
    // 高精度场景须数据类型相同
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return CaseOption::INVALID);
    OP_CHECK_DTYPE_NOT_SAME(x1, y, return CaseOption::INVALID);
    if (CheckEmptyOptionalTensor(bias, "bias")) {
        OP_CHECK_DTYPE_NOT_SAME(x1, bias, return CaseOption::INVALID);
    }
    return CaseOption::HIGH_ACCURACY;
}

static enum CaseOption CheckLowAccuracyCase(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                                            const aclTensor *x2Scale)
{
    // 先x1 x2指针判空
    if (!CheckEmptyTensor(x1, "x1") || !CheckEmptyTensor(x2, "x2")) {
        return CaseOption::INVALID;
    }
    // 矩阵入参为Hifloat8时，x1x2必须类型相同。
    if (x1->GetDataType() == op::DataType::DT_HIFLOAT8 || x1->GetDataType() == op::DataType::DT_FLOAT4_E2M1) {
        OP_CHECK_DTYPE_NOT_SAME(x1, x2, return CaseOption::INVALID);
    }
    if (!CheckEmptyTensor(x1Scale, "x1Scale") || !CheckEmptyTensor(x2Scale, "x2Scale")) {
        return CaseOption::INVALID;
    }
    if ((x1Scale->GetViewShape().GetDimNum() == DIM_NUM_ONE) ||
        ((x1Scale->GetViewShape().GetDimNum() == DIM_NUM_THREE) && (x1Scale->GetDataType() == ge::DT_FLOAT8_E8M0))) {
        return CaseOption::LOW_ACCURACY_PER_TENSOR_WITH_QUANT_AMAX;
    } else if (x1Scale->GetViewShape().GetDimNum() == DIM_NUM_TWO) {
        // perblock其余判定放在tiling侧进行
        return CaseOption::LOW_ACCURACY_PER_BLOCK;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMatmulReduceScatterV2GetWorkspaceSize", "x1Scale", "unexpected dim",
                                              "Invalid scene in low accuracy, please check input x1Scale.");
        return CaseOption::INVALID;
    }
}

static enum CaseOption CheckCase(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *y,
                                 const aclTensor *amax, const aclTensor *x1Scale, const aclTensor *x2Scale)
{
    if (amax != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMatmulReduceScatterV2GetWorkspaceSize", "amaxOutOptional",
                                              "non-nullptr", "Does not support non-nullptr amaxOutOptional.");
        return CaseOption::INVALID;
    }
    if (CheckType(x1->GetDataType(), INPUT_SUPPORT_TYPE_HIGH_ACCURACY) &&
        CheckType(x2->GetDataType(), INPUT_SUPPORT_TYPE_HIGH_ACCURACY)) { // 高精度场景
        return CheckHighAccuracyCase(x1, x2, bias, y);
    } else if (CheckType(x1->GetDataType(), INPUT_SUPPORT_TYPE_LOW_ACCURACY) &&
               CheckType(x2->GetDataType(), INPUT_SUPPORT_TYPE_LOW_ACCURACY)) { // 低精度场景
        return CheckLowAccuracyCase(x1, x2, x1Scale, x2Scale);
    } else {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "aclnnMatmulReduceScatterV2GetWorkspaceSize", "x1/x2",
            (std::string(op::ToString(x1->GetDataType()).GetString()) + "/" +
             op::ToString(x2->GetDataType()).GetString())
                .c_str(),
            "Does not match any available case, please read the documents and check the input.");
        return CaseOption::INVALID;
    }
}

static const aclTensor *TransX2Tensor(const aclTensor *x2)
{
    uint64_t storageShapeDimNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = x2->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = x2->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = x2->GetViewShape().GetDim(i);
    }
    // transpose the viewshape last two dimensions
    viewDim[0] = x2->GetViewShape().GetDim(1);
    viewDim[1] = x2->GetViewShape().GetDim(0);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    std::vector<int64_t> stride(viewShapeDimNum);
    auto transStride = x2->GetViewStrides();
    stride = std::vector<int64_t>(transStride.begin(), transStride.end());
    // transpose the two dimensions
    stride[0] = transStride[1];
    stride[1] = transStride[0];

    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, x2->GetTensor()->GetAddr());
}

aclnnStatus matmulReduceScatterV2GetWorkSpaceSizeA5(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                                    const aclTensor *x1Scale, const aclTensor *x2Scale,
                                                    const aclTensor *quantScale, int64_t blockSize, const char *group,
                                                    const char *reduceOp, int64_t commTurn, int64_t streamMode,
                                                    int64_t groupSize, const char *commMode, aclTensor *output,
                                                    aclTensor *amaxOutOptional, uint64_t *workspaceSize,
                                                    aclOpExecutor **executor)
{
    OP_LOGD("aclnnmatmulReduceScatterV2GetWorkSpaceSizeA5 start, comm_mode is %s.", commMode);
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    // 固定写法，参数检查
    auto retParam = CheckParams(x1, x2, bias, streamMode, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    CommType commModeEnum = CommType::INVALID;
    auto retCommmode = CheckAndSetCommMode(commMode, commModeEnum);
    CHECK_RET(retCommmode == ACLNN_SUCCESS, retCommmode);
    if (std::strcmp(commMode, "ai_cpu") != 0 && std::strcmp(commMode, "ccu") != 0 &&
        std::strcmp(commMode, GRAPH_CCU_MODE) != 0) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulReduceScatterV2GetWorkspaceSize", "commMode", commMode,
                                  "\"ccu\" or \"ai_cpu\"");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (commModeEnum >= CommType::INVALID) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnmatmulReduceScatterV2GetWorkSpaceSize", "commModeEnum", "invalid",
                                              "Does not match any available case, please check the input commMode.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 处理空tensor
    if (x1->IsEmpty() || x2->IsEmpty()) {
        OP_LOGD("MatmulReduceScatter, dealing with empty tensor.");
        // 固定写法，创建OpExecutor
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        if (workspaceSize != nullptr) {
            *workspaceSize = 0;
        }
        uniqueExecutor.ReleaseTo(executor);
        if ((executor != nullptr) && (*executor != nullptr)) {
            SetNnopbaseHcclServerTypeByArch(*executor, commModeEnum);
        }
        return ACLNN_SUCCESS;
    }
    OP_LOGD("X1 is %s. X2 is %s.", x1->ToString().GetString(), x2->ToString().GetString());

    bool transposeX1 = Ops::Transformer::IsTransposeLastTwoDims(x1);
    bool transposeX2 = Ops::Transformer::IsTransposeLastTwoDims(x2);
    CHECK_RET(CheckShape(x1, x2, transposeX1), ACLNN_ERR_PARAM_INVALID);
    CaseOption caseIndex = CheckCase(x1, x2, bias, output, amaxOutOptional, x1Scale, x2Scale);
    if (caseIndex >= CaseOption::INVALID) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnMatmulReduceScatterV2GetWorkspaceSize", "input case", "invalid",
            "Does not match any available case, please read the documents and check the input.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("Input fit case %u.", static_cast<uint32_t>(caseIndex));
    bool isAmaxOut = false;
    if (caseIndex == CaseOption::LOW_ACCURACY_PER_TENSOR_WITH_QUANT_AMAX) {
        isAmaxOut = true;
    }
    uint32_t rankSize = 0;
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    aclnnStatus ret = ACLNN_SUCCESS;
    auto transX2Scale = x2Scale;
    if ((x2Scale != nullptr) && MC2Aclnn::IsNeedScaleTrans(x2Scale)) {
        transX2Scale = TransX2Tensor(x2Scale);
    }
    auto transX2 = x2;
    if (transposeX2) {
        // x2转置时将两轴shape调换
        transX2 = TransX2Tensor(x2);
        OP_LOGD("X2 dim0 is %ld, dim1 is %ld.", x2->GetViewShape().GetDim(0), x2->GetViewShape().GetDim(1));
    }
    // Keep peer-only as an ACLNN-only optimization. Graph execution uses its own comm-mode marker.
    const char *innerCommMode = commMode;
    if (std::strcmp(commMode, "ccu") == 0) {
        innerCommMode = ACLNN_CCU_PEER_ONLY_MODE;
    } else if (std::strcmp(commMode, GRAPH_CCU_MODE) == 0) {
        innerCommMode = "ccu";
    }
    ret = aclnnInnerMatmulReduceScatterV2GetWorkspaceSize(
        x1, transX2, bias, x1Scale, transX2Scale, quantScale, const_cast<char *>(group), const_cast<char *>(reduceOp),
        transposeX1, transposeX2, commTurn, rankSize, blockSize, groupSize, isAmaxOut, yDtype,
        const_cast<char *>(innerCommMode), output, amaxOutOptional, workspaceSize, executor);
    if ((ret == ACLNN_SUCCESS) && (executor != nullptr) && (*executor != nullptr)) {
        SetNnopbaseHcclServerTypeByArch(*executor, commModeEnum);
        void *args = reinterpret_cast<void *>(static_cast<uintptr_t>(commModeEnum));
        NnopbaseSetUserHandle(*executor, args);
    }
    OP_LOGD("MatmulReduceScatterV2, end ret %d.", ret);
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

static bool MatmulReduceScatterV2IsWeightNZFormat(const aclTensor *x2)
{
    aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
    aclGetFormat(x2, &format);
    if (format == aclFormat::ACL_FORMAT_ND) {
        OP_LOGD("MatmulReduceScatterV2, Received weight format is ACL_FORMAT_ND");
    }
    if (format == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        OP_LOGD("MatmulReduceScatterV2, Received weight format is ACL_FORMAT_FRACTAL_NZ");
        uint64_t storageShapeDimNum = x2->GetStorageShape().GetDimNum();
        OP_LOGD("MatmulReduceScatterV2, Shape is %lu", storageShapeDimNum);
        const uint64_t transdataNzDim = 4U;
        if (storageShapeDimNum == transdataNzDim) {
            return true;
        }
    }
    return false;
}

aclnnStatus matmulReduceScatterV2GetWorkSpaceSizeAivMode(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *x1Scale, const aclTensor *x2Scale,
    const aclTensor *quantScale, int64_t blockSize, const char *group, const char *reduceOp, int64_t commTurn,
    int64_t streamMode, int64_t groupSize, const char *commMode, aclTensor *output, aclTensor *amaxOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnMatmulReduceScatterV2GetWorkspaceSizeAivMode start");
    auto ret_param = CheckAivModeParams(x1, x2, bias, streamMode, output);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    bool transposeX1 = Ops::Transformer::IsTransposeLastTwoDims(x1);
    bool transposeX2 = Ops::Transformer::IsTransposeLastTwoDims(x2);
    uint32_t rankSize = 0;
    bool isAmaxOut = false;
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    (void)MatmulReduceScatterV2IsWeightNZFormat(x2);
    CHECK_RET(CheckShape(x1, x2, transposeX1), ACLNN_ERR_PARAM_INVALID);
    // 【A2、A3】检查x2矩阵非连续合法性
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        if (!Ops::Transformer::IsTransposeLastTwoDims(x2) && !MC2Aclnn::IsTensorContiguous(x2)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMatmulReduceScatterV2GetWorkspaceSizeAivMode", "x2",
                                                  "non-contiguous", "x2 without transpose must be contiguous.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    aclnnStatus ret = aclnnInnerMatmulReduceScatterV2GetWorkspaceSize(
        x1, x2, bias, x1Scale, x2Scale, quantScale, const_cast<char *>(group), const_cast<char *>(reduceOp),
        transposeX1, transposeX2, commTurn, rankSize, blockSize, groupSize, isAmaxOut, yDtype,
        const_cast<char *>(commMode), output, amaxOutOptional, workspaceSize, executor);
    if ((ret == ACLNN_SUCCESS) && (executor != nullptr) && (*executor != nullptr)) {
        CommType commModeEnum = CommType::AIV;
        SetNnopbaseHcclServerTypeByArch(*executor, commModeEnum);
    }
    OP_LOGD("MatmulReduceScatterV2AivMode, aclnnInnerGetWorkspaceSize ret = %d.", ret);
    return ret;
}

aclnnStatus aclnnMatmulReduceScatterV2GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                                       const aclTensor *x1Scale, const aclTensor *x2Scale,
                                                       const aclTensor *quantScale, int64_t blockSize,
                                                       const char *group, const char *reduceOp, int64_t commTurn,
                                                       int64_t streamMode, int64_t groupSize, const char *commMode,
                                                       aclTensor *output, aclTensor *amaxOutOptional,
                                                       uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus ret = ACLNN_ERR_INNER;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        ret = matmulReduceScatterV2GetWorkSpaceSizeA5(x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, group,
                                                      reduceOp, commTurn, streamMode, groupSize, commMode, output,
                                                      amaxOutOptional, workspaceSize, executor);
    } else if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        OP_LOGD("[MatmulReduceScatterV2] NpuArch is 2201, support aiv commmode only.");
        ret = matmulReduceScatterV2GetWorkSpaceSizeAivMode(x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, group,
                                                           reduceOp, commTurn, streamMode, groupSize, commMode, output,
                                                           amaxOutOptional, workspaceSize, executor);
    } else {
        OP_LOGE(ACLNN_ERR_INNER, "[MatmulReduceScatterV2] Unsupported NPU arch, "
                                 "only DAV_3510 and DAV_2201 are supported.");
        return ACLNN_ERR_INNER;
    }

    return ret;
}

aclnnStatus aclnnMatmulReduceScatterV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream)
{
    if ((workspace == nullptr) || (workspaceSize == 0UL)) {
        OP_LOGD("Skip the api for empty tensor, workspace size %lu.", workspaceSize);
        return ACLNN_SUCCESS;
    }

    CommType commModeEnum = CommType::INVALID;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        void *arg = NnopbaseGetUserHandle(executor);
        uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
        commModeEnum = static_cast<CommType>(handleVal);
    } else if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        commModeEnum = CommType::AIV;
    }

    if (commModeEnum >= CommType::INVALID) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMatmulReduceScatterV2", "commModeEnum", "invalid",
                                              "Does not match any available case, please check the input commMode.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    SetNnopbaseHcclServerTypeByArch(executor, commModeEnum);
    aclnnStatus ret = aclnnInnerMatmulReduceScatterV2(workspace, workspaceSize, executor, stream);
    if (ret != 0) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnMatmulReduceScatterV2", "This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
