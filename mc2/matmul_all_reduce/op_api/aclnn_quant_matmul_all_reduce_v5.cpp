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
 * \file aclnn_quant_matmul_all_reduce_v5.cpp
 * \brief
 */
#include "aclnn_quant_matmul_all_reduce_v5.h"
#include "aclnnInner_matmul_all_reduce.h"
#include "common/op_api/mc2_aclnn_util.h"
#include "matmul_all_reduce_util.h"
#include "mc2_comm_utils.h"
#include "mc2_log_compat.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void *NnopbaseGetUserHandle(void *executor);

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_BIAS = {op::DataType::DT_INT32,
                                                                            op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_QUANT = {
    op::DataType::DT_INT8, op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_HIFLOAT8,
    op::DataType::DT_FLOAT4_E2M1};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_QUANT_FP4 = {op::DataType::DT_FLOAT4_E2M1};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_QUANT_FP8 = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_HIFLOAT8};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DEQUANT = {
    op::DataType::DT_UINT64, op::DataType::DT_INT64, op::DataType::DT_BF16, op::DataType::DT_FLOAT,
    op::DataType::DT_FLOAT8_E8M0};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_PERTOKEN = {op::DataType::DT_FLOAT,
                                                                                op::DataType::DT_FLOAT8_E8M0};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WITHOUT_FLOAT = {op::DataType::DT_FLOAT16,
                                                                                     op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_FLOAT = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *x2Scale, const aclTensor *output,
                         const char *reduceOp)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(x2Scale, return false);
    OP_CHECK_NULL(output, return false);
    if (reduceOp == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input reduceOp is nullptr.");
        return false;
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *x2Scale,
                            const aclTensor *x1Scale, const aclTensor *x3, const aclTensor *commQuantScale1Optional,
                            const aclTensor *commQuantScale2Optional, const aclTensor *output)
{
    // 检查x1、x2、bias、scale、offset、x3、output的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST_QUANT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST_QUANT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2Scale, DTYPE_SUPPORT_LIST_DEQUANT, return false);
    // hif8/fp8/mxfp4 支持fp32的输出
    const auto outputSupportedDtype = (CheckType(x1->GetDataType(), DTYPE_SUPPORT_LIST_QUANT_FP8) ||
                                       CheckType(x1->GetDataType(), DTYPE_SUPPORT_LIST_QUANT_FP4)) ?
                                          DTYPE_SUPPORT_LIST_FLOAT :
                                          DTYPE_SUPPORT_LIST_WITHOUT_FLOAT;
    OP_CHECK_DTYPE_NOT_SUPPORT(output, outputSupportedDtype, return false);
    // 检查bias、offset、x3的数据类型是否在算子的支持列表内
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST_BIAS, return false);
    }
    if (x3 != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x3, outputSupportedDtype, return false);
        // 检查x3和output的数据类型是否相同
        OP_CHECK_DTYPE_NOT_SAME(x3, output, return false);
    }

    // BF16场景，x2Scale的dtype与output一致，为BF16。fp8/fp4场景不检查
    if (((output->GetDataType() == op::DataType::DT_BF16) || (x2Scale->GetDataType() == op::DataType::DT_BF16)) &&
        (!CheckType(x1->GetDataType(), DTYPE_SUPPORT_LIST_QUANT_FP8) &&
         !CheckType(x1->GetDataType(), DTYPE_SUPPORT_LIST_QUANT_FP4))) {
        OP_CHECK_DTYPE_NOT_SAME(x2Scale, output, return false);
    }

    if (x1Scale != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x1Scale, DTYPE_SUPPORT_LIST_PERTOKEN, return false);
    }

    if (commQuantScale1Optional != nullptr && commQuantScale2Optional != nullptr) {
        // 检查commQuantScale1 commQuantScale2的数据类型是否在算子的支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(commQuantScale1Optional, DTYPE_SUPPORT_LIST_WITHOUT_FLOAT, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(commQuantScale2Optional, DTYPE_SUPPORT_LIST_WITHOUT_FLOAT, return false);
        // 检查commQuantScale1Optional commQuantScale2Optional和output的数据类型是否相同
        OP_CHECK_DTYPE_NOT_SAME(commQuantScale1Optional, output, return false);
        OP_CHECK_DTYPE_NOT_SAME(commQuantScale2Optional, output, return false);
    }
    return true;
}

// 检查传入的reduction数值是否在可选范围内
static bool CheckAttr(const char *reduceOp, int64_t streamMode)
{
    if (strncmp(reduceOp, REDUCE_OP_SUM, NUM_THREE) != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected reduceOp to be sum, but got=%s.", reduceOp);
        return false;
    }
    if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected streamMode to be 1, but got=%ld.", streamMode);
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *x3,
                       const aclTensor *output)
{
    OP_CHECK_MIN_DIM(x1, TWO_DIMS, return false);
    OP_CHECK_MAX_DIM(x1, THREE_DIMS, return false);
    if (MatmulAllReduceIsWeightNZFormat(x2)) {
        return true;
    }
    // x2的维度为2维,x1的维度为2D或者3D，output的维数与x1一致,weightNZ场景下，x2可能为4维
    if (x2->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulAllReduceV5", "x2",
                                                 (std::to_string(x2->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x2 must be 2D.");
        return false;
    }

    uint64_t x2Dim0 =
        QuantMatmulAllReduceIsAclnnPreTransposed(x2) ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);
    uint64_t x2Dim1 =
        QuantMatmulAllReduceIsAclnnPreTransposed(x2) ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    auto x2ShapeStr = op::ToString(x2->GetViewShape());
    QuantMatmulAllReduceProcessTransposedX2(x2, x2Dim0, x2Dim1, x2ShapeStr);
    // 仅支持x2矩阵转置，x1不支持转置, x1.GetDimNum(1) == x2.GetDimNum(0)
    const size_t x1Len = x1->GetViewShape().GetDimNum();
    if (static_cast<uint64_t>(x1->GetViewShape().GetDim(x1Len - 1)) != x2Dim0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected last dim of x1 to be equal to first dim of x2, but got x1 shape=%s, x2 shape=%s.",
                op::ToString(x1->GetViewShape()).GetString(), x2ShapeStr.GetString());
        return false;
    }

    // output的最后一维与x2的最后一维相同
    op::Shape outShape = x1->GetViewShape();
    outShape.SetDim(x1Len - 1, x2Dim1);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outShape, return false);

    // 判断output是否为空tensor
    if (output->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output is empty tensor, output shape is: %s",
                op::ToString(output->GetViewShape()).GetString());
        return false;
    }

    // x1 shape [s,m,k], x2 shape [k,n], output shape [s,m,n], bias shape [n]
    if (bias != nullptr) {
        if (bias->GetViewShape().GetDimNum() != ONE_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulAllReduceV5", "bias",
                                                     (std::to_string(bias->GetViewShape().GetDimNum()) + "D").c_str(),
                                                     "The shape of bias must be 1D.");
            return false;
        }
        op::Shape biasShape;
        biasShape.AppendDim(output->GetViewShape().GetDim(x1Len - 1));
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(bias, biasShape, return false);
    }

    // x3 shape [s,m,n]
    if (x3 != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(x3, output, return false);
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                               const aclTensor *x2Scale, const aclTensor *x1Scale, const aclTensor *x3,
                               const aclTensor *commQuantScale1Optional, const aclTensor *commQuantScale2Optional,
                               const char *reduceOp, const char *commMode, int64_t streamMode, const aclTensor *output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, x2Scale, output, reduceOp), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(
        CheckDtypeValid(x1, x2, bias, x2Scale, x1Scale, x3, commQuantScale1Optional, commQuantScale2Optional, output),
        ACLNN_ERR_PARAM_INVALID);

    // 3. 检查attr是否符合规则
    CHECK_RET(CheckAttr(reduceOp, streamMode), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出shape
    CHECK_RET(CheckShape(x1, x2, bias, x3, output), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查commMode是否合法
    CHECK_RET(IsCommModeValid(commMode), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor *CopyTensor(const aclTensor *x2)
{
    uint64_t storageDimCount = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDimVec(storageDimCount);
    for (size_t i = 0; i < storageDimCount; i++) {
        storageDimVec[i] = x2->GetStorageShape().GetDim(i);
    }
    OP_LOGD("MatmulAllReduce, CopyTensor storageDimsNum=%lu.", storageDimCount);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    auto stride = x2->GetViewStrides();
    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
    auto stgFormat = ge::GetPrimaryFormat(x2->GetStorageFormat());
    if (stgFormat == Format::FORMAT_ND) {
        OP_LOGD("MatmulAllReduce, CopyTensor format is ACL_FORMAT_ND.");
        format = aclFormat::ACL_FORMAT_ND;
    } else if (stgFormat == Format::FORMAT_FRACTAL_NZ) {
        format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    }
    return aclCreateTensor(storageDimVec.data(), storageDimCount, dataType, stride.data(), offset, format,
                           storageDimVec.data(), storageDimCount, x2->GetTensor()->GetAddr());
}

aclnnStatus aclnnQuantMatmulAllReduceV5GetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional, const aclTensor *x3Optional,
    const aclTensor *x1ScaleOptional, const aclTensor *x2Scale, const aclTensor *commQuantScale1Optional,
    const aclTensor *commQuantScale2Optional, const char *group, const char *reduceOp, const char *commMode,
    int64_t commTurn, int64_t streamMode, int64_t groupSize, int64_t commQuantMode, const aclTensor *output,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    // 固定写法，参数检查
    auto retParam = CheckParams(x1, x2, biasOptional, x2Scale, x1ScaleOptional, x3Optional, commQuantScale1Optional,
                                commQuantScale2Optional, reduceOp, commMode, streamMode, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // x2Scale转为uint64
    auto dequant = const_cast<aclTensor *>(x2Scale);
    if (dequant == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantMatmulAllReduce, dequant is nullptr.");
        return ACLNN_ERR_INNER;
    }
    if (dequant->GetDataType() == op::DataType::DT_INT64) {
        dequant->SetDataType(op::DataType::DT_UINT64);
    }

    // 目前不支持x1进行transpose
    bool transposeX1 = false;
    bool transposeX2 = IsTransposeLastTwoDims(x2) || QuantMatmulAllReduceIsAclnnPreTransposed(x2);

    aclTensor *scale = nullptr;
    aclTensor *offset = nullptr;
    int64_t antiquantGroupSize = 0;
    auto tempX2 = x2;
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2002 && MatmulAllReduceIsWeightNZFormat(x2)) {
        if (x2->GetTensor() == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Tensor of x2 is null.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
        tempX2 = CopyTensor(x2);
    }
    auto transX2 = tempX2;
    auto transX2Scale = x2Scale;
    if (transposeX2) {
        if (tempX2->GetTensor() == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Tensor is null.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
        transX2 = QuantMatmulAllReduceTransTensor(tempX2);
        // mxfp是3维scale，转置的是前两维，需要特殊判断，perblock场景复用判断转置接口
        if (MC2Aclnn::IsNeedScaleTrans(x2Scale)) {
            transX2Scale = QuantMatmulAllReduceTransTensor(x2Scale);
        }
    }

    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    aclnnStatus ret = aclnnInnerMatmulAllReduceGetWorkspaceSize(
        x1, transX2, biasOptional, x3Optional, scale, offset, transX2Scale, x1ScaleOptional, commQuantScale1Optional,
        commQuantScale2Optional, const_cast<char *>(group), const_cast<char *>(reduceOp), transposeX1, transposeX2,
        commTurn, antiquantGroupSize, groupSize, yDtype, commQuantMode, const_cast<char *>(commMode), output,
        workspaceSize, executor);
    if (ret == ACLNN_SUCCESS && executor != nullptr && *executor != nullptr && commMode != nullptr) {
        // SetUserHandle to pass comm_mode to aclnnQuantMatmulAllReduceV5
        uint8_t commModeEnum = 0;
        if (strcmp(commMode, COMM_MODE_AICPU) == 0) {
            commModeEnum = Mc2Comm::COMM_MODE_AICPU;
        } else if (strcmp(commMode, COMM_MODE_CCU) == 0) {
            commModeEnum = Mc2Comm::COMM_MODE_CCU;
        } else {
            OP_LOGE_WITH_INVALID_ATTR("aclnnQuantMatmulAllReduceV5GetWorkspaceSize", "commMode", commMode,
                                      "'ccu' or 'ai_cpu'");
            return ACLNN_ERR_PARAM_INVALID;
        }
        void *arg = reinterpret_cast<void *>(static_cast<uintptr_t>(commModeEnum));
        NnopbaseSetUserHandle(*executor, arg);
    }
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

aclnnStatus aclnnQuantMatmulAllReduceV5(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        const aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Param executor is nullptr.");
        return ACLNN_ERR_INNER;
    }
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            void *arg = NnopbaseGetUserHandle(executor);
            uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
            uint8_t commMode = static_cast<uint8_t>(handleVal);
            if (commMode == Mc2Comm::COMM_MODE_CCU) {
                OP_LOGD("A5 aclnnMatmulAllReduceV3: NnopbaseHcclServerType, use CCU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            } else {
                OP_LOGD("A5 aclnnMatmulAllReduceV3: NnopbaseHcclServerType, use AICPU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            }
        } else {
            OP_LOGD("A2 aclnnQuantMatmulAllReduceV5: NnopbaseHcclServerType, use AICPU mode");
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        }
    }
    aclnnStatus ret = aclnnInnerMatmulAllReduce(workspace, workspaceSize, executor, stream);
    OP_LOGD("QuantMatmulAllReduce, aclnnQuantMatmulAllReduceV5 ret=%d.", ret);

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "QuantMatmulAllReduce, This is an error in launch aicore.");
        return ACLNN_ERR_INNER;
    }

    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
