/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmul_all_reduce_util.h"
#include "aclnnInner_matmul_all_reduce.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "common/op_api/mc2_aclnn_util.h"
#include "mc2_comm_utils.h"
#include "mc2_log_compat.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void *executor, const size_t irIndex);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// 根据API定义，需要列出所能支持的所有dtype
const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_310P = {op::DataType::DT_FLOAT16};

// quant
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_BIAS = {op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_QUANT = {op::DataType::DT_INT8};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DEQUANT = {
    op::DataType::DT_UINT64, op::DataType::DT_INT64, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_PERTOKEN = {op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DEQUANT_310P = {op::DataType::DT_UINT64,
                                                                                    op::DataType::DT_INT64};

bool MatmulAllReduceCheckFormat(const aclTensor *x2)
{
    // 非量化不支持B矩阵转置
    bool isWeightNZ = MatmulAllReduceIsWeightNZFormat(x2);
    OP_LOGI("MatmulAllReduce, is Weight NZ: %d", isWeightNZ);
    if (isWeightNZ) {
        OP_LOGE_FOR_INVALID_FORMAT("MatmulAllReduce", "x2", "FRACTAL_NZ", "ND");
        return false;
    }
    return true;
}

bool MatmulAllReduceCheckValidEmptyTensor(const aclTensor *x1, const aclTensor *x2)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    // 当输入tensor的shape小于2或者大于6的时候，返回错误
    if (x2->GetViewShape().GetDimNum() < 2 || x2->GetViewShape().GetDimNum() > 6) {
        return false;
    }
    int64_t dim1 = x2->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = x2->GetViewShape().GetDimNum() - 2;
    int64_t m = x1->GetViewShape().GetDim(dim1);
    int64_t k = x2->GetViewShape().GetDim(dim1);
    int64_t n = x2->GetViewShape().GetDim(dim2);
    // 空tensor放行
    if (m == 0 || k == 0 || n == 0) {
        return true;
    }
    return false;
}

bool MatmulAllReduceCheckValidContiguous(const aclTensor *tensor, const char *tensorName)
{
    bool isTransposeX2 = IsTransposeLastTwoDims(tensor);
    // x2非连续时仅支持转置场景
    if (!isTransposeX2 && !MC2Aclnn::IsTensorContiguous(tensor)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("MatmulAllReduce", tensorName, "non-contiguous",
                                              "The value of tensor without transpose must be contiguous");
        return false;
    }
    return true;
}

aclnnStatus MatmulAllReduceCheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *x3,
                                       const aclTensor *bias, const char *reduceOp, int64_t streamMode,
                                       const aclTensor *output)
{
    const static bool is310P = op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2002;

    // 1. 检查参数是否为空指针
    CHECK_RET(MatmulAllReduceCheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(MatmulAllReduceCheckDtypeValid(x1, x2, x3, bias, output, is310P), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查attr是否符合规则
    CHECK_RET(MatmulAllReduceCheckAttr(reduceOp, streamMode), ACLNN_ERR_PARAM_INVALID);

    // 4. 非量化场景不支持B矩阵Nz格式 除了310P
    if (!is310P) {
        CHECK_RET(MatmulAllReduceCheckFormat(x2), ACLNN_ERR_PARAM_INVALID);
    }

    // 5. 检查输出shape
    CHECK_RET(MatmulAllReduceCheckShape(x1, x2, x3, bias, output), ACLNN_ERR_PARAM_INVALID);

    // 6.【A2】检查x2矩阵非连续合法性
    if (op::GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B &&
        !MatmulAllReduceCheckValidEmptyTensor(x1, x2)) {
        CHECK_RET(MatmulAllReduceCheckValidContiguous(x2, "x2"), ACLNN_ERR_PARAM_INVALID);
    }

    if (is310P && x3 != nullptr) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

// 检查入参是否为nullptr
bool MatmulAllReduceCheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *output)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(output, return false);
    return true;
}

bool MatmulAllReduceCheckDtypeValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *x3,
                                    const aclTensor *bias, const aclTensor *output, bool is310P)
{
    const auto &dtypeSupportList = is310P ? DTYPE_SUPPORT_LIST_310P : DTYPE_SUPPORT_LIST;
    // 检查x1、x2、bias、output的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, dtypeSupportList, return false);
    // 检查bias的数据类型是否在算子的支持列表内
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, dtypeSupportList, return false);
    }

    if (x3 != nullptr) {
        // 检查x3的数据类型是否在算子的支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(x3, dtypeSupportList, return false);
        // 检查x1和x3的数据类型是否相同
        OP_CHECK_DTYPE_NOT_SAME(x1, x3, return false);
    }
    // 检查x1和x2的数据类型是否相同
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    // 检查x1和output的数据类型是否相同
    OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);
    // 检查output和bias的数据类型是否相同
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(bias, x1, return false);
    }
    return true;
}

// 检查传入的reduction数值是否在可选范围内
bool MatmulAllReduceCheckAttr(const char *reduceOp, int64_t streamMode)
{
    if (reduceOp == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE("MatmulAllReduce", "reduceOp", "nullptr", "\"sum\"");
        return false;
    }
    bool flag = (strcmp(reduceOp, REDUCE_OP_SUM) == 0);
    if (!flag) {
        OP_LOGE_FOR_INVALID_VALUE("MatmulAllReduce", "reduceOp", reduceOp, "\"sum\"");
        return false;
    }
    if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
        OP_LOGE_FOR_INVALID_VALUE("MatmulAllReduce", "streamMode", std::to_string(streamMode).c_str(), "\"1\"");
        return false;
    }
    return true;
}

bool MatmulAllReduceCheckShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *x3, const aclTensor *bias,
                               const aclTensor *output)
{
    bool isWeightNZ = MatmulAllReduceIsWeightNZFormat(x2);
    uint64_t weightDim = isWeightNZ ? FOUR_DIMS : TWO_DIMS;
    // x2的维度为2维,x1的维度为2D或者3D，output的维数与x1一致,weightNZ场景下，x2可能为4维
    if (x2->GetViewShape().GetDimNum() != weightDim) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "MatmulAllReduce", "x2", (std::to_string(x2->GetViewShape().GetDimNum()) + "D").c_str(),
            ("The shape of x2 must be " + std::to_string(weightDim) + "D.").c_str());
        return false;
    }
    OP_CHECK_MIN_DIM(x1, TWO_DIMS, return false);
    OP_CHECK_MAX_DIM(x1, THREE_DIMS, return false);
    if (isWeightNZ) {
        return true;
    }
    // 仅支持x2矩阵转置，x1不支持转置, x1.GetDimNum(1) == x2.GetDimNum(0)
    const size_t x1_len = x1->GetViewShape().GetDimNum();
    if (x1->GetViewShape().GetDim(x1_len - 1) != x2->GetViewShape().GetDim(0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "MatmulAllReduce", "x1", op::ToString(x1->GetViewShape()).GetString(),
            std::string("The last dim of x1 must be equal to the first dim of x2, but x2 shape is " +
                        std::string(op::ToString(x2->GetViewShape()).GetString()))
                .c_str());
        return false;
    }

    // output的最后一维与x2的最后一维相同
    op::Shape outShape = x1->GetViewShape();
    outShape.SetDim(x1_len - 1, x2->GetViewShape().GetDim(1));
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outShape, return false);

    if (x3 != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(output, x3, return false);
    }

    // x1 shape [s,m,k], x2 shape [k,n], output shape [s,m,n], bias shape [n]
    if (bias != nullptr) {
        if (bias->GetViewShape().GetDimNum() != ONE_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("MatmulAllReduce", "bias",
                                                     (std::to_string(bias->GetViewShape().GetDimNum()) + "D").c_str(),
                                                     "The shape of bias must be 1D.");
            return false;
        }
        op::Shape biasShape;
        biasShape.AppendDim(output->GetViewShape().GetDim(x1_len - 1));
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(bias, biasShape, return false);
    }
    return true;
}

// 全量化场景
aclnnStatus QuantMatmulAllReduceCheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                            const aclTensor *dequantScale, const aclTensor *pertokenScale,
                                            const aclTensor *x3, const char *reduceOp, int64_t streamMode,
                                            const aclTensor *output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(MatmulAllReduceCheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(dequantScale, return ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(QuantMatmulAllReduceCheckDtypeValid(x1, x2, bias, dequantScale, pertokenScale, x3, output),
              ACLNN_ERR_PARAM_INVALID);

    // 3. 检查attr是否符合规则
    CHECK_RET(MatmulAllReduceCheckAttr(reduceOp, streamMode), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出shape
    CHECK_RET(QuantMatmulAllReduceCheckShape(x1, x2, bias, dequantScale, pertokenScale, x3, output),
              ACLNN_ERR_PARAM_INVALID);

    // 5.【A2】检查x2矩阵非连续合法性
    if (op::GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B &&
        !MatmulAllReduceCheckValidEmptyTensor(x1, x2)) {
        CHECK_RET(MatmulAllReduceCheckValidContiguous(x2, "x2"), ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

bool QuantMatmulAllReduceCheckDtypeValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                         const aclTensor *dequantScale, const aclTensor *pertokenScale,
                                         const aclTensor *x3, const aclTensor *output)
{
    const auto &dequantDtypeSupport = op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2002 ?
                                          DTYPE_SUPPORT_LIST_DEQUANT_310P :
                                          DTYPE_SUPPORT_LIST_DEQUANT;

    const auto &outDtypeSupport = op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2002 ?
                                      DTYPE_SUPPORT_LIST_310P :
                                      DTYPE_SUPPORT_LIST;

    // 检查x1、x2、bias、scale、offset、x3、output的数据类型是否在算子的支持列表内
    // 对于量化来说，x1,x2只为INT8
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST_QUANT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST_QUANT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(dequantScale, dequantDtypeSupport, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, outDtypeSupport, return false);
    // 检查bias、offset、x3的数据类型是否在算子的支持列表内
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST_BIAS, return false);
    }
    if (x3 != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x3, outDtypeSupport, return false);
        // 检查x3和output的数据类型是否相同
        OP_CHECK_DTYPE_NOT_SAME(x3, output, return false);
    }
    // 检查x1和x2的数据类型是否相同
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    // BF16场景，dequantScale的dtype与output一致，为BF16
    if (output->GetDataType() == op::DataType::DT_BF16 || dequantScale->GetDataType() == op::DataType::DT_BF16) {
        OP_CHECK_DTYPE_NOT_SAME(dequantScale, output, return false);
    }
    // pertokenScale必须为float32
    if (pertokenScale != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(pertokenScale, DTYPE_SUPPORT_LIST_PERTOKEN, return false);
        // FP16场景，dequantScale的dtype与pertokenScale一致，为float32
        if (output->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_CHECK_DTYPE_NOT_SAME(dequantScale, pertokenScale, return false);
        }
    }
    return true;
}

bool MatmulAllReduceIsWeightNZFormat(const aclTensor *x2)
{
    aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
    aclGetFormat(x2, &format);
    if (format == aclFormat::ACL_FORMAT_ND) {
        OP_LOGD("MatmulAllReduce, Received weight format is ACL_FORMAT_ND");
    }
    if (format == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        OP_LOGD("MatmulAllReduce, Received weight format is ACL_FORMAT_FRACTAL_NZ");
        uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
        OP_LOGD("MatmulAllReduce, Shape is %lu", storageDimsNum);
        const uint64_t transdataNzDim = 4U;
        if (storageDimsNum == transdataNzDim) {
            return true;
        }
    }
    return false;
}

bool QuantMatmulAllReduceIsWeightNZFormat(const aclTensor *x2)
{
    auto format = ge::GetPrimaryFormat(x2->GetStorageFormat());
    OP_LOGD("MatmulAllReduce, Received weight format is %d", format);
    if (format == Format::FORMAT_FRACTAL_NZ) {
        uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
        OP_LOGD("MatmulAllReduce, Shape is %lu", storageDimsNum);
        const uint64_t transdataNzDim = 4U;
        if (storageDimsNum == transdataNzDim) {
            return true;
        }
    }
    return false;
}

// 通过TransMatmulWeight接口预处理成NZ格式场景；
bool QuantMatmulAllReduceIsAclnnPreTransposed(const aclTensor *x2)
{
    auto viewFormat = ge::GetPrimaryFormat(x2->GetViewFormat());
    auto storageFormat = ge::GetPrimaryFormat(x2->GetStorageFormat());
    bool isAclnnPreTransposed = op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2002 &&
                                viewFormat == Format::FORMAT_ND && storageFormat == Format::FORMAT_FRACTAL_NZ;
    OP_LOGD("MatmulAllReduce, IsAclnnPreTransposed is %d", isAclnnPreTransposed);
    return isAclnnPreTransposed;
}

void QuantMatmulAllReduceProcessTransposedX2(const aclTensor *x2, uint64_t &x2Dim0, uint64_t &x2Dim1,
                                             ge::AscendString &x2ShapeStr)
{
    op::Shape x2ViewShape = x2->GetViewShape();
    x2ViewShape.SetDim(0, x2Dim0);
    x2ViewShape.SetDim(1, x2Dim1);
    if (QuantMatmulAllReduceIsAclnnPreTransposed(x2)) {
        x2ShapeStr = op::ToString(x2ViewShape);
    }
    OP_LOGD("MatmulAllReduce, x2 view shape is %s", x2ShapeStr.GetString());
}

bool QuantMatmulAllReduceCheckPertokenScaleShape(const aclTensor *pertokenScale, const aclTensor *x1,
                                                 const size_t x1Len)
{
    // pertokenScale的shape为[m]
    if (pertokenScale != nullptr) {
        OP_CHECK_MIN_DIM(pertokenScale, ONE_DIM, return false);
        OP_CHECK_MAX_DIM(pertokenScale, ONE_DIM, return false);
        const size_t pertokenScaleLen = pertokenScale->GetViewShape().GetDimNum();
        size_t x1_m = 1;
        for (size_t dim = 0; dim < x1Len - 1; dim++) {
            x1_m *= x1->GetViewShape().GetDim(dim);
        }
        if (pertokenScaleLen != DIM_LEN_ONE || static_cast<size_t>(pertokenScale->GetViewShape().GetDim(0)) != x1_m) {
            OP_LOGE_FOR_INVALID_SHAPE("MatmulAllReduce", "pertokenScale",
                                      op::ToString(pertokenScale->GetViewShape()).GetString(),
                                      std::string("[" + std::to_string(x1_m) + "]").c_str());
            return false;
        }
    }
    return true;
}

bool QuantMatmulAllReduceCheckDequantScale(const aclTensor *dequantScale, const op::Shape &outShape, size_t x1Len)
{
    OP_CHECK_MIN_DIM(dequantScale, ONE_DIM, return false);
    OP_CHECK_MAX_DIM(dequantScale, TWO_DIMS, return false);
    const size_t scaleLen = dequantScale->GetViewShape().GetDimNum();
    if (!(scaleLen == DIM_LEN_ONE && (dequantScale->GetViewShape().GetDim(0) == NUM_ONE ||
                                      dequantScale->GetViewShape().GetDim(0) == outShape.GetDim(x1Len - 1))) &&
        !(scaleLen == DIM_LEN_TWO && dequantScale->GetViewShape().GetDim(0) == NUM_ONE &&
          dequantScale->GetViewShape().GetDim(1) == outShape.GetDim(x1Len - 1))) {
        OP_LOGE_FOR_INVALID_SHAPE("MatmulAllReduce", "dequantScale",
                                  op::ToString(dequantScale->GetViewShape()).GetString(),
                                  std::string("[1] or [n] or [1,n], last dim should be " +
                                              std::to_string(outShape.GetDim(x1Len - 1)) + " or 1")
                                      .c_str());
        return false;
    }
    return true;
}

bool QuantMatmulAllReduceCheckShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                    const aclTensor *dequantScale, const aclTensor *pertokenScale, const aclTensor *x3,
                                    const aclTensor *output)
{
    OP_CHECK_MIN_DIM(x1, TWO_DIMS, return false);
    OP_CHECK_MAX_DIM(x1, THREE_DIMS, return false);
    if (QuantMatmulAllReduceIsWeightNZFormat(x2)) {
        return true;
    }
    // x2的维度为2维,x1的维度为2D或者3D，output的维数与x1一致,weightNZ场景下，x2可能为4维
    if (x2->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("MatmulAllReduce", "x2",
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
    if (x1Len < 1 || x1Len > THREE_DIMS) {
        return false;
    }
    if (static_cast<uint64_t>(x1->GetViewShape().GetDim(x1Len - 1)) != x2Dim0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "MatmulAllReduce", "x1", op::ToString(x1->GetViewShape()).GetString(),
            std::string("The last dim of x1 must be equal to the first dim of x2, but x2 shape is " +
                        std::string(x2ShapeStr.GetString()))
                .c_str());
        return false;
    }
    // output的最后一维与x2的最后一维相同
    op::Shape outShape = x1->GetViewShape();
    outShape.SetDim(x1Len - 1, x2Dim1);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outShape, return false);

    // 判断output是否为空tensor
    if (output->IsEmpty()) {
        OP_LOGE_FOR_INVALID_SHAPE("MatmulAllReduce", "output", op::ToString(output->GetViewShape()).GetString(),
                                  "non-empty tensor");
        return false;
    }

    // x1 shape [s,m,k], x2 shape [k,n], output shape [s,m,n], bias shape [n]
    if (bias != nullptr) {
        if (bias->GetViewShape().GetDimNum() != ONE_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("MatmulAllReduce", "bias",
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
    // scale和offset为per-tensor则shape为[1]，为per-channel则shape为[n]或者[1,n]
    if (!QuantMatmulAllReduceCheckDequantScale(dequantScale, outShape, x1Len)) {
        return false;
    }
    if (!QuantMatmulAllReduceCheckPertokenScaleShape(pertokenScale, x1, x1Len)) {
        return false;
    }
    return true;
}

const aclTensor *QuantMatmulAllReduceTransTensor(const aclTensor *x2)
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

aclTensor *QuantMatmulAllReduceCopyTensor(const aclTensor *x2)
{
    uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDims(storageDimsNum);
    for (size_t i = 0; i < storageDimsNum; i++) {
        storageDims[i] = x2->GetStorageShape().GetDim(i);
    }
    OP_LOGD("MatmulAllReduce, CopyTensor storageDimsNum is %lu.", storageDimsNum);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    std::vector<int64_t> stride(storageDimsNum, 1);
    for (int64_t i = static_cast<int64_t>(storageDimsNum - DIM_LEN_TWO); i >= 0; i--) {
        stride[i] = storageDims[i + 1] * stride[i + 1];
    }
    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
    auto stgFormat = ge::GetPrimaryFormat(x2->GetStorageFormat());
    if (stgFormat == Format::FORMAT_ND) {
        OP_LOGD("MatmulAllReduce, CopyTensor format is ACL_FORMAT_ND");
        format = aclFormat::ACL_FORMAT_ND;
    } else if (stgFormat == Format::FORMAT_FRACTAL_NZ) {
        format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    }
    return aclCreateTensor(storageDims.data(), storageDimsNum, dataType, stride.data(), offset, format,
                           storageDims.data(), storageDimsNum, x2->GetTensor()->GetAddr());
}

// quant Inner
aclnnStatus InnerQuantMatmulAllReduceGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                      const aclTensor *biasOptional, const aclTensor *x3Optional,
                                                      const aclTensor *dequant, const aclTensor *pertokenScaleOptional,
                                                      const char *group, const char *reduceOp, int64_t commTurn,
                                                      const aclTensor *output, uint64_t *workspaceSize,
                                                      aclOpExecutor **executor)
{
    // 目前不支持x1进行transpose
    bool transposeX1 = false;
    bool transposeX2 = IsTransposeLastTwoDims(x2) || QuantMatmulAllReduceIsAclnnPreTransposed(x2);
    aclTensor *scale = nullptr;
    aclTensor *offset = nullptr;
    aclTensor *commQuantScale1Optional = nullptr;
    aclTensor *commQuantScale2Optional = nullptr;
    int64_t antiquantGroupSize = 0;
    auto tempX2 = x2;
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2002 && QuantMatmulAllReduceIsWeightNZFormat(x2)) {
        if (x2->GetTensor() == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT("MatmulAllReduce", "x2");
            return ACLNN_ERR_INNER_NULLPTR;
        }
        tempX2 = QuantMatmulAllReduceCopyTensor(x2);
    }
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        }
    }
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    const char *commModePtr = "ai_cpu";
    aclnnStatus ret = aclnnInnerMatmulAllReduceGetWorkspaceSize(
        x1, tempX2, biasOptional, x3Optional, scale, offset, dequant, pertokenScaleOptional, commQuantScale1Optional,
        commQuantScale2Optional, const_cast<char *>(group), const_cast<char *>(reduceOp), transposeX1, transposeX2,
        commTurn, antiquantGroupSize, 0, yDtype, 0, const_cast<char *>(commModePtr), output, workspaceSize, executor);

    OP_LOGI("Group %s, reduce op %s, trans flag %u %u, ret %d.", group, reduceOp, transposeX1, transposeX2, ret);
#ifdef MC2_UT
    ret = 0;
#endif
    if (ret == 0) {
        if (NnopbaseDisableOptionalInput != nullptr && executor != nullptr && *executor != nullptr) {
            NnopbaseDisableOptionalInput(*executor, 4U); // 4 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 5U); // 5 is input irIndex
            if (pertokenScaleOptional == nullptr) {
                NnopbaseDisableOptionalInput(*executor, 7U); // 7 is input irIndex
            }
            NnopbaseDisableOptionalInput(*executor, 8U); // 8 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 9U); // 9 is input irIndex
        }
    }
    return ret;
}

// 检查commMode入参是否合法
bool IsCommModeValid(const char *commModePtr)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        if (commModePtr == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Its new Aclnn interface for A5, commModePtr should not be nullptr!");
            return false;
        }
        if (strcmp(commModePtr, COMM_MODE_AICPU) == 0)
            return true;
        if (strcmp(commModePtr, COMM_MODE_CCU) == 0)
            return true;
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For A5, commMode only support 'ccu' or 'ai_cpu', but it is '%s'.",
                commModePtr);
    } else {
        if (commModePtr == nullptr)
            return true;
        if (strcmp(commModePtr, COMM_MODE_AICPU) == 0)
            return true;
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For A2, commMode only support nullptr or 'ai_cpu', but it is '%s'.",
                commModePtr);
    }
    return false;
}

#ifdef __cplusplus
}
#endif
