/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allto_all_quant_matmul_base.h"

#include "acl/acl.h"
#include "mc2_log_compat.h"
#include "aclnnInner_allto_all_matmul.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "common/op_api/mc2_aclnn_util.h"
#include "common/utils/hccl_util.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "mc2/common/utils/mc2_comm_utils.h"
#include "mc2/matmul_allto_all/op_api/matmul_allto_all_util.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "securec.h"

namespace {

using namespace op;
using namespace l0op;
using namespace matmul_allto_all_check;

constexpr size_t THREE_DIMS = 3U;
static constexpr int64_t INT4_NUMS_IN_INT32 = 8; // 一个int32包含8个int4

// 将int32的输入dtype修改为int4, 同时ViewShape和ViewStrides也从int32修改为int4所对应的
aclTensor *ConvertTensorToInt4(const aclTensor *input, aclOpExecutor *executor)
{
    auto viewShape = input->GetViewShape();
    viewShape[viewShape.GetDimNum() - 1] = viewShape[viewShape.GetDimNum() - 1] * INT4_NUMS_IN_INT32;
    auto inputTemp = executor->CreateView(input, viewShape, input->GetViewOffset());
    inputTemp->SetDataType(DataType::DT_INT4);
    OP_LOGD("The conversion from int32 to int4 is completed.");
    return inputTemp;
}

// 对x1和x2进行int32到int4的转换预处理
void InputPreProcessInt4(const aclTensor *&x1, const aclTensor *&x2, const aclTensor *&alltoallout,
                         aclOpExecutor *executor)
{
    if (x2 != nullptr && x2->GetDataType() == DataType::DT_INT32) {
        x2 = ConvertTensorToInt4(x2, executor);
    }
    if (x1 != nullptr && x1->GetDataType() == DataType::DT_INT32) {
        x1 = ConvertTensorToInt4(x1, executor);
    }
    if (alltoallout != nullptr && alltoallout->GetDataType() == DataType::DT_INT32) {
        alltoallout = ConvertTensorToInt4(alltoallout, executor);
    }
}

// 检查必要输入是否为空，必须非空
static bool CheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                         const aclTensor *x1ScaleOptional, const aclTensor *x2Scale, const aclTensor *output,
                         int64_t x1QuantMode)
{
    if (x1 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x1 should not be null.");
        return false;
    }
    if (x2 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x2 should not be null.");
        return false;
    }
    if (static_cast<QuantModeType>(x1QuantMode) == QuantModeType::MX_QUANT &&
        GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        if (x1ScaleOptional == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The current scenario is not pertoken dynamic quantization,"
                                             "input x1ScaleOptional should not be null.");
            return false;
        }
    }
    if (x2Scale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x2Scale should not be null.");
        return false;
    }
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Output should not be null.");
        return false;
    }
    return true;
}

// 检查是否有空tensor
static bool CheckNotEmptyTensor(const aclTensor *x1, const aclTensor *x2, bool transposeX2)
{
    auto mVal = x1->GetViewShape().GetDim(0);
    auto kVal1 = x1->GetViewShape().GetDim(1);
    auto kVal2 = transposeX2 ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    OP_API_CHECK((mVal == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "AlltoAllQuantMatmul, x1 is empty tensor with zero dimM, which is unsupported.");
        return false;
    });
    OP_API_CHECK((kVal1 == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "AlltoAllQuantMatmul, x1 is empty tensor with zero dimK, which is unsupported.");
        return false;
    });
    OP_API_CHECK((kVal2 == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "AlltoAllQuantMatmul, x2 is empty tensor with zero dimK, which is unsupported.");
        return false;
    });
    OP_API_CHECK((nVal == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "AlltoAllQuantMatmul, x2 is empty tensor with zero dimN, which is unsupported.");
        return false;
    });
    return true;
}

// 校验Scale为1D时的shape（KC动态量化）
static bool Check1DScaleShape(const aclTensor *x2, const aclTensor *x2Scale, bool transposeX2)
{
    if (x2Scale->GetViewShape().GetDimNum() != ONE_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnAlltoAllQuantMatmul", "x2Scale",
                                                 (std::to_string(x2Scale->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x2Scale must be 1D.");
        return false;
    }
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    auto x2ScaleDim = x2Scale->GetViewShape().GetDim(0);
    if (x2ScaleDim != nVal) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The n-axis of x2 and x2ScaleDim should be same, but x2's n-axis is: %ld and x2ScaleDim is: %ld.", nVal,
                x2ScaleDim);
        return false;
    }
    return true;
}

// 校验Scale为3D时的shape（MX量化）
static bool Check3DScaleShape(const aclTensor *x2, const aclTensor *x1Scale, const aclTensor *x2Scale, bool transposeX2)
{
    if (x1Scale->GetViewShape().GetDimNum() != THREE_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnAlltoAllQuantMatmul", "x1Scale",
                                                 (std::to_string(x1Scale->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x1Scale must be 3D.");
        return false;
    }
    if (x2Scale->GetViewShape().GetDimNum() != THREE_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnAlltoAllQuantMatmul", "x2Scale",
                                                 (std::to_string(x2Scale->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x2Scale must be 3D.");
        return false;
    }
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    auto x2ScaleNVal = transposeX2 ? x2Scale->GetViewShape().GetDim(0) : x2Scale->GetViewShape().GetDim(1);
    auto x1ScaleLastDim = x1Scale->GetViewShape().GetDim(2);
    auto x2ScaleLastDim = x2Scale->GetViewShape().GetDim(2);
    if (x2ScaleNVal != nVal) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The n-axis of x2 and x2ScaleDim should be same, but x2's n-axis is: %ld and x2ScaleDim is: %ld.", nVal,
                x2ScaleNVal);
        return false;
    }
    if (x1ScaleLastDim != TWO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last dim of x1scale should be 2, but now it is: %ld.", x1ScaleLastDim);
        return false;
    }
    if (x2ScaleLastDim != TWO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last dim of x2scale should be 2, but now it is: %ld.", x2ScaleLastDim);
        return false;
    }
    return true;
}

// 校验输入Scaleshape
static bool CheckScaleShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                            const aclTensor *x2Scale, int64_t x1QuantMode, int64_t x2QuantMode, bool transposeX2)
{
    bool scaleShapeValid = true;
    if (static_cast<QuantModeType>(x1QuantMode) == QuantModeType::MX_QUANT &&
        static_cast<QuantModeType>(x2QuantMode) == QuantModeType::MX_QUANT) {
        OP_API_CHECK(!transposeX2, {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In the mx quantization scenario, x2 must be transposed.");
            return false;
        });
        scaleShapeValid = Check3DScaleShape(x2, x1Scale, x2Scale, transposeX2);
    } else if (static_cast<QuantModeType>(x1QuantMode) == QuantModeType::DYN_PERTOKEN_QUANT &&
               static_cast<QuantModeType>(x2QuantMode) == QuantModeType::PERCHANNEL_QUANT) {
        scaleShapeValid = Check1DScaleShape(x2, x2Scale, transposeX2);
    }
    return scaleShapeValid;
}

// 根据API定义，列出allto_all_quant_matmul量化输入X1所能支持的所有dtype(A2)
static const std::initializer_list<op::DataType> X1_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT4, op::DataType::DT_INT32};
// 根据API定义，列出allto_all_quant_matmul量化输入X2所能支持的所有dtype(A2)
static const std::initializer_list<op::DataType> X2_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT8, op::DataType::DT_INT4,
                                                                          op::DataType::DT_INT32};
// 根据API定义，列出allto_all_quant_matmul量化输入SCALE所能支持的所有dtype(A2)
static const std::initializer_list<op::DataType> SCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
// 根据API定义，列出allto_all_quant_matmul量化输出Output所能支持的所有dtype(A2)
static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                              op::DataType::DT_BF16};

// 校验所有输入的参数类型是否正确
static bool CheckAllDtypesValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                int64_t x1QuantMode, const aclTensor *x1ScaleOptional, const aclTensor *x2Scale,
                                const aclTensor *output, const aclTensor *alltoAllOutOptional)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, X1_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, X2_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2Scale, SCALE_DTYPE_SUPPORT_LIST, return false);
    if (x1QuantMode == static_cast<int64_t>(QuantModeType::DYN_PERTOKEN_QUANT) && x1ScaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(x1ScaleOptional, x1, return false);
    } else if (x1QuantMode == static_cast<int64_t>(QuantModeType::PERTOKEN_QUANT) && x1ScaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x1ScaleOptional, SCALE_DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUTPUT_DTYPE_SUPPORT_LIST, return false);
    if (alltoAllOutOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(x1, alltoAllOutOptional, return false);
    }
    return true;
}

// 950数据类型校验
// 量化模式下X支持的FP16数据类型（PerToken动态量化）(A5)
static const std::initializer_list<op::DataType> X_DTYPE_FP16_SUPPORT_LIST_A5 = {op::DataType::DT_FLOAT16,
                                                                                 op::DataType::DT_BF16};
// 量化模式下Output支持的数据类型（KC动态量化、MXQuant量化）(A5)
static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST_A5 = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
// 量化模式下X支持的FP8数据类型（PerChannel量化）(A5)
static const std::initializer_list<op::DataType> X_DTYPE_FP8_SUPPORT_LIST_A5 = {op::DataType::DT_FLOAT8_E4M3FN,
                                                                                op::DataType::DT_FLOAT8_E5M2};
// 量化模式下X支持的FP4和FP8数据类型（MXQuant量化）(A5)
static const std::initializer_list<op::DataType> X_DTYPE_FP4ANDFP8_SUPPORT_LIST_A5 = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT4_E2M1};
// 量化模式下Scale支持的FP32数据类型（PerChannel量化）(A5)
static const std::initializer_list<op::DataType> SCALE_DTYPE_FP32_SUPPORT_LIST_A5 = {op::DataType::DT_FLOAT};
// 量化模式下Bias支持的数据类型（KC动态量化、MXQuant量化）(A5)
static const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST_A5 = {op::DataType::DT_FLOAT};
// 量化模式下Scale支持的FP8数据类型（MXQuant量化）(A5)
static const std::initializer_list<op::DataType> SCALE_DTYPE_FP8_SUPPORT_LIST_A5 = {op::DataType::DT_FLOAT8_E8M0};

// KC动态量化场景下，校验所有输入的参数类型是否正确(A5)
static bool CheckKCDynQuantDtypesValidA5(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                         const aclTensor *x1ScaleOptional, const aclTensor *x2Scale,
                                         int64_t x1QuantDtype, const aclTensor *output,
                                         const aclTensor *alltoAllOutOptional)
{
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", x1, X_DTYPE_FP16_SUPPORT_LIST_A5, KCDYN_SCENE,
                                             return false);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", x2, X_DTYPE_FP8_SUPPORT_LIST_A5, KCDYN_SCENE,
                                             return false);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", x2Scale, SCALE_DTYPE_FP32_SUPPORT_LIST_A5,
                                             KCDYN_SCENE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", output, OUTPUT_DTYPE_SUPPORT_LIST_A5,
                                             KCDYN_SCENE, return false);
    if (x1QuantDtype != op::DataType::DT_FLOAT8_E4M3FN && x1QuantDtype != op::DataType::DT_FLOAT8_E5M2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "X1QuantDtype can be set only to 35(ACL_FLOAT8_E5M2) or 36(ACL_FLOAT8_E4M3FN), but the value is %ld",
                x1QuantDtype);
        return false;
    }
    if (biasOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", biasOptional, BIAS_DTYPE_SUPPORT_LIST_A5,
                                                 KCDYN_SCENE, return false);
    }
    if (x1ScaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", x1, x1ScaleOptional, KCDYN_SCENE,
                                              return false);
    }
    if (alltoAllOutOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", x1, alltoAllOutOptional, KCDYN_SCENE,
                                              return false);
    }
    return true;
}

// MX量化场景下，校验所有输入的参数类型是否正确(A5)
static bool CheckMXQuantDtypesValidA5(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                      const aclTensor *x1ScaleOptional, const aclTensor *x2Scale,
                                      const aclTensor *output, const aclTensor *alltoAllOutOptional)
{
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", x1, X_DTYPE_FP4ANDFP8_SUPPORT_LIST_A5,
                                             MX_SCENE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", x2, X_DTYPE_FP4ANDFP8_SUPPORT_LIST_A5,
                                             MX_SCENE, return false);
    if (x1->GetDataType() != x2->GetDataType() &&
        (x1->GetDataType() == op::DataType::DT_FLOAT4_E2M1 || x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In mxquant scenario, x1 and x2 must have same dtype when one is fp4_e2m1, but found x1: %s, x2: %s.",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", x1ScaleOptional,
                                             SCALE_DTYPE_FP8_SUPPORT_LIST_A5, MX_SCENE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", x2Scale, SCALE_DTYPE_FP8_SUPPORT_LIST_A5,
                                             MX_SCENE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", output, OUTPUT_DTYPE_SUPPORT_LIST_A5, MX_SCENE,
                                             return false);
    if (biasOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", biasOptional, BIAS_DTYPE_SUPPORT_LIST_A5,
                                                 MX_SCENE, return false);
    }
    if (alltoAllOutOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME_WITH_SCENARIO("aclnnAlltoAllQuantMatmul", x1, alltoAllOutOptional, MX_SCENE,
                                              return false);
    }
    return true;
}

// 校验所有场景的数据类型是否在各自的支持列表中
static bool CheckDtypesValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                             const aclTensor *x1ScaleOptional, const aclTensor *x2Scale, int64_t x1QuantMode,
                             int64_t x2QuantMode, int64_t x1QuantDtype, const aclTensor *output,
                             const aclTensor *alltoAllOutOptional)
{
    bool isAllDtypesValid = false;
    // 根据量化场景进入不同分支判断
    if (static_cast<QuantModeType>(x1QuantMode) == QuantModeType::DYN_PERTOKEN_QUANT &&
        static_cast<QuantModeType>(x2QuantMode) == QuantModeType::PERCHANNEL_QUANT) {
        isAllDtypesValid = CheckKCDynQuantDtypesValidA5(x1, x2, biasOptional, x1ScaleOptional, x2Scale, x1QuantDtype,
                                                        output, alltoAllOutOptional);
    } else if (static_cast<QuantModeType>(x1QuantMode) == QuantModeType::MX_QUANT &&
               static_cast<QuantModeType>(x2QuantMode) == QuantModeType::MX_QUANT) {
        isAllDtypesValid =
            CheckMXQuantDtypesValidA5(x1, x2, biasOptional, x1ScaleOptional, x2Scale, output, alltoAllOutOptional);
    } else {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnAlltoAllQuantMatmul", "x1QuantMode/x2QuantMode",
            (std::to_string(x1QuantMode) + "/" + std::to_string(x2QuantMode)).c_str(),
            "The value of x1QuantMode and x2QuantMode must be within the supported range.");
    }
    return isAllDtypesValid;
}

// 检查所有要用到的输入format是否为ND，不支持私有格式，如果内部不为ND格式，会打印warning日志，并将format转换为ND格式
static bool CheckFormat(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                        const aclTensor *x1ScaleOptional, const aclTensor *x2Scale, const aclTensor *output,
                        const aclTensor *alltoAllOutOptional)
{
    // 输入格式不支持私有格式
    if (IsPrivateFormat(x1->GetStorageFormat())) {
        OP_LOGE_WITH_INVALID_INPUT_FORMAT("aclnnAlltoAllQuantMatmul", "x1",
                                          op::ToString(x1->GetStorageFormat()).GetString(), "ND");
        return false;
    }
    if (IsPrivateFormat(x2->GetStorageFormat())) {
        OP_LOGE_WITH_INVALID_INPUT_FORMAT("aclnnAlltoAllQuantMatmul", "x2",
                                          op::ToString(x2->GetStorageFormat()).GetString(), "ND");
        return false;
    }
    if (biasOptional != nullptr) {
        if (IsPrivateFormat(biasOptional->GetStorageFormat())) {
            OP_LOGE_WITH_INVALID_INPUT_FORMAT("aclnnAlltoAllQuantMatmul", "bias",
                                              op::ToString(biasOptional->GetStorageFormat()).GetString(), "ND");
            return false;
        }
    }
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 && x1ScaleOptional != nullptr) {
        if (IsPrivateFormat(x1ScaleOptional->GetStorageFormat())) {
            OP_LOGE_WITH_INVALID_INPUT_FORMAT("aclnnAlltoAllQuantMatmul", "x1Scale",
                                              op::ToString(x1ScaleOptional->GetStorageFormat()).GetString(), "ND");
            return false;
        }
    }
    if (IsPrivateFormat(x2Scale->GetStorageFormat())) {
        OP_LOGE_WITH_INVALID_INPUT_FORMAT("aclnnAlltoAllQuantMatmul", "x2Scale",
                                          op::ToString(x2Scale->GetStorageFormat()).GetString(), "ND");
        return false;
    }
    if (IsPrivateFormat(output->GetStorageFormat())) {
        OP_LOGE_WITH_INVALID_INPUT_FORMAT("aclnnAlltoAllQuantMatmul", "output",
                                          op::ToString(output->GetStorageFormat()).GetString(), "ND");
        return false;
    }
    if (alltoAllOutOptional != nullptr) {
        if (IsPrivateFormat(alltoAllOutOptional->GetStorageFormat())) {
            OP_LOGE_WITH_INVALID_INPUT_FORMAT("aclnnAlltoAllQuantMatmul", "alltoAllOut",
                                              op::ToString(alltoAllOutOptional->GetStorageFormat()).GetString(), "ND");
            return false;
        }
    }
    return true;
}

// 兼容性处理，非ND格式转换为ND格式
static bool ReFormatNotND(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                          const aclTensor *x1ScaleOptional, const aclTensor *x2Scale, const aclTensor *output,
                          const aclTensor *alltoAllOutOptional)
{
    // 内部只处理ND格式，这里做reformat操作
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x1 origin format is %s.", op::ToString(x1->GetStorageFormat()).GetString());
        x1 = l0op::ReFormat(x1, op::Format::FORMAT_ND);
        CHECK_RET(x1 != nullptr, false);
    }
    if (x2->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x2 origin format is %s.", op::ToString(x2->GetStorageFormat()).GetString());
        x2 = l0op::ReFormat(x2, op::Format::FORMAT_ND);
        CHECK_RET(x2 != nullptr, false);
    }
    if (biasOptional != nullptr) {
        if (biasOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("bias origin format is %s.", op::ToString(biasOptional->GetStorageFormat()).GetString());
            biasOptional = l0op::ReFormat(biasOptional, op::Format::FORMAT_ND);
            CHECK_RET(biasOptional != nullptr, false);
        }
    }
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 && x1ScaleOptional != nullptr) {
        if (x1ScaleOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("x1ScaleOptional origin format is %s.",
                    op::ToString(x1ScaleOptional->GetStorageFormat()).GetString());
            x1ScaleOptional = l0op::ReFormat(x1ScaleOptional, op::Format::FORMAT_ND);
            CHECK_RET(x1ScaleOptional != nullptr, false);
        }
    }
    if (x2Scale->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x2Scale origin format is %s.", op::ToString(x2Scale->GetStorageFormat()).GetString());
        x2Scale = l0op::ReFormat(x2Scale, op::Format::FORMAT_ND);
        CHECK_RET(x2Scale != nullptr, false);
    }
    if (output->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("output origin format is %s.", op::ToString(output->GetStorageFormat()).GetString());
        output = l0op::ReFormat(output, op::Format::FORMAT_ND);
        CHECK_RET(output != nullptr, false);
    }
    if (alltoAllOutOptional != nullptr) {
        if (alltoAllOutOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("alltoallout origin format is %s.",
                    op::ToString(alltoAllOutOptional->GetStorageFormat()).GetString());
            alltoAllOutOptional = l0op::ReFormat(alltoAllOutOptional, op::Format::FORMAT_ND);
            CHECK_RET(alltoAllOutOptional != nullptr, false);
        }
    }
    return true;
}

static aclnnStatus CheckAndHandleParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                        const aclTensor *x1ScaleOptional, const aclTensor *x2Scale,
                                        const aclIntArray *alltoAllAxesOptional, const char *group, int64_t x1QuantMode,
                                        int64_t x2QuantMode, int64_t x1QuantDtype, bool transposeX1, bool transposeX2,
                                        const char *commMode, const aclTensor *output,
                                        const aclTensor *alltoAllOutOptional)
{
    // 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, biasOptional, x1ScaleOptional, x2Scale, output, x1QuantMode),
              ACLNN_ERR_PARAM_NULLPTR);
    // 检查空tensor
    CHECK_RET(CheckNotEmptyTensor(x1, x2, transposeX2), ACLNN_ERR_PARAM_INVALID);
    // 检查shape
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        CHECK_RET(
            CheckShapeAAMM("allto_all_quant_matmul", x1, x2, biasOptional, transposeX2, output, alltoAllOutOptional),
            ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckScaleShape(x1, x2, x1ScaleOptional, x2Scale, x1QuantMode, x2QuantMode, transposeX2),
                  ACLNN_ERR_PARAM_INVALID);
    }
    // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据芯片型号和api定义校验
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        CHECK_RET(CheckAllDtypesValid(x1, x2, biasOptional, x1QuantMode, x1ScaleOptional, x2Scale, output,
                                      alltoAllOutOptional),
                  ACLNN_ERR_PARAM_INVALID);
    } else if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        CHECK_RET(CheckDtypesValid(x1, x2, biasOptional, x1ScaleOptional, x2Scale, x1QuantMode, x2QuantMode,
                                   x1QuantDtype, output, alltoAllOutOptional),
                  ACLNN_ERR_PARAM_INVALID);
    }
    // 检查输入的数据格式是否为ND
    CHECK_RET(CheckFormat(x1, x2, biasOptional, x1ScaleOptional, x2Scale, output, alltoAllOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 兼容性处理非ND格式
    CHECK_RET(ReFormatNotND(x1, x2, biasOptional, x1ScaleOptional, x2Scale, output, alltoAllOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 检查alltoallAxes是否为空或者[-2,-1]
    CHECK_RET(CheckAlltoAllAxes("allto_all_quant_matmul", alltoAllAxesOptional, false), ACLNN_ERR_PARAM_INVALID);
    // 检查transposeX1是否合法, 目前不能为true
    CHECK_RET(CheckTransposeX1("allto_all_quant_matmul", transposeX1), ACLNN_ERR_PARAM_INVALID);
    // 检查group长度是否小于等于128
    CHECK_RET(CheckGroupLength("allto_all_quant_matmul", group), ACLNN_ERR_PARAM_INVALID);
    // 检查commMode非空指针
    if (commMode == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "commMode should not be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    // 如果所有检查都通过，且reformat也通过，输出参数检查成功
    OP_LOGD("aclnnAlltoAllQuantMatmul checkParams success");
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);

extern "C" void *NnopbaseGetUserHandle(void *executor);

// 非量化L2接口调用L0时需要设置较多默认值，通过InnerAlltoAllQuantMatmulGetWorkspaceSize完成默认值传参和调用L0层接口
extern "C" aclnnStatus InnerAlltoAllQuantMatmulGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional, const aclTensor *x1ScaleOptional,
    const aclTensor *x2Scale, const aclTensor *commScaleOptional, const aclTensor *x1OffsetOptional,
    const aclTensor *x2OffsetOptional, const char *group, const char *commMode, const aclIntArray *alltoAllAxesOptional,
    int64_t x1QuantMode, int64_t x2QuantMode, int64_t commQuantMode, int64_t commQuantDtype, int64_t x1QuantDtype,
    int64_t groupSize, bool transposeX1, bool transposeX2, const aclTensor *output, const aclTensor *all2AllOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // Inner接口部分入参类型和aclnn接口不一致，需要重新包装，同时Inner接口额外需要部分参数，按算子原型模板和实际业务逻辑生成
    char *str_group = const_cast<char *>(group);
    char *str_commMode = const_cast<char *>(commMode);
    int64_t worldSize = -1;                 // worldSize的默认值，实际值在建立通信域时获取
    int64_t yDtype = output->GetDataType(); // yDtype根据实际output的类型赋值，图模式需要该参数
    bool all2AllOutFlag = IsAll2AllOut("allto_all_quant_matmul", all2AllOutOptional);
    // 部分参数根据芯片型号不同，需要设置不同的默认值
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        // ACL和GE的datatype枚举值对undefined定义不同，inner接口进入到算子内部，需要使用GE枚举值
        commQuantDtype = op::DataType::DT_UNDEFINED;
    }

    uint8_t commModeEnum = 0;
    aclnnStatus checkCommModeRet = CheckAndHandleCommMode("allto_all_quant_matmul", group, commMode, commModeEnum);
    CHECK_RET(checkCommModeRet == ACLNN_SUCCESS, checkCommModeRet);

    OP_LOGD("AlltoAllQuantMatmul commmode is . %s", str_commMode);
    aclnnStatus ret = aclnnInnerAlltoAllMatmulGetWorkspaceSize(
        x1, x2, biasOptional, x1ScaleOptional, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
        str_group, worldSize, alltoAllAxesOptional, yDtype, x1QuantMode, x2QuantMode, commQuantMode, x1QuantDtype,
        commQuantDtype, transposeX1, transposeX2, groupSize, str_commMode, all2AllOutFlag, output, all2AllOutOptional,
        workspaceSize, executor);
    OP_LOGD("AlltoAllQuantMatmul, aclnnInnerGetWorkspaceSize ret %d.", ret);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "This is an error in launch aicore, aclnnAlltoAllQuantMatmulGetWorkspaceSize interface call failed.");
    }

    if (ret == ACLNN_SUCCESS && *executor != nullptr) {
        void *args = reinterpret_cast<void *>(static_cast<uint8_t>(commModeEnum));
        NnopbaseSetUserHandle(*executor, args);
    }

    return ret;
}

// 两段式接口
extern "C" aclnnStatus aclnnAlltoAllQuantMatmulBaseGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional, const aclTensor *x1ScaleOptional,
    const aclTensor *x2Scale, const aclTensor *commScaleOptional, const aclTensor *x1OffsetOptional,
    const aclTensor *x2OffsetOptional, const char *group, const char *commMode, const aclIntArray *alltoAllAxesOptional,
    int64_t x1QuantMode, int64_t x2QuantMode, int64_t commQuantMode, int64_t commQuantDtype, int64_t x1QuantDtype,
    int64_t groupSize, bool transposeX1, bool transposeX2, const aclTensor *output,
    const aclTensor *alltoAllOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 处理非连续Tensor，目前只有支持转置的x2涉及该处理
    aclnnStatus checkX2Ret = CheckX2Valid("allto_all_quant_matmul", x2);
    CHECK_RET(checkX2Ret == ACLNN_SUCCESS, checkX2Ret);                  // 先检查x2是否合法，避免非法操作
    auto transX2 = x2;                                                   // 复制一个x2
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) { // 只有当非连续时，才会涉及到转连续等情况
        bool notContiguous =
            IsTransposeLastTwoDims(x2); // notContiguous标识x2是否是非连续的，通常在pytorch经过.t()会导致x2非连续
        OP_LOGI("The notContiguous is: %d , and transposeX2 is: %d", notContiguous, transposeX2);
        if (notContiguous && transposeX2) { // 当非连续和转置同时生效时，判断为错误用法，直接报错
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 not contiguous, and set x2 transpose, it is error!");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (notContiguous) {
            transposeX2 = true;
            // 把非连续x2转成连续
            transX2 = TransX2Tensor(x2);
            CHECK_RET(transX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
            OP_LOGD("X2 is a non-contiguous tensor. The original dim0 is %ld, and dim1 is %ld. "
                    "After processing, transX2 dim0 is %ld, and dim1 is %ld.",
                    x2->GetViewShape().GetDim(0), x2->GetViewShape().GetDim(1), transX2->GetViewShape().GetDim(0),
                    transX2->GetViewShape().GetDim(1));
        }
    } else {
        // 对于 A2 和 A3，非连续则报错
        OP_API_CHECK(!MC2Aclnn::IsTensorContiguous(x2), {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The x2 must be contiguous, but it is non-contiguous.");
            return ACLNN_ERR_PARAM_INVALID;
        });
    }
    // 只在DAV_2201架构上对x1和x2进行int32到int4的转换预处理
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 && executor != nullptr) {
        auto uniqueExecutor = CREATE_EXECUTOR();
        InputPreProcessInt4(x1, transX2, alltoAllOutOptional, uniqueExecutor.get());
        uniqueExecutor.ReleaseTo(executor);
    }
    aclnnStatus retParam = CheckAndHandleParams(x1, transX2, biasOptional, x1ScaleOptional, x2Scale,
                                                alltoAllAxesOptional, group, x1QuantMode, x2QuantMode, x1QuantDtype,
                                                transposeX1, transposeX2, commMode, output, alltoAllOutOptional);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    uint8_t commModeEnum = 0;
    aclnnStatus checkCommModeRet = CheckAndHandleCommMode("allto_all_quant_matmul", group, commMode, commModeEnum);
    CHECK_RET(checkCommModeRet == ACLNN_SUCCESS, checkCommModeRet);

    aclnnStatus ret = InnerAlltoAllQuantMatmulGetWorkspaceSize(
        x1, transX2, biasOptional, x1ScaleOptional, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
        group, commMode, alltoAllAxesOptional, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, x1QuantDtype,
        groupSize, transposeX1, transposeX2, output, alltoAllOutOptional, workspaceSize, executor);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllQuantMatmulBase(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            void *arg = NnopbaseGetUserHandle(executor);
            uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
            uint8_t commMode = static_cast<uint8_t>(handleVal);
            if (commMode == Mc2Comm::COMM_MODE_CCU) {
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            } else {
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            }
        } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    aclnnStatus ret = aclnnInnerAlltoAllMatmul(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in launch aicore, aclnnInnerAlltoAllMatmul interface call failed.");
        return ret;
    }
    return ACLNN_SUCCESS;
}
