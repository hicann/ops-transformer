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
 * \file mx_quant_matmul_allto_all_tiling_base.cpp
 * \brief
 */
#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "mx_quant_matmul_allto_all_tiling_base.h"
#include "matmul_allto_all_fit_balance_tiling.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace MC2Tiling {

/**
 * @brief 工具函数：判断指定value是否存在于list中
 *
 * @param list: 有效值列表
 * @param value: 给定值
 * @return
 */
static bool IsContains(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::count(list.begin(), list.end(), value) > 0;
}

gert::StorageShape mxQuantStorageShape = gert::StorageShape();

/**
 * @brief 当前量化过程的准入条件
 * @return true
 */
bool MxQuantMatmulAllToAllTilingBase::IsCapable()
{
    int64_t x1QuantMode = 0;
    int64_t x2QuantMode = 0;
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    if (const int64_t *ptr = attrs->GetAttrPointer<int64_t>(ATTR_X1_QUANTMODE_INDEX)) {
        x1QuantMode = *ptr;
    }
    if (const int64_t *ptr = attrs->GetAttrPointer<int64_t>(ATTR_X2_QUANTMODE_INDEX)) {
        x2QuantMode = *ptr;
    }
    if (x1QuantMode == X1_QUANTMODE_VALUES && x2QuantMode == X2_QUANTMODE_VALUES) {
        OP_LOGI(opName_, "Start with MxQuantMatmulAlltoAll tiling.");
        return true;
    }
    OP_LOGI(opName_, "Skip MxQuantMatmulAlltoAll tiling when not MX_QUANT.");
    return false;
}

/**
 * @brief 校验输入信息是否合规:attr,Dtype,shape等，使用通用校验util中的check方法
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckOpInputInfo()
{
    OP_TILING_CHECK(
        MatmulAlltoAllTilingUtil::CheckAttrsInfo(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) != ge::GRAPH_SUCCESS,
        OP_LOGE(opName_, "Tiling check Attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckX2Transpose(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check x2transpose failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMxTensorFormat(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check format failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMxQuantTensorDataType(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "tiling check Dtype failed in mx quant matmul all to all."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMxQuantShapeInfo(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check mx quant shape info failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMxQuantMatrixMulShapes(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check mx quant matrix shape failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验参数的format::是否为私有格式
 *
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName: 算子名称
 * @return
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckMxTensorFormat(const gert::TilingContext *context,
                                                                     const char *opName)
{
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckTensorFormat(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check format failed."), return ge::GRAPH_FAILED);
    auto x1ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    OP_TILING_CHECK((x1ScaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1Scale"),
                    return ge::GRAPH_FAILED);
    ge::Format x1ScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x1ScaleTensorDesc->GetStorageFormat()));
    OP_TILING_CHECK(x1ScaleFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName, "x1Scale", Ops::Base::ToString(x1ScaleFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    auto x2ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((x2ScaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2Scale"),
                    return ge::GRAPH_FAILED);
    ge::Format x2ScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x2ScaleTensorDesc->GetStorageFormat()));
    OP_TILING_CHECK(x2ScaleFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName, "x2Scale", Ops::Base::ToString(x2ScaleFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 量化场景校验x2transpose是否一定为true
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckX2Transpose(const gert::TilingContext *context,
                                                                  const char *opName,
                                                                  const OpAttrIndexSchema &indexSchema)
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    const bool *isTransX2 = attrs->GetAttrPointer<bool>(indexSchema.x2Transpose);
    OP_TILING_CHECK(!(*isTransX2), OP_LOGE_WITH_INVALID_ATTR(opName, "transpose_x2", "false", "true"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 根据输入设置tiling参数
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::InitTilingContextParameters()
{
    MC2_CHECK_LOG_RET(
        opName_, MatmulAlltoAllTilingUtil::SetAttrsInfo(context_, opName_, contextInfo, MATMUL_ALLTOALL_INDEX_SCHEMA));
    MC2_CHECK_LOG_RET(opName_, SetMxDataTypeInfo(context_, opName_, contextInfo));
    MC2_CHECK_LOG_RET(opName_, MatmulAlltoAllTilingUtil::SetShapeInfo(context_, contextInfo));
    contextInfo.quantMode = QuantMode::MX_QUANT;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl参数；进行通算切分, 获取mm tiling等
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::DoOpTiling()
{
    // 输入参数的校验:Attrs,Dtype,Shape等
    MC2_CHECK_LOG_RET(opName_, CheckOpInputInfo());
    // 参数校验通过后赋值给全局上下文变量
    MC2_CHECK_LOG_RET(opName_, InitTilingContextParameters());
    // 进行通算切分
    MC2_CHECK_LOG_RET(opName_, TileCommAndCompute());
    // 调用量化Matmul的tiling方法进行切分
    MC2_CHECK_LOG_RET(opName_, DoMxQuantMMTiling());
    // hccl的tiling参数赋值处理
    MC2_CHECK_LOG_RET(opName_, SetHcclTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 量化场景校验参数的DType
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @param runInfo 过程信息
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckMxQuantTensorDataType(const gert::TilingContext *context,
                                                                            const char *opName)
{
    // 获取并校验输入张量描述符
    auto x1TensorDesc = context->GetInputDesc(INPUT_X1_INDEX);
    auto x2TensorDesc = context->GetInputDesc(INPUT_X2_INDEX);
    OP_TILING_CHECK((x1TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2"), return ge::GRAPH_FAILED);
    // 获取数据类型并校验一致性与范围
    ge::DataType x1Dtype = x1TensorDesc->GetDataType();
    ge::DataType x2Dtype = x2TensorDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_X_DTYPE_LIST, x1Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName, "x1", Ops::Base::ToString(x1Dtype).c_str(),
                        "The dtype of x1 must be in the mx-quant range (float8_e4m3fn/float8_e5m2/float4_e2m1)"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(MX_QUANT_X_DTYPE_LIST, x2Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName, "x2", Ops::Base::ToString(x2Dtype).c_str(),
                        "The dtype of x2 must be in the mx-quant range (float8_e4m3fn/float8_e5m2/float4_e2m1)"),
                    return ge::GRAPH_FAILED);
    if (x1Dtype == ge::DataType::DT_FLOAT4_E2M1 || x2Dtype == ge::DataType::DT_FLOAT4_E2M1) {
        isMxfp4_ = true;
        OP_TILING_CHECK(
            x1Dtype != x2Dtype,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                opName, "x1,x2", (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype)).c_str(),
                "The dtype of x1 and x2 must be the same in mxfp4 quant mode"),
            return ge::GRAPH_FAILED);
    }
    // 校验 bias 数据类型（如果存在）
    auto biasTensorDesc = context->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    if (biasTensorDesc != nullptr) {
        ge::DataType biasDtype = biasTensorDesc->GetDataType();
        OP_TILING_CHECK((biasDtype != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "bias", Ops::Base::ToString(biasDtype).c_str(),
                                                              "The dtype of bias must be float"),
                        return ge::GRAPH_FAILED);
    }
    // 校验 x1Scale和x2Scale 数据类型
    OP_TILING_CHECK(CheckMxQuantScaleDataType(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check scale dtype failed."), return ge::GRAPH_FAILED);
    // 校验输出张量数据类型
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "y"), return ge::GRAPH_FAILED);
    ge::DataType yDtype = yDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_Y_DTYPE_LIST, yDtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "y", Ops::Base::ToString(yDtype).c_str(),
                                                          "The dtype of y must be float16, bfloat16 or float"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 量化场景校验x1Scale和x2Scale参数的DType
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @param runInfo 过程信息
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckMxQuantScaleDataType(const gert::TilingContext *context,
                                                                           const char *opName)
{
    // 校验 scale 张量不为空（量化场景）
    auto x1ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    auto x2ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((x1ScaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1Scale"),
                    return ge::GRAPH_FAILED);
    ge::DataType x1scaleDtype = x1ScaleTensorDesc->GetDataType();
    OP_TILING_CHECK(
        (x1scaleDtype != ge::DataType::DT_FLOAT8_E8M0),
        OP_LOGE_FOR_INVALID_DTYPE(opName, "x1Scale", Ops::Base::ToString(x1scaleDtype).c_str(), "FLOAT8_E8M0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2ScaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2Scale"),
                    return ge::GRAPH_FAILED);
    ge::DataType x2scaleDtype = x2ScaleTensorDesc->GetDataType();
    OP_TILING_CHECK(
        (x2scaleDtype != ge::DataType::DT_FLOAT8_E8M0),
        OP_LOGE_FOR_INVALID_DTYPE(opName, "x2Scale", Ops::Base::ToString(x2scaleDtype).c_str(), "FLOAT8_E8M0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验量化tiling scale shape的Dim数量信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
static ge::graphStatus CheckMxScaleShapeDimensions(const gert::StorageShape *shape, const char *shapeName,
                                                   const char *opName)
{
    uint64_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((dimNum != 3),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(opName, shapeName, (std::to_string(dimNum) + "D").c_str(), "3D"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling shape的Dim数量信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
static ge::graphStatus CheckShapeDimensions(const gert::StorageShape *shape, const char *shapeName, const char *opName)
{
    uint64_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((dimNum != 2),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(opName, shapeName, (std::to_string(dimNum) + "D").c_str(), "2D"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling输入的bias的shape信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @param indexSchema 存放算子index差异的结构体
 * @return ge::graphStatus
 */
static ge::graphStatus CheckBiasShape(const gert::TilingContext *context, const char *opName,
                                      const OpAttrIndexSchema &indexSchema)
{
    const gert::StorageShape *biasShape = context->GetOptionalInputShape(INPUT_BIAS_INDEX);
    if (biasShape != nullptr) {
        uint64_t biasShapeDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            (biasShapeDimNum != 1),
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "bias", (std::to_string(biasShapeDimNum) + "D").c_str(), "1D"),
            return ge::GRAPH_FAILED);
        uint64_t biasDim0 = biasShape->GetStorageShape().GetDim(0);
        const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
        const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
        uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
        uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
        uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);

        bool x2TransFlag = false;
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const bool *isTransX2 = attrs->GetAttrPointer<bool>(indexSchema.x2Transpose);
        if (isTransX2) {
            x2TransFlag = *isTransX2;
        }
        uint64_t nAxis = (x2TransFlag) ? x2Dim0 : x2Dim1;
        OP_TILING_CHECK(
            (biasDim0 != nAxis),
            OP_LOGE_FOR_INVALID_VALUE(opName, "bias", std::to_string(biasDim0).c_str(), std::to_string(nAxis).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling输入Input的Dim范围信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @param indexSchema 存放算子index差异的结构体
 * @return ge::graphStatus
 */
static ge::graphStatus CheckShapeDimRange(const gert::TilingContext *context, const char *opName,
                                          const OpAttrIndexSchema &indexSchema)
{
    // attr非空在CheckAttrsInfo校验过
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    uint64_t x1Dim0 = x1Shape->GetStorageShape().GetDim(0);
    uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);
    // 获取n轴
    bool x2TransFlag = false;
    const bool *isTransX2 = attrs->GetAttrPointer<bool>(indexSchema.x2Transpose);
    if (isTransX2) {
        x2TransFlag = *isTransX2;
    }
    uint64_t nAxis = (x2TransFlag) ? x2Dim0 : x2Dim1;
    uint64_t kAxis = (x2TransFlag) ? x2Dim1 : x2Dim0;
    // 校验M,当前M为0的话，走公式化tiling切分实际是不支持的,后面可去除
    OP_TILING_CHECK(x1Dim0 == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "x1", std::to_string(x1Dim0).c_str(),
                                                          "The value of dim 0 (m) of x1 cannot be 0"),
                    return ge::GRAPH_FAILED);
    // 校验M不能大于int32的最大值
    OP_TILING_CHECK(x1Dim0 > MAX_INT32_VALUE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "x1", std::to_string(x1Dim0).c_str(),
                                                          "The value of dim 0 (m) of x1 must not exceed INT32_MAX"),
                    return ge::GRAPH_FAILED);
    // 校验K,K的范围应该在[1, 65535]
    OP_TILING_CHECK(kAxis > K_MAX_VALUE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "k", std::to_string(kAxis).c_str(),
                                                          "The value of k must not exceed 65535"),
                    return ge::GRAPH_FAILED);
    // 校验N, N不为空
    OP_TILING_CHECK(nAxis == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "x2", std::to_string(nAxis).c_str(),
                                                          "The value of N of x2 cannot be 0"),
                    return ge::GRAPH_FAILED);
    // 校验N不能大雨int32的
    OP_TILING_CHECK(nAxis > MAX_INT32_VALUE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "x2", std::to_string(nAxis).c_str(),
                                                          "The value of N of x2 must not exceed INT32_MAX"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验量化tiling输入的shape信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @param indexSchema 存放输入参数索引差别的结构体
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckMxQuantShapeInfo(const gert::TilingContext *context,
                                                                       const char *opName,
                                                                       const OpAttrIndexSchema &indexSchema)
{
    ge::graphStatus status;
    // 校验输入量化Input Shape是否为空
    status = CheckMxQuantInputShapesValid(context, opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验维度数目是否合法
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *x1ScaleShape = context->GetOptionalInputShape(INPUT_X1_SCALE_INDEX);
    const gert::StorageShape *x2ScaleShape = context->GetOptionalInputShape(INPUT_X2_SCALE_INDEX);
    status = CheckShapeDimensions(x1Shape, "mx quant input x1", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    status = CheckShapeDimensions(x2Shape, "mx quant input x2", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    status = CheckMxScaleShapeDimensions(x1ScaleShape, "mx quant input x1scale", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    status = CheckMxScaleShapeDimensions(x2ScaleShape, "mx quant input x2scale", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    // 校验输出
    const gert::StorageShape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "y"), return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(yShape, "mx quant output y", opName);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    // 校验bias的shape信息
    status = CheckBiasShape(context, opName, indexSchema);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    // 校验shape的dim范围
    status = CheckShapeDimRange(context, opName, indexSchema);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验量化tiling inputshape非空
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckMxQuantInputShapesValid(const gert::TilingContext *context,
                                                                              const char *opName)
{
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *x1ScaleShape = context->GetOptionalInputShape(INPUT_X1_SCALE_INDEX);
    const gert::StorageShape *x2ScaleShape = context->GetOptionalInputShape(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((x1Shape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2Shape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x1ScaleShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1Scale"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2ScaleShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2Scale"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验MX量化MatmulAlltoAll在不同转置情况下的x1,x2,output的shape关系,以及需要满足n/rankSize的整除关系
 * 需要满足 x1(BS,H1), x2(H2, H1) if trans else x2(H1, H2)
 * output(BS*rankSize, H2/rankSize)
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckMxQuantMatrixMulShapes(const gert::TilingContext *context,
                                                                             const char *opName)
{
    OP_TILING_CHECK(MatmulAllToAllTilingBase::Check2DMatrixMulShapes(context, opName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Mx quant tiling check x1 x2 and y shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckQuantGroupSize(context, opName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Mx quant tiling check groupSize failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMxQuantScaleShapes(context, opName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Mx quant tiling check scale shape failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验MXFP8量化scale的维度
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckMxQuantScaleShapes(const gert::TilingContext *context,
                                                                         const char *opName)
{
    Matrix2DShapes shapeInfo;
    MatmulAlltoAllTilingUtil::GetMatrix2DShapes(context, shapeInfo);
    const gert::StorageShape *x1ScaleShape = context->GetOptionalInputShape(INPUT_X1_SCALE_INDEX);
    const gert::StorageShape *x2ScaleShape = context->GetOptionalInputShape(INPUT_X2_SCALE_INDEX);
    uint64_t x1ScaleDim0 = x1ScaleShape->GetStorageShape().GetDim(0);
    uint64_t x1ScaleDim1 = x1ScaleShape->GetStorageShape().GetDim(1);
    uint64_t x1ScaleDim2 = x1ScaleShape->GetStorageShape().GetDim(2);
    uint64_t x2ScaleDim0 = x2ScaleShape->GetStorageShape().GetDim(0);
    uint64_t x2ScaleDim1 = x2ScaleShape->GetStorageShape().GetDim(1);
    uint64_t x2ScaleDim2 = x2ScaleShape->GetStorageShape().GetDim(2);
    uint64_t x1Dim1DivMxFp8Size =
        static_cast<uint64_t>(((static_cast<int64_t>(shapeInfo.x1Dim1) + MX_SCALE_OFFSET - 1) / MX_SCALE_OFFSET));
    uint64_t x2Dim0DivMxFp8Size =
        static_cast<uint64_t>(((static_cast<int64_t>(shapeInfo.x2Dim0) + MX_SCALE_OFFSET - 1) / MX_SCALE_OFFSET));
    uint64_t x2Dim1DivMxFp8Size =
        static_cast<uint64_t>(((static_cast<int64_t>(shapeInfo.x2Dim1) + MX_SCALE_OFFSET - 1) / MX_SCALE_OFFSET));
    // mxfp4场景中，k和ceil(k/32)需要为偶数
    if (isMxfp4_) {
        uint64_t x1Dim1DivMxFp4Size =
            static_cast<uint64_t>(((static_cast<int64_t>(shapeInfo.x1Dim1) + MX_GROUP_SIZE_K - 1) / MX_GROUP_SIZE_K));
        OP_TILING_CHECK((shapeInfo.x1Dim1 % 2 != 0) || (x1Dim1DivMxFp4Size % 2 != 0),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "x1", std::to_string(shapeInfo.x1Dim1).c_str(),
                                                              "in mxfp4 quant mode, k and ceil(k/32) must be even"),
                        return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(
        (x1ScaleDim0 != shapeInfo.x1Dim0) || (x1ScaleDim1 != x1Dim1DivMxFp8Size) || (x1ScaleDim2 != EVEN_ALIGN),
        OP_LOGE_FOR_INVALID_SHAPE(opName, "x1Scale",
                                  (std::string("(") + std::to_string(x1ScaleDim0) + "," + std::to_string(x1ScaleDim1) +
                                   "," + std::to_string(x1ScaleDim2) + ")")
                                      .c_str(),
                                  (std::string("(") + std::to_string(shapeInfo.x1Dim0) + "," +
                                   std::to_string(x1Dim1DivMxFp8Size) + "," + std::to_string(EVEN_ALIGN) + ")")
                                      .c_str()),
        return ge::GRAPH_FAILED);
    // Transposed Scenario
    OP_TILING_CHECK(
        (x2ScaleDim0 != shapeInfo.x2Dim0) || (x2ScaleDim1 != x2Dim1DivMxFp8Size) || (x2ScaleDim2 != EVEN_ALIGN),
        OP_LOGE_FOR_INVALID_SHAPE(opName_, "x2Scale",
                                  (std::string("(") + std::to_string(x2ScaleDim0) + "," + std::to_string(x2ScaleDim1) +
                                   "," + std::to_string(x2ScaleDim2) + ")")
                                      .c_str(),
                                  (std::string("(") + std::to_string(shapeInfo.x2Dim0) + "," +
                                   std::to_string(x2Dim1DivMxFp8Size) + "," + std::to_string(EVEN_ALIGN) + ")")
                                      .c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验MX量化GroupSize
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckQuantGroupSize(const gert::TilingContext *context,
                                                                     const char *opName)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    auto groupSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_SIZE_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "groupSize"), return false);
    uint64_t groupSizeK = static_cast<uint64_t>(*groupSizePtr) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    mc2tiling::Mc2MatmulShapeInfo shapeInfo = {context->GetInputShape(INPUT_X1_INDEX),
                                               context->GetInputShape(INPUT_X2_INDEX),
                                               context->GetOptionalInputShape(INPUT_X1_SCALE_INDEX),
                                               context->GetOptionalInputShape(INPUT_X2_SCALE_INDEX),
                                               true,
                                               *attrs->GetAttrPointer<bool>(MATMUL_ALLTOALL_INDEX_SCHEMA.x2Transpose),
                                               opName};

    OP_TILING_CHECK(!mc2tiling::Mc2TilingUtils::InferGroupSize(shapeInfo, groupSizeM, groupSizeN, groupSizeK),
                    OP_LOGE(opName_, "Failed to execute inferGroupSize."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (groupSizeM != MX_GROUP_SIZE_M) || (groupSizeN != MX_GROUP_SIZE_N) || (groupSizeK != MX_GROUP_SIZE_K),
        OP_LOGE_WITH_INVALID_ATTR(opName_, "groupSize",
                                  (std::string("[groupSizeM=") + std::to_string(groupSizeM) + ", groupSizeN=" +
                                   std::to_string(groupSizeN) + ", groupSizeK=" + std::to_string(groupSizeK) + "]")
                                      .c_str(),
                                  "[groupSizeM=1, groupSizeN=1, groupSizeK=32]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置算子的数据类型信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @param contextInfo 存储了tiling的过程信息
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::SetMxDataTypeInfo(const gert::TilingContext *context,
                                                                   const char *opName, TilingContextInfo &contextInfo)
{
    const gert::StorageShape *matrixBias = context->GetOptionalInputShape(INPUT_BIAS_INDEX);
    ge::DataType aType = context->GetInputDesc(INPUT_X1_INDEX)->GetDataType();
    ge::DataType bType = context->GetInputDesc(INPUT_X2_INDEX)->GetDataType();
    ge::DataType cType = context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    ge::DataType biasType;
    bool isBias = true;
    if (matrixBias == nullptr) {
        isBias = false;
        biasType = cType;
    } else {
        biasType = context->GetOptionalInputDesc(INPUT_BIAS_INDEX)->GetDataType();
    }

    contextInfo.args_.outputDtypeSize = mc2tiling::GetDataTypeSize(opName, cType);
    contextInfo.args_.inputDtypeSize = mc2tiling::GetDataTypeSize(opName, aType);
    contextInfo.args_.isBias = isBias;
    contextInfo.args_.geCType = cType;
    contextInfo.args_.geBiasType = biasType;
    contextInfo.args_.geAType = aType;
    contextInfo.args_.geBType = bType;
    contextInfo.args_.cType = mc2tiling::ConvertGeTypeToMmType(opName, cType);
    contextInfo.args_.aType = mc2tiling::ConvertGeTypeToMmType(opName, aType);
    contextInfo.args_.bType = mc2tiling::ConvertGeTypeToMmType(opName, bType);
    contextInfo.args_.biasType = mc2tiling::ConvertGeTypeToMmType(opName, biasType);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl的config,进行hccl对应的通信任务设置
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::SetHcclTiling()
{
    OP_TILING_CHECK(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geCType) ==
                        mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
                    OP_LOGE(opName_, "Cannot find HcclDataType according to ge datatype = %d.",
                            static_cast<int32_t>(contextInfo.args_.geCType)),
                    return ge::GRAPH_FAILED;);
    Mc2CcTilingConfigBuilder matmulAllToAllBuilder =
        Mc2CcTilingConfigBuilder::create(contextInfo.group, mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL,
                                         Mc2CcTilingConfigBuilder::AlgConfigType::ALL_TO_ALL);

    // 获取commMode
    uint8_t engineType = 0;
    if (MatmulAlltoAllTilingUtil::GetAndConvertCommMode(context_, opName_, contextInfo, MATMUL_ALLTOALL_INDEX_SCHEMA,
                                                        engineType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    AscendC::Mc2CcTilingConfig matmulAllToAllTilingConfig =
        matmulAllToAllBuilder
            .withReduceType(opName_, mc2tiling::HcclReduceOp::HCCL_REDUCE_SUM, contextInfo.args_.geCType,
                            contextInfo.args_.geCType)
            .withCommEngine(engineType)
            .build();
    if (!matmulAllToAllBuilder.isSuccess()) {
        OP_LOGE(opName_, "Mx quant matmul allto all build hccl tiling config failed: %s",
                matmulAllToAllBuilder.errorMsg().c_str());
        return ge::GRAPH_FAILED;
    }
    matmulAllToAllTilingConfig.GetTiling(localTilingData_.mc2InitTiling);
    matmulAllToAllTilingConfig.GetTiling(localTilingData_.mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 进行通算切分之后单个块的MM Tiling
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::DoMxQuantMMTiling()
{
    // 设置MM切前信息
    mmMvalueLen_ = inferredInfo.tileM;
    MxQuantMatmulAlltoAllHelper mmTile(*this, localTilingData_.mc2QuantBmmV3TileTilingData, mmMvalueLen_);
    MC2_CHECK_LOG_RET(opName_, mmTile.DoTiling());
    if (inferredInfo.tailCnt == 0) {
        return ge::GRAPH_SUCCESS;
    }
    mmMvalueLen_ = inferredInfo.tailM;
    MxQuantMatmulAlltoAllHelper mmTail(*this, localTilingData_.mc2QuantBmmV3TailTilingData, mmMvalueLen_);
    MC2_CHECK_LOG_RET(opName_, mmTail.DoTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 重写获取MM index的信息
 * 由于本算子的context和MM不一样，需要重写获取MM
 * index的一些信息，把我们的context传给Matmul，来达到可以调用MM策略的目的。
 * @return ge::graphStatus
 */
const gert::Shape MxQuantMatmulAlltoAllHelper::GetX1Shape(const size_t index)
{
    (void)index;
    return gert::Shape({static_cast<int64_t>(mmLen_), static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue)});
}
const gert::Shape MxQuantMatmulAlltoAllHelper::GetX2Shape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.contextInfo.args_.isBTrans) {
        return gert::Shape({static_cast<int64_t>(tilingProcesser_.contextInfo.args_.nValue),
                            static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue)});
    }
    return gert::Shape({static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue),
                        static_cast<int64_t>(tilingProcesser_.contextInfo.args_.nValue)});
}

const gert::Shape &MxQuantMatmulAlltoAllHelper::GetScaleShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(INPUT_X2_SCALE_INDEX))->GetStorageShape();
}

const gert::StorageShape *MxQuantMatmulAlltoAllHelper::GetOffsetShape(const size_t index)
{
    (void)index;
    return (gert::StorageShape *)nullptr;
}

const gert::StorageShape *MxQuantMatmulAlltoAllHelper::GetPertokenShape(const size_t index)
{
    (void)index;
    mxQuantStorageShape = gert::StorageShape({static_cast<int64_t>(mmLen_)}, {static_cast<int64_t>(mmLen_)});
    return &mxQuantStorageShape;
}

const gert::StorageShape *MxQuantMatmulAlltoAllHelper::GetBiasShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(INPUT_BIAS_INDEX));
}

ge::graphStatus MxQuantMatmulAlltoAllHelper::GetShapeAttrsInfo()
{
    OP_LOGD(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
    auto &&tilingArgs = tilingProcesser_.contextInfo.args_;
    inputParams_.opName = tilingProcesser_.opName_;
    inputParams_.transB = tilingArgs.isBTrans;
    inputParams_.hasBias = tilingArgs.isBias;
    inputParams_.transA = false;
    inputParams_.aDtype = tilingArgs.geAType;
    inputParams_.bDtype = tilingArgs.geBType;
    inputParams_.libApiWorkSpaceSize = tilingProcesser_.libApiWorkSpaceSize_;
    int64_t yDType = *context_->GetAttrs()->GetAttrPointer<int64_t>(ATTR_Y_DTYPE_INDEX);
    auto scaleTensorDesc = context_->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((scaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(tilingProcesser_.opName_, "scale tensor"),
                    return ge::GRAPH_FAILED);
    inputParams_.scaleDtype = scaleTensorDesc->GetDataType();
    auto perTokenScaleTensorDesc = context_->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    OP_TILING_CHECK((perTokenScaleTensorDesc == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(tilingProcesser_.opName_, "perToken scale tensor"),
                    return ge::GRAPH_FAILED);
    inputParams_.perTokenScaleDtype = perTokenScaleTensorDesc->GetDataType();
    inputParams_.outDtype = static_cast<int64_t>(yDType);
    inputParams_.cDtype = static_cast<ge::DataType>(yDType);
    OP_LOGD(tilingProcesser_.opName_, "yDType is %ld", inputParams_.outDtype);
    inputParams_.biasDtype = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;
    if ((scaleTensorDesc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0) &&
        (perTokenScaleTensorDesc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0)) {
        inputParams_.groupSizeK = MX_SCALE_OFFSET;
    }
    MC2_CHECK_TRUE_RET(tilingProcesser_.opName_, AnalyzeInputs());
    PrintTilingInputParam(inputParams_);
    return ge::GRAPH_SUCCESS;
}

void MxQuantMatmulAlltoAllHelper::PrintTilingInputParam(Mc2QuantBatchMatmulInfo &quantMatmulInfo)
{
    OP_LOGD(tilingProcesser_.opName_,
            "aDtype_ %d bDtype_ %d cDtype_ %d biasDtype_ %d outDtype %ld scaleDtype %d perTokenScaleDtype %d",
            static_cast<int32_t>(quantMatmulInfo.aDtype), static_cast<int32_t>(quantMatmulInfo.bDtype),
            static_cast<int32_t>(quantMatmulInfo.cDtype), static_cast<int32_t>(quantMatmulInfo.biasDtype),
            quantMatmulInfo.outDtype, static_cast<int32_t>(quantMatmulInfo.scaleDtype),
            static_cast<int32_t>(quantMatmulInfo.perTokenScaleDtype));
    OP_LOGD(tilingProcesser_.opName_, "mSize_ %ld kSize_ %ld nSize_ %ld libApiWorkSpaceSize %u", quantMatmulInfo.mSize,
            quantMatmulInfo.kSize, quantMatmulInfo.nSize, quantMatmulInfo.libApiWorkSpaceSize);
    OP_LOGD(tilingProcesser_.opName_, "Check isPerTensor=%d, isDoubleScale=%d.",
            static_cast<int32_t>(quantMatmulInfo.isPerTensor), static_cast<int32_t>(quantMatmulInfo.isDoubleScale));
    OP_LOGD(tilingProcesser_.opName_, "Check groupSizeM=%d, groupSizeK=%d, groupSizeN=%d",
            static_cast<int32_t>(quantMatmulInfo.groupSizeM), static_cast<int32_t>(quantMatmulInfo.groupSizeK),
            static_cast<int32_t>(quantMatmulInfo.groupSizeN));
}

ge::graphStatus MxQuantMatmulAlltoAllHelper::DoLibApiTiling()
{
    MC2_CHECK_LOG_RET(tilingProcesser_.opName_, Mc2AdaptiveSlidingWindowTiling::DoLibApiTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 重写友元类PostTiling方法
 * PostTiling主要做的是拷贝tilingdata的活，但是本算子拷贝tilingdata是在大结构体中拷贝，不需要在此处拷贝。
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAlltoAllHelper::PostTiling()
{
    tilingProcesser_.workspaceSize_ = std::max(tilingProcesser_.workspaceSize_, workspaceSize_);
    OP_LOGD(tilingProcesser_.opName_, "set mm workspace size %lu to mc2", tilingProcesser_.workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 构造函数，创建一个MxQuantMatmulAlltoAllHelper对象
 *
 * @param context
 */
MxQuantMatmulAlltoAllHelper::MxQuantMatmulAlltoAllHelper(
    MxQuantMatmulAllToAllTilingBase &mxQuantMatmulAllToAllTilingBase,
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &data, uint64_t &mmMvalueLen)
    : Mc2AdaptiveSlidingWindowTiling(mxQuantMatmulAllToAllTilingBase.context_, &data),
      tilingProcesser_(mxQuantMatmulAllToAllTilingBase),
      mmLen_(mmMvalueLen)
{}

/**
 * @brief 打印量化matmul tiling的信息
 *
 * @param opName
 * @param tiling
 */
void MxQuantMatmulAllToAllTilingBase::PrintMxQuantMMV3TilingData(
    const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling)
{
    PrintTCubeTilingData(opName, tiling.matmulTiling);
    PrintExtendMatmulTiling(opName, tiling);
}

/**
 * @brief 打印执行过程中的matmul tiling信息
 *
 * @param opName
 * @param tiling
 */
void MxQuantMatmulAllToAllTilingBase::PrintExtendMatmulTiling(const std::string &opName,
                                                              DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling)
{
    OP_LOGD(opName, "QuantBmmV3Params.batchA=%u.", tiling.params.batchA);
    OP_LOGD(opName, "QuantBmmV3Params.batchA1=%u.", tiling.params.batchA1);
    OP_LOGD(opName, "QuantBmmV3Params.batchA2=%u.", tiling.params.batchA2);
    OP_LOGD(opName, "QuantBmmV3Params.batchA3=%u.", tiling.params.batchA3);
    OP_LOGD(opName, "QuantBmmV3Params.batchA4=%u.", tiling.params.batchA4);
    OP_LOGD(opName, "QuantBmmV3Params.batchB=%u.", tiling.params.batchB);
    OP_LOGD(opName, "QuantBmmV3Params.batchB1=%u.", tiling.params.batchB1);
    OP_LOGD(opName, "QuantBmmV3Params.batchB2=%u.", tiling.params.batchB2);
    OP_LOGD(opName, "QuantBmmV3Params.batchB3=%u.", tiling.params.batchB3);
    OP_LOGD(opName, "QuantBmmV3Params.batchB4=%u.", tiling.params.batchB4);
    OP_LOGD(opName, "QuantBmmV3Params.batchC=%u.", tiling.params.batchC);
    OP_LOGD(opName, "QuantBmmV3Params.batchC1=%u.", tiling.params.batchC1);
    OP_LOGD(opName, "QuantBmmV3Params.batchC2=%u.", tiling.params.batchC2);
    OP_LOGD(opName, "QuantBmmV3Params.batchC3=%u.", tiling.params.batchC3);
    OP_LOGD(opName, "QuantBmmV3Params.batchC4=%u.", tiling.params.batchC4);
    OP_LOGD(opName, "QuantBmmV3Params.singleCoreBatch=%u.", tiling.params.singleCoreBatch);
    OP_LOGD(opName, "QuantBmmV3Params.isPerTensor=%u.", tiling.params.isPerTensor);
    OP_LOGD(opName, "QuantBmmV3Params.isPertoken=%u.", tiling.params.isPertoken);
    OP_LOGD(opName, "QuantBmmV3Params.isDoubleScale=%u.", tiling.params.isDoubleScale);
    OP_LOGD(opName, "QuantBmmV3Params.biasThreeDim=%u.", tiling.params.biasThreeDim);
    OP_LOGD(opName, "QuantBmmV3Params.needUbBuffer=%u.", tiling.params.needUbBuffer);
    OP_LOGD(opName, "QuantBmmV3Params.ubCalcM=%u.", tiling.params.ubCalcM);
    OP_LOGD(opName, "QuantBmmV3Params.ubCalcN=%u.", tiling.params.ubCalcN);
    OP_LOGD(opName, "QuantBmmV3Params.realSingleCoreM=%u.", tiling.params.realSingleCoreM);
    OP_LOGD(opName, "QuantBmmV3Params.realSingleCoreN=%u.", tiling.params.realSingleCoreN);
    OP_LOGD(opName, "QuantBmmV3Params.biasDtype=%u.", tiling.params.biasDtype);
    OP_LOGD(opName, "QuantBmmV3Params.ubSize=%u.", tiling.params.ubSize);
    OP_LOGD(opName, "QuantBmmV3Params.isMClash=%u.", tiling.params.isMClash);
    OP_LOGD(opName, "QuantBmmV3Params.isNClash=%u.", tiling.params.isNClash);
    OP_LOGD(opName, "QuantBmmV3Params.groupSizeM=%u.", tiling.params.groupSizeM);
    OP_LOGD(opName, "QuantBmmV3Params.groupSizeK=%u.", tiling.params.groupSizeK);
    OP_LOGD(opName, "QuantBmmV3Params.groupSizeN=%u.", tiling.params.groupSizeN);
    OP_LOGD(opName, "AdaptiveSlidingWin.mTailTile=%u.", tiling.adaptiveSlidingWin.mTailTile);
    OP_LOGD(opName, "AdaptiveSlidingWin.nTailTile=%u.", tiling.adaptiveSlidingWin.nTailTile);
    OP_LOGD(opName, "TileL2cacheTiling.mTileCntL2=%u.", tiling.tileL2cacheTiling.mTileCntL2);
    OP_LOGD(opName, "TileL2cacheTiling.nTileCntL2=%u.", tiling.tileL2cacheTiling.nTileCntL2);
    OP_LOGD(opName, "TileL2cacheTiling.mTileBlock=%u.", tiling.tileL2cacheTiling.mTileBlock);
    OP_LOGD(opName, "TileL2cacheTiling.nTileBlock=%u.", tiling.tileL2cacheTiling.nTileBlock);
    OP_LOGD(opName, "TileL2cacheTiling.calOrder=%u.", tiling.tileL2cacheTiling.calOrder);
    OP_LOGD(opName, "TileL2cacheTiling.isBasicTiling=%u.", tiling.tileL2cacheTiling.isBasicTiling);
}

/**
 * @brief 打印tilingInfo信息
 *
 * @param opName
 * @param tilingInfo
 */
void MxQuantMatmulAllToAllTilingBase::PrintMxQuantMatmulAlltoAllTilingInfo(const std::string &opName,
                                                                           MatmulAlltoAllTilingInfo &tilingInfo)
{
    OP_LOGD(opName, "tilingInfo.rankDim: %u", tilingInfo.rankDim);
    OP_LOGD(opName, "tilingInfo.tileCnt: %u", tilingInfo.tileCnt);
    OP_LOGD(opName, "tilingInfo.tileM: %u", tilingInfo.tileM);
    OP_LOGD(opName, "tilingInfo.tailCnt: %u", tilingInfo.tailCnt);
    OP_LOGD(opName, "tilingInfo.tailM: %u", tilingInfo.tailM);
    OP_LOGD(opName, "tilingInfo.rankM: %u", tilingInfo.rankM);
    OP_LOGD(opName, "tilingInfo.rankK: %u", tilingInfo.rankK);
    OP_LOGD(opName, "tilingInfo.rankN: %u", tilingInfo.rankN);
    OP_LOGD(opName, "tilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "tilingInfo.permuteLen: %lu", tilingInfo.permuteLen);
    OP_LOGD(opName, "tilingInfo.mmResultLen: %lu", tilingInfo.mmResultLen);
    OP_LOGD(opName, "tilingInfo.aicCoreNum: %u", tilingInfo.aicCoreNum);
    OP_LOGD(opName, "tilingInfo.hcclDataType: %lu", tilingInfo.hcclDataType);
}

/**
 * @brief 打印传递给kernel的tilingData
 *
 * @param outTilingData tilingData参数
 */
void MxQuantMatmulAllToAllTilingBase::PrintMxQuantMatmulAlltoAllTilingData(QuantMatmulAlltoAllTilingData &outTilingData)
{
    PrintMxQuantMatmulAlltoAllTilingInfo(opName_, outTilingData.quantMatmulAlltoAllTilingInfo);
    PrintMxQuantMMV3TilingData(opName_, outTilingData.mc2QuantBmmV3TileTilingData);
    if (outTilingData.quantMatmulAlltoAllTilingInfo.tailCnt == 0) {
        return;
    }
    OP_LOGD(opName_, "MxQuantMatmulAlltoall has tail");
    PrintMxQuantMMV3TilingData(opName_, outTilingData.mc2QuantBmmV3TailTilingData);
}

/**
 * @brief 保存量化tiling数据到context
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::PostTiling()
{
    context_->SetScheduleMode(1);
    SetTilingInfo(localTilingData_.quantMatmulAlltoAllTilingInfo);
    QuantMatmulAlltoAllTilingData *outTilingData = context_->GetTilingData<QuantMatmulAlltoAllTilingData>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "Failed to get tiling data from context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tilingBufCap < sizeof(localTilingData_)),
                    OP_LOGE(opName_, "TilingBuffer capacity too small, capacity = %zu, need = %zu.", tilingBufCap,
                            sizeof(localTilingData_)),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(outTilingData, tilingBufCap, &localTilingData_, sizeof(localTilingData_));
    if (ret != EOK) {
        OP_LOGE(opName_, "MatmulAlltoAll postTiling: memcpy_s tiling data failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.", sizeof(QuantMatmulAlltoAllTilingData),
            context_->GetRawTilingData()->GetCapacity());
    context_->SetBlockDim(contextInfo.args_.aicCoreNum);
    context_->GetRawTilingData()->SetDataSize(sizeof(QuantMatmulAlltoAllTilingData));
    PrintMxQuantMatmulAlltoAllTilingData(*outTilingData);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingInfo结构体
 *
 * @param tilingInfo 目标结构体
 */
void MxQuantMatmulAllToAllTilingBase::SetTilingInfo(MatmulAlltoAllTilingInfo &tilingInfo) const
{
    // 基本字段拷贝
    tilingInfo.tileCnt = inferredInfo.tileCnt;
    tilingInfo.tileM = inferredInfo.tileM;
    tilingInfo.tailCnt = inferredInfo.tailCnt;
    tilingInfo.tailM = inferredInfo.tailM;
    tilingInfo.rankM = contextInfo.args_.mValue;
    tilingInfo.rankN = contextInfo.args_.nValue;
    tilingInfo.rankK = contextInfo.args_.kValue;
    tilingInfo.biasLen = inferredInfo.biasLen;
    tilingInfo.mmResultLen = inferredInfo.mmResultLen;
    tilingInfo.permuteLen = inferredInfo.permuteLen;
    tilingInfo.rankDim = contextInfo.args_.rankDim;
    tilingInfo.aicCoreNum = contextInfo.args_.aicCoreNum;
    tilingInfo.hcclDataType =
        (static_cast<uint64_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geCType))); // hccl数据类型
}

/**
 * @brief 获取对应的tilingKey
 * 使用QUANT_MODE来区分tilingKey,此处的QUANT_MODE指的是x1,x2的QUANT模式组合，以x1为pertoken量化(K)，x2为perchannel量化(C)
 * 为例子，K-C量化就代表一种组合
 *
 * @return uint64_t tilingKey结果
 */
uint64_t MxQuantMatmulAllToAllTilingBase::GetTilingKey() const
{
    // 按照量化组合模式，是否转置，bias数据类型进行展开
    bool x2TransposeFlag = contextInfo.args_.isBTrans ? true : false;
    uint32_t biasDType = DTYPE_BIAS_FP32;

    uint8_t engineType = 0;
    if (MatmulAlltoAllTilingUtil::GetAndConvertCommMode(context_, opName_, contextInfo, MATMUL_ALLTOALL_INDEX_SCHEMA,
                                                        engineType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint8_t commMode = (engineType == mc2tiling::A5_CCU_ENGINE) ? COMMMODE_CCU : COMMMODE_AICPU;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(MX_QUANT_MODE, x2TransposeFlag, biasDType, commMode, ADDBIAS_OFF);
    OP_LOGD(opName_,
            "MXQUANTMODE,X2TRANSPOSE,DTYPEBIAS,COMMMODE,ADDBIASSPLITMODE: [%d,%d,%d,%d,0], TilingKey is [%lu].",
            MX_QUANT_MODE, x2TransposeFlag, biasDType, commMode, tilingKey);
    return tilingKey;
}

/**
 * @brief 获取tiling切分结果（arch35覆盖）
 *
 * @return CutResult
 */
CutResult MxQuantMatmulAllToAllTilingBase::GetTilingResult()
{
    uint8_t engineType = 0;
    if (MatmulAlltoAllTilingUtil::GetAndConvertCommMode(context_, opName_, contextInfo, MATMUL_ALLTOALL_INDEX_SCHEMA,
                                                        engineType) != ge::GRAPH_SUCCESS) {
        OP_LOGD(opName_, "GetTilingResult: failed to get commMode, default to AICPU.");
    }

    uint8_t commMode = (engineType == mc2tiling::A5_CCU_ENGINE) ? Mc2Comm::COMM_MODE_CCU : Mc2Comm::COMM_MODE_AICPU;
    return GetArch35TilingResult(contextInfo.args_, KernelType::ALL_TO_ALL, SocVersion::SOC950, npuArch_,
                                 QuantMode::MX_QUANT, commMode);
}

/**
 * @brief 构造函数，创建一个MxQuantMatmulAllToAllTilingBase对象
 *
 * @param context
 */
MxQuantMatmulAllToAllTilingBase::MxQuantMatmulAllToAllTilingBase(gert::TilingContext *context)
    : MatmulAllToAllTilingBase(context)
{}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(MatmulAlltoAll, MxQuantMatmulAllToAllTilingBase,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 2);

} // namespace MC2Tiling
