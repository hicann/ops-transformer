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
 * \file matmul_allto_all_util_tiling.cpp
 * \brief
 */
#include "matmul_allto_all_util_tiling.h"

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

/**
 * @brief 获取和校验rank数的函数，
 *
 * @param context 上下文信息
 * @param opName 算子名称
 * @param group 通信域名称
 * @param rankDim rank数
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTilingUtil::GetAndValidateRankSize(const gert::TilingContext *context, const char *opName,
                                                                 const char *group, int64_t &rankDim)
{
    // attrs非空在checkAttrs校验过
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *worldSize = attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX);
    if ((worldSize == nullptr) || (*worldSize == RANK_DEFAULT_NUM)) {
        OP_TILING_CHECK(!mc2tiling::GetRankSize(opName, group, rankDim), OP_LOGE(opName, "GetRankSize failed."),
                        return ge::GRAPH_FAILED);
    } else {
        rankDim = *worldSize;
    }
    OP_TILING_CHECK(SUPPORT_RANK_SIZE.find(rankDim) == SUPPORT_RANK_SIZE.end(),
                    OP_LOGE_WITH_INVALID_ATTR(opName, "world_size", std::to_string(rankDim).c_str(), "2/4/8/16"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 从参数中获取通信引擎
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @param contextInfo 存储了tiling的过程信息
 * @param commMode 返回通信引擎模式
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTilingUtil::GetAndConvertCommMode(const gert::TilingContext *context, const char *opName,
                                                                const TilingContextInfo &contextInfo,
                                                                const OpAttrIndexSchema &indexSchema, uint8_t &commMode)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(opName, "Failed to get attrs."), return ge::GRAPH_FAILED);
    const char *commModeStr = attrs->GetAttrPointer<char>(indexSchema.commMode);
    OP_TILING_CHECK(commModeStr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "comm_mode attr"),
                    return ge::GRAPH_FAILED);

    const size_t maxLength = 7UL;
    if (strncmp(commModeStr, "ai_cpu", maxLength) == 0) {
        commMode = mc2tiling::A5_AICPU_TS_ENGINE;
    } else if (strncmp(commModeStr, "ccu", maxLength) == 0) {
        commMode = mc2tiling::A5_CCU_ENGINE;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "comm_mode", commModeStr,
                                              "The value of comm_mode must be 'ai_cpu' or 'ccu'");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置二维输入输出的shape信息, 仅支持x1,x2,output都是二维的情况
 *
 * @param context 上下文信息
 * @param shapes shapeinfo,包含x1,x2和output
 */
void MatmulAlltoAllTilingUtil::GetMatrix2DShapes(const gert::TilingContext *context, Matrix2DShapes &shapes)
{
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);

    shapes.x1Dim0 = x1Shape->GetStorageShape().GetDim(0);
    shapes.x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
    shapes.x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    shapes.x2Dim1 = x2Shape->GetStorageShape().GetDim(1);
    shapes.yDim0 = yShape->GetStorageShape().GetDim(0);
    shapes.yDim1 = yShape->GetStorageShape().GetDim(1);
}

/**
 * @brief 校验attrs信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @param indexSchema 存放输入参数索引差别的结构体
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTilingUtil::CheckAttrsInfo(const gert::TilingContext *context, const char *opName,
                                                         const OpAttrIndexSchema &indexSchema)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(opName, "Failed to get attrs."), return ge::GRAPH_FAILED);

    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    const char *commMode = attrs->GetAttrPointer<char>(indexSchema.commMode);
    // 判断为空或者空字符串
    OP_TILING_CHECK(group == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "group"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(group[0] == '\0', OP_LOGE_WITH_INVALID_INPUT(opName, "group"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "comm_mode attr"), return ge::GRAPH_FAILED);
    // 判断group是否超过127
    size_t groupLen = strlen(group);
    OP_TILING_CHECK(groupLen > MAX_GROUP_NAME_LEN,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "group", std::to_string(groupLen).c_str(),
                                                          "The value of group length must not exceed 127"),
                    return ge::GRAPH_FAILED);
    // 通路输入world_size为可选，如果默认值不为-1，则调用mc2tiling的获取rankSize的方法
    int64_t rankDim = 0;
    if (GetAndValidateRankSize(context, opName, group, rankDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const bool *isTransX1 = attrs->GetAttrPointer<bool>(indexSchema.x1Transpose);
    bool x1TransposeFlag = (isTransX1 != nullptr) ? *isTransX1 : false;
    OP_TILING_CHECK(x1TransposeFlag, OP_LOGE_WITH_INVALID_ATTR(opName, "transpose_x1", "true", "false"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验参数的format::是否为私有格式
 *
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
ge::graphStatus MatmulAlltoAllTilingUtil::CheckTensorFormat(const gert::TilingContext *context, const char *opName)
{
    // x1:require
    auto x1TensorDesc = context->GetInputDesc(INPUT_X1_INDEX);
    OP_TILING_CHECK((x1TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1"), return ge::GRAPH_FAILED);
    ge::Format x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1TensorDesc->GetStorageFormat()));
    OP_TILING_CHECK(x1Format != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName, "x1", Ops::Base::ToString(x1Format).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    // x2:require
    auto x2TensorDesc = context->GetInputDesc(INPUT_X2_INDEX);
    OP_TILING_CHECK((x2TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2"), return ge::GRAPH_FAILED);
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2TensorDesc->GetStorageFormat()));
    OP_TILING_CHECK(x2Format != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName, "x2", Ops::Base::ToString(x2Format).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    // y:require
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "y"), return ge::GRAPH_FAILED);
    ge::Format yFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetStorageFormat()));
    OP_TILING_CHECK(yFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName, "y", Ops::Base::ToString(yFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    // bias:optional
    auto biasTensorDesc = context->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    if (biasTensorDesc != nullptr) {
        ge::Format biasFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(biasTensorDesc->GetStorageFormat()));
        OP_TILING_CHECK(biasFormat != ge::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(opName, "bias", Ops::Base::ToString(biasFormat).c_str(), "ND"),
                        return ge::GRAPH_FAILED);
    }
    // alltoallout:optional
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    bool allToAllOutFlag = false;
    // 先检查属性数量是否大于索引12，避免越界访问
    if (attrs->GetAttrNum() > ALLTOALLMATMUL_ATTR_ALLTO_ALL_OUT_FLAG_INDEX) {
        const bool *allToAllOutFlagPtr = attrs->GetAttrPointer<bool>(ALLTOALLMATMUL_ATTR_ALLTO_ALL_OUT_FLAG_INDEX);
        allToAllOutFlag = (allToAllOutFlagPtr != nullptr) ? *allToAllOutFlagPtr : false;
    }
    if (allToAllOutFlag) {
        auto allToAllOutDesc = context->GetOutputDesc(ALLTO_ALL_OUT_INDEX);
        if (allToAllOutDesc != nullptr) {
            ge::Format allToAllFormat =
                static_cast<ge::Format>(ge::GetPrimaryFormat(allToAllOutDesc->GetStorageFormat()));
            OP_TILING_CHECK(
                allToAllFormat != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMAT(opName, "alltoallOut", Ops::Base::ToString(allToAllFormat).c_str(), "ND"),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 非量化场景校验参数的DType
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTilingUtil::CheckNonQuantTensorDataType(const gert::TilingContext *context,
                                                                      const char *opName)
{
    // 获取并校验输入张量描述符
    auto x1TensorDesc = context->GetInputDesc(INPUT_X1_INDEX);
    OP_TILING_CHECK((x1TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1"), return ge::GRAPH_FAILED);
    auto x2TensorDesc = context->GetInputDesc(INPUT_X2_INDEX);
    OP_TILING_CHECK((x2TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2"), return ge::GRAPH_FAILED);
    // 获取数据类型并校验一致性与范围
    ge::DataType x1Dtype = x1TensorDesc->GetDataType();
    ge::DataType x2Dtype = x2TensorDesc->GetDataType();
    OP_TILING_CHECK((x1Dtype != x2Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName, "x1,x2", (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype)).c_str(),
                        "The dtype of x1 and x2 must be the same"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(NON_QUANT_X_DTYPE_LIST, x1Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName, "x", (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype)).c_str(),
                        "The dtype of x must be float16 or bf16"),
                    return ge::GRAPH_FAILED);

    // 校验 bias 数据类型（如果存在）
    auto biasTensorDesc = context->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    if (biasTensorDesc != nullptr) {
        ge::DataType biasDtype = biasTensorDesc->GetDataType();
        OP_TILING_CHECK((x1Dtype != biasDtype && biasDtype != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE(opName, "bias", Ops::Base::ToString(biasDtype).c_str(),
                                                  (Ops::Base::ToString(x1Dtype) + " or FLOAT32").c_str()),
                        return ge::GRAPH_FAILED);
    }

    // 校验 scale 张量为空（非量化场景）
    auto x1ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    auto x2ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((x1ScaleTensorDesc != nullptr || x2ScaleTensorDesc != nullptr),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "x1Scale/x2Scale", "non-null",
                                                          "x1Scale and x2Scale must be nullptr in non-quant mode"),
                    return ge::GRAPH_FAILED);

    // 校验输出张量数据类型
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "y"), return ge::GRAPH_FAILED);
    ge::DataType yDtype = yDesc->GetDataType();
    OP_TILING_CHECK((yDtype != x1Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName, "y", Ops::Base::ToString(yDtype).c_str(),
                                              Ops::Base::ToString(x1Dtype).c_str()),
                    return ge::GRAPH_FAILED);

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
ge::graphStatus MatmulAlltoAllTilingUtil::CheckKcQuantTensorDataType(const gert::TilingContext *context,
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
    OP_TILING_CHECK(!IsContains(KC_QUANT_X_DTYPE_LIST, x1Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "x1", Ops::Base::ToString(x1Dtype).c_str(),
                                                          "The dtype of x1 must be float8_e4m3fn or float8_e5m2"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(KC_QUANT_X_DTYPE_LIST, x2Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "x2", Ops::Base::ToString(x2Dtype).c_str(),
                                                          "The dtype of x2 must be float8_e4m3fn or float8_e5m2"),
                    return ge::GRAPH_FAILED);
    // 校验 bias 数据类型（如果存在）
    auto biasTensorDesc = context->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    QuantMode mode = MatmulAlltoAllTilingUtil::GetQuantMode(context, opName);
    if (biasTensorDesc != nullptr && mode == QuantMode::KC_QUANT) {
        ge::DataType biasDtype = biasTensorDesc->GetDataType();
        OP_TILING_CHECK((biasDtype != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE(opName, "bias", Ops::Base::ToString(biasDtype).c_str(), "FLOAT32"),
                        return ge::GRAPH_FAILED);
    }
    // 校验 scale 张量不为空（量化场景）
    auto x1ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    auto x2ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((x1ScaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1Scale"),
                    return ge::GRAPH_FAILED);
    ge::DataType x1scaleDtype = x1ScaleTensorDesc->GetDataType();
    OP_TILING_CHECK((x1scaleDtype != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE(opName, "x1Scale", Ops::Base::ToString(x1scaleDtype).c_str(), "FLOAT32"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2ScaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2Scale"),
                    return ge::GRAPH_FAILED);
    ge::DataType x2scaleDtype = x2ScaleTensorDesc->GetDataType();
    OP_TILING_CHECK((x2scaleDtype != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE(opName, "x2Scale", Ops::Base::ToString(x2scaleDtype).c_str(), "FLOAT32"),
                    return ge::GRAPH_FAILED);
    // 校验输出张量数据类型
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "y"), return ge::GRAPH_FAILED);
    ge::DataType yDtype = yDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(KC_QUANT_Y_DTYPE_LIST, yDtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "y", Ops::Base::ToString(yDtype).c_str(),
                                                          "The dtype of y must be float16, bfloat16 or float"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling inputshape非空
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
static ge::graphStatus CheckInputShapesValid(const gert::TilingContext *context, const char *opName)
{
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    OP_TILING_CHECK((x1Shape == nullptr) || (x2Shape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1/x2"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验量化tiling inputshape非空
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
static ge::graphStatus CheckKcQuantInputShapesValid(const gert::TilingContext *context, const char *opName)
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
 * @brief 校验量化tiling scale shape的Dim数量信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
static ge::graphStatus CheckScaleShapeDimensions(const gert::StorageShape *shape, const char *shapeName,
                                                 const char *opName)
{
    uint64_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((dimNum != 1),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(opName, shapeName, (std::to_string(dimNum) + "D").c_str(), "1D"),
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
    // 校验N不能大于int32的
    OP_TILING_CHECK(nAxis > MAX_INT32_VALUE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "x2", std::to_string(nAxis).c_str(),
                                                          "The value of N of x2 must not exceed INT32_MAX"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling输入的shape信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @param indexSchema 存放输入参数索引差别的结构体
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTilingUtil::CheckShapeInfo(const gert::TilingContext *context, const char *opName,
                                                         const OpAttrIndexSchema &indexSchema)
{
    ge::graphStatus status;
    // 校验输入Input Shape是否为空
    status = CheckInputShapesValid(context, opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验维度数目是否合法
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    status = CheckShapeDimensions(x1Shape, "input x1", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    status = CheckShapeDimensions(x2Shape, "input x2", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验输出是否为空及维度数目
    const gert::StorageShape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "y"), return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(yShape, "output y", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验bias的shape信息
    status = CheckBiasShape(context, opName, indexSchema);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验shape的dim范围
    status = CheckShapeDimRange(context, opName, indexSchema);
    if (status != ge::GRAPH_SUCCESS)
        return status;
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
ge::graphStatus MatmulAlltoAllTilingUtil::CheckKcQuantShapeInfo(const gert::TilingContext *context, const char *opName,
                                                                const OpAttrIndexSchema &indexSchema)
{
    ge::graphStatus status;
    // 校验输入量化Input Shape是否为空
    status = CheckKcQuantInputShapesValid(context, opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验维度数目是否合法
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *x1ScaleShape = context->GetOptionalInputShape(INPUT_X1_SCALE_INDEX);
    const gert::StorageShape *x2ScaleShape = context->GetOptionalInputShape(INPUT_X2_SCALE_INDEX);
    status = CheckShapeDimensions(x1Shape, "kc quant input x1", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    status = CheckShapeDimensions(x2Shape, "kc quant input x2", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    status = CheckScaleShapeDimensions(x1ScaleShape, "kc quant input x1scale", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    status = CheckScaleShapeDimensions(x2ScaleShape, "kc quant input x2scale", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    // 校验输出
    const gert::StorageShape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "y"), return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(yShape, "kc quant output y", opName);
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
 * @brief 设置attrs信息，group， rank等
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @param contextInfo 过程信息
 * @param indexSchema 存放输入参数索引差别的结构体
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTilingUtil::SetAttrsInfo(const gert::TilingContext *context, const char *opName,
                                                       TilingContextInfo &contextInfo,
                                                       const OpAttrIndexSchema &indexSchema)
{
    // 在CheckAttrsInfo已经校验过attrs和
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    contextInfo.group = group;
    int64_t rankDim = 0;
    if (GetAndValidateRankSize(context, opName, group, rankDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    contextInfo.args_.rankDim = rankDim;
    // x1不支持转置
    contextInfo.args_.isATrans = false;
    const bool *isTransX2 = attrs->GetAttrPointer<bool>(indexSchema.x2Transpose);
    contextInfo.args_.isBTrans = isTransX2 ? *isTransX2 : false;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置算子的shape信息，供MatmulAlltoAll单独使用
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param contextInfo 存储了tiling的过程信息
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTilingUtil::SetShapeInfo(const gert::TilingContext *context,
                                                       TilingContextInfo &contextInfo)
{
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    uint64_t x1Dim0 = x1Shape->GetStorageShape().GetDim(0);
    uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);

    contextInfo.args_.orgMValue = x1Dim0;
    contextInfo.args_.orgNValue = (contextInfo.args_.isBTrans) ? x2Dim0 : x2Dim1;
    contextInfo.args_.orgKValue = x1Dim1;

    // rank上的值,对于MatmulAlltoAll，取的就是rank上实际的m,n,k
    contextInfo.args_.mValue = x1Dim0;
    contextInfo.args_.kValue = x1Dim1;
    contextInfo.args_.nValue = (contextInfo.args_.isBTrans) ? x2Dim0 : x2Dim1;
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
ge::graphStatus MatmulAlltoAllTilingUtil::SetDataTypeInfo(const gert::TilingContext *context, const char *opName,
                                                          TilingContextInfo &contextInfo)
{
    const gert::StorageShape *matrixBias = context->GetOptionalInputShape(INPUT_BIAS_INDEX);
    ge::DataType aType = context->GetInputDesc(INPUT_X1_INDEX)->GetDataType();
    ge::DataType bType = context->GetInputDesc(INPUT_X2_INDEX)->GetDataType();
    ge::DataType biasType;
    bool isBias = true;
    ge::DataType cType = aType;

    if (matrixBias == nullptr) {
        isBias = false;
        biasType = cType;
    } else {
        biasType = context->GetOptionalInputDesc(INPUT_BIAS_INDEX)->GetDataType();
    }

    contextInfo.args_.inputDtypeSize = mc2tiling::GetDataTypeSize(opName, aType);
    contextInfo.args_.outputDtypeSize = mc2tiling::GetDataTypeSize(opName, cType);
    contextInfo.args_.isBias = isBias;
    contextInfo.args_.geAType = aType;
    contextInfo.args_.geBType = bType;
    contextInfo.args_.geCType = cType;
    contextInfo.args_.geBiasType = biasType;
    contextInfo.args_.aType = mc2tiling::ConvertGeTypeToMmType(opName, aType);
    contextInfo.args_.bType = mc2tiling::ConvertGeTypeToMmType(opName, bType);
    contextInfo.args_.cType = mc2tiling::ConvertGeTypeToMmType(opName, cType);
    contextInfo.args_.biasType = mc2tiling::ConvertGeTypeToMmType(opName, biasType);
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
ge::graphStatus MatmulAlltoAllTilingUtil::SetKcDataTypeInfo(const gert::TilingContext *context, const char *opName,
                                                            TilingContextInfo &contextInfo)
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
 * @brief 功能函数：获取算子对应的QUANT类型
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return QuantMode
 */
QuantMode MatmulAlltoAllTilingUtil::GetQuantMode(const gert::TilingContext *context, const char *opName)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(opName, "Failed to get attrs.");
        return QuantMode::ERROR;
    }
    // 获取量化模式属性（默认为0，表示非量化）
    int64_t x1QuantMode = 0;
    int64_t x2QuantMode = 0;
    if (const int64_t *ptr = attrs->GetAttrPointer<int64_t>(ATTR_X1_QUANTMODE_INDEX)) {
        x1QuantMode = *ptr;
    }
    if (const int64_t *ptr = attrs->GetAttrPointer<int64_t>(ATTR_X2_QUANTMODE_INDEX)) {
        x2QuantMode = *ptr;
    }
    // 获取输入的x1,x2的数据类型
    auto x1TensorDesc = context->GetInputDesc(INPUT_X1_INDEX);
    auto x2TensorDesc = context->GetInputDesc(INPUT_X2_INDEX);
    if (x1TensorDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "x1");
        return QuantMode::ERROR; // 返回一个异常值，代表没有落入量化组合模式范围内
    }
    if (x2TensorDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "x2");
        return QuantMode::ERROR;
    }
    ge::DataType aType = x1TensorDesc->GetDataType();
    ge::DataType bType = x2TensorDesc->GetDataType();
    if (x1QuantMode == 0 && x2QuantMode == 0 && aType == bType && (aType == ge::DT_BF16 || aType == ge::DT_FLOAT16)) {
        return QuantMode::NON_QUANT;
    }
    if (x1QuantMode == X1_QUANTMODE_VALUE && x2QuantMode == X2_QUANTMODE_VALUE) {
        return QuantMode::KC_QUANT;
    } else {
        OP_LOGD(opName,
                "Quantization mode error, KC quantization X1 should be three, X2 should be two."
                "currently X1=%d, X2=%d.",
                x1QuantMode, x2QuantMode);
    }
    return QuantMode::ERROR;
}

Mc2CcTilingConfigBuilder::Mc2CcTilingConfigBuilder(const std::string &groupName, uint32_t opType,
                                                   const std::string &algConfig)
    : mc2CcTilingConfig(groupName, opType, algConfig)
{
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::withReduceType(const char *opName, mc2tiling::HcclReduceOp reduceType,
                                                                   ge::DataType dstDataType, ge::DataType srcDataType)
{
    this->reduceType = static_cast<uint32_t>(reduceType);
    this->dstDataType = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName, dstDataType));
    this->srcDataType = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName, srcDataType));
    this->hasReduceType = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::withStepSize(uint8_t stepSize)
{
    this->stepSize = stepSize;
    this->hasStepSize = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::isLocalRankDataToLocalDst(bool flag)
{
    this->localRankDataToLocalDst = flag ? CONSTANTS_ONE : CONSTANTS_ZERO;
    this->hasLocalRankDataToLocalDst = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::isSrcDataFromWindow(bool fromWindow)
{
    this->srcDataFromWindow = fromWindow ? CONSTANTS_TWO : CONSTANTS_ONE;
    this->hasSrcDataFromWindow = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::withDebugMode(uint8_t debugMode)
{
    this->debugMode = debugMode;
    this->hasDebugMode = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::withCommBlockNum(uint16_t commBlockNum)
{
    this->commBlockNum = commBlockNum;
    this->hasCommBlockNum = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::withQueueNum(uint16_t queueNum)
{
    this->queueNum = queueNum;
    this->hasQueueNum = true;
    return *this;
}

Mc2CcTilingConfigBuilder &Mc2CcTilingConfigBuilder::withCommEngine(uint8_t commEngine)
{
    this->commEngine = commEngine;
    this->hasCommEngine = true;
    return *this;
}

AscendC::Mc2CcTilingConfig Mc2CcTilingConfigBuilder::build()
{
    if (this->hasReduceType) {
        if (this->SUCCESS != mc2CcTilingConfig.SetReduceType(reduceType, dstDataType, srcDataType)) {
            this->errorSet.insert(ConfigFile::REDUCE_TYPE);
        } else {
            this->errorSet.erase(ConfigFile::REDUCE_TYPE);
        }
    }
    //  setConfig
    auto setConfig = [this](const char *fieldName, bool has, const std::function<int()> &setter) {
        if (has) {
            if (this->SUCCESS != setter()) {
                this->errorSet.insert(fieldName);
            } else {
                this->errorSet.erase(fieldName);
            }
        }
    };

    setConfig(ConfigFile::STEP_SIZE, this->hasStepSize,
              [this]() { return this->mc2CcTilingConfig.SetStepSize(this->stepSize); });
    setConfig(ConfigFile::SKIP_LOCAL_RANK_COPY, this->hasLocalRankDataToLocalDst,
              [this]() { return this->mc2CcTilingConfig.SetSkipLocalRankCopy(this->localRankDataToLocalDst); });
    setConfig(ConfigFile::SKIP_BUFFER_WINDOW_COPY, this->hasSrcDataFromWindow,
              [this]() { return this->mc2CcTilingConfig.SetSkipBufferWindowCopy(this->srcDataFromWindow); });
    setConfig(ConfigFile::DEBUG_MODE, this->hasDebugMode,
              [this]() { return this->mc2CcTilingConfig.SetDebugMode(this->debugMode); });
    setConfig(ConfigFile::COMM_BLOCK_NUM, this->hasCommBlockNum,
              [this]() { return this->mc2CcTilingConfig.SetCommBlockNum(this->commBlockNum); });
    setConfig(ConfigFile::QUEUE_NUM, this->hasQueueNum,
              [this]() { return this->mc2CcTilingConfig.SetQueueNum(this->queueNum); });
    setConfig(ConfigFile::COMM_ENGINE, this->hasCommEngine,
              [this]() { return this->mc2CcTilingConfig.SetCommEngine(this->commEngine); });
    return this->mc2CcTilingConfig;
}

bool Mc2CcTilingConfigBuilder::isSuccess() const
{
    return this->errorSet.empty();
}

std::string Mc2CcTilingConfigBuilder::errorMsg() const
{
    std::string msg;
    for (auto &error : this->errorSet) {
        msg += error + ",";
    }
    return msg;
}

void Mc2CcTilingConfigBuilder::setInitTilingData(::Mc2InitTiling initTiling)
{
    if (!this->hasGotInit) {
        this->hasGotInit = true;
        mc2CcTilingConfig.GetTiling(initTiling);
    }
}

Mc2CcTilingConfigBuilder Mc2CcTilingConfigBuilder::create(const std::string &groupName, mc2tiling::AicpuComType opType,
                                                          const std::string &algConfig)
{
    return Mc2CcTilingConfigBuilder(groupName, static_cast<uint32_t>(opType), algConfig);
}

} // namespace MC2Tiling
