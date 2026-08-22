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
 * \file quant_reduce_scatter_util_tiling.cpp
 * \brief
 */
#include "quant_reduce_scatter_util_tiling.h"

namespace MC2Tiling {

using namespace ops;

/**
 * @brief 工具函数：判断指定value是否存在于list中
 * @param list: 有效值列表
 * @param value: 给定值
 * @return
 */
static bool IsContains(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::find(list.begin(), list.end(), value) != list.end();
}

/**
 * @brief 校验attrs，并设置group
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus CheckAttrsInfo(const gert::TilingContext *context, TilingRunInfo &runInfo)
{
    const char *nodeName = context->GetNodeName();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);
    // 校验group是否为空
    const char *groupPtr = attrs->GetAttrPointer<char>(GROUP_INDEX);
    OP_TILING_CHECK(groupPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "group"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(std::string(groupPtr).empty(), OP_LOGE_WITH_INVALID_INPUT(nodeName, "group"),
                    return ge::GRAPH_FAILED);
    runInfo.groupPtr = groupPtr;
    runInfo.group = std::string(groupPtr);
    // 校验reduce_op的类型是否为sum
    const char *reduceOpPtr = attrs->GetAttrPointer<char>(REDUCE_OP_INDEX);
    OP_TILING_CHECK(reduceOpPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "reduce_op"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(std::strcmp(reduceOpPtr, REDUCE_OP_TYPE.c_str()) != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "reduce_op", reduceOpPtr, REDUCE_OP_TYPE.c_str()),
                    return ge::GRAPH_FAILED);
    // 校验output_dtype
    const int64_t *outputTypePtr = attrs->GetAttrPointer<int64_t>(OUTPUT_DTYPE_INDEX);
    OP_TILING_CHECK(outputTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "outputType"),
                    return ge::GRAPH_FAILED);
    ge::DataType outputType = static_cast<ge::DataType>(*outputTypePtr);
    OP_TILING_CHECK(!IsContains(OUTPUT_DTYPE_LIST, outputType),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "outputType", Ops::Base::ToString(outputType).c_str(),
                                              "bfloat16/float/float16"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置rankSize
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @return
 */
static ge::graphStatus SetRankSize(const gert::TilingContext *context, TilingRunInfo &runInfo)
{
    const char *nodeName = context->GetNodeName();
    // attrs在函数CheckAttrsInfo中已做校验
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *rankSizePtr = attrs->GetAttrPointer<int64_t>(WORLD_SIZE_INDEX);
    if (rankSizePtr == nullptr || *rankSizePtr == RANK_SIZE_DEFAULT) {
        int64_t rankSize = 0;
        OP_TILING_CHECK(!mc2tiling::GetRankSize(nodeName, runInfo.groupPtr, rankSize),
                        OP_LOGE(nodeName, "Get rankSize failed."), return ge::GRAPH_FAILED);
        runInfo.rankSize = rankSize;
    } else {
        runInfo.rankSize = *rankSizePtr;
    }
    OP_TILING_CHECK(std::find(RANK_SIZE_LIST.begin(), RANK_SIZE_LIST.end(), runInfo.rankSize) >= RANK_SIZE_LIST.end(),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "rankSize", std::to_string(runInfo.rankSize).c_str(),
                                              VectorToString(RANK_SIZE_LIST).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置量化模式
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @return
 */
static bool SetQuantMode(const gert::TilingContext *context, TilingRunInfo &runInfo)
{
    const char *nodeName = context->GetNodeName();
    // context->GetInputDesc在函数CheckTensorDataType中已经校验
    ge::DataType xDtype = context->GetInputDesc(X_INDEX)->GetDataType();
    ge::DataType scalesDtype = context->GetInputDesc(SCALES_INDEX)->GetDataType();
    // 0: 无量化模式; 1: TG量化; 2: MX量化
    uint32_t quantMode = 0;
    if (IsContains(X_DTYPE_LIST, xDtype) && scalesDtype == ge::DT_FLOAT) {
        quantMode = TG_QUANT_MOD;
    } else if ((xDtype == ge::DT_FLOAT8_E4M3FN || xDtype == ge::DT_FLOAT8_E5M2) && scalesDtype == ge::DT_FLOAT8_E8M0) {
        quantMode = MX_QUANT_MOD;
    }
    OP_TILING_CHECK(
        !static_cast<bool>(quantMode),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "quantMode",
                                              ("x=" + std::string(Ops::Base::ToString(xDtype).c_str()) +
                                               ", scales=" + std::string(Ops::Base::ToString(scalesDtype).c_str()))
                                                  .c_str(),
                                              "The value of quantMode must match a known quantization mode (TG or MX)"),
        return false);
    // 设置quantMode
    runInfo.quantMode = quantMode;
    return true;
}

/**
 * @brief 校验所有参数的dtype，并设置量化模式
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @return
 */
static bool CheckTensorDataType(const gert::TilingContext *context, TilingRunInfo &runInfo)
{
    const char *nodeName = context->GetNodeName();
    // 校验x的dtype
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return false);
    ge::DataType xDtype = context->GetInputDesc(X_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(X_DTYPE_LIST, xDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDtype).c_str(),
                                              "int8/hifloat8/float8_e4m3fn/float8_e5m2"),
                    return false);
    // 校验scales的dtype
    auto scalesDesc = context->GetInputDesc(SCALES_INDEX);
    OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scales"), return false);
    ge::DataType scalesDtype = context->GetInputDesc(SCALES_INDEX)->GetDataType();
    OP_TILING_CHECK(
        !IsContains(SCALES_DTYPE_LIST, scalesDtype),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales", Ops::Base::ToString(scalesDtype).c_str(), "float/float8_e8m0"),
        return false);
    // 校验output的dtype
    auto outputDesc = context->GetOutputDesc(OUTPUT_INDEX);
    OP_TILING_CHECK(outputDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "output"), return false);
    ge::DataType outputType = outputDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(OUTPUT_DTYPE_LIST, outputType),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "output", Ops::Base::ToString(outputType).c_str(),
                                              "float16/bfloat16/float"),
                    return false);
    // 设置量化模式
    OP_TILING_CHECK(!SetQuantMode(context, runInfo), OP_LOGE(nodeName, "get quantMode error."), return false);
    return true;
}

/**
 * @brief 校验x维度的合法性，必须为2维或3维
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opType: 当前op类型
 * @return
 */
static bool CheckXDimValid(const gert::TilingContext *context, const OpType opType)
{
    (void)opType; // Reserved for future extension
    const char *nodeName = context->GetNodeName();
    // context->GetInputShape在函数CheckInputTensorDim中已经校验
    size_t xDimNum = context->GetInputShape(X_INDEX)->GetStorageShape().GetDimNum();
    // quant_all_reduce和quant_reduce_scatter算子的x可能是2维或者3维，即x.shape(bs, h)或x.shape(b, s, h)
    bool inValidDimNum = (xDimNum != TWO_DIMS) && (xDimNum != THREE_DIMS);
    OP_TILING_CHECK(inValidDimNum,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "x", (std::to_string(xDimNum) + "D").c_str(), "2D or 3D"),
                    return false);

    return true;
}

/**
 * @brief 校验x的Shape, 包括是否为空tensor，BS轴，H轴是否合法
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @param opType: 当前op类型
 * @return
 */
static bool CheckXShapeValid(const gert::TilingContext *context, TilingRunInfo &runInfo, const OpType opType)
{
    (void)opType; // Reserved for future extension
    const char *nodeName = context->GetNodeName();
    // context->GetInputShape在函数CheckInputTensorDim中已经校验
    const gert::StorageShape *xShape = context->GetInputShape(X_INDEX);
    // 获取x各维度值
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    uint64_t xValueOne = xShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueTwo = xShape->GetStorageShape().GetDim(DIM_ONE);

    // 当x是2维时，x.shape = (BS, H)
    uint64_t xValueBS = xValueOne;
    uint64_t xValueH = xValueTwo;
    bool emptyTensor = xValueOne == 0 || xValueTwo == 0;
    // 当x是3维时，x.shape = (B, S, H)
    if (xDimNum == THREE_DIMS) {
        xValueBS = xValueOne * xValueTwo;
        xValueH = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        emptyTensor = emptyTensor || xValueH == 0;
    }

    // 校验x是否为空tensor
    OP_TILING_CHECK(
        emptyTensor,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x", Ops::Base::ToString(xShape->GetStorageShape()).c_str(),
                                              "The shape of x must have all dimensions positive (>=1)"),
        return false);

    // 校验BS是否被worldSize整除
    const char *dimDesc = (xDimNum == THREE_DIMS) ? "B*S" : "BS"; // 报错信息二维时为BS, 三维时为B*S
    OP_TILING_CHECK(xValueBS % runInfo.rankSize != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x",
                                                          ("shape=" + Ops::Base::ToString(xShape->GetStorageShape()) +
                                                           ", " + std::string(dimDesc) + "=" + std::to_string(xValueBS))
                                                              .c_str(),
                                                          (std::string(dimDesc) + " must be divisible by world_size (" +
                                                           std::to_string(runInfo.rankSize) + ")")
                                                              .c_str()),
                    return false);

    // 校验H是否在[1024, 8192]之间，且能被128整除
    OP_TILING_CHECK(xValueH < H_VALUE_LOWER_LIMIT || xValueH > H_VALUE_UPPER_LIMIT || xValueH % TG_QUANT_NUMBER != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x", std::to_string(xValueH).c_str(),
                                                          "H dimension must be in [1024, 8192] and 128 multiple"),
                    return false);
    return true;
}

/**
 * @brief 根据x的形状和量化模式计算正确的scales形状
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @return 计算出的正确scales维度向量
 */
static std::vector<uint64_t> CalculateExpectedScalesShape(const gert::TilingContext *context, TilingRunInfo &runInfo)
{
    const gert::StorageShape *xShape = context->GetInputShape(X_INDEX);

    // 获取x的维度和值
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    std::vector<uint64_t> expectedScalesDims;

    // 根据x的维度构建基础维度
    if (xDimNum == TWO_DIMS) {
        // x.shape = (BS, H)
        uint64_t bs = xShape->GetStorageShape().GetDim(DIM_ZERO);
        uint64_t h = xShape->GetStorageShape().GetDim(DIM_ONE);

        if (runInfo.quantMode == TG_QUANT_MOD) {
            // TG量化: scales.shape(BS, H/128)
            expectedScalesDims.push_back(bs);
            expectedScalesDims.push_back(ops::CeilDiv(h, TG_QUANT_NUMBER));
        } else if (runInfo.quantMode == MX_QUANT_MOD) {
            // MX量化: scales.shape(BS, H/64, 2)
            expectedScalesDims.push_back(bs);
            expectedScalesDims.push_back(ops::CeilDiv(h, MX_QUANT_NUMBER));
            expectedScalesDims.push_back(MX_SCALE_LAST_DIM);
        }
    } else if (xDimNum == THREE_DIMS) {
        // x.shape = (B, S, H)
        uint64_t b = xShape->GetStorageShape().GetDim(DIM_ZERO);
        uint64_t s = xShape->GetStorageShape().GetDim(DIM_ONE);
        uint64_t h = xShape->GetStorageShape().GetDim(DIM_TWO);

        if (runInfo.quantMode == TG_QUANT_MOD) {
            // TG量化: scales.shape(B, S, H/128)
            expectedScalesDims.push_back(b);
            expectedScalesDims.push_back(s);
            expectedScalesDims.push_back(ops::CeilDiv(h, TG_QUANT_NUMBER));
        } else if (runInfo.quantMode == MX_QUANT_MOD) {
            // MX量化: scales.shape(B, S, H/64, 2)
            expectedScalesDims.push_back(b);
            expectedScalesDims.push_back(s);
            expectedScalesDims.push_back(ops::CeilDiv(h, MX_QUANT_NUMBER));
            expectedScalesDims.push_back(MX_SCALE_LAST_DIM);
        }
    }

    return expectedScalesDims;
}

// 辅助函数：格式化shape为字符串
static std::string FormatShape(const std::vector<uint64_t> &dims)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0)
            ss << ", ";
        ss << dims[i];
    }
    ss << "]";
    return ss.str();
}

/**
 * @brief 校验传入的scales是否符合预期
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param expectedScalesDims: 预期的scales
 * @param runInfo: 封装的doTiling所需要的参数（包含quantMode信息）
 * @return true: 校验通过, false: 校验失败
 */
static bool CheckScalesValid(const gert::TilingContext *context, const std::vector<uint64_t> &expectedScalesDims,
                             const TilingRunInfo &runInfo)
{
    const char *nodeName = context->GetNodeName();
    const gert::StorageShape *scalesShape = context->GetInputShape(SCALES_INDEX);

    // 将quantMode转换为可读字符串
    const char *quantModeStr = "";
    if (runInfo.quantMode == TG_QUANT_MOD) {
        quantModeStr = "TG";
    } else if (runInfo.quantMode == MX_QUANT_MOD) {
        quantModeStr = "MX";
    }

    // 检查维度数量是否一致
    size_t actualDimNum = scalesShape->GetStorageShape().GetDimNum();
    if (actualDimNum != expectedScalesDims.size()) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "scales", FormatShape(expectedScalesDims).c_str(),
            ("The shape dim of scales must be " + std::to_string(expectedScalesDims.size()) + "D").c_str());
        return false;
    }

    // 检查每个维度的值是否一致
    for (size_t i = 0; i < expectedScalesDims.size(); ++i) {
        uint64_t expectedDim = expectedScalesDims[i];
        uint64_t actualDim = scalesShape->GetStorageShape().GetDim(i);
        if (expectedDim != actualDim) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                nodeName, "scales", std::to_string(actualDim).c_str(),
                ("The shape of scales dim " + std::to_string(i) + " must be " + std::to_string(expectedDim)).c_str());
            return false;
        }
    }
    return true;
}

/**
 * @brief 校验入参x和scales是否合法
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @param opType:当前op类型
 * @return
 */
static bool CheckInputTensorDim(const gert::TilingContext *context, TilingRunInfo &runInfo, const OpType opType)
{
    const char *nodeName = context->GetNodeName();
    // 1.校验x相关
    const gert::StorageShape *xShape = context->GetInputShape(X_INDEX);
    // 校验x不为空
    OP_TILING_CHECK(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xShape"), return false);
    // 校验x维度数量合法性
    if (!CheckXDimValid(context, opType)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "x", "x_dim_invalid", "x dim check failed");
        return false;
    }
    // 校验x.shape合法性
    if (!CheckXShapeValid(context, runInfo, opType)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x", "x_shape_invalid", "x shape check failed");
        return false;
    }

    // 2.校验scales
    const gert::StorageShape *scalesShape = context->GetInputShape(SCALES_INDEX);
    // 根据x计算正确的scales, 当scale形状不匹配时，会打印预期的形状和实际的形状
    std::vector<uint64_t> expectedScalesDims = CalculateExpectedScalesShape(context, runInfo);
    // 校验scales不为空
    OP_TILING_CHECK(scalesShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scales"), return false);
    // 校验scales维度和shape是否正确
    if (!CheckScalesValid(context, expectedScalesDims, runInfo)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "scales", "scales_shape_invalid", "scales shape check failed");
        return false;
    }
    return true;
}

/**
 * @brief 检查输出维度大小的合法性
 */
static bool CheckOutputDimSize(const gert::TilingContext *context, size_t outputDim, size_t xDimNum, OpType opType,
                               const char *nodeName)
{
    (void)context; // Reserved for future extension
    bool invalidOutputDim = false;
    if (opType == OpType::OP_QUANT_ALL_REDUCE) {
        // 对于quant_all_reduce，输出维度必须与与输入维度一致, 必须是2维或3维
        invalidOutputDim = outputDim != xDimNum;
        OP_TILING_CHECK(invalidOutputDim,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "output", (std::to_string(outputDim) + "D").c_str(),
                                                     (std::to_string(xDimNum) + "D").c_str()),
                        return false);
    } else {
        // 对于quant_reduce_scatter，输出维度必须是2维
        invalidOutputDim = outputDim != TWO_DIMS;
        OP_TILING_CHECK(
            invalidOutputDim,
            OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "output", (std::to_string(outputDim) + "D").c_str(), "2D"),
            return false);
    }
    return true;
}

/**
 * @brief 检查quant_all_reduce的输出形状
 */
static bool CheckAllReduceOutputShape(const gert::TilingContext *context, const gert::StorageShape *outputShape,
                                      size_t outputDim, size_t xDimNum, TilingRunInfo &runInfo, const char *nodeName)
{
    (void)xDimNum; // Reserved for future extension
    uint64_t outputValueOne = outputShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t outputValueTwo = outputShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t xValueOne = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueTwo = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);

    // 对于quant_all_reduce算子，output.shape必须等于x.shape
    bool invalidShape = (xValueOne != outputValueOne) || (xValueTwo != outputValueTwo); // 校验前两维的大小

    // quant_all_reduce算子支持三维，output可能需要校验第3维
    if (outputDim == THREE_DIMS) {
        uint64_t outputValueThree = outputShape->GetStorageShape().GetDim(DIM_TWO);
        uint64_t xValueThree = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        OP_LOGI(nodeName, "output dim2 is %lu, x dim2 is %lu", outputValueThree, xValueThree);
        invalidShape = invalidShape || (xValueThree != outputValueThree); // 校验第三维的大小
        OP_TILING_CHECK(invalidShape,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            nodeName, "output",
                            ("(" + std::to_string(outputValueOne) + ", " + std::to_string(outputValueTwo) + ", " +
                             std::to_string(outputValueThree) + ")")
                                .c_str(),
                            ("The shape of output must match x shape (" + std::to_string(xValueOne) + ", " +
                             std::to_string(xValueTwo) + ", " + std::to_string(xValueThree) + ")")
                                .c_str()),
                        return false);
    } else {
        OP_TILING_CHECK(
            invalidShape,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                nodeName, "output",
                ("(" + std::to_string(outputValueOne) + ", " + std::to_string(outputValueTwo) + ")").c_str(),
                ("The shape of output must match x shape (" + std::to_string(xValueOne) + ", " +
                 std::to_string(xValueTwo) + ")")
                    .c_str()),
            return false);
    }

    return true;
}

/**
 * @brief 检查quant_reduce_scatter的输出形状, 当输入x为3D时
 */
static bool CheckReduceScatter3DShape(const gert::TilingContext *context, uint64_t outputValueOne,
                                      uint64_t outputValueTwo, uint64_t xValueOne, uint64_t xValueTwo,
                                      TilingRunInfo &runInfo, const char *nodeName)
{
    // 若X为3维，则要对b,s进行合轴，再与output判断是否合法
    uint64_t xValueBS = xValueOne * xValueTwo;
    uint64_t xValueThree = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_TWO);
    bool invalidShape = xValueBS / runInfo.rankSize != outputValueOne; // 校验bs轴
    invalidShape = invalidShape || (xValueThree != outputValueTwo);    // 校验h轴
    OP_TILING_CHECK(invalidShape,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "output",
                        ("(" + std::to_string(outputValueOne) + ", " + std::to_string(outputValueTwo) + ")").c_str(),
                        ("The shape of output must match x shape/rankSize (BS=" + std::to_string(xValueBS) + "/" +
                         std::to_string(runInfo.rankSize) + ", H=" + std::to_string(xValueThree) + ")")
                            .c_str()),
                    return false);
    return true;
}

/**
 * @brief 检查quant_reduce_scatter的的输出形状, 当输入x为2D时
 */
static bool CheckReduceScatter2DShape(uint64_t outputValueOne, uint64_t outputValueTwo, uint64_t xValueOne,
                                      uint64_t xValueTwo, TilingRunInfo &runInfo, const char *nodeName)
{
    // 若X为2维, 逐个校验即可
    bool invalidShape = xValueOne / runInfo.rankSize != outputValueOne; // 校验bs轴
    invalidShape = invalidShape || (xValueTwo != outputValueTwo);       // 校验h轴
    OP_TILING_CHECK(invalidShape,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "output",
                        ("(" + std::to_string(outputValueOne) + ", " + std::to_string(outputValueTwo) + ")").c_str(),
                        ("The shape of output must match x shape/rankSize (BS=" + std::to_string(xValueOne) + "/" +
                         std::to_string(runInfo.rankSize) + ", H=" + std::to_string(xValueTwo) + ")")
                            .c_str()),
                    return false);
    return true;
}

/**
 * @brief 检查quant_reduce_scatter的输出形状
 */
static bool CheckReduceScatterOutputShape(const gert::TilingContext *context, const gert::StorageShape *outputShape,
                                          size_t outputDim, size_t xDimNum, TilingRunInfo &runInfo,
                                          const char *nodeName)
{
    (void)outputDim; // Reserved for future extension
    uint64_t outputValueOne = outputShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t outputValueTwo = outputShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t xValueOne = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueTwo = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);

    // 对于quant_reduce_scatter算子, 输出output一定为2维，判断x维度大小决定是否b,s合轴
    if (xDimNum == THREE_DIMS) {
        return CheckReduceScatter3DShape(context, outputValueOne, outputValueTwo, xValueOne, xValueTwo, runInfo,
                                         nodeName);
    } else {
        return CheckReduceScatter2DShape(outputValueOne, outputValueTwo, xValueOne, xValueTwo, runInfo, nodeName);
    }
}

/**
 * @brief 校验output的维度
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @return
 */
static bool CheckOutputDim(const gert::TilingContext *context, TilingRunInfo &runInfo, const OpType opType)
{
    const char *nodeName = context->GetNodeName();
    // context->GetOutputShape在函数CheckOutputTensorDim中已经校验
    const gert::StorageShape *outputShape = context->GetOutputShape(OUTPUT_INDEX);
    size_t outputDim = outputShape->GetStorageShape().GetDimNum();
    // context->GetInputShape在函数CheckInputTensorDim中已经校验
    size_t xDimNum = context->GetInputShape(X_INDEX)->GetStorageShape().GetDimNum();
    // 检查output的维度大小
    if (!CheckOutputDimSize(context, outputDim, xDimNum, opType, nodeName)) {
        return false;
    }

    // 检查输出output形状与输入x形状的关系
    if (opType == OpType::OP_QUANT_ALL_REDUCE) {
        return CheckAllReduceOutputShape(context, outputShape, outputDim, xDimNum, runInfo, nodeName);
    } else {
        return CheckReduceScatterOutputShape(context, outputShape, outputDim, xDimNum, runInfo, nodeName);
    }
}

/**
 * @brief 校验所有出参的维度
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @param opType: 当前op类型
 * @return
 */
static bool CheckOutputTensorDim(const gert::TilingContext *context, TilingRunInfo &runInfo, const OpType opType)
{
    const char *nodeName = context->GetNodeName();
    // 红线校验
    const gert::StorageShape *outputShape = context->GetOutputShape(OUTPUT_INDEX);
    OP_TILING_CHECK(outputShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "output"), return false);
    return CheckOutputDim(context, runInfo, opType);
}

/**
 * @brief 校验所有参数的format
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static bool CheckTensorFormat(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    // context->GetInputDesc在CheckTensorDataType函数中已经校验
    auto xDesc = context->GetInputDesc(X_INDEX);
    ge::Format xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()));
    OP_TILING_CHECK(xFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x", Ops::Base::ToString(xFormat).c_str(), "ND"),
                    return false);
    auto scalesDesc = context->GetInputDesc(SCALES_INDEX);
    ge::Format scalesFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat()));
    OP_TILING_CHECK(scalesFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "scales", Ops::Base::ToString(scalesFormat).c_str(), "ND"),
                    return false);
    // context->GetOutputDesc在CheckTensorDataType函数中已经校验
    auto outputDesc = context->GetOutputDesc(OUTPUT_INDEX);
    ge::Format outPutFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(outputDesc->GetStorageFormat()));
    OP_TILING_CHECK(outPutFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "output", Ops::Base::ToString(outPutFormat).c_str(), "ND"),
                    return false);
    return true;
}

/**
 * @brief 校验win区大小
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @return
 */
static bool CheckWindowSize(const gert::TilingContext *context, const TilingRunInfo &runInfo)
{
    const char *nodeName = context->GetNodeName();
    // 获取量化模式，数据类型
    uint64_t xValueOne = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueTwo = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t scalesValueOne = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t scalesValueTwo = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_ONE);

    // 计算xDataSize
    uint64_t xValue = xValueOne * xValueTwo;
    uint64_t scalesValue = scalesValueOne * scalesValueTwo;
    uint32_t scalesLastDim = DIM_TWO;
    size_t xDimNum = context->GetInputShape(X_INDEX)->GetStorageShape().GetDimNum();
    if (xDimNum == THREE_DIMS) {
        uint64_t xValueThree = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        xValue = xValue * xValueThree;
        uint64_t scalesValueThree = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        scalesValue = scalesValue * scalesValueThree;
        scalesLastDim = DIM_THREE;
    }
    uint64_t xDataSize = ((xValue * X_DTYPE_SIZE_ONE + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    OP_LOGD(nodeName, "current xDataSize is: [%lu]MB.", ops::CeilDiv(xDataSize, MB_SIZE));

    // 计算scalesDataSize
    uint64_t scalesSize = 0UL;
    if (runInfo.quantMode == TG_QUANT_MOD) {
        scalesSize = scalesValue * SCALE_DTYPE_SIZE_FOUR;
    } else if (runInfo.quantMode == MX_QUANT_MOD) {
        // scales的最后一维一定为2
        uint64_t scalesValueLast = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(scalesLastDim);
        scalesSize = scalesValue * scalesValueLast * SCALE_DTYPE_SIZE_ONE;
    }
    uint64_t scalesDataSize = ((scalesSize + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    OP_LOGD(nodeName, "current scalesDataSize is: [%lu]MB.", ops::CeilDiv(scalesDataSize, MB_SIZE));

    // 实际的windowSize = 数据区（x和scales）+ 状态区（1Mb）
    uint64_t actualWinSize = xDataSize + scalesDataSize + MB_SIZE;
    uint64_t maxWinSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
    OP_TILING_CHECK(
        HCCL_BUFFSIZE_FACTOR * actualWinSize > maxWinSize,
        OP_LOGE(nodeName,
                "The HCCL_BUFFSIZE is too small. The current HCCL_BUFFSIZE in the environment is [%lu] MB,"
                "but the NEED HCCL_BUFFSIZE is [%lu] MB. Please check HCCL_BUFFSIZE config.",
                ops::CeilDiv(maxWinSize, MB_SIZE), ops::CeilDiv(actualWinSize, MB_SIZE) * HCCL_BUFFSIZE_FACTOR),
        return false);
    return true;
}

/**
 * @brief 设置workspace
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus SetWorkSpace(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workSpaces"), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验NpuArch
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
ge::graphStatus QuantReduceScatterUtilTiling::CheckNpuArch(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    // 校验NpuArch
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfoPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "platformInfoPtr"),
                    return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    OP_TILING_CHECK(npuArch != NpuArch::DAV_3510,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "npuArch", "non-DAV_3510",
                                                          "The value of npuArch must be DAV_3510"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 功能函数：quant_all_reduce tiling部分总的校验函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param runInfo: 封装的doTiling所需要的参数
 * @param opType: 当前op类型
 * @return
 */
ge::graphStatus QuantReduceScatterUtilTiling::CheckTilingFunc(gert::TilingContext *context, TilingRunInfo &runInfo,
                                                              const OpType opType)
{
    const char *nodeName = context->GetNodeName();
    // set group
    if (CheckAttrsInfo(context, runInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(nodeName, "CheckAttrsInfo failed");
        return ge::GRAPH_FAILED;
    }
    // set rankSize
    OP_TILING_CHECK(SetRankSize(context, runInfo) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "set rankSize failed."),
                    return ge::GRAPH_FAILED);
    // set quantMode
    if (!CheckTensorDataType(context, runInfo)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "tensor", "tensor_dtype_invalid", "check dtype failed");
        return ge::GRAPH_FAILED;
    }
    if (!CheckInputTensorDim(context, runInfo, opType)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "input tensor", "input_dim_invalid",
                                                 "check input dim failed");
        return ge::GRAPH_FAILED;
    }
    if (!CheckOutputTensorDim(context, runInfo, opType)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "output tensor", "output_dim_invalid",
                                                 "check output dim failed");
        return ge::GRAPH_FAILED;
    }
    if (!CheckTensorFormat(context)) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(nodeName, "tensor", "tensor_format_invalid", "check format failed");
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(!CheckWindowSize(context, runInfo), OP_LOGE(nodeName, "HCCL_BUFFSIZE is too small."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetWorkSpace(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "set workspace failed."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace MC2Tiling
