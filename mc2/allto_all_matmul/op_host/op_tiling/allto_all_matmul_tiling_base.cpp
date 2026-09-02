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
 * \file allto_all__matmul_tiling_base.cpp
 * \brief
 */
#include "allto_all_matmul_tiling_base.h"

#include "mc2_log.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace MC2Tiling {

/**
 * @brief 工具函数：判断指定value是否存在于list中
 *
 * @param list: 有效值列表
 * @param value: 给定值
 * @return
 */
static bool IsContain(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::count(list.begin(), list.end(), value) > 0;
}

/**
 * @brief 基类private私有方法，仅用于算子名称初始化
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMatmulTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
};

/**
 * @brief 获取平台相关信息
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMatmulTilingBase::GetPlatformInfo()
{
    fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Fail to get platform info"), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    contextInfo_.args_.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
};

/**
 * @brief 基类private私有方法，原本是用于高阶api，但在业务中实际没有使用到
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMatmulTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllToAllMatmulTilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed"), return ge::GRAPH_FAILED);
    SetUserWorkSpace();
    uint64_t workspaceSize =
        libApiWorkSpaceSize_ + inferredInfo_.commLen + inferredInfo_.permuteLen + inferredInfo_.biasLen;
    workspaces[0] = workspaceSize;
    OP_LOGD(opName_, "Workspaces[0] size=%zu, biasLen=%u, commlen=%lu", workspaces[0], inferredInfo_.biasLen,
            inferredInfo_.commLen);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取对应的tilingKey
 * @return uint64_t tilingKey结果
 */
uint64_t AllToAllMatmulTilingBase::GetTilingKey() const
{
    return 0;
}

/**
 * @brief 设置额外需要的空间，包括通信结果地址，重排地址，偏移地址等
 *
 */
void AllToAllMatmulTilingBase::SetUserWorkSpace()
{
    constexpr uint64_t alignAddrLen = 512;
    // AlltoAllMatmul先进行通信，需要有对应的空间先存放结果，假设x1(m,k),假设原始rank上X1的第0维为M，这里的m根据kernel需要取完整的M,
    // m已经在前面获取输入参数的时候进行过处理
    inferredInfo_.commLen = mc2tiling::AlignUp(
        contextInfo_.args_.orgMValue * contextInfo_.args_.orgKValue * contextInfo_.args_.inputDtypeSize, alignAddrLen);
    // 重排空间等于通信结果结果空间,如果存在alltoallout空间的话，不需要申请这块
    if (!contextInfo_.allToAllOutFlag) {
        inferredInfo_.permuteLen = inferredInfo_.commLen;
    }
    if (contextInfo_.args_.isBias) {
        inferredInfo_.biasLen =
            mc2tiling::AlignUp(contextInfo_.args_.nValue, mc2tiling::SHAPE_ALIGN_SIZE) * sizeof(float);
    }
}

CutResult AllToAllMatmulTilingBase::GetCutResOfCommAndCompute()
{
    OP_LOGD(opName_, "Start to find proper tile by formulaic tiling.");
    std::string socVersionStr = mc2tiling::GetSocVersion(context_);
    OP_LOGD(opName_, "Current SocVersion is : %s", socVersionStr.c_str());

    SocVersion nowSocVersion = SocVersion::SOC950;
    if (socVersionStr == "Ascend910_93") {
        nowSocVersion = SocVersion::SOC910_93;
    }
    AlltoAllMM alltoallMatmulTileFormulate(contextInfo_.args_, contextInfo_.args_.rankDim, KernelType::ALL_TO_ALL,
                                           nowSocVersion, true);
    alltoallMatmulTileFormulate.GetTiling();
    return alltoallMatmulTileFormulate.tilingM_.cutRes;
}

/**
 * @brief 进行通算切分:使用公式化tiling的方式，当前阶段公式化tiling只是个预估，AlltoAllMatmul传递的内轴为K,与
 * MatmulAlltoAll的要区分开，用一个额外参数进行隔离
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMatmulTilingBase::TileCommAndCompute()
{
    CutResult cutRes = GetCutResOfCommAndCompute();
    if (!cutRes.shortTileAtBack && cutRes.numShortTile > 0) {
        inferredInfo_.tileM = cutRes.shortTileLen;
        inferredInfo_.tileCnt = cutRes.numShortTile;
        if (cutRes.numLongTile > 0) {
            inferredInfo_.tailM = cutRes.longTileLen;
            inferredInfo_.tailCnt = cutRes.numLongTile;
        }
    } else {
        inferredInfo_.tileM = cutRes.longTileLen;
        inferredInfo_.tileCnt = cutRes.numLongTile;
        inferredInfo_.tailM = 0;
        inferredInfo_.tailCnt = 0;
        if (cutRes.numShortTile > 0) {
            inferredInfo_.tailM = cutRes.shortTileLen;
            inferredInfo_.tailCnt = cutRes.numShortTile;
        }
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验alltoAllOut的shape约束情况
 *
 * @param context
 * @param opName
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMatmulTilingBase::CheckAlltoAllOutShape(const gert::TilingContext *context, const char *opName)
{
    // shape校验,假设x1的shape为（BS,H), 则alltoallout的shape应该为(BS/rankSize,
    // H*rankSize),在checkShapeInfo的时候校验 了BS能整除rankSize
    // 前面校验过，如果为-1的话，说明要调用接口获取
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    int64_t rankDim = 0;
    if (MatmulAlltoAllTilingUtil::GetAndValidateRankSize(context, opName, group, rankDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    uint64_t x1Dim0 = x1Shape->GetStorageShape().GetDim(0);
    uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
    const gert::StorageShape *allToAllOutShape = context->GetOutputShape(ALLTO_ALL_OUT_INDEX);
    uint64_t outDim0 = allToAllOutShape->GetStorageShape().GetDim(0);
    uint64_t outDim1 = allToAllOutShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        x1Dim0 != (outDim0 * rankDim),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x1 and allToAllOut",
                                              (Ops::Base::ToString(x1Shape->GetStorageShape()) + " and " +
                                               Ops::Base::ToString(allToAllOutShape->GetStorageShape()))
                                                  .c_str(),
                                              "The first dim of x1 should be rankSize times of allToAllOut first dim"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief allToAllMatmul专有，单独校验
 *
 * @param context
 * @param opName
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMatmulTilingBase::CheckAlltoAllOut(const gert::TilingContext *context, const char *opName)
{
    // 非空校验在CheckAttrsInfo中已经校验过
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const bool *allToAllOutFlag = attrs->GetAttrPointer<bool>(ALLTOALLMATMUL_ATTR_ALLTO_ALL_OUT_FLAG_INDEX);
    contextInfo_.allToAllOutFlag = (allToAllOutFlag != nullptr) ? *allToAllOutFlag : false;
    // 当alltoAlloutFlag非空时，需要进行校验:数据类型、DIM维度数、shape
    if (contextInfo_.allToAllOutFlag) {
        // 数据类型校验, x1的数据类型已经在前面校验过
        auto x1TensorDesc = context->GetInputDesc(INPUT_X1_INDEX);
        ge::DataType x1DType = x1TensorDesc->GetDataType();
        auto allToAllOutDesc = context->GetOutputDesc(ALLTO_ALL_OUT_INDEX);
        OP_TILING_CHECK((allToAllOutDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "alltoAllOut"),
                        return ge::GRAPH_FAILED);
        ge::DataType allToAlloutDtype = allToAllOutDesc->GetDataType();
        OP_TILING_CHECK(
            (allToAlloutDtype != x1DType),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "alltoAllOut", Ops::Base::ToString(allToAlloutDtype).c_str(),
                                                  "The dtype of allToAllOut must be same as input x"),
            return ge::GRAPH_FAILED);
        // dim维度校验
        const gert::StorageShape *allToAllOutShape = context->GetOutputShape(ALLTO_ALL_OUT_INDEX);
        OP_TILING_CHECK((allToAllOutShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "allToAllOut"),
                        return ge::GRAPH_FAILED);
        uint64_t dimNum = allToAllOutShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            (dimNum != TWO_DIMS),
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "allToAllOut", (std::to_string(dimNum) + "D").c_str(),
                                                     "The shape dim of allToAllOut must be 2D"),
            return ge::GRAPH_FAILED);
        // shape校验
        OP_TILING_CHECK(CheckAlltoAllOutShape(context, opName) != ge::GRAPH_SUCCESS,
                        OP_LOGE(opName_, "Tiling check allToAllOutShape failed."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验AlltoAllMatmul在不同转置情况下的x1,x2,output的shape关系
 * 需要满足 x1(BS,H1), x2(H2, H1*rankSize) if trans else x2(H1*rankSize, H2)
 * output(BS/rankSize, H2)
 *
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMatmulTilingBase::CheckMatrixMulShapes(const gert::TilingContext *context, const char *opName)
{
    // attr及其元素的非空校验在前置的Check方法里都校验过，所以这里不需要额外判断
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    int64_t rankDim = 0;
    if (MatmulAlltoAllTilingUtil::GetAndValidateRankSize(context, opName, group, rankDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    bool x2TransFlag = false;
    const bool *isTransX2 = attrs->GetAttrPointer<bool>(ALLTOALLMATMUL_ATTR_X2_TRANSPOSE_INDEX);
    if (isTransX2) {
        x2TransFlag = *isTransX2;
    }
    Matrix2DShapes shapeInfo;
    MatmulAlltoAllTilingUtil::GetMatrix2DShapes(context, shapeInfo);
    uint64_t kAxis = x2TransFlag ? shapeInfo.x2Dim1 : shapeInfo.x2Dim0;
    uint64_t nAxis = x2TransFlag ? shapeInfo.x2Dim0 : shapeInfo.x2Dim1;
    // AlltoAllMatmul,m要整除rankSize
    OP_TILING_CHECK(shapeInfo.x1Dim0 % static_cast<uint64_t>(rankDim) != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x1", std::to_string(shapeInfo.x1Dim0).c_str(),
                                                          "The shape of x1 must be divisible by rankSize on first dim"),
                    return ge::GRAPH_FAILED);
    // AlltoAllMatmul: x2 K-axis = x1Dim1 * rankDim
    OP_TILING_CHECK((kAxis != (shapeInfo.x1Dim1 * rankDim)),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName, "x1 and x2", std::to_string(kAxis).c_str(),
                        "The shape of x1 and x2 must satisfy K-axis of x2 equals second dim of x1 times rankSize"),
                    return ge::GRAPH_FAILED);
    // AlltoAllMatmul: x1Dim0 = yDim0 * rankDim and x2 N-axis = yDim1
    OP_TILING_CHECK(
        ((shapeInfo.x1Dim0 != (shapeInfo.yDim0 * rankDim)) || (nAxis != shapeInfo.yDim1)),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName, "x1, x2 and y", (std::to_string(shapeInfo.x1Dim0) + ", " + std::to_string(nAxis)).c_str(),
            "The shapes of x1, x2 and y must satisfy first dim of x1 equals rankDim times first dim of y "
            "and N-axis of x2 equals second dim of y"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置AlltoAllMatmul算子的shape信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param contextInfo 存储了tiling的过程信息
 * @return ge::graphStatus
 */
ge::graphStatus AllToAllMatmulTilingBase::SetAlltoAllMatmulShapeInfo(const gert::TilingContext *context,
                                                                     TilingContextInfo &contextInfo)
{
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    uint64_t x1Dim0 = x1Shape->GetStorageShape().GetDim(0);
    uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);

    contextInfo.args_.orgMValue = x1Dim0;
    contextInfo.args_.orgNValue = (contextInfo.args_.isBTrans) ? x2Dim0 : x2Dim1;
    contextInfo.args_.orgKValue = x1Dim1;

    // 对于AlltoAllMatmul, m要除以rank数，k取完整的k,也就是x2的k维度
    contextInfo.args_.mValue = x1Dim0 / contextInfo.args_.rankDim;
    contextInfo.args_.kValue = (contextInfo.args_.isBTrans) ? x2Dim1 : x2Dim0;
    contextInfo.args_.nValue = (contextInfo.args_.isBTrans) ? x2Dim0 : x2Dim1;
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
ge::graphStatus AllToAllMatmulTilingBase::CheckKcQuantTensorDataType(const gert::TilingContext *context,
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
    const std::vector<uint32_t> KC_QUANT_X1_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
    const std::vector<uint32_t> KC_QUANT_X2_DTYPE_LIST = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
    OP_TILING_CHECK(!IsContain(KC_QUANT_X1_DTYPE_LIST, x1Dtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "x1", Ops::Base::ToString(x1Dtype).c_str(),
                                                          "The dtype of x1 must be float16 or bf16 in kc-quant range"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !IsContain(KC_QUANT_X2_DTYPE_LIST, x2Dtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "x2", Ops::Base::ToString(x2Dtype).c_str(),
                                              "The dtype of x2 must be float8_e4m3fn or float8_e5m2 in kc-quant range"),
        return ge::GRAPH_FAILED);
    // 校验 bias 数据类型（如果存在）
    auto biasTensorDesc = context->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    if (biasTensorDesc != nullptr) {
        ge::DataType biasDtype = biasTensorDesc->GetDataType();
        OP_TILING_CHECK((biasDtype != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "bias", Ops::Base::ToString(biasDtype).c_str(),
                                                              "The dtype of bias must be FLOAT"),
                        return ge::GRAPH_FAILED);
    }
    auto x2ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((x2ScaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2Scale"),
                    return ge::GRAPH_FAILED);
    ge::DataType x2scaleDtype = x2ScaleTensorDesc->GetDataType();
    OP_TILING_CHECK((x2scaleDtype != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "x2Scale", Ops::Base::ToString(x2scaleDtype).c_str(),
                                                          "The dtype of x2Scale must be FLOAT in kc quant mode"),
                    return ge::GRAPH_FAILED);
    // 校验输出张量数据类型
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "y"), return ge::GRAPH_FAILED);
    ge::DataType yDtype = yDesc->GetDataType();
    OP_TILING_CHECK(!IsContain(MC2Tiling::KC_QUANT_Y_DTYPE_LIST, yDtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "y", Ops::Base::ToString(yDtype).c_str(),
                                                          "The dtype of y must be float16, bfloat16 or float"),
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
ge::graphStatus AllToAllMatmulTilingBase::CheckKcQuantShapeInfo(const gert::TilingContext *context, const char *opName,
                                                                const OpAttrIndexSchema &indexSchema)
{
    OP_TILING_CHECK(
        MatmulAlltoAllTilingUtil::CheckShapeInfo(context, opName, ALLTOALL_MATMUL_INDEX_SCHEMA) != ge::GRAPH_SUCCESS,
        OP_LOGE(opName, "Tiling common info check shape failed."), return ge::GRAPH_FAILED);
    ge::graphStatus status;
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *x2ScaleShape = context->GetOptionalInputShape(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((x1Shape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2Shape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2ScaleShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2Scale"), return ge::GRAPH_FAILED);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);
    uint64_t x2ScaleDimNum = x2ScaleShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        (x2ScaleDimNum != 1),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "x2Scale", (std::to_string(x2ScaleDimNum) + "D").c_str(),
                                                 "The shape dim of x2Scale must be 1D"),
        return ge::GRAPH_FAILED);
    uint64_t x2ScaleDim0 = x2ScaleShape->GetStorageShape().GetDim(0);
    bool x2IsTransFlag = false;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const bool *isTransX2 = attrs->GetAttrPointer<bool>(indexSchema.x2Transpose);
    if (isTransX2) {
        x2IsTransFlag = *isTransX2;
    }
    uint64_t nAxis = (x2IsTransFlag) ? x2Dim0 : x2Dim1;
    OP_TILING_CHECK((x2ScaleDim0 != nAxis),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x2Scale", std::to_string(x2ScaleDim0).c_str(),
                                                          "The shape of x2Scale dim0 must be " + std::to_string(nAxis)),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace MC2Tiling
