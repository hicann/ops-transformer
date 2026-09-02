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
 * \file matmul_allto_all_tiling_base.cpp
 * \brief
 */
#include "matmul_allto_all_tiling_base.h"
#include "mc2_log.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace MC2Tiling {

/**
 * @brief 基类private私有方法，仅用于算子名称初始化
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAllToAllTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
};

/**
 * @brief 获取平台相关信息
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAllToAllTilingBase::GetPlatformInfo()
{
    fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Fail to get platform info."), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    contextInfo.args_.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
};

/**
 * @brief 基类private私有方法，原本是用于高阶api，但在业务中实际没有使用到
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAllToAllTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取额外申请的空间
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAllToAllTilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "Get workspace failed"), return ge::GRAPH_FAILED);
    SetUserWorkSpace();
    uint64_t workspaceSize_ = libApiWorkSpaceSize_ + inferredInfo.mmResultLen + inferredInfo.permuteLen;
    workspaces[0] = workspaceSize_;
    OP_LOGD(opName_, "Workspaces[0] size=%zu, mmResultLen=%lu", workspaces[0], inferredInfo.mmResultLen);
    return ge::GRAPH_SUCCESS;
}

CutResult MatmulAllToAllTilingBase::GetTilingResult()
{
    SocVersion nowSocVersion = SocVersion::SOC950;
    std::string socVersionStr = mc2tiling::GetSocVersion(context_);
    if (socVersionStr == "Ascend910_93") {
        nowSocVersion = SocVersion::SOC910_93;
    }
    AlltoAllMM formulaicTiling(contextInfo.args_, contextInfo.args_.rankDim, KernelType::ALL_TO_ALL, nowSocVersion);
    formulaicTiling.GetTiling();
    return formulaicTiling.tilingM_.cutRes;
}

/**
 * @brief 返回具体的切分结果
 *
 */
ge::graphStatus MatmulAllToAllTilingBase::TileCommAndCompute()
{
    OP_LOGD(opName_, "Start to find proper tile by formulaic tiling.");
    CutResult mCutMMAlltoAll = GetTilingResult();
    // 根据 shortTileAtBack 和 numShortTile 确定主块和尾块
    const bool useLongAsMain = mCutMMAlltoAll.shortTileAtBack || mCutMMAlltoAll.numShortTile == 0;

    inferredInfo.tileCnt = useLongAsMain ? mCutMMAlltoAll.numLongTile : mCutMMAlltoAll.numShortTile;
    inferredInfo.tileM = useLongAsMain ? mCutMMAlltoAll.longTileLen : mCutMMAlltoAll.shortTileLen;
    inferredInfo.tailCnt = useLongAsMain ? mCutMMAlltoAll.numShortTile : mCutMMAlltoAll.numLongTile;
    inferredInfo.tailM = useLongAsMain ? mCutMMAlltoAll.shortTileLen : mCutMMAlltoAll.longTileLen;

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置额外需要的空间，包括计算结果地址，重排地址，偏移地址等
 *
 */
void MatmulAllToAllTilingBase::SetUserWorkSpace()
{
    constexpr uint64_t alignAddrLen = 512;
    // MatmulAlltoAll先进行计算，需要有对应的空间先存放结果，假设x1(m,k),
    // x2(k,n),那么计算结果大小为m*n,这里申请的是一块总的空间，通算切分的头尾块偏移由kernel侧自行计算
    inferredInfo.mmResultLen = mc2tiling::AlignUp(
        contextInfo.args_.mValue * contextInfo.args_.nValue * contextInfo.args_.outputDtypeSize, alignAddrLen);
    // 重排空间等于mm计算结果空间
    inferredInfo.permuteLen = inferredInfo.mmResultLen;
}

/**
 * @brief 校验MatmulAlltoAll在不同转置情况下的x1,x2,output的shape关系,以及需要满足n/rankSize的整除关系
 * 需要满足 x1(BS,H1), x2(H2, H1) if trans else x2(H1, H2)
 * output(BS*rankSize, H2/rankSize)
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAllToAllTilingBase::Check2DMatrixMulShapes(const gert::TilingContext *context, const char *opName)
{
    bool x2TransFlag = false;
    // attr及其元素的非空校验在前置的Check方法里都校验过，所以这里不需要额外判断
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const bool *isTransX2 = attrs->GetAttrPointer<bool>(ATTR_X2_TRANSPOSE_INDEX);
    if (isTransX2) {
        x2TransFlag = *isTransX2;
    }
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);

    int64_t rankDim = 0;
    if (MatmulAlltoAllTilingUtil::GetAndValidateRankSize(context, opName, group, rankDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    Matrix2DShapes shapeInfo;
    MatmulAlltoAllTilingUtil::GetMatrix2DShapes(context, shapeInfo);
    uint64_t kAxis = x2TransFlag ? shapeInfo.x2Dim1 : shapeInfo.x2Dim0;
    uint64_t nAxis = x2TransFlag ? shapeInfo.x2Dim0 : shapeInfo.x2Dim1;
    // MatmulAlltoAll, n要整除rankSize
    OP_TILING_CHECK(nAxis % static_cast<uint64_t>(rankDim) != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "n", std::to_string(nAxis).c_str(),
                                                          "The value of n must be divisible by rankSize"),
                    return ge::GRAPH_FAILED);
    // MatmulAlltoAll: x1Dim1 = x2 K-axis
    OP_TILING_CHECK(
        (shapeInfo.x1Dim1 != kAxis),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName, "x1dim1 and x2Kdim", (std::to_string(shapeInfo.x1Dim1) + " vs " + std::to_string(kAxis)).c_str(),
            "The second dim of x1 must be equal to the K-axis dim of x2"),
        return ge::GRAPH_FAILED);
    // MatmulAlltoAll: yDim0 = x1Dim0 * rankDim and x2 N-axis = yDim1 * rankDim
    OP_TILING_CHECK(
        (((shapeInfo.x1Dim0 * rankDim) != shapeInfo.yDim0) || (nAxis != (shapeInfo.yDim1 * rankDim))),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName, "x1_x2_y",
            (std::string("x1Dim0=") + std::to_string(shapeInfo.x1Dim0) + " yDim0=" + std::to_string(shapeInfo.yDim0) +
             " nAxis=" + std::to_string(nAxis) + " yDim1=" + std::to_string(shapeInfo.yDim1))
                .c_str(),
            "The value of yDim0 must equal x1Dim0 * rankSize and nAxis must equal yDim1 * rankSize"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验KC量化MatmulAlltoAll在不同转置情况下的x1,x2,output的shape关系,以及需要满足n/rankSize的整除关系
 * 需要满足 x1(BS,H1), x2(H2, H1) if trans else x2(H1, H2)
 * output(BS*rankSize, H2/rankSize)
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAllToAllTilingBase::CheckKcQuantMatrixMulShapes(const gert::TilingContext *context,
                                                                      const char *opName)
{
    OP_TILING_CHECK(Check2DMatrixMulShapes(context, opName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Kc quant tiling check x1 x2 and y shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckKcQuantScaleShapes(context, opName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Kc quant tiling check scale shape failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验KC量化scale的维度
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAllToAllTilingBase::CheckKcQuantScaleShapes(const gert::TilingContext *context,
                                                                  const char *opName)
{
    bool TransX2Flag = false;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const bool *isX2TransX2 = attrs->GetAttrPointer<bool>(ATTR_X2_TRANSPOSE_INDEX);

    if (isX2TransX2) {
        TransX2Flag = *isX2TransX2;
    }
    Matrix2DShapes shapeInfo;
    MatmulAlltoAllTilingUtil::GetMatrix2DShapes(context, shapeInfo);
    uint64_t nAxis = TransX2Flag ? shapeInfo.x2Dim0 : shapeInfo.x2Dim1;
    const gert::StorageShape *x1ScaleShape = context->GetOptionalInputShape(INPUT_X1_SCALE_INDEX);
    const gert::StorageShape *x2ScaleShape = context->GetOptionalInputShape(INPUT_X2_SCALE_INDEX);
    uint64_t x1ScaleDim = x1ScaleShape->GetStorageShape().GetDim(0);
    uint64_t x2ScaleDim = x2ScaleShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(
        (x1ScaleDim != shapeInfo.x1Dim0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName, "x1Scale", (std::to_string(x1ScaleDim) + " vs " + std::to_string(shapeInfo.x1Dim0)).c_str(),
            "The dim of x1Scale must be equal to the first dim of x1"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2ScaleDim != nAxis),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName, "x2Scale", (std::to_string(x2ScaleDim) + " vs " + std::to_string(nAxis)).c_str(),
                        "The dim of x2Scale must be equal to the second dim of x2"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace MC2Tiling
