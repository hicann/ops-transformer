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
 * \file mhc_sinkhorn_tiling.cpp
 * \brief mhc_sinkhorn_tiling
 */

#include <vector>
#include <string>
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "mhc_sinkhorn_tiling.h"

using namespace AscendC;
namespace optiling {
constexpr int64_t X_IDX = 0;
constexpr int64_t Y_IDX = 0;
constexpr int64_t ATTR_EPS_IDX = 0;
constexpr int64_t ATTR_NUM_ITERS_IDX = 1;
constexpr int64_t ATTR_OUT_FLAG_IDX = 2;

constexpr int64_t DIM_NUM_3 = 3;
constexpr int64_t DIM_NUM_4 = 4;
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;

constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t NUM_ONE_HUNDRED = 100;

constexpr int64_t N_NUM_4 = 4;
constexpr int64_t N_NUM_6 = 6;
constexpr int64_t N_NUM_8 = 8;

constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 0;

static const std::set<ge::DataType> X_DTYPE = {ge::DT_FLOAT};

bool MhcSinkhornTiling::IsCapable()
{
    return true;
}

ge::graphStatus MhcSinkhornTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const MhcSinkhornCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    if (ubSize_ <= 0) {
        OP_LOGE(opName_, "ub size less than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornTiling::GetShapeAttrsInfo()
{
    OP_LOGD(opName_, "MhcSinkhorn tiling GetShapeAttrsInfo.");
    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto epsPtr = attrs->GetAttrPointer<float>(ATTR_EPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, epsPtr);
    eps_ = static_cast<float>(*epsPtr);
    auto numItersPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ITERS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numItersPtr);
    num_iters_ = static_cast<int64_t>(*numItersPtr);
    if (num_iters_ < NUM_ONE || num_iters_ > NUM_ONE_HUNDRED) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "num_iters", std::to_string(num_iters_).c_str(),
                                              "num_iters must be between 1 and 100");
        return ge::GRAPH_FAILED;
    }
    auto outFlagPtr = attrs->GetAttrPointer<int64_t>(ATTR_OUT_FLAG_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outFlagPtr);
    outFlag_ = static_cast<int64_t>(*outFlagPtr);
    if (outFlag_ != NUM_ZERO && outFlag_ != NUM_ONE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "out_flag", std::to_string(outFlag_).c_str(),
                                              "out_flag must be 0 or 1");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(CheckInputDtype() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "input dtype check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckInputShape() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "input shape check failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornTiling::CheckInputDtype()
{
    auto xPtr = context_->GetInputDesc(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xPtr);
    xDtype_ = xPtr->GetDataType();
    if (X_DTYPE.find(xDtype_) == X_DTYPE.end()) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "x", ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str(),
                                              "x dtype only supports float32, please check");
        return ge::GRAPH_FAILED;
    }
    xDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(xDtype_));

    auto outputPtr = context_->GetOutputDesc(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputPtr);
    auto outputDtype = outputPtr->GetDataType();
    if (outputDtype != xDtype_) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "y", ge::TypeUtils::DataTypeToSerialString(outputDtype).c_str(),
                                               "the dtype of y must be equal to the dtype of x");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornTiling::CheckInputShape()
{
    auto xShapePtr = context_->GetInputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    if (xShape.GetShapeSize() == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "x", "empty tensor", "each dimension must be >= 1");
        return ge::GRAPH_FAILED;
    }
    xDimNum_ = static_cast<int64_t>(xShape.GetDimNum());
    if (xDimNum_ != DIM_NUM_3 && xDimNum_ != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "x", std::to_string(xDimNum_).c_str(),
                                                 "the dim of x must be 3 or 4");
        return ge::GRAPH_FAILED;
    }
    auto n0 = xShape.GetDim(DIM_TWO);
    if (xDimNum_ == 3) {
        n0 = xShape.GetDim(DIM_ONE);
        n_ = xShape.GetDim(DIM_TWO);
        T_ = xShape.GetDim(DIM_ZERO);
    } else {
        n_ = xShape.GetDim(DIM_THREE);
        T_ = xShape.GetDim(DIM_ZERO) * xShape.GetDim(DIM_ONE);
    }
    if (n_ != n0) {
        std::string shapeMsg = "{" + std::to_string(n0) + ", " + std::to_string(n_) + "}";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "n0, n1", shapeMsg.c_str(), "n0 must be equal to n1");
        return ge::GRAPH_FAILED;
    }
    if (n_ != N_NUM_4 && n_ != N_NUM_6 && n_ != N_NUM_8) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "x_n", std::to_string(n_).c_str(),
                                                 "x nDim must be 4, 6, or 8");
        return ge::GRAPH_FAILED;
    }

    auto yShapePtr = context_->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();
    yDimNum_ = static_cast<int64_t>(yShape.GetDimNum());
    if (yDimNum_ != DIM_NUM_3 && yDimNum_ != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "y", std::to_string(yDimNum_).c_str(),
                                                 "the dim of y must be 3 or 4");
        return ge::GRAPH_FAILED;
    }
    int64_t n = yShape.GetDim(DIM_TWO);
    if (n != 4 && n != 6 && n != 8) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "y_n", std::to_string(n).c_str(),
                                                 "y nDim must be 4, 6, or 8");
        return ge::GRAPH_FAILED;
    }
    if (yDimNum_ != xDimNum_) {
        std::string dimMsg = "y=" + std::to_string(yDimNum_) + ", x=" + std::to_string(xDimNum_);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "x, y", dimMsg.c_str(), "y dim must equal x dim");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void MhcSinkhornTiling::SplitByCoreNum()
{
    auto splitInfo = BaseSplitCores(T_, n_, xDtypeSize_, outFlag_);

    usedCoreNum_ = splitInfo.usedCoreNum;
    tNormCore_ = splitInfo.tNormCore;
    tNormCoreLoop_ = splitInfo.tNormCoreLoop;
    tUbFactor_ = splitInfo.tUbFactor;
    tUbFactorTail_ = splitInfo.tUbFactorTail;
    tTailCoreLoop_ = splitInfo.tTailCoreLoop;
    tUbTailTail_ = splitInfo.tUbTailTail;
}

ge::graphStatus MhcSinkhornTiling::DoOpTiling()
{
    OP_LOGD(opName_, "MhcSinkhorn tiling DoOpTiling.");

    SplitByCoreNum();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void MhcSinkhornTiling::SetTilingData()
{
    MhcSinkhornTilingData *tilingData = context_->GetTilingData<MhcSinkhornTilingData>();
    tilingData->eps = eps_;
    tilingData->num_iters = num_iters_;
    tilingData->out_flag = outFlag_;
    tilingData->n = n_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->tNormCoreLoop = tNormCoreLoop_;
    tilingData->tUbFactor = tUbFactor_;
    tilingData->tUbFactorTail = tUbFactorTail_;
    tilingData->tTailCoreLoop = tTailCoreLoop_;
    tilingData->tUbTailTail = tUbTailTail_;
    tilingData->tNormCore = tNormCore_;
}

ge::graphStatus MhcSinkhornTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcSinkhornTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    if (outFlag_ == 1) {
        tilingKey = 1;
    }
    return GET_TPL_TILING_KEY(tilingKey);
}

ge::graphStatus MhcSinkhornTiling::GetWorkspaceSize()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    auto res = context_->SetLocalMemorySize(ubSize_);
    if (res != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "SetLocalMemorySize ubSize = %ld failed.", ubSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void MhcSinkhornTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "\n eps: " << eps_;
    info << "\n numIters: " << num_iters_;
    info << "\n outFlag: " << outFlag_;
    info << "\n T: " << T_;
    info << "\n n: " << n_;
    info << "\n tilingKey: " << GetTilingKey();
    info << "\n usedCoreNum: " << usedCoreNum_;
    info << "\n tNormCoreLoop: " << tNormCoreLoop_;
    info << "\n tUbFactor: " << tUbFactor_;
    info << "\n tUbFactorTail: " << tUbFactorTail_;
    info << "\n tTailCoreLoop: " << tTailCoreLoop_;
    info << "\n tUbTailTail: " << tUbTailTail_;
    info << "\n tNormCore: " << tNormCore_;
    OP_LOGI(opName_, "%s", info.str().c_str());
}

static ge::graphStatus TilingForMhcSinkhorn(gert::TilingContext *context)
{
    MhcSinkhornTiling tiling(context);
    auto ret = tiling.DoTiling();
    return ret;
}

static ge::graphStatus TilingPrepareForMhcSinkhorn([[maybe_unused]] gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepareForMhcSinkhorn entering.");
    auto compileInfo = context->GetCompiledInfo<MhcSinkhornCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->coreNum <= 0) {
        OP_LOGE(context, "Failed to get core num.");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    if (compileInfo->ubSize <= 0) {
        OP_LOGE(context, "Failed to get ub size.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcSinkhorn)
    .Tiling(TilingForMhcSinkhorn)
    .TilingParse<MhcSinkhornCompileInfo>(TilingPrepareForMhcSinkhorn);
} // namespace optiling
