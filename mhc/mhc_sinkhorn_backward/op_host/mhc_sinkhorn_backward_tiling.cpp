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
 * \file mhc_sinkhorn_backward_tiling.cpp
 * \brief mhc_sinkhorn_backward_tiling
 */

#include <vector>
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "mhc_sinkhorn_backward_tiling.h"

using namespace AscendC;
namespace optiling {

const static int64_t GRAD_Y_INPUT_INDEX = 0;
const static int64_t NORM_INPUT_INDEX = 1;
const static int64_t SUM_INPUT_INDEX = 2;
const static int64_t GRAD_INPUT_OUTPUT_INDEX = 0;

const static int64_t GRAD_Y_INPUT_BS_DIMNUM = 4;
const static int64_t GRAD_Y_INPUT_T_DIMNUM = 3;
const static int64_t NORM_INPUT_DIMNUM = 1;
const static int64_t SUM_INPUT_DIMNUM = 1;
const static int64_t GRAD_INPUT_OUTPUT_BS_DIMNUM = 4;
const static int64_t GRAD_INPUT_OUTPUT_T_DIMNUM = 3;

const static int64_t MIN_PER_CORE_ELEMENTS = 32;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;

constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 0;

ge::graphStatus MhcSinkhornBackwardTiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();

    // 获取输入shape信息
    auto gradYShapePtr = context_->GetInputShape(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYShapePtr);
    gradYShape_ = &gradYShapePtr->GetStorageShape();

    auto normShapePtr = context_->GetInputShape(NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, normShapePtr);
    normShape_ = &normShapePtr->GetStorageShape();

    auto sumShapePtr = context_->GetInputShape(SUM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sumShapePtr);
    sumShape_ = &sumShapePtr->GetStorageShape();

    // 获取输出shape
    auto gradInputShapePtr = context_->GetOutputShape(GRAD_INPUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradInputShapePtr);
    gradInputShape_ = &gradInputShapePtr->GetStorageShape();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const MhcSinkhornBackwardCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::CheckDtype()
{
    // 验证输入数据类型
    auto gradYDesc = context_->GetInputDesc(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYDesc);
    auto gradYDtype = gradYDesc->GetDataType();
    OP_CHECK_IF((gradYDtype != ge::DataType::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "grad_y",
                                                      ge::TypeUtils::DataTypeToSerialString(gradYDtype).c_str(),
                                                      "the dtype of grad_y must be float32"),
                return ge::GRAPH_FAILED);

    inputDTypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(gradYDesc->GetDataType()));

    auto normDesc = context_->GetInputDesc(NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, normDesc);
    auto normDtype = normDesc->GetDataType();
    OP_CHECK_IF(
        (normDtype != ge::DataType::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "norm", ge::TypeUtils::DataTypeToSerialString(normDtype).c_str(),
                                              "the dtype of norm must be float32"),
        return ge::GRAPH_FAILED);

    auto sumDesc = context_->GetInputDesc(SUM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sumDesc);
    auto sumDtype = sumDesc->GetDataType();
    OP_CHECK_IF(
        (sumDtype != ge::DataType::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "sum", ge::TypeUtils::DataTypeToSerialString(sumDtype).c_str(),
                                              "the dtype of sum must be float32"),
        return ge::GRAPH_FAILED);

    // 验证输出数据类型
    auto gradInputDesc = context_->GetOutputDesc(GRAD_INPUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradInputDesc);
    auto gradInputDtype = gradInputDesc->GetDataType();
    OP_CHECK_IF((gradInputDtype != ge::DataType::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "grad_input",
                                                      ge::TypeUtils::DataTypeToSerialString(gradInputDtype).c_str(),
                                                      "the dtype of grad_input must be float32"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::CheckGradYShape()
{
    if (isTShape_) {
        if (gradYShape_->GetDim(DIM_2) != nSize_) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "grad_y",
                                                  ("dim2=" + std::to_string(gradYShape_->GetDim(DIM_2)) +
                                                   ", dim3=" + std::to_string(gradYShape_->GetDim(DIM_3)))
                                                      .c_str(),
                                                  "dim2 and dim3 of grad_y must be equal");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (gradYShape_->GetDim(DIM_3) != nSize_) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "grad_y",
                                                  ("dim2=" + std::to_string(gradYShape_->GetDim(DIM_2)) +
                                                   ", dim3=" + std::to_string(gradYShape_->GetDim(DIM_3)))
                                                      .c_str(),
                                                  "dim2 and dim3 of grad_y must be equal");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::CheckNormShape()
{
    if (normShape_->GetDim(DIM_0) != 2 * numIters_ * totalLength_ * nSize_ * nAlignSize_) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "norm", std::to_string(normShape_->GetDim(DIM_0)).c_str(),
            ("should be " + std::to_string(2 * numIters_ * totalLength_ * nSize_ * nAlignSize_)).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::CheckSumShape()
{
    if (sumShape_->GetDim(DIM_0) != 2 * numIters_ * totalLength_ * nAlignSize_) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "sum", std::to_string(sumShape_->GetDim(DIM_0)).c_str(),
            ("should be " + std::to_string(2 * numIters_ * totalLength_ * nAlignSize_)).c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t numIters = sumShape_->GetDim(DIM_0) / (2 * totalLength_ * nAlignSize_);
    if (numIters != numIters_) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "numIters", std::to_string(numIters).c_str(),
                                              ("should be " + std::to_string(numIters_)).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::CheckGradInputShape()
{
    if (isTShape_) {
        if (gradInputShape_->GetDimNum() != GRAD_INPUT_OUTPUT_T_DIMNUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "grad_input",
                                                     std::to_string(gradInputShape_->GetDimNum()).c_str(),
                                                     ("must be " + std::to_string(GRAD_INPUT_OUTPUT_T_DIMNUM)).c_str());
            return ge::GRAPH_FAILED;
        }

        if (gradInputShape_->GetDim(DIM_0) != tSize_ || gradInputShape_->GetDim(DIM_1) != nSize_ ||
            gradInputShape_->GetDim(DIM_2) != nSize_) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "grad_input",
                                                  ("{" + std::to_string(gradInputShape_->GetDim(DIM_0)) + ", " +
                                                   std::to_string(gradInputShape_->GetDim(DIM_1)) + ", " +
                                                   std::to_string(gradInputShape_->GetDim(DIM_2)) + "}")
                                                      .c_str(),
                                                  ("should be {" + std::to_string(tSize_) + ", " +
                                                   std::to_string(nSize_) + ", " + std::to_string(nSize_) + "}")
                                                      .c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (gradInputShape_->GetDimNum() != GRAD_INPUT_OUTPUT_BS_DIMNUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                opName_, "grad_input", std::to_string(gradInputShape_->GetDimNum()).c_str(),
                ("must be " + std::to_string(GRAD_INPUT_OUTPUT_BS_DIMNUM)).c_str());
            return ge::GRAPH_FAILED;
        }

        if (gradInputShape_->GetDim(DIM_0) != bSize_ || gradInputShape_->GetDim(DIM_1) != sSize_ ||
            gradInputShape_->GetDim(DIM_2) != nSize_ || gradInputShape_->GetDim(DIM_3) != nSize_) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "grad_input",
                ("{" + std::to_string(gradInputShape_->GetDim(DIM_0)) + ", " +
                 std::to_string(gradInputShape_->GetDim(DIM_1)) + ", " +
                 std::to_string(gradInputShape_->GetDim(DIM_2)) + ", " +
                 std::to_string(gradInputShape_->GetDim(DIM_3)) + "}")
                    .c_str(),
                ("should be {" + std::to_string(bSize_) + ", " + std::to_string(sSize_) + ", " +
                 std::to_string(nSize_) + ", " + std::to_string(nSize_) + "}")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::CheckShape()
{
    OP_CHECK_IF(gradYShape_ == nullptr, OP_LOGE(opName_, "grad_y shape is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(normShape_ == nullptr, OP_LOGE(opName_, "norm shape is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(sumShape_ == nullptr, OP_LOGE(opName_, "sum shape is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gradInputShape_ == nullptr, OP_LOGE(opName_, "grad_input shape is null"), return ge::GRAPH_FAILED);

    if (normShape_->GetDimNum() != NORM_INPUT_DIMNUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "norm", std::to_string(normShape_->GetDimNum()).c_str(),
                                                 ("must be " + std::to_string(NORM_INPUT_DIMNUM)).c_str());
        return ge::GRAPH_FAILED;
    }
    if (sumShape_->GetDimNum() != SUM_INPUT_DIMNUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "sum", std::to_string(sumShape_->GetDimNum()).c_str(),
                                                 ("must be " + std::to_string(SUM_INPUT_DIMNUM)).c_str());
        return ge::GRAPH_FAILED;
    }
    if (gradYShape_->GetDimNum() != GRAD_Y_INPUT_BS_DIMNUM && gradYShape_->GetDimNum() != GRAD_Y_INPUT_T_DIMNUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "grad_y", std::to_string(gradYShape_->GetDimNum()).c_str(),
                                                 "must be 3 or 4");
        return ge::GRAPH_FAILED;
    }

    isTShape_ = gradYShape_->GetDimNum() == GRAD_Y_INPUT_T_DIMNUM ? true : false;
    if (isTShape_) {
        tSize_ = gradYShape_->GetDim(DIM_0);
        nSize_ = gradYShape_->GetDim(DIM_1);
        totalLength_ = tSize_;
    } else {
        bSize_ = gradYShape_->GetDim(DIM_0);
        sSize_ = gradYShape_->GetDim(DIM_1);
        nSize_ = gradYShape_->GetDim(DIM_2);
        totalLength_ = bSize_ * sSize_;
    }

    if (nSize_ != 4 && nSize_ != 6 && nSize_ != 8) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "grad_y", std::to_string(nSize_).c_str(),
                                              "the n dim of grad_y must be 4, 6, or 8");
        return ge::GRAPH_FAILED;
    }

    int64_t blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t inputNumPerBlock = Ops::Base::FloorDiv(blockSize, inputDTypeSize_);
    nAlignSize_ = Ops::Base::CeilAlign(nSize_, inputNumPerBlock);

    int64_t normDim0 = normShape_->GetDim(DIM_0);
    if (normDim0 & 1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "norm", std::to_string(normDim0).c_str(),
                                              "dim0 of norm must be an even number");
        return ge::GRAPH_FAILED;
    }
    numIters_ = normDim0 / (2 * totalLength_ * nSize_ * nAlignSize_);

    if (numIters_ > 100 || numIters_ < 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "numIters", std::to_string(numIters_).c_str(),
                                              "must be in [1, 100]");
        return ge::GRAPH_FAILED;
    }

    if (CheckGradYShape() != ge::GRAPH_SUCCESS || CheckNormShape() != ge::GRAPH_SUCCESS ||
        CheckSumShape() != ge::GRAPH_SUCCESS || CheckGradInputShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::SplitCores()
{
    // tiling分核参数和正向对齐取较小值
    auto splitInfo = BaseSplitCores(totalLength_, nSize_, inputDTypeSize_);

    needCoreNum_ = splitInfo.usedCoreNum;
    // 核内切分：普通核
    coreTSize_ = splitInfo.tNormCore;
    normCoreTLoops_ = splitInfo.tNormCoreLoop;
    normCorePerLoopTSize_ = splitInfo.tUbFactor;
    normCoreLastLoopTSize_ = splitInfo.tUbFactorTail;
    // 核内切分：尾核
    tailCoreTSize_ = splitInfo.tTailCore;
    tailCoreLoops_ = splitInfo.tTailCoreLoop;
    tailCorePerLoopTSize_ = splitInfo.tUbFactor;
    tailCoreLastLoopTSize_ = splitInfo.tUbTailTail;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::DoOpTiling()
{
    auto ret = CheckDtype();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = SplitCores();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

void MhcSinkhornBackwardTiling::SetTilingData()
{
    MhcSinkhornBackwardTilingData *tilingData = context_->GetTilingData<MhcSinkhornBackwardTilingData>();
    tilingData->totalLength = totalLength_;
    tilingData->tSize = tSize_;
    tilingData->bSize = bSize_;
    tilingData->sSize = sSize_;
    tilingData->nSize = nSize_;
    tilingData->nAlignSize = nAlignSize_;
    tilingData->numIters = numIters_;
    tilingData->needCoreNum = needCoreNum_;
    tilingData->coreTSize = coreTSize_;
    tilingData->normCoreTLoops = normCoreTLoops_;
    tilingData->normCorePerLoopTSize = normCorePerLoopTSize_;
    tilingData->normCoreLastLoopTSize = normCoreLastLoopTSize_;
    tilingData->tailCoreTSize = tailCoreTSize_;
    tilingData->tailCoreLoops = tailCoreLoops_;
    tilingData->tailCorePerLoopTSize = tailCorePerLoopTSize_;
    tilingData->tailCoreLastLoopTSize = tailCoreLastLoopTSize_;
}

ge::graphStatus MhcSinkhornBackwardTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcSinkhornBackwardTiling::GetTilingKey() const
{
    uint64_t tilingKey = 1;
    return GET_TPL_TILING_KEY(tilingKey);
}

ge::graphStatus MhcSinkhornBackwardTiling::GetWorkspaceSize()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornBackwardTiling::PostTiling()
{
    context_->SetBlockDim(needCoreNum_);
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MhcSinkhornBackwardTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "\n totalLength: " << totalLength_;
    info << "\n tSize: " << tSize_;
    info << "\n bSize: " << bSize_;
    info << "\n sSize: " << sSize_;
    info << "\n nSize: " << nSize_;
    info << "\n nAlignSize: " << nAlignSize_;
    info << "\n numIters: " << numIters_;
    info << "\n needCoreNum: " << needCoreNum_;
    OP_LOGI(opName_, "%s", info.str().c_str());
}

static ge::graphStatus TilingForMhcSinkhornBackward(gert::TilingContext *context)
{
    MhcSinkhornBackwardTiling tiling(context);
    auto ret = tiling.DoTiling();
    return ret;
}

static ge::graphStatus TilingPrepareForMhcSinkhornBackward([[maybe_unused]] gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<MhcSinkhornBackwardCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context, "Failed to get core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcSinkhornBackward)
    .Tiling(TilingForMhcSinkhornBackward)
    .TilingParse<MhcSinkhornBackwardCompileInfo>(TilingPrepareForMhcSinkhornBackward);
} // namespace optiling
