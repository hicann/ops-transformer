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
 * \file mhc_post_tiling_nohres_arch35.cpp
 * \brief MhcPost tiling implementation (arch35 nohres - h_res optional path)
 * Formula: x_{l+1} = x_l + h_{l}^{out} * H_{t}^{post}
 */

#include <algorithm>
#include <vector>
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_base.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "util/math_util.h"
#include "../mhc_post_tiling.h"
#include "../../../op_kernel/arch35/mhc_post_tiling_data.h"
#include "../../../op_kernel/arch35/mhc_post_tiling_key.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t BF16_FP16_ALIGN_SIZE = 16;
constexpr int64_t DB_BYTES_F32 = 8;
constexpr int64_t DB_BYTES_F16 = 4;

const static int64_t X_INPUT_INDEX = 0;
const static int64_t H_RES_INPUT_INDEX = 1;
const static int64_t H_OUT_INPUT_INDEX = 1;
const static int64_t H_POST_INPUT_INDEX = 2;
const static int64_t H_POST_FULL_INPUT_INDEX = 3;

const static int64_t OUTPUT_INDEX = 0;

const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
static const int64_t DIM_NUM_2 = 2;
static const int64_t DIM_NUM_3 = 3;
static const int64_t DIM_NUM_4 = 4;

class MhcPostTilingNoHResArch35 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MhcPostTilingNoHResArch35(gert::TilingContext* context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus CheckNullptr();
    ge::graphStatus CheckInputShapePositive(int64_t idx) const;
    ge::graphStatus CheckShapeAllPositive();
    ge::graphStatus CheckDataType();
    ge::graphStatus CheckShape3D();
    ge::graphStatus CheckShape4D();
    ge::graphStatus CheckShapeConsistency();
    ge::graphStatus CheckParam();

    void ComputeTiling();

    const gert::Shape* xShape_ = nullptr;

    int64_t b_ = 0;
    int64_t s_ = 0;
    int64_t n_ = 0;
    int64_t d_ = 0;
    int64_t totalItems_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t bsInner_ = 0;
    int64_t bsOuter_ = 0;
    int64_t bsTail_ = 0;
    int64_t dInner_ = 0;
    int64_t dOuter_ = 0;
    int64_t dTail_ = 0;
    int64_t nInner_ = 0;
    int64_t nOuter_ = 0;
    int64_t nTail_ = 0;

    MhcPostRegbaseTilingData* tilingData_ = context_->GetTilingData<MhcPostRegbaseTilingData>();
};

bool MhcPostTilingNoHResArch35::IsCapable()
{
    auto input3 = context_->GetInputDesc(H_POST_FULL_INPUT_INDEX);
    if (input3 != nullptr && input3->GetDataType() == ge::DT_FLOAT) {
        return false;
    }

    auto desc = context_->GetInputDesc(H_POST_INPUT_INDEX);
    if (desc == nullptr) {
        return false;
    }
    return (desc->GetDataType() == ge::DT_FLOAT);
}

ge::graphStatus MhcPostTilingNoHResArch35::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_, "fail to get platform info");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingNoHResArch35::GetShapeAttrsInfo()
{
    auto input3 = context_->GetInputDesc(H_POST_FULL_INPUT_INDEX);
    if (input3 != nullptr && input3->GetDataType() == ge::DT_FLOAT) {
        auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
        if (xShapePtr != nullptr) {
            xShape_ = &xShapePtr->GetStorageShape();
        }
        return ge::GRAPH_SUCCESS;
    }

    auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
    OP_CHECK_IF(
        xShapePtr == nullptr,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", "null", "input shape cannot be null"),
        return ge::GRAPH_FAILED);
    xShape_ = &xShapePtr->GetStorageShape();
    OP_CHECK_IF(
        xShape_ == nullptr,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", "null", "input shape cannot be null"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckParam() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", "invalid", "CheckParam failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingNoHResArch35::CheckNullptr()
{
    const std::vector<int64_t> requiredInputs = {X_INPUT_INDEX, H_OUT_INPUT_INDEX, H_POST_INPUT_INDEX};
    for (int64_t i : requiredInputs) {
        auto desc = context_->GetInputDesc(i);
        OP_CHECK_IF(desc == nullptr,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(),
                                                          (std::string("input_") + std::to_string(i)).c_str(), "null",
                                                          "input desc cannot be null"),
                    return ge::GRAPH_FAILED);
        auto shape = context_->GetInputShape(i);
        OP_CHECK_IF(shape == nullptr,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(),
                                                          std::string("input_") + std::to_string(i).c_str(), "null",
                                                          "input shape cannot be null"),
                    return ge::GRAPH_FAILED);
    }

    auto desc = context_->GetOutputDesc(OUTPUT_INDEX);
    OP_CHECK_IF(
        desc == nullptr,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "output", "null", "output desc cannot be null"),
        return ge::GRAPH_FAILED);
    auto shape = context_->GetOutputShape(OUTPUT_INDEX);
    OP_CHECK_IF(
        shape == nullptr,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "output", "null", "output shape cannot be null"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingNoHResArch35::CheckInputShapePositive(int64_t idx) const
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "input",
                                                          std::to_string(shape.GetDim(i)).c_str(),
                                                          "shape dimension value must be positive"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingNoHResArch35::CheckShapeAllPositive()
{
    const std::vector<int64_t> requiredInputs = {X_INPUT_INDEX, H_OUT_INPUT_INDEX, H_POST_INPUT_INDEX};
    for (int64_t i : requiredInputs) {
        OP_CHECK_IF(CheckInputShapePositive(i) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "input", "0",
                                                          "shape dimension value must be positive"),
                    return ge::GRAPH_FAILED);
    }

    auto shape = context_->GetOutputShape(OUTPUT_INDEX)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "output",
                                                          std::to_string(shape.GetDim(i)).c_str(),
                                                          "shape dimension value must be positive"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingNoHResArch35::CheckDataType()
{
    auto dtype_ = context_->GetInputDesc(X_INPUT_INDEX)->GetDataType();

    const std::vector<ge::DataType> supportedDtype = {ge::DT_BF16, ge::DT_FLOAT16};
    OP_CHECK_IF(std::find(supportedDtype.begin(), supportedDtype.end(), dtype_) == supportedDtype.end(),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(dtype_).c_str(),
                                                      "dtype must be DT_BF16 or DT_FLOAT16"),
                return ge::GRAPH_FAILED);

    auto hOutType = context_->GetInputDesc(H_OUT_INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(hOutType != dtype_,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "h_out",
                                          ge::TypeUtils::DataTypeToSerialString(hOutType).c_str(),
                                          ge::TypeUtils::DataTypeToSerialString(dtype_).c_str()),
                return ge::GRAPH_FAILED);

    auto hPostType = context_->GetInputDesc(H_POST_INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(hPostType != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "h_post",
                                          ge::TypeUtils::DataTypeToSerialString(hPostType).c_str(), "DT_FLOAT"),
                return ge::GRAPH_FAILED);

    auto outputType = context_->GetOutputDesc(OUTPUT_INDEX)->GetDataType();
    OP_CHECK_IF(outputType != dtype_,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "output",
                                          ge::TypeUtils::DataTypeToSerialString(outputType).c_str(),
                                          ge::TypeUtils::DataTypeToSerialString(dtype_).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingNoHResArch35::CheckShape3D()
{
    uint32_t dimNum = xShape_->GetDimNum();
    int64_t T = static_cast<int64_t>(totalItems_);

    auto hOutShapePtr = context_->GetInputShape(H_OUT_INPUT_INDEX);
    const gert::Shape* hOutShape = &hOutShapePtr->GetStorageShape();
    OP_CHECK_IF(hOutShape->GetDimNum() != DIM_NUM_2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "h_out",
                                             std::to_string(hOutShape->GetDimNum()).c_str(), "2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        hOutShape->GetDim(DIM_0) != T || hOutShape->GetDim(DIM_1) != d_,
        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "h_out",
                                  (std::string("[") + std::to_string(hOutShape->GetDim(DIM_0)) + ", " +
                                   std::to_string(hOutShape->GetDim(DIM_1)) + "]")
                                      .c_str(),
                                  (std::string("[") + std::to_string(T) + ", " + std::to_string(d_) + "]").c_str()),
        return ge::GRAPH_FAILED);

    auto hPostShapePtr = context_->GetInputShape(H_POST_INPUT_INDEX);
    const gert::Shape* hPostShape = &hPostShapePtr->GetStorageShape();
    OP_CHECK_IF(hPostShape->GetDimNum() != DIM_NUM_2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "h_post",
                                             std::to_string(hPostShape->GetDimNum()).c_str(), "2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        hPostShape->GetDim(DIM_0) != T || hPostShape->GetDim(DIM_1) != n_,
        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "h_post",
                                  (std::string("[") + std::to_string(hPostShape->GetDim(DIM_0)) + ", " +
                                   std::to_string(hPostShape->GetDim(DIM_1)) + "]")
                                      .c_str(),
                                  (std::string("[") + std::to_string(T) + ", " + std::to_string(n_) + "]").c_str()),
        return ge::GRAPH_FAILED);

    auto outputShapePtr = context_->GetOutputShape(OUTPUT_INDEX);
    const gert::Shape* outputShape = &outputShapePtr->GetStorageShape();
    OP_CHECK_IF(outputShape->GetDimNum() != dimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "output",
                                                         std::to_string(outputShape->GetDimNum()).c_str(),
                                                         "dimNum must match x's dimNum"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outputShape->GetDim(DIM_0) != T || outputShape->GetDim(DIM_1) != n_ || outputShape->GetDim(DIM_2) != d_,
        OP_LOGE_FOR_INVALID_SHAPE(
            context_->GetNodeName(), "output",
            (std::string("[") + std::to_string(outputShape->GetDim(DIM_0)) + ", " +
             std::to_string(outputShape->GetDim(DIM_1)) + ", " + std::to_string(outputShape->GetDim(DIM_2)) + "]")
                .c_str(),
            (std::string("[") + std::to_string(T) + ", " + std::to_string(n_) + ", " + std::to_string(d_) + "]")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingNoHResArch35::CheckShape4D()
{
    uint32_t dimNum = xShape_->GetDimNum();

    auto hOutShapePtr = context_->GetInputShape(H_OUT_INPUT_INDEX);
    const gert::Shape* hOutShape = &hOutShapePtr->GetStorageShape();
    OP_CHECK_IF(hOutShape->GetDimNum() != DIM_NUM_3,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "h_out",
                                             std::to_string(hOutShape->GetDimNum()).c_str(), "3"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        hOutShape->GetDim(DIM_0) != b_ || hOutShape->GetDim(DIM_1) != s_ || hOutShape->GetDim(DIM_2) != d_,
        OP_LOGE_FOR_INVALID_SHAPE(
            context_->GetNodeName(), "h_out",
            (std::string("[") + std::to_string(hOutShape->GetDim(DIM_0)) + ", " +
             std::to_string(hOutShape->GetDim(DIM_1)) + ", " + std::to_string(hOutShape->GetDim(DIM_2)) + "]")
                .c_str(),
            (std::string("[") + std::to_string(b_) + ", " + std::to_string(s_) + ", " + std::to_string(d_) + "]")
                .c_str()),
        return ge::GRAPH_FAILED);

    auto hPostShapePtr = context_->GetInputShape(H_POST_INPUT_INDEX);
    const gert::Shape* hPostShape = &hPostShapePtr->GetStorageShape();
    OP_CHECK_IF(hPostShape->GetDimNum() != DIM_NUM_3,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "h_post",
                                             std::to_string(hPostShape->GetDimNum()).c_str(), "3"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        hPostShape->GetDim(DIM_0) != b_ || hPostShape->GetDim(DIM_1) != s_ || hPostShape->GetDim(DIM_2) != n_,
        OP_LOGE_FOR_INVALID_SHAPE(
            context_->GetNodeName(), "h_post",
            (std::string("[") + std::to_string(hPostShape->GetDim(DIM_0)) + ", " +
             std::to_string(hPostShape->GetDim(DIM_1)) + ", " + std::to_string(hPostShape->GetDim(DIM_2)) + "]")
                .c_str(),
            (std::string("[") + std::to_string(b_) + ", " + std::to_string(s_) + ", " + std::to_string(n_) + "]")
                .c_str()),
        return ge::GRAPH_FAILED);

    auto outputShapePtr = context_->GetOutputShape(OUTPUT_INDEX);
    const gert::Shape* outputShape = &outputShapePtr->GetStorageShape();
    OP_CHECK_IF(outputShape->GetDimNum() != dimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "output",
                                                         std::to_string(outputShape->GetDimNum()).c_str(),
                                                         "dimNum must match x's dimNum"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputShape->GetDim(DIM_0) != b_ || outputShape->GetDim(DIM_1) != s_ ||
                    outputShape->GetDim(DIM_2) != n_ || outputShape->GetDim(DIM_3) != d_,
                OP_LOGE_FOR_INVALID_SHAPE(
                    context_->GetNodeName(), "output",
                    (std::string("[") + std::to_string(outputShape->GetDim(DIM_0)) + ", " +
                     std::to_string(outputShape->GetDim(DIM_1)) + ", " + std::to_string(outputShape->GetDim(DIM_2)) +
                     ", " + std::to_string(outputShape->GetDim(DIM_3)) + "]")
                        .c_str(),
                    (std::string("[") + std::to_string(b_) + ", " + std::to_string(s_) + ", " + std::to_string(n_) +
                     ", " + std::to_string(d_) + "]")
                        .c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingNoHResArch35::CheckShapeConsistency()
{
    uint32_t dimNum = xShape_->GetDimNum();
    if (dimNum == DIM_NUM_4) {
        b_ = xShape_->GetDim(DIM_0);
        s_ = xShape_->GetDim(DIM_1);
        n_ = xShape_->GetDim(DIM_2);
        d_ = xShape_->GetDim(DIM_3);
        totalItems_ = b_ * s_;
        OP_LOGI(context_, "BSND format: B=%ld, S=%ld, n=%ld, D=%ld, totalItems=%ld", b_, s_, n_, d_, totalItems_);
        OP_CHECK_IF(
            CheckShape4D() != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", "invalid", "CheckShape4D failed"),
            return ge::GRAPH_FAILED);
    } else if (dimNum == DIM_NUM_3) {
        b_ = 1;
        s_ = 1;
        totalItems_ = xShape_->GetDim(DIM_0);
        n_ = xShape_->GetDim(DIM_1);
        d_ = xShape_->GetDim(DIM_2);
        OP_LOGI(context_, "TND format: T=%ld, n=%ld, D=%ld", totalItems_, n_, d_);
        OP_CHECK_IF(
            CheckShape3D() != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", "invalid", "CheckShape3D failed"),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(dimNum).c_str(),
                                                 "dimNum must be 3 (TND format) or 4 (BSND format)");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingNoHResArch35::CheckParam()
{
    OP_CHECK_IF(CheckNullptr() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", "null", "CheckNullptr failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckDataType() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", "invalid", "CheckDataType failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckShapeConsistency() != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", "invalid", "CheckShapeConsistency failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShapeAllPositive() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", "non-positive",
                                                      "CheckShapeAllPositive failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MhcPostTilingNoHResArch35::ComputeTiling()
{
    uint32_t coreNum = static_cast<uint32_t>(aicoreParams_.numBlocks);
    usedCoreNum_ = (totalItems_ < coreNum) ? totalItems_ : coreNum;
    bsInner_ = Ops::Base::CeilDiv(totalItems_, usedCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(totalItems_, bsInner_);
    bsOuter_ = usedCoreNum_;
    bsTail_ = totalItems_ - (usedCoreNum_ - 1) * bsInner_;

    const int64_t UB_SIZE = static_cast<int64_t>(aicoreParams_.ubSize);
    const int64_t D_MIN = 32;
    const int64_t D_ALIGN = D_MIN;
    const int64_t N_MIN = 8;
    const int64_t N_ALIGN = N_MIN;

    for (int64_t dOuter = 1;; dOuter++) {
        int64_t dInnerMin = Ops::Base::CeilDiv(d_, dOuter);
        dInner_ = Ops::Base::CeilAlign(dInnerMin, D_ALIGN);
        dInner_ = (dInner_ < D_MIN) ? D_MIN : dInner_;
        int64_t ubWithMinN = DB_BYTES_F32 * N_MIN + DB_BYTES_F16 * dInner_ + DB_BYTES_F16 * N_MIN * dInner_ +
                             static_cast<int64_t>(sizeof(float)) * N_MIN * dInner_ + DB_BYTES_F16 * N_MIN * dInner_;
        if (ubWithMinN < UB_SIZE) {
            dOuter_ = dOuter;
            break;
        }
    }
    dTail_ = d_ - (dOuter_ - 1) * dInner_;

    for (int64_t nOuter = 1;; nOuter++) {
        int64_t nInnerMin = Ops::Base::CeilDiv(n_, nOuter);
        nInner_ = Ops::Base::CeilAlign(nInnerMin, N_ALIGN);
        nInner_ = (nInner_ < N_MIN) ? N_MIN : nInner_;
        int64_t ub = DB_BYTES_F32 * nInner_ + DB_BYTES_F16 * dInner_ + DB_BYTES_F16 * nInner_ * dInner_ +
                     static_cast<int64_t>(sizeof(float)) * nInner_ * dInner_ + DB_BYTES_F16 * nInner_ * dInner_;
        if (ub < UB_SIZE) {
            nOuter_ = nOuter;
            break;
        }
    }
    nTail_ = n_ - (nOuter_ - 1) * nInner_;
}

ge::graphStatus MhcPostTilingNoHResArch35::DoOpTiling()
{
    ComputeTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingNoHResArch35::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingNoHResArch35::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_, "fail to get platform info");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingNoHResArch35::PostTiling()
{
    tilingData_->n = n_;
    tilingData_->d = d_;
    tilingData_->usedCoreNum = usedCoreNum_;
    tilingData_->bsInner = bsInner_;
    tilingData_->bsOuter = bsOuter_;
    tilingData_->bsTail = bsTail_;
    tilingData_->nInner = nInner_;
    tilingData_->nOuter = nOuter_;
    tilingData_->nTail = nTail_;
    tilingData_->dInner = dInner_;
    tilingData_->dOuter = dOuter_;
    tilingData_->dTail = dTail_;

    context_->SetBlockDim(usedCoreNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcPostTilingNoHResArch35::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(0, 1, 0);
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(MhcPost, MhcPostTilingNoHResArch35,
                                   std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510),
                                                         static_cast<int32_t>(NpuArch::DAV_9201),
                                                         static_cast<int32_t>(NpuArch::DAV_9202)}),
                                   5);
} // namespace optiling
