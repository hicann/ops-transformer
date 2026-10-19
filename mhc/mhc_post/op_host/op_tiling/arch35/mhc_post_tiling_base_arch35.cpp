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
 * \file mhc_post_tiling_base_arch35.cpp
 * \brief MhcPost tiling implementation
 */

#include <cmath>
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

// Memory alignment constants (in elements)
constexpr uint32_t BF16_FP16_ALIGN_SIZE = 16; // 16 elements = 32 bytes for bf16/fp16
constexpr uint32_t ALIGN_SIZE_512B = 256;     // 256 elements = 512 bytes for bf16/fp16

constexpr uint32_t SIZE_OF_16BIT = 2;
constexpr uint32_t SIZE_OF_32BIT = 4;

// Double Buffer configuration
constexpr uint32_t DOUBLE_BUFFER_DEPTH = 2; // Double Buffer depth for data tiles
constexpr uint32_t SINGLE_BUFFER_DEPTH = 1; // Single Buffer depth for weights

// Tiling parameter constants
constexpr uint32_t NUM_DATA_STREAMS = 3;   // hOut, x, output (TQue + TBuf each)
constexpr uint64_t NON_X_BUFFER_COUNT = 2; // hOut + output (x excluded, counted by n_)

// Input indices
const static int64_t X_INPUT_INDEX = 0;      // x (B, S, n, D)
const static int64_t H_RES_INPUT_INDEX = 1;  // h_res (B, S, n, n)
const static int64_t H_OUT_INPUT_INDEX = 2;  // h_out (B, S, D)
const static int64_t H_POST_INPUT_INDEX = 3; // h_post (B, S, n)

// Output indices
const static int64_t OUTPUT_INDEX = 0; // output (B, S, n, D)

// Dim indices
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
static const int64_t DIM_NUM_2 = 2;
static const int64_t DIM_NUM_3 = 3;
static const int64_t DIM_NUM_4 = 4;

class MhcPostTilingBaseArch35 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MhcPostTilingBaseArch35(gert::TilingContext* context)
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
    // Check functions
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
    int64_t normalCoreProcessNum_ = 0;
    int64_t tailCoreProcessNum_ = 0;
    int64_t bsInner_ = 0;
    int64_t bsOuter_ = 0;
    int64_t bsTail_ = 0;
    int64_t dInner_ = 0;
    int64_t dOuter_ = 0;
    int64_t dTail_ = 0;
    int64_t dTailAlign_ = 0;

    uint16_t usePermanentX_ = 0;

    MhcPostTilingData* tilingData_ = context_->GetTilingData<MhcPostTilingData>();
};

bool MhcPostTilingBaseArch35::IsCapable()
{
    return true;
}

ge::graphStatus MhcPostTilingBaseArch35::GetPlatformInfo()
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

ge::graphStatus MhcPostTilingBaseArch35::GetShapeAttrsInfo()
{
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

ge::graphStatus MhcPostTilingBaseArch35::CheckNullptr()
{
    // Check all input desc and shape
    for (int64_t i = X_INPUT_INDEX; i <= H_POST_INPUT_INDEX; i++) {
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

    // Check output desc and shape
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

ge::graphStatus MhcPostTilingBaseArch35::CheckInputShapePositive(int64_t idx) const
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

ge::graphStatus MhcPostTilingBaseArch35::CheckShapeAllPositive()
{
    // Check all inputs
    for (int64_t i = X_INPUT_INDEX; i <= H_POST_INPUT_INDEX; i++) {
        OP_CHECK_IF(CheckInputShapePositive(i) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "input", "0",
                                                          "shape dimension value must be positive"),
                    return ge::GRAPH_FAILED);
    }

    // Check output
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

ge::graphStatus MhcPostTilingBaseArch35::CheckDataType()
{
    // Get x dtype as reference
    auto dtype_ = context_->GetInputDesc(X_INPUT_INDEX)->GetDataType();

    // Check supported dtype
    const std::vector<ge::DataType> supportedDtype = {ge::DT_BF16, ge::DT_FLOAT16};
    OP_CHECK_IF(std::find(supportedDtype.begin(), supportedDtype.end(), dtype_) == supportedDtype.end(),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(dtype_).c_str(),
                                                      "dtype must be DT_BF16 or DT_FLOAT16"),
                return ge::GRAPH_FAILED);

    // Check x and h_out have same dtype (bf16/fp16)
    auto hOutType = context_->GetInputDesc(H_OUT_INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(hOutType != dtype_,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "h_out", Ops::Base::ToString(hOutType).c_str(),
                                          Ops::Base::ToString(dtype_).c_str()),
                return ge::GRAPH_FAILED);

    // Check h_res and h_post are float32
    auto hResType = context_->GetInputDesc(H_RES_INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(
        hResType != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "h_res", Ops::Base::ToString(hResType).c_str(), "DT_FLOAT"),
        return ge::GRAPH_FAILED);

    auto hPostType = context_->GetInputDesc(H_POST_INPUT_INDEX)->GetDataType();
    OP_CHECK_IF(hPostType != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "h_post", Ops::Base::ToString(hPostType).c_str(),
                                          "DT_FLOAT"),
                return ge::GRAPH_FAILED);

    // Check output dtype matches x dtype
    auto outputType = context_->GetOutputDesc(OUTPUT_INDEX)->GetDataType();
    OP_CHECK_IF(outputType != dtype_,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "output", Ops::Base::ToString(outputType).c_str(),
                                          Ops::Base::ToString(dtype_).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBaseArch35::CheckShape3D()
{
    uint32_t dimNum = xShape_->GetDimNum();

    // TND format validation
    int64_t T = static_cast<int64_t>(totalItems_);
    // Validate h_res: (T, n, n)
    auto hResShapePtr = context_->GetInputShape(H_RES_INPUT_INDEX);
    const gert::Shape* hResShape = &hResShapePtr->GetStorageShape();
    OP_CHECK_IF(hResShape->GetDimNum() != dimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "h_res",
                                                         std::to_string(hResShape->GetDimNum()).c_str(),
                                                         "dimNum must match x's dimNum"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        hResShape->GetDim(DIM_0) != T || hResShape->GetDim(DIM_1) != n_ || hResShape->GetDim(DIM_2) != n_,
        OP_LOGE_FOR_INVALID_SHAPE(
            context_->GetNodeName(), "h_res",
            (std::string("[ ") + std::to_string(hResShape->GetDim(DIM_0)) + ", " +
             std::to_string(hResShape->GetDim(DIM_1)) + ", " + std::to_string(hResShape->GetDim(DIM_2)) + " ]")
                .c_str(),
            (std::string("[ ") + std::to_string(T) + ", " + std::to_string(n_) + ", " + std::to_string(n_) + " ]")
                .c_str()),
        return ge::GRAPH_FAILED);

    // Validate h_out: (T, D)
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

    // Validate h_post: (T, n)
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

    // Validate output: (T, n, D)
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

ge::graphStatus MhcPostTilingBaseArch35::CheckShape4D()
{
    uint32_t dimNum = xShape_->GetDimNum();
    // BSND format validation
    // Validate h_res: (B, S, n, n)
    auto hResShapePtr = context_->GetInputShape(H_RES_INPUT_INDEX);
    const gert::Shape* hResShape = &hResShapePtr->GetStorageShape();
    OP_CHECK_IF(hResShape->GetDimNum() != dimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "h_res",
                                                         std::to_string(hResShape->GetDimNum()).c_str(),
                                                         "dimNum must match x's dimNum"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(hResShape->GetDim(DIM_0) != b_ || hResShape->GetDim(DIM_1) != s_ || hResShape->GetDim(DIM_2) != n_ ||
                    hResShape->GetDim(DIM_3) != n_,
                OP_LOGE_FOR_INVALID_SHAPE(
                    context_->GetNodeName(), "h_res",
                    (std::string("[") + std::to_string(hResShape->GetDim(DIM_0)) + ", " +
                     std::to_string(hResShape->GetDim(DIM_1)) + ", " + std::to_string(hResShape->GetDim(DIM_2)) + ", " +
                     std::to_string(hResShape->GetDim(DIM_3)) + "]")
                        .c_str(),
                    (std::string("[") + std::to_string(b_) + ", " + std::to_string(s_) + ", " + std::to_string(n_) +
                     ", " + std::to_string(n_) + "]")
                        .c_str()),
                return ge::GRAPH_FAILED);

    // Validate h_out: (B, S, D)
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

    // Validate h_post: (B, S, n)
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

    // Validate output: (B, S, n, D)
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

ge::graphStatus MhcPostTilingBaseArch35::CheckShapeConsistency()
{
    // Support both BSND (4D) and TND (3D) formats
    // BSND: (B, S, n, D) -> totalItems = B * S
    // TND:  (T, n, D)    -> totalItems = T
    uint32_t dimNum = xShape_->GetDimNum();
    if (dimNum == DIM_NUM_4) {
        // BSND format: (B, S, n, D)
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
        // TND format: (T, n, D)
        b_ = 1; // Not used in TND format
        s_ = 1; // Not used in TND format
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

    OP_LOGI(context_, "All input and output shapes validated successfully");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBaseArch35::CheckParam()
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

void MhcPostTilingBaseArch35::ComputeTiling()
{
    // Core Partitioning - handle remainder properly
    uint32_t coreNum = static_cast<uint32_t>(aicoreParams_.numBlocks);
    uint32_t halfCoreNum = coreNum / 2;
    bsOuter_ = totalItems_;
    bsInner_ = 1;
    bsTail_ = 1;

    const uint32_t UB_SIZE = static_cast<uint32_t>(aicoreParams_.ubSize);

    // Calculate bytes per tileD element
    // TQue bf16: 3 * 2 bytes (hOut:1, x:1, output:1)
    // TBuf f32:  3 * 4 bytes (hOutF32:1, xF32:1, outF32:1)
    uint32_t bytesPerTileD =
        NUM_DATA_STREAMS * (DOUBLE_BUFFER_DEPTH * SIZE_OF_16BIT + SINGLE_BUFFER_DEPTH * SIZE_OF_32BIT);
    uint32_t maxTileD = UB_SIZE / bytesPerTileD;
    dOuter_ = 1;
    dInner_ = d_;
    dTail_ = d_;

    while (bsOuter_ * dOuter_ <= halfCoreNum || dInner_ >= maxTileD) {
        if (dInner_ <= ALIGN_SIZE_512B) {
            break;
        }
        dOuter_ = dOuter_ * 2;
        dInner_ = d_ / dOuter_;
    }
    dInner_ = Ops::Base::CeilAlign(dInner_, static_cast<int64_t>(BF16_FP16_ALIGN_SIZE));
    dOuter_ = Ops::Base::CeilDiv(d_, dInner_);
    dTail_ = d_ - (dOuter_ - 1) * dInner_;
    dTailAlign_ = Ops::Base::CeilAlign(dTail_, static_cast<int64_t>(BF16_FP16_ALIGN_SIZE));

    int64_t totalCount = bsOuter_ * dOuter_;
    usedCoreNum_ = (totalCount < coreNum) ? totalCount : coreNum;
    normalCoreProcessNum_ = Ops::Base::CeilDiv(totalCount, usedCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(totalCount, normalCoreProcessNum_);
    tailCoreProcessNum_ = totalCount - (usedCoreNum_ - 1) * normalCoreProcessNum_;

    uint64_t fullyBytesPerTileD = (n_ + NON_X_BUFFER_COUNT) * (DOUBLE_BUFFER_DEPTH * SIZE_OF_16BIT + SIZE_OF_32BIT);
    if (fullyBytesPerTileD * dInner_ <= UB_SIZE) {
        usePermanentX_ = 1;
    }
}

ge::graphStatus MhcPostTilingBaseArch35::DoOpTiling()
{
    ComputeTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBaseArch35::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBaseArch35::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_, "fail to get platform info");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_LOGI(context_, "Workspace size: %ld bytes (%.2f MB)", workspaceSize_, workspaceSize_ / (1024.0 * 1024.0));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostTilingBaseArch35::PostTiling()
{
    tilingData_->n = n_;
    tilingData_->d = d_;
    tilingData_->usedCoreNum = usedCoreNum_;
    tilingData_->normalCoreProcessNum = normalCoreProcessNum_;
    tilingData_->tailCoreProcessNum = tailCoreProcessNum_;
    tilingData_->bsInner = bsInner_;
    tilingData_->bsOuter = bsOuter_;
    tilingData_->bsTail = bsTail_;
    tilingData_->dInner = dInner_;
    tilingData_->dOuter = dOuter_;
    tilingData_->dTail = dTail_;
    tilingData_->dTailAlign = dTailAlign_;

    context_->SetBlockDim(usedCoreNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcPostTilingBaseArch35::GetTilingKey() const
{
    OP_LOGI(context_, "Tiling: usePermanentX_=%u", usePermanentX_);
    return GET_TPL_TILING_KEY(usePermanentX_, 0, 1);
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(MhcPost, MhcPostTilingBaseArch35,
                                   std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510),
                                                         static_cast<int32_t>(NpuArch::DAV_9201),
                                                         static_cast<int32_t>(NpuArch::DAV_9202)}),
                                   20);
} // namespace optiling
