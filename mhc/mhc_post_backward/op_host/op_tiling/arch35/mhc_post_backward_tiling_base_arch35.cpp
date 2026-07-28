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
 * \file mhc_post_backward_tiling_base_arch35.cpp
 * \brief
 */

#include <cmath>
#include <algorithm>
#include <vector>
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_base.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "../mhc_post_backward_tiling.h"
#include "../../../op_kernel/arch35/mhc_post_backward_tiling_data_arch35.h"
#include "../../../op_kernel/arch35/mhc_post_backward_tiling_key_arch35.h"

namespace optiling {

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return ((value + align - 1) / align) * align;
}

inline uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return (value / align) * align;
}

constexpr uint64_t TILING_KEY_GENERALIZED = 0;
constexpr uint32_t BF16_FP16_ALIGN_SIZE = 16;
constexpr uint32_t FLOAT32_ALIGN_SIZE = 8;

const static int64_t GRAD_OUTPUT_INPUT_INDEX = 0;
const static int64_t X_INPUT_INDEX = 1;
const static int64_t H_RES_INPUT_INDEX = 2;
const static int64_t H_OUT_INPUT_INDEX = 3;
const static int64_t H_POST_INPUT_INDEX = 4;

const static int64_t GRAD_X_OUTPUT_INDEX = 0;
const static int64_t GRAD_H_RES_OUTPUT_INDEX = 1;
const static int64_t GRAD_H_OUT_OUTPUT_INDEX = 2;
const static int64_t GRAD_H_POST_OUTPUT_INDEX = 3;

class MhcPostBackwardTilingBaseArch35 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MhcPostBackwardTilingBaseArch35(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~MhcPostBackwardTilingBaseArch35() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    ge::graphStatus CheckNullptr();
    ge::graphStatus CheckShapeAllPositive(int64_t idx) const;
    ge::graphStatus CheckShapeAllPositive();
    ge::graphStatus CheckDataType();
    ge::graphStatus CheckShapeConsistency();
    ge::graphStatus CheckSpecConstraints();
    ge::graphStatus CheckParam();
    ge::graphStatus ComputeTiling();

    const gert::Shape *gradOutputShape_ = nullptr;

    uint32_t B_ = 0;
    uint32_t S_ = 0;
    uint32_t n_ = 0;
    uint32_t D_ = 0;
    uint32_t totalItems_ = 0;
    uint32_t usedCores_ = 0;
    uint32_t itemsPerCore_ = 0;
    uint32_t remainderItems_ = 0;
    uint32_t tileD_ = 0;
    uint32_t nTilesD_ = 0;
    uint32_t usedAic_ = 0;
    uint32_t itemsPerAic_ = 0;
    uint32_t remainderItemsAic_ = 0;
    uint32_t isNAligned_ = 0;
    uint32_t isNNAligned_ = 0;
    uint32_t isDAligned_ = 0;
    bool isHResActive_ = true;

    const char *opName_ = "MhcPostBackward";
    ge::DataType dtype_ = ge::DT_UNDEFINED;
};

ge::graphStatus MhcPostBackwardTilingBaseArch35::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_, "fail to get platform info");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::GetShapeAttrsInfo()
{
    auto totalInputs = context_->GetComputeNodeInputNum();
    OP_LOGI(context_, "GetShapeAttrsInfo: totalInputs=%zu", totalInputs);
    for (size_t i = 0; i < totalInputs; i++) {
        auto desc = context_->GetInputDesc(i);
        auto shape = context_->GetInputShape(i);
        if (desc == nullptr || shape == nullptr) {
            OP_LOGI(context_, "  input[%zu]: desc=%p, shape=%p", i, desc, shape);
        } else {
            const auto &s = shape->GetStorageShape();
            std::string shapeStr = "[";
            for (size_t d = 0; d < s.GetDimNum(); d++) {
                shapeStr += std::to_string(s.GetDim(d));
                if (d + 1 < s.GetDimNum())
                    shapeStr += ",";
            }
            shapeStr += "]";
            OP_LOGI(context_, "  input[%zu]: dtype=%d, shape=%s, shapeSize=%ld", i,
                    static_cast<int32_t>(desc->GetDataType()), shapeStr.c_str(), s.GetShapeSize());
        }
    }

    auto gradOutputShapePtr = context_->GetInputShape(GRAD_OUTPUT_INPUT_INDEX);
    if (gradOutputShapePtr == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "Storage shape of grad_output", "null",
                                              "grad_output shape must not be null");
        return ge::GRAPH_FAILED;
    }
    gradOutputShape_ = &gradOutputShapePtr->GetStorageShape();

    auto hResDesc = context_->GetInputDesc(H_RES_INPUT_INDEX);
    if (hResDesc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "Descriptor of input tensor h_res", "null",
                                              "h_res descriptor must not be null");
        return ge::GRAPH_FAILED;
    }
    auto hResShape = context_->GetInputShape(H_RES_INPUT_INDEX);
    if (hResShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "Storage shape of input tensor h_res", "null",
                                              "h_res shape must not be null");
        return ge::GRAPH_FAILED;
    }
    isHResActive_ = hResShape->GetStorageShape().GetShapeSize() != 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckNullptr()
{
    for (int64_t i = GRAD_OUTPUT_INPUT_INDEX; i <= H_POST_INPUT_INDEX; i++) {
        auto desc = context_->GetInputDesc(i);
        if (desc == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), ("Descriptor of input tensor at index " + std::to_string(i)).c_str(), "null",
                ("input tensor descriptor at index " + std::to_string(i) + " must not be null").c_str());
            return ge::GRAPH_FAILED;
        }
        auto shape = context_->GetInputShape(i);
        if (shape == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), ("Storage shape of input tensor at index " + std::to_string(i)).c_str(),
                "null", ("input tensor shape at index " + std::to_string(i) + " must not be null").c_str());
            return ge::GRAPH_FAILED;
        }
    }

    for (int64_t i = GRAD_X_OUTPUT_INDEX; i <= GRAD_H_POST_OUTPUT_INDEX; i++) {
        auto desc = context_->GetOutputDesc(i);
        if (desc == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), ("Descriptor of output tensor at index " + std::to_string(i)).c_str(), "null",
                ("output tensor descriptor at index " + std::to_string(i) + " must not be null").c_str());
            return ge::GRAPH_FAILED;
        }
        auto shape = context_->GetOutputShape(i);
        if (shape == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), ("Storage shape of output tensor at index " + std::to_string(i)).c_str(),
                "null", ("output tensor shape at index " + std::to_string(i) + " must not be null").c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckShapeAllPositive(int64_t idx) const
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), ("Dimension value of input tensor at index " + std::to_string(idx)).c_str(),
                std::to_string(shape.GetDim(i)).c_str(),
                ("dimension at axis " + std::to_string(i) + " of input tensor at index " + std::to_string(idx) +
                 " must be positive")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckShapeAllPositive()
{
    const int64_t requiredIndices[] = {GRAD_OUTPUT_INPUT_INDEX, X_INPUT_INDEX, H_OUT_INPUT_INDEX, H_POST_INPUT_INDEX};
    for (auto i : requiredIndices) {
        if (CheckShapeAllPositive(i) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_, "input %ld has non-positive shape", i);
            return ge::GRAPH_FAILED;
        }
    }

    if (isHResActive_) {
        if (CheckShapeAllPositive(H_RES_INPUT_INDEX) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_, "input %ld has non-positive shape", H_RES_INPUT_INDEX);
            return ge::GRAPH_FAILED;
        }
    }

    for (int64_t i = GRAD_X_OUTPUT_INDEX; i <= GRAD_H_POST_OUTPUT_INDEX; i++) {
        auto shape = context_->GetOutputShape(i)->GetStorageShape();
        for (size_t j = 0; j < shape.GetDimNum(); j++) {
            if (shape.GetDim(j) <= 0) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), ("Dimension value of output tensor at index " + std::to_string(i)).c_str(),
                    std::to_string(shape.GetDim(j)).c_str(),
                    ("dimension at axis " + std::to_string(j) + " of output tensor at index " + std::to_string(i) +
                     " must be positive")
                        .c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckDataType()
{
    dtype_ = context_->GetInputDesc(GRAD_OUTPUT_INPUT_INDEX)->GetDataType();

    const std::vector<ge::DataType> supportedDtype = {ge::DT_BF16, ge::DT_FLOAT16};
    if (std::find(supportedDtype.begin(), supportedDtype.end(), dtype_) == supportedDtype.end()) {
        std::string dtypeStr = std::to_string(static_cast<int32_t>(dtype_));
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dtype of grad_output", dtypeStr.c_str(),
                                  "[27(DT_BF16), 1(DT_FLOAT16)]");
        return ge::GRAPH_FAILED;
    }

    auto xType = context_->GetInputDesc(X_INPUT_INDEX)->GetDataType();
    if (xType != dtype_) {
        std::string xTypeStr = std::to_string(static_cast<int32_t>(xType));
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dtype of x", xTypeStr.c_str(),
                                              "x dtype must be same as grad_output dtype");
        return ge::GRAPH_FAILED;
    }

    auto hOutType = context_->GetInputDesc(H_OUT_INPUT_INDEX)->GetDataType();
    if (hOutType != dtype_) {
        std::string hOutTypeStr = std::to_string(static_cast<int32_t>(hOutType));
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dtype of h_out", hOutTypeStr.c_str(),
                                              "h_out dtype must be same as grad_output dtype");
        return ge::GRAPH_FAILED;
    }

    auto hResType = context_->GetInputDesc(H_RES_INPUT_INDEX)->GetDataType();
    if (hResType != ge::DT_FLOAT) {
        std::string hResTypeStr = std::to_string(static_cast<int32_t>(hResType));
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dtype of h_res", hResTypeStr.c_str(), "[0(DT_FLOAT)]");
        return ge::GRAPH_FAILED;
    }

    auto hPostType = context_->GetInputDesc(H_POST_INPUT_INDEX)->GetDataType();
    if (hPostType != ge::DT_FLOAT) {
        std::string hPostTypeStr = std::to_string(static_cast<int32_t>(hPostType));
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dtype of h_post", hPostTypeStr.c_str(), "[0(DT_FLOAT)]");
        return ge::GRAPH_FAILED;
    }

    auto gradXType = context_->GetOutputDesc(GRAD_X_OUTPUT_INDEX)->GetDataType();
    if (gradXType != dtype_) {
        std::string gradXTypeStr = std::to_string(static_cast<int32_t>(gradXType));
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dtype of grad_x", gradXTypeStr.c_str(),
                                              "grad_x dtype must be same as grad_output dtype");
        return ge::GRAPH_FAILED;
    }

    auto gradHOutType = context_->GetOutputDesc(GRAD_H_OUT_OUTPUT_INDEX)->GetDataType();
    if (gradHOutType != dtype_) {
        std::string gradHOutTypeStr = std::to_string(static_cast<int32_t>(gradHOutType));
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dtype of grad_h_out", gradHOutTypeStr.c_str(),
                                              "grad_h_out dtype must be same as grad_output dtype");
        return ge::GRAPH_FAILED;
    }

    auto gradHResType = context_->GetOutputDesc(GRAD_H_RES_OUTPUT_INDEX)->GetDataType();
    if (gradHResType != ge::DT_FLOAT) {
        std::string gradHResTypeStr = std::to_string(static_cast<int32_t>(gradHResType));
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dtype of grad_h_res", gradHResTypeStr.c_str(),
                                  "[0(DT_FLOAT)]");
        return ge::GRAPH_FAILED;
    }

    auto gradHPostType = context_->GetOutputDesc(GRAD_H_POST_OUTPUT_INDEX)->GetDataType();
    if (gradHPostType != ge::DT_FLOAT) {
        std::string gradHPostTypeStr = std::to_string(static_cast<int32_t>(gradHPostType));
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dtype of grad_h_post", gradHPostTypeStr.c_str(),
                                  "[0(DT_FLOAT)]");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckShapeConsistency()
{
    if (gradOutputShape_ == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "Storage shape of grad_output", "null",
                                              "grad_output shape must not be null");
        return ge::GRAPH_FAILED;
    }

    uint32_t dimNum = gradOutputShape_->GetDimNum();
    if (dimNum == 4) {
        int64_t B_int = gradOutputShape_->GetDim(0);
        int64_t S_int = gradOutputShape_->GetDim(1);
        int64_t n_int = gradOutputShape_->GetDim(2);
        int64_t D_int = gradOutputShape_->GetDim(3);

        B_ = static_cast<uint32_t>(B_int);
        S_ = static_cast<uint32_t>(S_int);
        n_ = static_cast<uint32_t>(n_int);
        D_ = static_cast<uint32_t>(D_int);
        totalItems_ = B_ * S_;
        OP_LOGI(context_, "BSND format: B=%u, S=%u, n=%u, D=%u, totalItems=%u", B_, S_, n_, D_, totalItems_);
    } else if (dimNum == 3) {
        int64_t T_int = gradOutputShape_->GetDim(0);
        int64_t n_int = gradOutputShape_->GetDim(1);
        int64_t D_int = gradOutputShape_->GetDim(2);

        B_ = 1;
        S_ = 1;
        totalItems_ = static_cast<uint32_t>(T_int);
        n_ = static_cast<uint32_t>(n_int);
        D_ = static_cast<uint32_t>(D_int);
        OP_LOGI(context_, "TND format: T=%u, n=%u, D=%u", totalItems_, n_, D_);
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of grad_output",
                                     std::to_string(dimNum).c_str(), "3(TND) or 4(BSND)");
        return ge::GRAPH_FAILED;
    }

    if (dimNum == 4) {
        int64_t B = static_cast<int64_t>(B_);
        int64_t S = static_cast<int64_t>(S_);
        int64_t n = static_cast<int64_t>(n_);
        int64_t D = static_cast<int64_t>(D_);

        auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
        const gert::Shape *xShape = &xShapePtr->GetStorageShape();
        if (xShape->GetDimNum() != dimNum) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of x",
                                         std::to_string(xShape->GetDimNum()).c_str(), std::to_string(dimNum).c_str());
            return ge::GRAPH_FAILED;
        }
        if (xShape->GetDim(0) != B || xShape->GetDim(1) != S || xShape->GetDim(2) != n || xShape->GetDim(3) != D) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "x, grad_output",
                (std::to_string(xShape->GetDim(0)) + "," + std::to_string(xShape->GetDim(1)) + "," +
                 std::to_string(xShape->GetDim(2)) + "," + std::to_string(xShape->GetDim(3)))
                    .c_str(),
                "x shape must match grad_output shape (B,S,n,D)");
            return ge::GRAPH_FAILED;
        }

        if (isHResActive_) {
            auto hResShapePtr = context_->GetInputShape(H_RES_INPUT_INDEX);
            const gert::Shape *hResShape = &hResShapePtr->GetStorageShape();
            if (hResShape->GetDimNum() != dimNum) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of h_res",
                                             std::to_string(hResShape->GetDimNum()).c_str(),
                                             std::to_string(dimNum).c_str());
                return ge::GRAPH_FAILED;
            }
            if (hResShape->GetDim(0) != B || hResShape->GetDim(1) != S || hResShape->GetDim(2) != n ||
                hResShape->GetDim(3) != n) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "h_res, grad_output",
                    (std::to_string(hResShape->GetDim(0)) + "," + std::to_string(hResShape->GetDim(1)) + "," +
                     std::to_string(hResShape->GetDim(2)) + "," + std::to_string(hResShape->GetDim(3)))
                        .c_str(),
                    "h_res shape must be (B,S,n,n)");
                return ge::GRAPH_FAILED;
            }
        }

        auto hOutShapePtr = context_->GetInputShape(H_OUT_INPUT_INDEX);
        const gert::Shape *hOutShape = &hOutShapePtr->GetStorageShape();
        if (hOutShape->GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of h_out",
                                         std::to_string(hOutShape->GetDimNum()).c_str(), "3");
            return ge::GRAPH_FAILED;
        }
        if (hOutShape->GetDim(0) != B || hOutShape->GetDim(1) != S || hOutShape->GetDim(2) != D) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "h_out, grad_output",
                                                   (std::to_string(hOutShape->GetDim(0)) + "," +
                                                    std::to_string(hOutShape->GetDim(1)) + "," +
                                                    std::to_string(hOutShape->GetDim(2)))
                                                       .c_str(),
                                                   "h_out shape must be (B,S,D)");
            return ge::GRAPH_FAILED;
        }

        auto hPostShapePtr = context_->GetInputShape(H_POST_INPUT_INDEX);
        const gert::Shape *hPostShape = &hPostShapePtr->GetStorageShape();
        if (hPostShape->GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of h_post",
                                         std::to_string(hPostShape->GetDimNum()).c_str(), "3");
            return ge::GRAPH_FAILED;
        }
        if (hPostShape->GetDim(0) != B || hPostShape->GetDim(1) != S || hPostShape->GetDim(2) != n) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "h_post, grad_output",
                                                   (std::to_string(hPostShape->GetDim(0)) + "," +
                                                    std::to_string(hPostShape->GetDim(1)) + "," +
                                                    std::to_string(hPostShape->GetDim(2)))
                                                       .c_str(),
                                                   "h_post shape must be (B,S,n)");
            return ge::GRAPH_FAILED;
        }

        auto gradXShapePtr = context_->GetOutputShape(GRAD_X_OUTPUT_INDEX);
        const gert::Shape *gradXShape = &gradXShapePtr->GetStorageShape();
        if (gradXShape->GetDimNum() != dimNum) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of grad_x",
                                         std::to_string(gradXShape->GetDimNum()).c_str(),
                                         std::to_string(dimNum).c_str());
            return ge::GRAPH_FAILED;
        }
        if (gradXShape->GetDim(0) != B || gradXShape->GetDim(1) != S || gradXShape->GetDim(2) != n ||
            gradXShape->GetDim(3) != D) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "grad_x, grad_output",
                (std::to_string(gradXShape->GetDim(0)) + "," + std::to_string(gradXShape->GetDim(1)) + "," +
                 std::to_string(gradXShape->GetDim(2)) + "," + std::to_string(gradXShape->GetDim(3)))
                    .c_str(),
                "grad_x shape must match grad_output shape (B,S,n,D)");
            return ge::GRAPH_FAILED;
        }

        auto gradHResShapePtr = context_->GetOutputShape(GRAD_H_RES_OUTPUT_INDEX);
        const gert::Shape *gradHResShape = &gradHResShapePtr->GetStorageShape();
        if (gradHResShape->GetDimNum() != dimNum) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of grad_h_res",
                                         std::to_string(gradHResShape->GetDimNum()).c_str(),
                                         std::to_string(dimNum).c_str());
            return ge::GRAPH_FAILED;
        }
        if (gradHResShape->GetDim(0) != B || gradHResShape->GetDim(1) != S || gradHResShape->GetDim(2) != n ||
            gradHResShape->GetDim(3) != n) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "grad_h_res, grad_output",
                (std::to_string(gradHResShape->GetDim(0)) + "," + std::to_string(gradHResShape->GetDim(1)) + "," +
                 std::to_string(gradHResShape->GetDim(2)) + "," + std::to_string(gradHResShape->GetDim(3)))
                    .c_str(),
                "grad_h_res shape must be (B,S,n,n)");
            return ge::GRAPH_FAILED;
        }

        auto gradHOutShapePtr = context_->GetOutputShape(GRAD_H_OUT_OUTPUT_INDEX);
        const gert::Shape *gradHOutShape = &gradHOutShapePtr->GetStorageShape();
        if (gradHOutShape->GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of grad_h_out",
                                         std::to_string(gradHOutShape->GetDimNum()).c_str(), "3");
            return ge::GRAPH_FAILED;
        }
        if (gradHOutShape->GetDim(0) != B || gradHOutShape->GetDim(1) != S || gradHOutShape->GetDim(2) != D) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_h_out, grad_output",
                                                   (std::to_string(gradHOutShape->GetDim(0)) + "," +
                                                    std::to_string(gradHOutShape->GetDim(1)) + "," +
                                                    std::to_string(gradHOutShape->GetDim(2)))
                                                       .c_str(),
                                                   "grad_h_out shape must be (B,S,D)");
            return ge::GRAPH_FAILED;
        }

        auto gradHPostShapePtr = context_->GetOutputShape(GRAD_H_POST_OUTPUT_INDEX);
        const gert::Shape *gradHPostShape = &gradHPostShapePtr->GetStorageShape();
        if (gradHPostShape->GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of grad_h_post",
                                         std::to_string(gradHPostShape->GetDimNum()).c_str(), "3");
            return ge::GRAPH_FAILED;
        }
        if (gradHPostShape->GetDim(0) != B || gradHPostShape->GetDim(1) != S || gradHPostShape->GetDim(2) != n) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_h_post, grad_output",
                                                   (std::to_string(gradHPostShape->GetDim(0)) + "," +
                                                    std::to_string(gradHPostShape->GetDim(1)) + "," +
                                                    std::to_string(gradHPostShape->GetDim(2)))
                                                       .c_str(),
                                                   "grad_h_post shape must be (B,S,n)");
            return ge::GRAPH_FAILED;
        }
    } else {
        int64_t T = static_cast<int64_t>(totalItems_);
        int64_t n = static_cast<int64_t>(n_);
        int64_t D = static_cast<int64_t>(D_);

        auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
        const gert::Shape *xShape = &xShapePtr->GetStorageShape();
        if (xShape->GetDimNum() != dimNum) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of x",
                                         std::to_string(xShape->GetDimNum()).c_str(), std::to_string(dimNum).c_str());
            return ge::GRAPH_FAILED;
        }
        if (xShape->GetDim(0) != T || xShape->GetDim(1) != n || xShape->GetDim(2) != D) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x, grad_output",
                                                   (std::to_string(xShape->GetDim(0)) + "," +
                                                    std::to_string(xShape->GetDim(1)) + "," +
                                                    std::to_string(xShape->GetDim(2)))
                                                       .c_str(),
                                                   "x shape must match grad_output shape (T,n,D)");
            return ge::GRAPH_FAILED;
        }

        if (isHResActive_) {
            auto hResShapePtr = context_->GetInputShape(H_RES_INPUT_INDEX);
            const gert::Shape *hResShape = &hResShapePtr->GetStorageShape();
            if (hResShape->GetDimNum() != dimNum) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of h_res",
                                             std::to_string(hResShape->GetDimNum()).c_str(),
                                             std::to_string(dimNum).c_str());
                return ge::GRAPH_FAILED;
            }
            if (hResShape->GetDim(0) != T || hResShape->GetDim(1) != n || hResShape->GetDim(2) != n) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "h_res, grad_output",
                                                       (std::to_string(hResShape->GetDim(0)) + "," +
                                                        std::to_string(hResShape->GetDim(1)) + "," +
                                                        std::to_string(hResShape->GetDim(2)))
                                                           .c_str(),
                                                       "h_res shape must be (T,n,n)");
                return ge::GRAPH_FAILED;
            }
        }

        auto hOutShapePtr = context_->GetInputShape(H_OUT_INPUT_INDEX);
        const gert::Shape *hOutShape = &hOutShapePtr->GetStorageShape();
        if (hOutShape->GetDimNum() != 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of h_out",
                                         std::to_string(hOutShape->GetDimNum()).c_str(), "2");
            return ge::GRAPH_FAILED;
        }
        if (hOutShape->GetDim(0) != T || hOutShape->GetDim(1) != D) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "h_out, grad_output",
                (std::to_string(hOutShape->GetDim(0)) + "," + std::to_string(hOutShape->GetDim(1))).c_str(),
                "h_out shape must be (T,D)");
            return ge::GRAPH_FAILED;
        }

        auto hPostShapePtr = context_->GetInputShape(H_POST_INPUT_INDEX);
        const gert::Shape *hPostShape = &hPostShapePtr->GetStorageShape();
        if (hPostShape->GetDimNum() != 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of h_post",
                                         std::to_string(hPostShape->GetDimNum()).c_str(), "2");
            return ge::GRAPH_FAILED;
        }
        if (hPostShape->GetDim(0) != T || hPostShape->GetDim(1) != n) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "h_post, grad_output",
                (std::to_string(hPostShape->GetDim(0)) + "," + std::to_string(hPostShape->GetDim(1))).c_str(),
                "h_post shape must be (T,n)");
            return ge::GRAPH_FAILED;
        }

        auto gradXShapePtr = context_->GetOutputShape(GRAD_X_OUTPUT_INDEX);
        const gert::Shape *gradXShape = &gradXShapePtr->GetStorageShape();
        if (gradXShape->GetDimNum() != dimNum) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of grad_x",
                                         std::to_string(gradXShape->GetDimNum()).c_str(),
                                         std::to_string(dimNum).c_str());
            return ge::GRAPH_FAILED;
        }
        if (gradXShape->GetDim(0) != T || gradXShape->GetDim(1) != n || gradXShape->GetDim(2) != D) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_x, grad_output",
                                                   (std::to_string(gradXShape->GetDim(0)) + "," +
                                                    std::to_string(gradXShape->GetDim(1)) + "," +
                                                    std::to_string(gradXShape->GetDim(2)))
                                                       .c_str(),
                                                   "grad_x shape must match grad_output shape (T,n,D)");
            return ge::GRAPH_FAILED;
        }

        auto gradHResShapePtr = context_->GetOutputShape(GRAD_H_RES_OUTPUT_INDEX);
        const gert::Shape *gradHResShape = &gradHResShapePtr->GetStorageShape();
        if (gradHResShape->GetDimNum() != dimNum) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of grad_h_res",
                                         std::to_string(gradHResShape->GetDimNum()).c_str(),
                                         std::to_string(dimNum).c_str());
            return ge::GRAPH_FAILED;
        }
        if (gradHResShape->GetDim(0) != T || gradHResShape->GetDim(1) != n || gradHResShape->GetDim(2) != n) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_h_res, grad_output",
                                                   (std::to_string(gradHResShape->GetDim(0)) + "," +
                                                    std::to_string(gradHResShape->GetDim(1)) + "," +
                                                    std::to_string(gradHResShape->GetDim(2)))
                                                       .c_str(),
                                                   "grad_h_res shape must be (T,n,n)");
            return ge::GRAPH_FAILED;
        }

        auto gradHOutShapePtr = context_->GetOutputShape(GRAD_H_OUT_OUTPUT_INDEX);
        const gert::Shape *gradHOutShape = &gradHOutShapePtr->GetStorageShape();
        if (gradHOutShape->GetDimNum() != 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of grad_h_out",
                                         std::to_string(gradHOutShape->GetDimNum()).c_str(), "2");
            return ge::GRAPH_FAILED;
        }
        if (gradHOutShape->GetDim(0) != T || gradHOutShape->GetDim(1) != D) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "grad_h_out, grad_output",
                (std::to_string(gradHOutShape->GetDim(0)) + "," + std::to_string(gradHOutShape->GetDim(1))).c_str(),
                "grad_h_out shape must be (T,D)");
            return ge::GRAPH_FAILED;
        }

        auto gradHPostShapePtr = context_->GetOutputShape(GRAD_H_POST_OUTPUT_INDEX);
        const gert::Shape *gradHPostShape = &gradHPostShapePtr->GetStorageShape();
        if (gradHPostShape->GetDimNum() != 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "Dimension count of grad_h_post",
                                         std::to_string(gradHPostShape->GetDimNum()).c_str(), "2");
            return ge::GRAPH_FAILED;
        }
        if (gradHPostShape->GetDim(0) != T || gradHPostShape->GetDim(1) != n) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "grad_h_post, grad_output",
                (std::to_string(gradHPostShape->GetDim(0)) + "," + std::to_string(gradHPostShape->GetDim(1))).c_str(),
                "grad_h_post shape must be (T,n)");
            return ge::GRAPH_FAILED;
        }
    }

    OP_LOGI(context_, "All input and output shapes validated successfully");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckSpecConstraints()
{
    if (totalItems_ == 0) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "Value of totalItems", "0", "greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (n_ != 4 && n_ != 6 && n_ != 8) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "Value of n", std::to_string(n_).c_str(), "[4, 6, 8]");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::CheckParam()
{
    if (CheckNullptr() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "CheckNullptr failed");
        return ge::GRAPH_FAILED;
    }

    if (CheckDataType() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "CheckDataType failed");
        return ge::GRAPH_FAILED;
    }

    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "CheckShapeConsistency failed");
        return ge::GRAPH_FAILED;
    }

    if (CheckSpecConstraints() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "CheckSpecConstraints failed");
        return ge::GRAPH_FAILED;
    }

    if (CheckShapeAllPositive() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "CheckShapeAllPositive failed");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::ComputeTiling()
{
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    usedCores_ = (totalItems_ < (coreNum * 2)) ? totalItems_ : (coreNum * 2);
    itemsPerCore_ = totalItems_ / usedCores_;
    remainderItems_ = totalItems_ % usedCores_;
    usedAic_ = (totalItems_ < coreNum) ? totalItems_ : coreNum;
    itemsPerAic_ = totalItems_ / usedAic_;
    remainderItemsAic_ = totalItems_ % usedAic_;

    const uint32_t UB_SIZE = static_cast<uint32_t>(aicoreParams_.ubSize);

    uint32_t alignedN = AlignUp(n_, FLOAT32_ALIGN_SIZE);
    uint32_t alignedNN = AlignUp(n_ * n_, FLOAT32_ALIGN_SIZE);

    uint32_t bytesPerTileD = (2 * n_ + 2) * 2 + (n_ + 3) * 4;
    uint32_t smallBufferBytes = (3 * alignedN + 2 * alignedNN) * 4;

    uint32_t availableUB = UB_SIZE - smallBufferBytes;
    uint32_t maxTileD = availableUB / bytesPerTileD;
    maxTileD = AlignDown(maxTileD, BF16_FP16_ALIGN_SIZE);

    uint32_t alignedD = AlignUp(D_, BF16_FP16_ALIGN_SIZE);
    if (alignedD <= maxTileD) {
        tileD_ = alignedD;
        nTilesD_ = 1;
    } else {
        tileD_ = maxTileD;
        nTilesD_ = CeilDiv(alignedD, tileD_);
    }

    uint32_t lastTileD = alignedD - (nTilesD_ - 1) * tileD_;

    isNAligned_ = (n_ == 8) ? 1 : 0;
    isNNAligned_ = ((n_ * n_) % FLOAT32_ALIGN_SIZE == 0) ? 1 : 0;
    isDAligned_ = ((D_ % BF16_FP16_ALIGN_SIZE == 0) && (nTilesD_ == 1)) ? 1 : 0;

    MhcPostBackwardTilingDataArch35 *tilingData_ = context_->GetTilingData<MhcPostBackwardTilingDataArch35>();
    tilingData_->totalItems = totalItems_;
    tilingData_->itemsPerCore = itemsPerCore_;
    tilingData_->remainderItems = remainderItems_;
    tilingData_->usedCores = usedCores_;
    tilingData_->S = S_;
    tilingData_->n = n_;
    tilingData_->D = D_;
    tilingData_->tileD = tileD_;
    tilingData_->nTilesD = nTilesD_;
    tilingData_->alignedD = alignedD;
    tilingData_->lastTileD = lastTileD;
    tilingData_->alignedN = alignedN;
    tilingData_->alignedNN = alignedNN;
    tilingData_->itemsPerAic = itemsPerAic_;
    tilingData_->remainderItemsAic = remainderItemsAic_;
    auto xDtype = context_->GetInputDesc(X_INPUT_INDEX)->GetDataType();
    matmul_tiling::MultiCoreMatmulTiling mm(ascendcPlatform);
    matmul_tiling::DataType halfDtype = matmul_tiling::DataType::DT_FLOAT16;
    if (xDtype == ge::DT_BF16) {
        halfDtype = matmul_tiling::DataType::DT_BFLOAT16;
    }
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype, true);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm.SetShape(n_, n_, D_);
    mm.SetOrgShape(n_, n_, D_);
    mm.EnableBias(false);
    mm.SetBufferSpace(-1, -1, -1);
    if (mm.GetTiling(tilingData_->matmulTiling) == -1) {
        OP_LOGE(context_, "fail to get matmul tiling");
        return ge::GRAPH_FAILED;
    }
    tilingData_->matmulTiling.usedCoreNum = usedAic_;
    OP_LOGI(context_, "Tiling: n=%u, D=%u, alignedD=%u, tileD=%u, lastTileD=%u, nTilesD=%u", n_, D_, alignedD, tileD_,
            lastTileD, nTilesD_);
    OP_LOGI(context_, "Tiling: alignedN=%u, alignedNN=%u, isNAligned=%u, isNNAligned=%u, isDAligned=%u", alignedN,
            alignedNN, isNAligned_, isNNAligned_, isDAligned_);
    OP_LOGI(context_, "Tiling: usedCores=%u, itemsPerCore=%u, remainderItems=%u, UB=%u, bytesPerTileD=%u", usedCores_,
            itemsPerCore_, remainderItems_, UB_SIZE, bytesPerTileD);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::DoOpTiling()
{
    auto ret = CheckParam();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ComputeTiling();
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPostBackwardTilingBaseArch35::GetWorkspaceSize()
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

ge::graphStatus MhcPostBackwardTilingBaseArch35::PostTiling()
{
    context_->SetBlockDim(usedAic_);
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcPostBackwardTilingBaseArch35::GetTilingKey() const
{
    return isHResActive_ ? GET_TPL_TILING_KEY(1, 1) : GET_TPL_TILING_KEY(1, 0);
}

void MhcPostBackwardTilingBaseArch35::Reset()
{
    opName_ = "MhcPostBackward";
    gradOutputShape_ = nullptr;
    B_ = 0;
    S_ = 0;
    n_ = 0;
    D_ = 0;
    totalItems_ = 0;
    usedCores_ = 0;
    itemsPerCore_ = 0;
    remainderItems_ = 0;
    tileD_ = 0;
    nTilesD_ = 0;
    usedAic_ = 0;
    itemsPerAic_ = 0;
    remainderItemsAic_ = 0;
    isHResActive_ = true;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(MhcPostBackward, MhcPostBackwardTilingBaseArch35,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 0);
} // namespace optiling
