/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_inplace_index_add_base_tiling.cpp
 * \brief
 */
#include "moe_inplace_index_add_tiling_arch35.h"
#include "op_host/tiling_templates_registry.h"
#include "moe_inplace_index_add_tiling.h"
#include <cstdint>

namespace optiling {
constexpr int64_t VAR_IDX = 0L;
constexpr int64_t INDICES_IDX = 1L;
constexpr int64_t UPDATES_IDX = 2L;
constexpr int64_t ALPHA_IDX = 3L;
constexpr int64_t ATTR_AXIS_IDX = 0L;
constexpr int64_t DCACHE_SIZE_SIMT = 32L * 1024L;
constexpr int64_t MIN_SIZE_SIMD_HANDLE = 128L;
constexpr int64_t THRESHOLD_USE_SORT = 5L;
constexpr int64_t SIMT_SORT_LASTDIM_LIMIT = 1024L;
static constexpr uint64_t CAST_INT32_TO_INT16 = 1; // int32 Cast int16
static constexpr uint64_t CAST_INT64_TO_INT32 = 2; // int64 Cast int32
static constexpr uint64_t CAST_INT64_TO_INT16 = 3; // int64 Cast int16
static constexpr uint64_t CAST_INT32_TO_UINT8 = 4; // int32 Cast uint8
static constexpr uint64_t CAST_INT64_TO_UINT8 = 5; // int64 Cast uint8

constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16L * 1024L * 1024L;
static const std::set<ge::DataType> VAR_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_INT64, ge::DT_INT32,
                                                 ge::DT_INT16, ge::DT_INT8,    ge::DT_UINT8, ge::DT_BOOL};
static const std::set<ge::DataType> SIMD_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32,
                                                  ge::DT_INT16, ge::DT_INT8,    ge::DT_BOOL};
static const std::set<ge::DataType> SIMT_SORT_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT64,
                                                       ge::DT_INT32};
static const std::set<ge::DataType> DETERMIN_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

bool MoeInplaceIndexAddTiling::IsCapable()
{
    return true;
}

void MoeInplaceIndexAddTiling::GetCastTypeForSort()
{
    indicesCastDtype_ = indicesDtype_;

    if (indicesDtype_ == ge::DT_INT32) {
        if (varInAxis_ < UINT8_MAX) {
            indicesCastMode_ = CAST_INT32_TO_UINT8; // int32 Cast uint8
            indicesCastDtype_ = ge::DT_UINT8;
        } else if (varInAxis_ < INT16_MAX) {
            indicesCastMode_ = CAST_INT32_TO_INT16; // int32 Cast int16
            indicesCastDtype_ = ge::DT_INT16;
        }
    } else {
        if (varInAxis_ < UINT8_MAX) {
            indicesCastMode_ = CAST_INT64_TO_UINT8; // int64 Cast uint8
            indicesCastDtype_ = ge::DT_UINT8;
        } else if (varInAxis_ < INT16_MAX) {
            indicesCastMode_ = CAST_INT64_TO_INT16; // int64 Cast int16
            indicesCastDtype_ = ge::DT_INT16;
        } else if (varInAxis_ < INT32_MAX) {
            indicesCastMode_ = CAST_INT64_TO_INT32; // int64 Cast int32
            indicesCastDtype_ = ge::DT_INT32;
        }
    }

    if (indicesCastMode_ != 0) {
        indicesCastDtypeSize_ = ge::GetSizeByDataType(indicesCastDtype_);
    }
}

void MoeInplaceIndexAddTiling::SelTemplateByInput()
{
    bool useSort = (updatesInAxis_ / varInAxis_ >= THRESHOLD_USE_SORT);
    bool useSortSimt = useSort && afterAxis_ <= SIMT_SORT_LASTDIM_LIMIT;

    if (context_->GetDeterministic() == 1 && (DETERMIN_DTYPE.find(dtype_) != DETERMIN_DTYPE.end())) {
        isDeterminstic_ = 1;
        return;
    }

    bool choseSimd = false;
    if (useSort) {
        choseSimd = afterAxis_ * varTypeSize_ >= MIN_SIZE_SIMD_HANDLE;
    } else {
        choseSimd = afterAxis_ >= totalCoreNum_ * MIN_SIZE_SIMD_HANDLE;
    }
    if (choseSimd && SIMD_DTYPE.find(dtype_) != SIMD_DTYPE.end()) {
        if (useSort) {
            isSimdSort_ = 1;
        } else {
            isSimdNoSort_ = 1;
        }
    } else {
        if (useSortSimt && SIMT_SORT_DTYPE.find(dtype_) != SIMT_SORT_DTYPE.end()) {
            isSimtSort_ = 1;
        } else {
            isSimtNoSort_ = 1;
        }
    }
}

ge::graphStatus MoeInplaceIndexAddTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const MoeInplaceIndexAddCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    ubSize_ = ubSize_ - DCACHE_SIZE_SIMT;
    SelTemplateByInput();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddTiling::GetShapeAttrsInfo()
{
    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto axis = attrs->GetAttrPointer<int64_t>(ATTR_AXIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, axis);
    int64_t dim = static_cast<int64_t>(*axis);

    auto varShapePtr = context_->GetInputShape(VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varShapePtr);
    auto varShape = varShapePtr->GetStorageShape();
    rank_ = static_cast<int64_t>(varShape.GetDimNum());

    int64_t dimMax = std::max(-1 * rank_, rank_ - 1);
    int64_t dimMin = std::min(-1 * rank_, rank_ - 1);
    MOE_OP_TILING_CHECK(
        (dim > dimMax || dim < dimMin),
        MOE_VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "attr axis must be in range of [%ld, %ld], but got [%ld].", dimMin, dimMax, dim),
        return ge::GRAPH_FAILED);

    dim_ = dim < 0 ? dim + rank_ : dim;

    MOE_OP_TILING_CHECK(CheckInputDtype() != ge::GRAPH_SUCCESS,
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "input dtype check failed."),
                        return ge::GRAPH_FAILED);

    MOE_OP_TILING_CHECK(CheckInputShape() != ge::GRAPH_SUCCESS,
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "input shape check failed."),
                        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddTiling::CheckInputDtype()
{
    auto varPtr = context_->GetInputDesc(VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varPtr);
    dtype_ = varPtr->GetDataType();
    MOE_OP_TILING_CHECK((VAR_DTYPE.find(dtype_) == VAR_DTYPE.end()),
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(
                            context_->GetNodeName(), "var dtype only support float32, float16, uint8, int8, int32, \
int16, bool, int64, bfloat16 currently, but got [%s].",
                            Ops::Base::ToString(dtype_).c_str()),
                        return ge::GRAPH_FAILED);
    varTypeSize_ = ge::GetSizeByDataType(dtype_);
    MOE_OP_TILING_CHECK(varTypeSize_ <= 0,
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "get dataType size fail."),
                        return ge::GRAPH_FAILED);

    auto indicesPtr = context_->GetInputDesc(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesPtr);
    indicesDtype_ = indicesPtr->GetDataType();
    indicesTypeSize_ = ge::GetSizeByDataType(indicesDtype_);
    bool dtypeInValid = indicesDtype_ != ge::DT_INT32 && indicesDtype_ != ge::DT_INT64;
    MOE_OP_TILING_CHECK(
        dtypeInValid,
        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                            "indices dtype only support int32 and int64 currently, but got [%s].",
                                            Ops::Base::ToString(indicesDtype_).c_str()),
        return ge::GRAPH_FAILED);

    auto updatesPtr = context_->GetInputDesc(UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updatesPtr);
    auto updatesDtype = updatesPtr->GetDataType();
    MOE_OP_TILING_CHECK((updatesDtype != dtype_),
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(
                            context_->GetNodeName(), "updates [%s] and var [%s] must have the same dtype.",
                            Ops::Base::ToString(updatesDtype).c_str(), Ops::Base::ToString(dtype_).c_str()),
                        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool MoeInplaceIndexAddTiling::CompareShape(const gert::Shape& shape1, const gert::Shape& shape2, int64_t dim)
{
    int64_t inputShapeSize = static_cast<int64_t>(shape1.GetDimNum());
    for (int64_t i = 0; i < inputShapeSize; ++i) {
        if (i != dim) {
            if (shape1.GetDim(i) != shape2.GetDim(i)) {
                return false;
            }
        }
    }
    return true;
}

void MoeInplaceIndexAddTiling::CombineAxis(const gert::Shape& varShape, const gert::Shape& updatesShape)
{
    for (int64_t i = 0; i < dim_; ++i) {
        preAxis_ *= varShape.GetDim(i);
    }

    varInAxis_ = varShape.GetDim(dim_);
    updatesInAxis_ = updatesShape.GetDim(dim_);

    for (int64_t j = dim_ + 1; j < rank_; ++j) {
        afterAxis_ *= varShape.GetDim(j);
    }
    return;
}

ge::graphStatus MoeInplaceIndexAddTiling::CheckInputShape()
{
    auto varShapePtr = context_->GetInputShape(VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varShapePtr);
    auto varShape = varShapePtr->GetStorageShape();
    varAxis_ = varShape.GetShapeSize();

    bool isView = context_->InputIsView(INDICES_IDX);
    if (isView) {
        auto stridePtr = context_->GetInputStride(INDICES_IDX);
        if (stridePtr != nullptr && stridePtr->GetDimNum() == 1) {
            gert::Stride indexStride = *(stridePtr);
            indicesStride_ = indexStride[0];
        }
    }
    auto indicesShapePtr = context_->GetInputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShapePtr);
    auto indicesShape = isView ? indicesShapePtr->GetShape() : indicesShapePtr->GetStorageShape();
    int64_t indicesDimNum = static_cast<int64_t>(indicesShape.GetDimNum());
    indicesAxis_ = indicesShape.GetShapeSize();

    auto updatesShapePtr = context_->GetInputShape(UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updatesShapePtr);
    auto updatesShape = updatesShapePtr->GetStorageShape();
    int64_t updatesDimNum = static_cast<int64_t>(updatesShape.GetDimNum());
    updatesAxis_ = updatesShape.GetShapeSize();

    auto alphaShapePtr = context_->GetInputShape(ALPHA_IDX);
    if (alphaShapePtr != nullptr) {
        withAlpha_ = 1UL;
        auto alphaShape = alphaShapePtr->GetStorageShape();
        MOE_OP_TILING_CHECK(alphaShape.GetShapeSize() != 1,
                            MOE_VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "the alpha size must be 1."),
                            return ge::GRAPH_FAILED);
    }

    MOE_OP_TILING_CHECK(
        (indicesDimNum > 1),
        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                            "index is supposed to be a vector, but got dim: %ld.", indicesDimNum),
        return ge::GRAPH_FAILED);

    if (varShape.IsScalar() && updatesAxis_ == 1 && indicesAxis_ == 1) {
        return ge::GRAPH_SUCCESS;
    } else {
        MOE_OP_TILING_CHECK((updatesDimNum != rank_),
                            MOE_VECTOR_INNER_ERR_REPORT_TILIING(
                                context_->GetNodeName(), "updates must have the same number of dimensions as var."),
                            return ge::GRAPH_FAILED);

        int64_t expectedNum = updatesShape.GetDim(dim_);
        MOE_OP_TILING_CHECK(
            indicesAxis_ != expectedNum,
            MOE_VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                "number of indices [%ld] should be equal to update.size(%ld): [%ld].",
                                                indicesAxis_, dim_, expectedNum),
            return ge::GRAPH_FAILED);

        MOE_OP_TILING_CHECK(
            !CompareShape(updatesShape, varShape, dim_),
            MOE_VECTOR_INNER_ERR_REPORT_TILIING(
                context_->GetNodeName(), "updates shape must match var shape, excluding the specified dimension %ld. \
Got var.shape = %s updates.shape = %s",
                dim_, Ops::Base::ToString(varShape).c_str(), Ops::Base::ToString(updatesShape).c_str()),
            return ge::GRAPH_FAILED);
        CombineAxis(varShape, updatesShape);
    }
    return ge::GRAPH_SUCCESS;
}

void MoeInplaceIndexAddTiling::GetCastTypeSize()
{
    if (dtype_ == ge::DT_INT16 || dtype_ == ge::DT_INT8 || dtype_ == ge::DT_UINT8) {
        castTypeSize_ = ge::GetSizeByDataType(ge::DT_INT32);
    } else if (dtype_ == ge::DT_BOOL) {
        castTypeSize_ = ge::GetSizeByDataType(ge::DT_FLOAT16);
    }
}

uint32_t MoeInplaceIndexAddTiling::GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend)
{
    std::vector<int64_t> shapeVec = {lastAxisNum};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
    OP_LOGI("RadixSortTilingForAscendC", "Need tmp buffer %u byte for ac sort api", maxValue);
    return maxValue;
}

ge::graphStatus MoeInplaceIndexAddTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInplaceIndexAddTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus MoeInplaceIndexAddTiling::GetWorkspaceSize()
{
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = ASCENDC_TOOLS_WORKSPACE + workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
