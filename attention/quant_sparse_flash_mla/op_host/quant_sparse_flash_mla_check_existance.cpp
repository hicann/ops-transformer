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
 * \file quant_sparse_flash_mla_check_existance.cpp
 * \brief
 */

#include "quant_sparse_flash_mla_check.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::pair;
using std::string;
namespace optiling {

static constexpr uint32_t TopK_SIZE = 512;
static constexpr uint32_t DIM_0 = 0;
static constexpr uint32_t DIM_1 = 1;
static constexpr uint32_t DIM_2 = 2;
static constexpr uint32_t DIM_3 = 3;

ge::graphStatus QSMLATilingCheck::CheckParaExistenceKv() const
{
    if (kvLayout_ == QSMLALayout::BSND) {
        return ge::GRAPH_SUCCESS;
    } else if (kvLayout_ == QSMLALayout::PA_BBND) {
        if (perfMode_ != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
            perfMode_ != QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            OP_CHECK_IF(opParamInfo_.sequsedOriKv.tensor == nullptr,
                        OP_LOGE(opName_, "when layout_kv is PA_BBND, sequsedOriKv must not be null."),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(opParamInfo_.oriBlockTable.tensor == nullptr,
                    OP_LOGE(opName_, "when layout_kv is PA_BBND, oriBlockTable must not be null."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckParaExistence()
{
    if (ge::GRAPH_SUCCESS != CheckCmpSparseIndicesExistence() || ge::GRAPH_SUCCESS != CheckSWAExistence() ||
        ge::GRAPH_SUCCESS != CheckHCAExistence() || ge::GRAPH_SUCCESS != CheckCSAExistence() ||
        ge::GRAPH_SUCCESS != CheckCmpRatioExistence() || ge::GRAPH_SUCCESS != CheckParaExistenceKv()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckCmpSparseIndicesExistence()
{
    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        if (qLayout_ == QSMLALayout::BSND) {
            if (opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_3) <= 0) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmpSparseIndices",
                    Ops::Base::ToString(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape()).c_str(),
                    "When qLayout is BSND, topK should be greater than 0, but got " +
                        std::to_string(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_3)));
                return ge::GRAPH_FAILED;
            }
            if (opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_1) != s1Size_) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmpSparseIndices",
                    Ops::Base::ToString(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape()).c_str(),
                    "When qLayout is BSND, cmpSparseIndices's S should be equal to s1Size:" + std::to_string(s1Size_) +
                        ", but got " +
                        std::to_string(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(1)));
                return ge::GRAPH_FAILED;
            }
        } else {
            if (opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_2) <= 0) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmpSparseIndices",
                    Ops::Base::ToString(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape()).c_str(),
                    "When qLayout is TND, topK should be greater than 0, but got " +
                        std::to_string(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(2)));
                return ge::GRAPH_FAILED;
            }
            if (opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_0) != qTSize_) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmpSparseIndices",
                    Ops::Base::ToString(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape()).c_str(),
                    "When qLayout is TND, cmpSparseIndices's T should be equal to qTSize:" + std::to_string(qTSize_) +
                        ", but got " +
                        std::to_string(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_0)));
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSWAExistence()
{
    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.oriBlockTable.tensor == nullptr &&
                    kvLayout_ == QSMLALayout::PA_BBND,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "oriBlockTable",
                    "oriBlockTable must not be empty when kvLayout is PA_BBND and cmpKv is not provided"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckHCAExistence()
{
    if (perfMode_ != QSMLATemplateMode::HCA_TEMPLATE_MODE) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        opParamInfo_.oriKv.tensor == nullptr && opParamInfo_.cmpKv.tensor != nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName_, "oriKv", "oriKv must not be empty when cmpKv is provided and cmpSparseIndices is not provided"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.cmpKv.tensor == nullptr &&
            opParamInfo_.cmpRatio != nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName_, "cmpKv", "cmpKv must not be empty when cmpKv is provided and cmpSparseIndices is not provided"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.cmpKv.tensor != nullptr &&
                    opParamInfo_.cmpRatio == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "cmpRatio",
                    "cmpRatio must not be empty when cmpKv is provided and cmpSparseIndices is not provided"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvLayout_ == QSMLALayout::PA_BBND && opParamInfo_.cmpBlockTable.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "cmpBlockTable", "cmpBlockTable must not be empty when kvLayout is PA_BBND in HCA mode"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvLayout_ == QSMLALayout::PA_BBND && opParamInfo_.sequsedCmpKv.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "sequsedCmpKv", "sequsedCmpKv must not be empty when kvLayout is PA_BBND in HCA mode"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvLayout_ == QSMLALayout::TND && opParamInfo_.cuSeqLensCmpKv.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "cuSeqLensCmpKv", "cuSeqLensCmpKv must not be empty when kvLayout is TND in HCA mode"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckCSAExistence()
{
    if (perfMode_ != QSMLATemplateMode::CSA_TEMPLATE_MODE) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.cmpKv.tensor == nullptr &&
                    opParamInfo_.cmpSparseIndices.tensor != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "cmpKv", "cmpKv must not be empty when cmpKv and cmpSparseIndices are provided"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr && opParamInfo_.cmpKv.tensor != nullptr &&
                    opParamInfo_.cmpSparseIndices.tensor != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "oriKv", "oriKv must not be empty when cmpKv and cmpSparseIndices are provided"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr && opParamInfo_.cmpKv.tensor == nullptr &&
                    opParamInfo_.cmpSparseIndices.tensor != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "oriKv and cmpKv",
                    "oriKv and cmpKv must not be empty when cmpKv and cmpSparseIndices are provided"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvLayout_ == QSMLALayout::PA_BBND && opParamInfo_.cmpBlockTable.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "cmpBlockTable", "cmpBlockTable must not be empty when kvLayout is PA_BBND in CSA mode"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvLayout_ == QSMLALayout::PA_BBND && opParamInfo_.sequsedCmpKv.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "sequsedCmpKv", "sequsedCmpKv must not be empty when kvLayout is PA_BBND in CSA mode"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvLayout_ == QSMLALayout::TND && opParamInfo_.cuSeqLensCmpKv.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "cuSeqLensCmpKv", "cuSeqLensCmpKv must not be empty when kvLayout is TND in CSA mode"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckCmpRatioExistence()
{
    OP_CHECK_IF(opParamInfo_.cmpRatio == nullptr, OP_LOGE(opName_, "cmpRatio is required, but got nullptr."),
                return ge::GRAPH_FAILED);

    if (perfMode_ == QSMLATemplateMode::SWA_TEMPLATE_MODE) {
        OP_CHECK_IF(
            *opParamInfo_.cmpRatio != 1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "cmpRatio", std::to_string(*opParamInfo_.cmpRatio).c_str(),
                                                  "When SWA mode, cmpRatio must be 1"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            *opParamInfo_.cmpRatio < 1 || *opParamInfo_.cmpRatio > 128,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "cmpRatio", std::to_string(*opParamInfo_.cmpRatio).c_str(),
                                                  "When non-SWA mode, cmpRatio must be in range [1, 128]"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
