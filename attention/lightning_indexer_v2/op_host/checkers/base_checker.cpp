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
 * \file base_checker.cpp
 * \brief Implements shared parameter validation for Lightning Indexer V2 checkers.
 */

#include "base_checker_lightning_indexer_v2.h"

#include <algorithm>
#include "graph/utils/type_utils.h"
#include "log/log.h"

namespace optiling {
namespace lightning_indexer_v2_checker {

ge::graphStatus LightningIndexerV2BaseChecker::CheckParaExistence(const LightningIndexerV2CheckerInfo &) const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerV2BaseChecker::CheckSinglePara(const LightningIndexerV2CheckerInfo &) const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerV2BaseChecker::CheckFeature(const LightningIndexerV2CheckerInfo &) const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerV2BaseChecker::CheckMultiPara(const LightningIndexerV2CheckerInfo &) const
{
    return ge::GRAPH_SUCCESS;
}

std::string LightningIndexerV2BaseChecker::DtypesToString(const std::vector<ge::DataType> &dtypes)
{
    std::string result;
    for (size_t i = 0; i < dtypes.size(); ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += Ops::Base::ToString(dtypes[i]);
    }
    return result;
}

std::string LightningIndexerV2BaseChecker::ShapeToString(const CheckerTensor &tensor)
{
    const gert::Shape *shape = tensor.GetShape();
    return shape == nullptr ? "null" : Ops::Base::ToString(*shape);
}

ge::graphStatus LightningIndexerV2BaseChecker::CheckDtype(const LightningIndexerV2CheckerInfo &info,
                                                          const CheckerTensor &tensor, const char *name,
                                                          const std::vector<ge::DataType> &expected) const
{
    if (!tensor.IsPresent()) {
        return ge::GRAPH_SUCCESS;
    }
    if (tensor.desc == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, name, "Tensor desc cannot be null");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType actualDtype = tensor.desc->GetDataType();
    if (std::find(expected.begin(), expected.end(), actualDtype) == expected.end()) {
        const std::string actual = Ops::Base::ToString(actualDtype);
        const std::string expectedText = DtypesToString(expected);
        OP_LOGE_FOR_INVALID_DTYPE(info.opName, name, actual.c_str(), expectedText.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerV2BaseChecker::CheckRank(const LightningIndexerV2CheckerInfo &info,
                                                         const CheckerTensor &tensor, const char *name,
                                                         uint32_t expected, const std::string &condition) const
{
    if (!tensor.IsPresent()) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape *shape = tensor.GetShape();
    if (shape == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, name, "Tensor shape cannot be null");
        return ge::GRAPH_FAILED;
    }
    const uint32_t actual = shape->GetDimNum();
    if (actual != expected) {
        const std::string actualText = std::to_string(actual) + "D";
        const std::string expectedText = std::to_string(expected) + "D";
        if (!condition.empty()) {
            const std::string reason = "When " + condition + ", the tensor must be " + expectedText;
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(info.opName, name, actualText.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
        OP_LOGE_FOR_INVALID_SHAPEDIM(info.opName, name, actualText.c_str(), expectedText.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerV2BaseChecker::CheckElementCount(const LightningIndexerV2CheckerInfo &info,
                                                                 const CheckerTensor &tensor, const char *name,
                                                                 int64_t expected) const
{
    if (!tensor.IsPresent()) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape *shape = tensor.GetShape();
    if (shape == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, name, "Tensor shape cannot be null");
        return ge::GRAPH_FAILED;
    }
    if (shape->GetShapeSize() != expected) {
        const std::string actual = ShapeToString(tensor);
        const std::string reason =
            "The tensor shape must be [B], where B is batch size (" + std::to_string(expected) + ")";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(info.opName, name, actual.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace lightning_indexer_v2_checker
} // namespace optiling
