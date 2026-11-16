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
 * \file sinks_checker.cpp
 * \brief Checker for the optional sinks parameter.
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../mqfa_fa_tiling_info.h"
#include "sinks_checker.h"

namespace optiling {
namespace mixed_quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus SinksChecker::CheckSinglePara(const FaTilingInfo& faInfo)
{
    if (!faInfo.sinksFlag) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        faInfo.opParamInfo.sinks.tensor == nullptr || faInfo.opParamInfo.sinks.desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "sinks", "sinks tensor or tensor desc cannot be null"),
        return ge::GRAPH_FAILED);

    auto dtype = faInfo.opParamInfo.sinks.desc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "sinks", DataTypeToSerialString(dtype).c_str(), "FLOAT"),
                return ge::GRAPH_FAILED);

    const auto& shape = faInfo.opParamInfo.sinks.tensor->GetStorageShape();
    OP_CHECK_IF(
        shape.GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "sinks", (std::to_string(shape.GetDimNum()) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(0) != faInfo.n1Size,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    faInfo.opName, "sinks", GetShapeStr(shape).c_str(),
                    ("When layout_q is " + LayoutToSerialString(faInfo.qLayout) +
                     ", the shape size of sinks must be "
                     "equal to the size (" +
                     std::to_string(faInfo.n1Size) + ") of the " + (faInfo.qLayout == FaLayout::BSND ? "2nd" : "1st") +
                     " axis (num heads) of q")
                        .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace mixed_quant_flash_attn
} // namespace optiling
