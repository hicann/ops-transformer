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
 * \brief Base checker implementation for quant_flash_mla_with_kvcache parameters
 */

#include "base_checker.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include <graph/utils/type_utils.h>

namespace optiling {
namespace quant_flash_mla_with_kvcache {

ge::graphStatus QmlaBaseChecker::CheckTensorContiguous(const gert::Shape& inputShape, const gert::Stride* strides,
                                                       int32_t& index) const
{
    if (strides == nullptr || strides->GetDimNum() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    const int32_t tensorDimNum = static_cast<int32_t>(inputShape.GetDimNum());
    if (tensorDimNum == 0 || tensorDimNum == 1) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t preStride = 1;
    for (index = tensorDimNum - 1; index >= 0; index--) {
        if (inputShape.GetDim(index) == 1) {
            continue;
        }
        if (preStride != strides->GetStride(index)) {
            return ge::GRAPH_FAILED;
        }
        preStride *= inputShape.GetDim(index);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaBaseChecker::CheckFormatSupport(const gert::CompileTimeTensorDesc* desc, const char* name) const
{
    if (desc != nullptr) {
        auto format = desc->GetOriginFormat();
        OP_CHECK_IF(format != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT("QuantFlashMla", name,
                                               ge::TypeUtils::FormatToSerialString(format).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling
