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
 * \brief Base checker implementation for quant_flash_attn parameters
 */

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "../qfa_tiling_info.h"
#include "base_checker_quant_flash_attn.h"
#include "log/log.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35QFA;

ge::graphStatus QfaBaseChecker::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
                                                  const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        "QuantFlashAttn", name.c_str(), DataTypeToSerialString(desc->GetDataType()).c_str(),
                        "The datatype support list of this parameter should be specified in DTYPE_SUPPORT_MAP"),
                    return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        if (std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end()) {
            std::string dtypeStr = DataTypeToSerialString(desc->GetDataType());
            std::string expectedDtypes;
            for (size_t i = 0; i < expectDtypeList.size(); i++) {
                if (i > 0) {
                    expectedDtypes += ", ";
                }
                expectedDtypes += DataTypeToSerialString(expectDtypeList[i]);
            }
            OP_LOGE_FOR_INVALID_DTYPE("QuantFlashAttn", name.c_str(), dtypeStr.c_str(), expectedDtypes.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaBaseChecker::CheckFormatSupport(const gert::CompileTimeTensorDesc *desc,
                                                   const std::string &name) const
{
    if (desc != nullptr) {
        auto format = desc->GetOriginFormat();
        OP_CHECK_IF(
            (FORMAT_SUPPORT_SET.find(format) == FORMAT_SUPPORT_SET.end()),
            OP_LOGE_FOR_INVALID_FORMAT("QuantFlashAttn", name.c_str(), Ops::Base::ToString(format).c_str(), "ND"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

std::string QfaBaseChecker::DataTypeToSerialString(ge::DataType type) const
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("QuantFlashAttn", "datatype", std::to_string(type).c_str(),
                                              "The datatype is not supported");
        return "UNDEFINED";
    }
}

ge::graphStatus QfaBaseChecker::CheckTensorContiguous(const uint32_t &tensorDimNum, const gert::Shape &inputShape,
                                                      const gert::Stride *strides, int32_t &index) const
{
    // 根据 shape 从最后一维向前累乘推算连续场景的期望 stride，若某一维实际 stride 不等于期望值则不连续
    if (strides == nullptr || strides->GetDimNum() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    // 维度为 0 或 1 的 tensor 始终连续
    if (tensorDimNum == 0 || tensorDimNum == 1) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t preStride = 1; // 连续场景最后一维的 stride 默认为 1
    for (index = static_cast<int32_t>(tensorDimNum) - 1; index >= 0; index--) {
        if (inputShape.GetDim(index) == 1) { // dim=1 时步长不影响连续性
            continue;
        }
        if (preStride != strides->GetStride(index)) {
            return ge::GRAPH_FAILED;
        }
        preStride *= inputShape.GetDim(index);
    }
    return ge::GRAPH_SUCCESS;
}

uint32_t QfaBaseChecker::GetTypeSize(ge::DataType dtype)
{
    constexpr uint32_t NUM_BYTES_FLOAT = 4;
    constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
    constexpr uint32_t NUM_BYTES_INT8 = 1;

    switch (dtype) {
        case ge::DT_FLOAT:
            return NUM_BYTES_FLOAT;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return NUM_BYTES_FLOAT16;
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_FLOAT8_E4M3FN:
        case ge::DT_FLOAT8_E8M0:
            return NUM_BYTES_INT8;
        default:
            return NUM_BYTES_FLOAT16;
    }
}

} // namespace quant_flash_attn
} // namespace optiling
