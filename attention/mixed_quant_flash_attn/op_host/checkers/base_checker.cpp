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
 * \brief
 */

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "../mqfa_fa_tiling_info.h"
#include "mqfa_base_checker.h"
#include "log/log.h"
namespace optiling {
namespace mixed_quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus FABaseChecker::CheckDtypeSupport(const gert::CompileTimeTensorDesc* desc, const std::string& name) const
{
    if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE("MixedQuantFlashAttn", "%s datatype support list should be specify in DTYPE_SUPPORT_MAP",
                            name.c_str()),
                    return ge::GRAPH_FAILED);
        auto& expectDtypeList = it->second;
        if (std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end()) {
            std::string dtypeStr = DataTypeToSerialString(desc->GetDataType());
            std::string expectedDtypes;
            for (size_t i = 0; i < expectDtypeList.size(); i++) {
                if (i > 0)
                    expectedDtypes += ", ";
                expectedDtypes += DataTypeToSerialString(expectDtypeList[i]);
            }
            OP_LOGE_FOR_INVALID_DTYPE("MixedQuantFlashAttn", name.c_str(), dtypeStr.c_str(), expectedDtypes.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FABaseChecker::CheckFormatSupport(const gert::CompileTimeTensorDesc* desc,
                                                  const std::string& name) const
{
    if (desc != nullptr) {
        auto format = desc->GetOriginFormat();
        OP_CHECK_IF(
            (FORMAT_SUPPORT_SET.find(format) == FORMAT_SUPPORT_SET.end()),
            OP_LOGE_FOR_INVALID_FORMAT("MixedQuantFlashAttn", name.c_str(), Ops::Base::ToString(format).c_str(), "ND"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

std::string FABaseChecker::DataTypeToSerialString(ge::DataType type) const
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("MixedQuantFlashAttn", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

// 获取数据类型大小，单位bit
uint32_t FABaseChecker::GetTypeSize(ge::DataType dtype)
{
    constexpr uint32_t NUM_BIT_FLOAT = 32;
    constexpr uint32_t NUM_BIT_FLOAT16 = 16;
    constexpr uint32_t NUM_BIT_INT8 = 8;
    constexpr uint32_t NUM_BIT_FP4 = 4;

    switch (dtype) {
        case ge::DT_FLOAT:
            return NUM_BIT_FLOAT;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return NUM_BIT_FLOAT16;
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_FLOAT8_E8M0:
        case ge::DT_FLOAT8_E4M3FN:
            return NUM_BIT_INT8;
        case ge::DT_FLOAT4_E2M1:
            return NUM_BIT_FP4;
        default:
            return NUM_BIT_FLOAT16;
    }
}
} // namespace mixed_quant_flash_attn
} // namespace optiling
