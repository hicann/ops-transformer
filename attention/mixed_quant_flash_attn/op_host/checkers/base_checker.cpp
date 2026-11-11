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

#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>
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

// CheckTensorContiguous会判断当前Tensor是否是连续的，如果非连续会计算最内层的非连续轴索引(index)用于后续判断
ge::graphStatus FABaseChecker::CheckTensorContiguous(const uint32_t tensorDimNum, const gert::Shape& inputShape,
                                                     const gert::Stride* strides, int32_t& index) const
{
    // 根据shape推导连续场景的strides，如果存在某一维不相等则表示不连续
    if (strides == nullptr || strides->GetDimNum() == 0 || strides->GetDimNum() != tensorDimNum) {
        index = -1;
        return ge::GRAPH_SUCCESS;
    }
    // 循环顶部expectedStride恒为当前维的连续期望stride（末维为1）；
    // 每轮末尾expectedStride乘以当前维大小即得外一维的expectedStride
    uint64_t expectedStride = 1;
    for (index = static_cast<int32_t>(tensorDimNum) - 1; index >= 0; index--) {
        // shape为1的维不影响访存，其stride可为任意值，跳过比对
        if (inputShape.GetDim(static_cast<size_t>(index)) != 1 &&
            strides->GetStride(static_cast<size_t>(index)) != static_cast<int64_t>(expectedStride)) {
            return ge::GRAPH_FAILED;
        }
        if (index == 0) {
            // 最外维比对完成，无需再推导外一维的期望stride
            break;
        }
        uint64_t dimSize = static_cast<uint64_t>(inputShape.GetDim(static_cast<size_t>(index)));
        // 外一维的期望stride超出int64可表示范围时，其stride不可能与之相等，判为不连续
        if (dimSize != 0 && expectedStride > static_cast<uint64_t>(INT64_MAX) / dimSize) {
            // 不匹配维为外一维
            index--;
            return ge::GRAPH_FAILED;
        }
        expectedStride = dimSize * expectedStride;
    }
    index = -1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FABaseChecker::CheckTensorStridedViewNoOverlap(const uint32_t tensorDimNum,
                                                               const gert::Shape& inputShape,
                                                               const gert::Stride* strides, int32_t& overlapAxisIdx,
                                                               int32_t& overlapPrevAxisIdx) const
{
    // 重叠视图校验：不同逻辑下标映射到同一底层元素
    overlapAxisIdx = -1;
    overlapPrevAxisIdx = -1;
    if (strides == nullptr || strides->GetDimNum() == 0 || strides->GetDimNum() != tensorDimNum) {
        return ge::GRAPH_SUCCESS;
    }
    std::vector<std::tuple<int64_t, int64_t, int32_t>> strideSizeAxis; // (stride, size, axisIdx)
    for (uint32_t i = 0; i < tensorDimNum; i++) {
        int64_t dimSize = inputShape.GetDim(static_cast<size_t>(i));
        if (dimSize > 1) {
            strideSizeAxis.push_back({strides->GetStride(static_cast<size_t>(i)), dimSize, static_cast<int32_t>(i)});
        }
    }
    // 按stride升序排序后逐级要求 stride[i] >= stride[i-1] * size[i-1]
    std::sort(strideSizeAxis.begin(), strideSizeAxis.end());
    for (size_t i = 1; i < strideSizeAxis.size(); i++) {
        const uint64_t prevStride = static_cast<uint64_t>(std::get<0>(strideSizeAxis[i - 1]));
        const uint64_t prevSize = static_cast<uint64_t>(std::get<1>(strideSizeAxis[i - 1]));
        const uint64_t currStride = static_cast<uint64_t>(std::get<0>(strideSizeAxis[i]));
        // 真实乘积超出uint64范围时，当前维stride必然小于该乘积
        if (prevStride > UINT64_MAX / prevSize || currStride < prevStride * prevSize) {
            overlapAxisIdx = std::get<2>(strideSizeAxis[i]);
            overlapPrevAxisIdx = std::get<2>(strideSizeAxis[i - 1]);
            return ge::GRAPH_FAILED;
        }
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
