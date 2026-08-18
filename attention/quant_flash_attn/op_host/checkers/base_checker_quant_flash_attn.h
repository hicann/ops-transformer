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
 * \file base_checker_quant_flash_attn.h
 * \brief Base checker class for quant_flash_attn parameters
 */

#ifndef BASE_CHECKER_QUANT_FLASH_ATTN_H
#define BASE_CHECKER_QUANT_FLASH_ATTN_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"

#include "../qfa_tiling_info.h"
#include "../qfa_tiling_shape.h"
#include "../quant_flash_attn_tiling_utils.h"

namespace optiling {
namespace quant_flash_attn {

class QfaBaseChecker {
public:
    QfaBaseChecker() = default;
    virtual ~QfaBaseChecker() = default;

    virtual ge::graphStatus CheckSinglePara(const QfaTilingInfo &qfaInfo)
    {
        (void)qfaInfo;
        return ge::GRAPH_SUCCESS;
    }
    virtual ge::graphStatus CheckParaExistence(const QfaTilingInfo &qfaInfo)
    {
        (void)qfaInfo;
        return ge::GRAPH_SUCCESS;
    }
    virtual ge::graphStatus CheckFeature(const QfaTilingInfo &qfaInfo)
    {
        (void)qfaInfo;
        return ge::GRAPH_SUCCESS;
    }
    virtual ge::graphStatus CheckMultiPara(const QfaTilingInfo &qfaInfo)
    {
        (void)qfaInfo;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    ge::graphStatus CheckFormatSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    template <typename T>
    ge::graphStatus CheckValueSupport(const T value, const std::vector<T> &expectValList) const
    {
        if (std::find(expectValList.begin(), expectValList.end(), value) == expectValList.end()) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    // 校验 tensor 是否连续：根据 shape 从最后一维向前累乘推算期望 stride，与实际 stride 比对。
    // 返回 GRAPH_SUCCESS 表示连续；GRAPH_FAILED 表示不连续，index 输出第一个不连续维度的下标。
    ge::graphStatus CheckTensorContiguous(const uint32_t &tensorDimNum, const gert::Shape &inputShape,
                                          const gert::Stride *strides, int32_t &index) const;

    std::string DataTypeToSerialString(ge::DataType type) const;
    static uint32_t GetTypeSize(ge::DataType dtype);
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // BASE_CHECKER_QUANT_FLASH_ATTN_H
