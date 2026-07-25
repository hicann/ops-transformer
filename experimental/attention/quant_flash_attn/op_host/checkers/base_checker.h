/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file base_checker.h
 * \brief Base checker class for quant_flash_attn parameters
 */

#ifndef QUANT_FLASH_ATTN_BASE_CHECKER_H
#define QUANT_FLASH_ATTN_BASE_CHECKER_H

#include <map>
#include <set>
#include <numeric>
#include <vector>
#include <string>
#include "tiling/tiling_api.h"

#include "../quant_flash_attn_tiling_info.h"

namespace optiling {
namespace quant_flash_attn {

// Shape/axis limits
constexpr uint32_t B_LIMIT = 65536;
constexpr uint32_t DIM_NUM_0 = 0;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_NUM_5 = 5;
constexpr uint32_t DIM_NUM_6 = 6;

// Supported dtype map (per parameter name)
const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME, {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1}},
    {KEY_NAME, {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1}},
    {VALUE_NAME, {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1}},
    {Q_DESCALE_NAME, {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT}},
    {K_DESCALE_NAME, {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT}},
    {V_DESCALE_NAME, {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT}},
    {BLOCK_TABLE_NAME, {ge::DT_INT32}},
    {P_SCALE_NAME, {ge::DT_FLOAT}},
    {ATTEN_MASK_NAME, {ge::DT_INT8}},
    {ATTEN_OUT_NAME, {ge::DT_BF16}},
    {SOFTMAX_LSE_NAME, {ge::DT_FLOAT}},
};

const std::set<ge::Format> FORMAT_SUPPORT_SET = {ge::FORMAT_ND};

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {{ge::DT_UNDEFINED, "DT_UNDEFINED"},
                                                                    {ge::DT_FLOAT, "DT_FLOAT"},
                                                                    {ge::DT_FLOAT16, "DT_FLOAT16"},
                                                                    {ge::DT_INT8, "DT_INT8"},
                                                                    {ge::DT_INT16, "DT_INT16"},
                                                                    {ge::DT_UINT16, "DT_UINT16"},
                                                                    {ge::DT_UINT8, "DT_UINT8"},
                                                                    {ge::DT_INT32, "DT_INT32"},
                                                                    {ge::DT_INT64, "DT_INT64"},
                                                                    {ge::DT_UINT32, "DT_UINT32"},
                                                                    {ge::DT_UINT64, "DT_UINT64"},
                                                                    {ge::DT_BOOL, "DT_BOOL"},
                                                                    {ge::DT_DOUBLE, "DT_DOUBLE"},
                                                                    {ge::DT_BF16, "DT_BFLOAT16"},
                                                                    {ge::DT_INT4, "DT_INT4"},
                                                                    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},
                                                                    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
                                                                    {ge::DT_FLOAT8_E8M0, "DT_FLOAT8_E8M0"},
                                                                    {ge::DT_FLOAT4_E2M1, "DT_FLOAT4_E2M1"}};

class QfaBaseChecker {
public:
    QfaBaseChecker() = default;
    virtual ~QfaBaseChecker() = default;

    virtual ge::graphStatus CheckSinglePara(const QuantFlashAttnTilingInfo &qfaInfo)
    {
        (void)qfaInfo;
        return ge::GRAPH_SUCCESS;
    }
    virtual ge::graphStatus CheckParaExistence(const QuantFlashAttnTilingInfo &qfaInfo)
    {
        (void)qfaInfo;
        return ge::GRAPH_SUCCESS;
    }
    virtual ge::graphStatus CheckFeature(const QuantFlashAttnTilingInfo &qfaInfo)
    {
        (void)qfaInfo;
        return ge::GRAPH_SUCCESS;
    }
    virtual ge::graphStatus CheckMultiPara(const QuantFlashAttnTilingInfo &qfaInfo)
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

    std::string DataTypeToSerialStr(ge::DataType type) const;
    static std::string LayoutToSerialStr(FiaLayout layout);
    static uint32_t GetTypeSize(ge::DataType dtype);
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // QUANT_FLASH_ATTN_BASE_CHECKER_H
