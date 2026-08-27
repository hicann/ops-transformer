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
 * \file qfa_tiling_info.cpp
 * \brief
 */

#include "qfa_tiling_info.h"

namespace optiling {
namespace quant_flash_attn {

std::string QfaLayoutToSerialString(QfaLayout layout)
{
    const std::map<QfaLayout, std::string> layout2Str = {
        {QfaLayout::BSND, "BSND"},       {QfaLayout::BNSD, "BNSD"},       {QfaLayout::TND, "TND"},
        {QfaLayout::PA_BBND, "PA_BBND"}, {QfaLayout::PA_BNBD, "PA_BNBD"}, {QfaLayout::PA_NZ, "PA_NZ"},
        {QfaLayout::LSE_BNS, "LSE_BNS"}, {QfaLayout::LSE_NT, "LSE_NT"},   {QfaLayout::N2TGD, "N2TGD"},
        {QfaLayout::NTD, "NTD"}};

    if (layout2Str.find(layout) != layout2Str.end()) {
        return layout2Str.at(layout);
    }
    return "UNKNOWN";
}

static const std::string QFA_AXIS_SERIAL_STRINGS[] = {"B",  "S",  "N",  "D",  "H",  "T",    "D1",
                                                      "D0", "S1", "S2", "Bn", "Bs", "CONST"};

std::string QfaAxisToSerialString(QfaAxis axis)
{
    uint32_t idx = static_cast<uint32_t>(axis);
    return (idx < sizeof(QFA_AXIS_SERIAL_STRINGS) / sizeof(QFA_AXIS_SERIAL_STRINGS[0])) ? QFA_AXIS_SERIAL_STRINGS[idx] :
                                                                                          "UNKNOWN";
}

std::string QfaQuantModeToSerialString(QfaQuantMode qfaQuantMode)
{
    const std::map<QfaQuantMode, std::string> quantMode2Str = {
        {QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32,
         "A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32"},
        {QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32,
         "A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32"},
        {QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32, "A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32"}};

    if (quantMode2Str.find(qfaQuantMode) != quantMode2Str.end()) {
        return quantMode2Str.at(qfaQuantMode);
    }
    return "UNKNOWN";
}

} // namespace quant_flash_attn
} // namespace optiling
