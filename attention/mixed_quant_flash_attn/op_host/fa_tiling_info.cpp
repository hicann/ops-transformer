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
 * \file fa_tiling_info.cpp
 * \brief
 */

#include "mqfa_fa_tiling_info.h"

namespace optiling {
namespace mixed_quant_flash_attn {

std::string LayoutToSerialString(FaLayout layout)
{
    const std::map<FaLayout, std::string> layout2Str = {{FaLayout::BSND, "BSND"},       {FaLayout::BNSD, "BNSD"},
                                                        {FaLayout::TND, "TND"},         {FaLayout::PA_BBND, "PA_BBND"},
                                                        {FaLayout::PA_BNBD, "PA_BNBD"}, {FaLayout::PA_NZ, "PA_NZ"},
                                                        {FaLayout::LSE_BNS, "LSE_BNS"}, {FaLayout::LSE_NT, "LSE_NT"}};

    if (layout2Str.find(layout) != layout2Str.end()) {
        return layout2Str.at(layout);
    }
    return "UNKNOWN";
}

std::string AxisToSerialString(FaAxis axis)
{
    switch (axis) {
        case FaAxis::B:
            return "B";
        case FaAxis::S:
            return "S";
        case FaAxis::N:
            return "N";
        case FaAxis::D:
            return "D";
        case FaAxis::H:
            return "H";
        case FaAxis::T:
            return "T";
        case FaAxis::D1:
            return "D1";
        case FaAxis::D0:
            return "D0";
        case FaAxis::S1:
            return "S1";
        case FaAxis::S2:
            return "S2";
        case FaAxis::Bn:
            return "Bn";
        case FaAxis::Bs:
            return "Bs";
        case FaAxis::CONST:
            return "CONST";
        default:
            return "UNKNOWN";
    }
}

} // namespace mixed_quant_flash_attn
} // namespace optiling
