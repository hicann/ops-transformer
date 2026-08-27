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
 * \file common_checker_quant_flash_attn.h
 * \brief Common checker for layout, shape, dtype parameters ( 公共参数组)
 */

#ifndef COMMON_CHECKER_QUANT_FLASH_ATTN_H
#define COMMON_CHECKER_QUANT_FLASH_ATTN_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker_quant_flash_attn.h"

namespace optiling {
namespace quant_flash_attn {

class CommonChecker : public QfaBaseChecker {
public:
    CommonChecker() = default;
    ~CommonChecker() override = default;

    ge::graphStatus CheckSinglePara(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckParaExistence(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckFeature(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckMultiPara(const QfaTilingInfo &qfaInfo) override;

private:
    // --- Layout checks ---
    ge::graphStatus CheckSingleParaLayout(const QfaTilingInfo &qfaInfo);

    // --- Dtype/shapeDim checks (SinglePara) ---
    ge::graphStatus CheckSingleParaDtype(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaShapeDim(const QfaTilingInfo &qfaInfo);

    // --- Attr checks (SinglePara) ---
    ge::graphStatus CheckSingleParaSoftmaxScale(const QfaTilingInfo &qfaInfo);

    // --- MultiPara: dtype consistency ---
    ge::graphStatus CheckQuantDataType(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckDtypeConsistency(const QfaTilingInfo &qfaInfo);

    // --- Feature: axis & headNum cross-check ---
    ge::graphStatus CheckAxis(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckHeadNum(const QfaTilingInfo &qfaInfo);
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // COMMON_CHECKER_QUANT_FLASH_ATTN_H
